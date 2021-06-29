# -*- coding: utf-8 -*-
# File: dg_filter.py                                                                                         #
# Project: Multi-object Filters                                                                              #
# File Created: Thursday, 10th June 2021 9:05:30 am                                                          #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the Discrete Gamma filter with labels as proposed in:                       #
#                                                                                                            #
# F. E. De Melo and S. Maskell, "A CPHD approximation based on a discrete-Gamma cardinality model,"          #
# IEEE Trans Signal Processing, Vol. 67, No. 2, pp. 336-350, 15 Jan.15, 2019.                                #
#                                                                                                            #
# BibTeX entry:                                                                                              #     
# @ARTICLE{DG2019,                                                                                           #
#  author={De Melo, Flávio Eler and Maskell, Simon},                                                         #
#  journal={IEEE Transactions on Signal Processing},                                                         #
#  title={A CPHD Approximation Based on a Discrete-Gamma Cardinality Model},                                 #
#  year={2019},                                                                                              #
#  volume={67},                                                                                              #
#  number={2},                                                                                               #
#  pages={336-350}}                                                                                          #
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 1:31:21 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
from numpy.lib.arraysetops import unique
from copy import deepcopy
from scipy.stats import chi2
from time import perf_counter
from copy import copy
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements_per_component import gate_measurements_per_component
from dependencies.esf import esf
from dependencies.kalman_update_multiple_per_component import kalman_update_multiple_per_component
from dependencies.gm_management import gm_prune_with_labels, gm_cap_with_labels
from dependencies.set_birth_model import set_birth_model
from dependencies.log_sum_exp import log_sum_exp

VAL_MIN = np.spacing(0)
LOG_VAL_MIN = np.log(VAL_MIN)
REAL_MIN = 2.0 ** -1022.0

class DGFilter(object):
    def __init__(self, model):
        # Multi-object filter id
        self.id = 'DG'
        self.has_labels = True
        # Number of time steps
        self.K = 0
        # Point process model
        self.model = deepcopy(model)

        # Estimates
        self.X = {}
        self.mu = {}
        self.var = {}
        self.N = {}
        self.labels = {}
        self.label_max = 0

        # Filter parameters
        self.max_num_of_components = 300 # limit on number of Gaussians
        self.prune_threshold = 1e-5 # pruning threshold
        self.merge_threshold = 4    # merging threshold

        # Specific to the CPHD
        self.N_max = 2 * model.num_of_targets

        self.p_g = 0.99                              # gate size in percentage
        self.gamma = chi2.ppf(self.p_g, model.n_z)   # inverse chi square cdf
        self.print_flag = False

        self.prd_time = 0.0
        self.gat_time = 0.0
        self.upd_time = 0.0
        self.mgm_time = 0.0
    
    # Reset
    def reset_estimates(self):
        # Number of time steps
        self.K = 0

        # Estimates
        self.X = {}
        self.mu = {}
        self.var = {}
        self.N = {}
        self.labels = {}

        self.prd_time = 0.0
        self.gat_time = 0.0
        self.upd_time = 0.0
        self.mgm_time = 0.0

    # Recursive filtering
    def run(self, measurement_set, print_flag=False):
        # Reset estimates
        self.reset_estimates()
        # Print flag
        self.print_flag = print_flag

        # Input parameters
        self.K = measurement_set.K

        w_update = np.array([])
        m_update = np.array([[]])
        P_update = np.array([[[]]])
        l_update = np.array([], dtype=int)
        model = self.model

        # Initial point process parameters
        a_update = 0.0
        b_update = 1.0
        mu_update = a_update / b_update

        mu_c = model.mu_c
        q_d = model.q_d
        p_s = model.p_s

        log_mu_c = np.log(model.mu_c)
        log_pdf_c = np.log(model.pdf_c)
        log_p_d = np.log(model.p_d)
        log_q_d = np.log(model.q_d)

        # Set initial model birth as null
        model.w_birth = np.array([])
        model.l_birth = np.array([], dtype=int)
        model.m_birth = np.array([[]])
        model.P_birth = np.array([[[]]])

        # Run recursion
        for k in range(self.K):
            # Prediction
            t_start = perf_counter()

            # Intensity prediction
            w_predict = p_s * w_update
            l_predict = l_update
            m_predict, P_predict = kalman_predict_multiple(model, m_update, P_update)

            if len(w_predict) > 0 and len(model.w_birth) > 0:
                w_predict = np.hstack([model.w_birth, w_predict])
                l_predict = np.hstack([model.l_birth, l_predict])
                m_predict = np.hstack([model.m_birth, m_predict])
                P_predict = np.dstack([model.P_birth, P_predict])
            elif len(model.w_birth) > 0:
                w_predict = copy(model.w_birth)
                l_predict = copy(model.l_birth)
                m_predict = copy(model.m_birth)
                P_predict = copy(model.P_birth)
            else:
                pass

            # Cardinality prediction 
            # Predicted number of targets
            mu_predict = np.sum(model.w_birth) + p_s * a_update / b_update
            # Predicted variance on number of targets
            var_predict = mu_predict + (p_s ** 2) * mu_update * (1.0 / b_update - 1.0)
        
            # Predict parameters
            if mu_predict == 0.0 and var_predict == 0.0:
                a_predict = VAL_MIN
                b_predict = 1.0
            else:
                a_predict = (mu_predict ** 2) / var_predict
                b_predict = mu_predict / var_predict
            self.prd_time += perf_counter() - t_start

            # Gating
            t_start = perf_counter()
            Z_g, Z_ng, valid_meas, innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat = \
                gate_measurements_per_component(
                    measurement_set.Z[k], 
                    self.gamma, model, 
                    m_predict, P_predict)
            self.gat_time += perf_counter() - t_start

            # Update
            t_start = perf_counter()
            # Number of measurements
            m = Z_g.shape[1]

            # Pre-calculation for Kalman update parameters
            if m > 0:
                log_q_z, m_filtered, P_filtered = \
                    kalman_update_multiple_per_component(
                        Z_g, m_predict, P_predict, model,
                        innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat, 
                        log_likelihood=True)

            # Pre-calculation of factors
            log_w_predict = np.log(w_predict)
            log_factor_pred = np.log(np.complex(a_predict + VAL_MIN)) - np.log(np.complex(b_predict))
            factor_pred = a_predict / b_predict
            xi_vals = np.zeros((m, ))
            for j in range(m):
                log_q_z_j = copy(log_q_z[:, j])
                log_q_z_j[np.isnan(log_q_z_j)] = LOG_VAL_MIN
                xi_vals[j] = np.exp(log_p_d + log_sum_exp(log_q_z_j + log_w_predict) - log_pdf_c) + VAL_MIN
            
            # Obtain log of elementary symmetric functions
            v = xi_vals / (factor_pred * mu_c)
            log_e_s = np.log(esf(v) + VAL_MIN)

            # Compute polynomial terms used to calculate coefficients
            log_Theta = self.compute_theta(m + 2, q_d, a_predict, b_predict, mu_predict)
            
            # Compute coefficients
            # Indexes
            u_00 = np.arange(m + 1).astype(int)
            u_11 = u_00 + 1
            u_22 = u_00 + 2

            log_terms_num_q = log_Theta[u_11] + log_e_s
            log_terms_den_q = log_Theta[u_00] + log_e_s
            log_sum_terms_den_q = log_sum_exp(log_terms_den_q)
            
            log_L_q = log_sum_exp(log_terms_num_q) - log_sum_terms_den_q

            # Coefficients for computing cardinality moments
            log_0_m = np.log(np.arange(m + 1).astype(float) + VAL_MIN)
            log_terms_num_p = log_0_m + log_terms_den_q
            log_l_p =  log_sum_exp(log_terms_num_p) - log_sum_terms_den_q
            
            log_terms_num_r = 2 * log_0_m + log_terms_den_q
            log_l_r =  log_sum_exp(log_terms_num_r) - log_sum_terms_den_q
            
            log_terms_num_q2 = log_0_m + log_terms_num_q
            log_l_q = log_sum_exp(log_terms_num_q2) - log_sum_terms_den_q
            
            log_terms_num_s = log_Theta[u_22] + log_e_s
            log_l_s = log_sum_exp(log_terms_num_s) - log_sum_terms_den_q

            log_L_p = np.zeros((m, 1)) + LOG_VAL_MIN
            u_10 = np.arange(m).astype(int) + 1
            if m > 1:
                for j in range(m):
                    inds = np.array([i for i in range(j)] + [i for i in range(j + 1, m)], dtype=int)
                    v = xi_vals[inds] / (factor_pred * mu_c)
                    log_e_s = np.log(esf(v) + REAL_MIN)
                    log_terms_num = log_Theta[u_10] + log_e_s
                    log_L_p[j] = log_sum_exp(log_terms_num) - log_sum_terms_den_q
            elif m == 1:
                log_e_s = 0.0
                log_terms_num = log_Theta[u_10] + log_e_s
                log_L_p[0] = log_terms_num - log_sum_terms_den_q
            else:
                pass

            # Missed detection term
            if len(l_predict) > 0:
                log_w_update = log_L_q - log_factor_pred + log_q_d + log_w_predict
            else:
                log_w_update = np.array([])
            l_update = copy(l_predict)
            m_update = m_predict
            P_update = P_predict
            
            if m > 0:
                # Detection terms (m)
                for j in range(m):
                    valid_idx = np.isfinite(log_q_z[:, j])
                    log_w_j = log_L_p[j] - log_factor_pred + log_p_d + log_q_z[valid_idx, j] \
                        - log_pdf_c - log_mu_c + log_w_predict[valid_idx]
                    log_w_update = np.hstack([log_w_update, log_w_j])
                    l_update = np.hstack([l_update, l_predict[valid_idx]])
                    m_update = np.hstack([m_update, m_filtered[:, valid_idx, j].reshape(model.n_x, -1)])
                    P_update = np.dstack([P_update, P_filtered[:, :, valid_idx].reshape(model.n_x, model.n_x, -1)])

            w_update = np.zeros(log_w_update.shape)
            w_update[:] = np.real(np.exp(log_w_update))

            # Cardinality update
            # Updated number of targets
            mu_update = np.exp(log_l_p) + np.exp(log_L_q + log_q_d)
            # Updated variance of number of targets
            var_update = np.exp(log_l_r) - np.exp(log_l_p) + 2*np.exp(log_l_q + log_q_d) \
                + np.exp(log_l_s + 2*log_q_d) - mu_update**2 + mu_update
            var_update = max(var_update, VAL_MIN)
            
            # Updated parameters of the discrete gamma distribution
            a_update = (mu_update ** 2) / var_update
            b_update = mu_update / var_update

            L_updated = len(w_update)
            self.upd_time += perf_counter() - t_start
            
            # Gaussian mixture management
            t_start = perf_counter()
            gm_prune_with_labels(w_update, m_update, P_update, l_update, self.prune_threshold)
            L_pruned = L_updated - len(w_update)
            gm_cap_with_labels(w_update, m_update, P_update, l_update, 
                self.max_num_of_components, round(mu_update))
            L_capped = L_updated - L_pruned - len(w_update)
            self.mgm_time += perf_counter() - t_start

            if len(w_update) == 0:
                a_update = 0.0
                b_update = 1.0

            # Estimates extraction
            self.extract_estimates(w_update, m_update, l_update, mu_update, var_update, k)

            # Display diagnostics
            if self.print_flag:
                cprint(
                    ('k = {:03d}, int = {:08.5f}, crd = {:08.5f}, var = {:08.5f}, ' + 
                    'comp. updated = {:04d}, comp. pruned = {:04d}, comp. capped = {:04d}')
                        .format(
                            k, self.mu[k], self.N[k], self.var[k],
                            L_updated, L_pruned, L_capped), 
                    'cyan')

            # Compose birth model for next step
            set_birth_model(model, Z_ng, unique(l_update))
    
    def extract_estimates(self, w_update, m_update, l_update, mu_update, var_update, k):
         # Save point process moments
        self.mu[k] = mu_update
        self.var[k] = var_update

        unique_labels = unique(l_update)
        N_k = round(min(mu_update, len(unique_labels)))
        idx_comp = np.argsort(-w_update)
        m_est = np.zeros((self.model.n_x, N_k))
        l_est = np.zeros((N_k, ), dtype=int)
        i = 0
        j = 0
        while np.any(l_est == 0):
            if not l_update[idx_comp[j]] in l_est:
                m_est[:, i] = m_update[:, idx_comp[j]]
                l_est[i] = l_update[idx_comp[j]]
                i += 1
            j += 1
        idx = np.argsort(l_est)
        self.X[k] = m_est[:, idx]
        self.N[k] = N_k
        self.labels[k] = l_est[idx]
        self.label_max = max(l_est.tolist() + [self.label_max])

    def compute_theta(self, r, s, alpha, beta, N):
        log_s = np.log(s)
        log_z = - beta + log_s
        log_Psi = np.zeros((r + 1,)).astype('float64') + LOG_VAL_MIN
        
        # Number of terms to approximate    
        epsilon = REAL_MIN
        nu = N
        if nu > VAL_MIN:
            for k in range(30):
                nu -= ( (alpha - 1.0) * np.log(nu) -beta * nu - np.log(epsilon)) / (2*((alpha - 1.0) / nu - beta))
        else:
            nu = r

        lb = 1
        if np.isnan(nu):
            ub = r
        else:
            ub = max(int(np.real(nu)), r)
        
        n = np.arange(lb, ub + 1)
        n_log_z = n * log_z
        log_n = np.log(n)
        am1_log_n = (alpha - 1.0) * log_n
        theta = am1_log_n + n_log_z
        theta_max = np.max(theta)
        theta_min = np.min(theta)
        d_theta = theta - (theta_min + theta_max) / 2
        
        log_Psi[0] = log_sum_exp(d_theta)
        
        # Loop
        log_s_j = log_s
        log_n_j = log_n
        for j in range(r):
            log_Psi[j + 1] = log_sum_exp(d_theta + log_n_j - log_s_j)
            log_n_j = log_n_j + np.log(np.maximum(n - (j + 1), VAL_MIN))
            log_s_j = log_s_j + log_s
        
        # Normalize and return
        idx_max = np.argmax(np.abs(log_Psi))
        log_Theta = log_Psi - log_Psi[idx_max]
        return log_Theta
