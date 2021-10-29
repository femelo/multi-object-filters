# -*- coding: utf-8 -*-
# File: phd_filter.py                                                                                        #
# Project: Multi-object Filters                                                                              #
# File Created: Monday, 7th June 2021 9:16:17 am                                                             #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the Gaussian mixture PHD filter as proposed in:                             #
#                                                                                                            #
# B.-N. Vo, and W. K. Ma, "The Gaussian mixture Probability Hypothesis Density Filter,"                      #
# IEEE Trans Signal Processing, Vol. 54, No. 11, pp. 4091-4104, 2006.                                        #
#                                                                                                            #
# BibTeX entry:                                                                                              #
# @ARTICLE{PHD2006,                                                                                          #
#  author={B.-N. Vo and W.-K. Ma},                                                                           #
#  journal={IEEE Transactions on Signal Processing},                                                         #
#  title={The Gaussian Mixture Probability Hypothesis Density Filter},                                       #
#  year={2006},                                                                                              #
#  month={Nov},                                                                                              #
#  volume={54},                                                                                              #
#  number={11},                                                                                              #
#  pages={4091-4104}}                                                                                        # 
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 1:41:10 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
from scipy.stats import chi2
from time import perf_counter
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements import gate_measurements
from dependencies.kalman_update_multiple import kalman_update_multiple
from dependencies.gm_management import gm_prune, gm_merge, gm_cap

class PHDFilter(object):
    def __init__(self, model, gate_flag=True):
        # Multi-object filter id
        self.id = 'PHD'
        self.has_labels = False
         # Number of time steps
        self.K = 0
        # Point process model
        self.model = model

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

        self.p_g = 0.99                              # gate size in percentage
        self.gamma = chi2.ppf(self.p_g, model.n_z)   # inverse chi square cdf
        self.gate_flag = gate_flag                   # gating on or off 1/0
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
        # Reset internal state variables
        self.reset_estimates()
        # Print flag
        self.print_flag = print_flag

        # Input parameters
        self.K = measurement_set.K

        # Initialize parameters
        w_update = np.array([])
        m_update = np.array([[]])
        P_update = np.array([[[]]])
        model = self.model

        # Run recursion
        for k in range(self.K):
            # Prediction
            t_start = perf_counter()
            w_predict = self.model.p_s * w_update
            m_predict, P_predict = kalman_predict_multiple(model, m_update, P_update)

            if len(w_predict) > 0:
                m_predict = np.hstack([model.m_birth, m_predict])
                P_predict = np.dstack([model.P_birth, P_predict])
                w_predict = np.hstack([model.w_birth, w_predict])
            else:
                m_predict = model.m_birth
                P_predict = model.P_birth
                w_predict = model.w_birth

            self.prd_time += (perf_counter() - t_start)

            # Gating
            t_start = perf_counter()
            if self.gate_flag:
                Z_k, _ = gate_measurements(measurement_set.Z[k], self.gamma, model, m_predict, P_predict)
            else:
                Z_k = measurement_set.Z[k]
            self.gat_time += (perf_counter() - t_start)

            # Update
            t_start = perf_counter()
            # Number of measurements
            m = Z_k.shape[1]

            # Missed detection term
            w_update = model.q_d * w_predict
            m_update = m_predict
            P_update = P_predict

            if m > 0:
                # Detection terms (m)
                q_z, m_filtered, P_filtered = kalman_update_multiple(Z_k, m_predict, P_predict, model)
                for j in range(m):
                    w_j = model.p_d * w_predict * q_z[:, j]
                    w_j /= (model.mu_c * model.pdf_c + np.sum(w_j))
                    w_update = np.hstack([w_update, w_j])
                    m_update = np.hstack([m_update, m_filtered[:, :, j]])
                    P_update = np.dstack([P_update, P_filtered])

            L_updated = len(w_update)
            self.upd_time += (perf_counter() - t_start)
            
            # Gaussian mixture management
            t_start = perf_counter()
            gm_prune(w_update, m_update, P_update, self.prune_threshold)
            L_pruned = L_updated - len(w_update)
            gm_merge(w_update, m_update, P_update, self.merge_threshold)
            L_merged = L_updated - L_pruned - len(w_update)
            gm_cap(w_update, m_update, P_update, self.max_num_of_components)
            self.mgm_time += (perf_counter() - t_start)

            # Estimates extraction
            self.extract_estimates(w_update, m_update, k)

            # Display diagnostics
            if self.print_flag:
                cprint(
                    ('k = {:03d}, int = {:08.5f}, crd = {:08.5f}, var = {:08.5f}, ' + 
                    'comp. updated = {:04d}, comp. pruned = {:04d}, comp. merged = {:04d}')
                        .format(
                            k, self.mu[k], self.N[k], self.var[k],
                            L_updated, L_pruned, L_merged), 
                    'cyan')

    def extract_estimates(self, w_update, m_update, k):
        # Save point process moments
        self.mu[k] = np.sum(w_update)
        self.var[k] = self.mu[k]

        idx = np.where(w_update > 0.5)[0]
        X_k = np.array([[]])
        N_k = 0
        for i in idx:
            N_i = int(round(w_update[i]))
            if X_k.shape[1] == 0:
                if N_i <= 1:
                    X_k = m_update[:, i, None]
                else:
                    X_k = np.hstack(N_i*[m_update[:, i, None]])
            else:
                X_k = np.hstack([X_k] + N_i*[m_update[:, i, None]])
            N_k += N_i
        self.X[k] = X_k
        self.N[k] = N_k
            
