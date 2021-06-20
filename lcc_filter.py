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

class LCCFilter(object):
    def __init__(self, model):
        # Multi-object filter id
        self.id = 'LCC'
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
        self.prune_threshold = 1e-6 # pruning threshold
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

        # Initial point process cumulants
        c1_update = 0.0
        c2_update = 1.0
        c1_clutter = model.mu_c
        c2_clutter = model.mu_c - model.var_c

        mu_c = model.mu_c
        pdf_c = model.pdf_c
        p_d = model.p_d
        q_d = model.q_d
        p_s = model.p_s
        q_s = model.q_s

        log_mu_c = np.log(model.mu_c)
        log_pdf_c = np.log(model.pdf_c)
        log_p_d = np.log(model.p_d)
        log_q_d = np.log(model.q_d)
        log_p_s = np.log(model.p_s)
        log_q_s = np.log(model.q_s)

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

            # Predict cumulants
            c1_predict = np.sum(w_predict)
            c2_predict = p_s ** 2 * c2_update
        
            # Calculate prior process parameters
            alpha_predict = (p_d * c1_predict + mu_c) ** 2 / (c2_predict + c2_clutter)
            beta_predict = alpha_predict

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
            log_num = np.log(np.complex(alpha_predict + m))
            log_den = np.log(np.complex(beta_predict + p_d * c1_predict + mu_c))
            log_theta_1 = log_num - log_den
            log_theta_2 = log_theta_1 - log_den
            
            # Missed detection term
            log_w_predict = np.log(w_predict)
            log_w_update = log_theta_1 + log_q_d + log_w_predict
            l_update = copy(l_predict)
            m_update = m_predict
            P_update = P_predict
            c2_missed_detections = np.real(np.sum(np.exp(log_theta_2 + 2*(log_q_d + log_w_predict))))
            
            log_w_sq = np.array([])
            if m > 0:
                # Detection terms (m)
                for j in range(m):
                    valid_idx = np.isfinite(log_q_z[:, j])
                    log_w_j = log_p_d + log_w_predict[valid_idx] + log_q_z[valid_idx, j]
                    log_w_j -= np.log(mu_c * pdf_c + np.sum(np.exp(log_w_j)))
                    log_w_sq = np.append(log_w_sq, 2.0 * log_w_j, axis=0)
                    log_w_update = np.hstack([log_w_update, log_w_j])
                    l_update = np.hstack([l_update, l_predict[valid_idx]])
                    m_update = np.hstack([m_update, m_filtered[:, valid_idx, j].reshape(model.n_x, -1)])
                    P_update = np.dstack([P_update, P_filtered[:, :, valid_idx].reshape(model.n_x, model.n_x, -1)])
            w_update = np.zeros(log_w_update.shape)
            w_update[:] = np.real(np.exp(log_w_update))

            # Update cumulants
            c2_detections = np.sum(np.exp(log_w_sq))
            c1_update = np.sum(w_update)
            c2_update = c2_missed_detections - c2_detections

            L_updated = len(w_update)
            self.upd_time += perf_counter() - t_start
            
            # Gaussian mixture management
            t_start = perf_counter()
            gm_prune_with_labels(w_update, m_update, P_update, l_update, self.prune_threshold)
            L_pruned = L_updated - len(w_update)
            gm_cap_with_labels(w_update, m_update, P_update, l_update, 
                self.max_num_of_components, round(c1_update))
            L_capped = L_updated - L_pruned - len(w_update)
            self.mgm_time += perf_counter() - t_start

            # In case all components where removed, reset cumulants
            if len(w_update) == 0:
                c1_update = 0.0
                c2_update = 1.0

            # Estimates extraction
            self.extract_estimates(w_update, m_update, l_update, c1_update, c2_update, k)

            # Display diagnostics
            if self.print_flag:
                cprint(
                    ('k = {:03d}, int = {:06.2f}, crd = {:06.2f}, var = {:06.2f}, ' + 
                    'comp. updated = {:04d}, comp. pruned = {:04d}, comp. capped = {:04d}')
                        .format(
                            k, self.mu[k], self.N[k], self.var[k],
                            L_updated, L_pruned, L_capped), 
                    'cyan')

            # Compose birth model for next step
            set_birth_model(model, Z_ng, unique(l_update))
    
    def extract_estimates(self, w_update, m_update, l_update, c1_update, c2_update, k):
         # Save point process moments
        self.mu[k] = c1_update
        self.var[k] = c1_update + c2_update

        unique_labels = unique(l_update)
        N_k = round(min(abs(c1_update), len(unique_labels)))
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
