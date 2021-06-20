
import numpy as np
from scipy.stats import chi2
from time import perf_counter
from copy import copy
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements import gate_measurements
from dependencies.esf import esf
from dependencies.kalman_update_multiple import kalman_update_multiple
from dependencies.gm_management import gm_prune, gm_merge, gm_cap

VAL_MIN = np.spacing(0)
LOG_VAL_MIN = np.log(VAL_MIN)
REAL_MIN = 2.0 ** -1022.0

class CPHDFilter(object):
    def __init__(self, model, gate_flag=True):
        # Multi-object filter id
        self.id = 'CPHD'
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
        self.prune_threshold = 1e-6 # pruning threshold
        self.merge_threshold = 4    # merging threshold

        # Specific to the CPHD
        self.N_max = 2 * model.num_of_targets

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
        # Reset estimates
        self.reset_estimates()
        # Print flag
        self.print_flag = print_flag

        # Input parameters
        self.K = measurement_set.K

        w_update = np.array([])
        m_update = np.array([[]])
        P_update = np.array([[[]]])
        prd_time = self.prd_time
        gat_time = self.gat_time
        upd_time = self.upd_time
        mgm_time = self.mgm_time
        model = self.model

        # Cardinality
        cdn_update = np.zeros((self.N_max + 1, ))
        cdn_update[0] = 1 # first positions is for the null cardinality (zero targets)
        survive_cdn_predict = np.zeros((self.N_max + 1, ))
        cdn_predict = np.zeros((self.N_max + 1, ))

        # Precompute factors
        log_1_n = np.log(np.arange(1, self.N_max + 1))
        sum_log_1_n = np.zeros((self.N_max, ))
        for n in range(self.N_max):
            sum_log_1_n[n] = np.sum(log_1_n[:n + 1])
        sum_log_0_n = np.zeros((self.N_max + 1, ))
        sum_log_0_n[1:] = sum_log_1_n

        p_d = self.model.p_d
        pdf_c = self.model.pdf_c
        p_s = self.model.p_s
        log_mu_c = np.log(self.model.mu_c)
        log_p_d = np.log(self.model.p_d)
        log_q_d = np.log(self.model.q_d)
        log_p_s = np.log(self.model.p_s)
        log_q_s = np.log(self.model.q_s)

        # Run recursion
        for k in range(self.K):
            # Prediction
            t_start = perf_counter()

            # Intensity prediction
            w_predict = p_s * w_update
            m_predict, P_predict = kalman_predict_multiple(model, m_update, P_update)

            if len(w_predict) > 0:
                w_predict = np.hstack([model.w_birth, w_predict])
                m_predict = np.hstack([model.m_birth, m_predict])
                P_predict = np.dstack([model.P_birth, P_predict])
            else:
                w_predict = model.w_birth
                m_predict = model.m_birth
                P_predict = model.P_birth

            # Cardinality prediction 
            # Surviving cardinality distribution
            for j in range(self.N_max + 1):
                terms = np.zeros((self.N_max + 1, ))
                for l in range(j, self.N_max + 1):
                    terms[l] = cdn_update[l] * np.exp(
                        sum_log_0_n[max(l, 0)] -sum_log_0_n[max(j, 0)] \
                        - sum_log_0_n[max(l - j, 0)] + j*log_p_s + (l - j)*log_q_s
                        )
                survive_cdn_predict[j] = np.sum(terms)

            # Convolution of birth and surviving cardinality distribution
            mu_birth = np.sum(model.w_birth)
            for n in range(self.N_max + 1):
                terms = np.zeros((self.N_max + 1, ))
                for j in range(n + 1):
                    terms[j] = survive_cdn_predict[j] * np.exp(
                        - mu_birth + (n - j)*np.log(mu_birth) \
                        - sum_log_0_n[max(0, n - j)]
                        )
                cdn_predict[n] = np.sum(terms)

            # Normalize predicted cardinality distribution
            cdn_predict /= np.sum(cdn_predict)
            prd_time += perf_counter() - t_start

            # Gating
            t_start = perf_counter()
            if self.gate_flag:
                Z_k, _ = gate_measurements(measurement_set.Z[k], self.gamma, model, m_predict, P_predict)        
            else:
                Z_k = measurement_set.Z[k]
            gat_time += perf_counter() - t_start

            # Update
            t_start = perf_counter()
            # Number of measurements
            m = Z_k.shape[1]

            # Pre-calculation for Kalman update parameters
            if m > 0:
                q_z, m_filtered, P_filtered = kalman_update_multiple(Z_k, m_predict, P_predict, model)


            # Pre-calculation of elementary symmetric functions
            xi_vals = np.zeros((m, ))
            for j in range(m):
                xi_vals[j] = model.p_d * np.dot(w_predict, q_z[:, j]) / model.pdf_c
            
            esf_vals_e = esf(xi_vals) # calculate elementary symmetric functions for entire observation set
            # calculate elementary symmetric functions with each observation index removed one-by-one
            esf_vals_d = np.zeros((m, m)) 
            for j in range(m):
                esf_vals_d[:, j] = esf(np.hstack([xi_vals[:j], xi_vals[(j + 1):m]]))
            
            # Pre-calculation for likelihood factors
            upsilon_0_e = np.zeros((self.N_max + 1, ))
            upsilon_1_e = np.zeros((self.N_max + 1, ))
            upsilon_1_d = np.zeros((self.N_max + 1, m))
            
            log_sum_w_predict = np.log(np.sum(w_predict))
            for n in range(self.N_max + 1):
                # Calcaulate upsilon_0_e[n]
                terms_0_e = np.zeros((min(m, n) + 1, ))
                for j in range(min(m, n) + 1):
                    terms_0_e[j] = esf_vals_e[j] * np.exp(
                        -model.mu_c + (-j)*log_mu_c + sum_log_0_n[max(n, 0)]
                        -sum_log_0_n[max(n - j, 0)] +(n - j)*log_q_d 
                        -j*log_sum_w_predict
                        )
                upsilon_0_e[n]= np.sum(terms_0_e)
                
                # Calcaulate upsilon_1_e[n]
                terms_1_e = np.zeros((min(m, n) + 1, ))
                for j in range(min(m, n) + 1):
                    if n >= j + 1:
                        terms_1_e[j] = esf_vals_e[j] * np.exp(
                            -model.mu_c + (-j)*log_mu_c + sum_log_0_n[max(n, 0)]
                            -sum_log_0_n[max(n - (j + 1), 0)] +(n - (j + 1))*log_q_d
                            -(j + 1)*log_sum_w_predict
                            )
                upsilon_1_e[n]= np.sum(terms_1_e)

                # Calcaulate upsilon_1_d[n, :]
                if m > 0:
                    terms_1_d = np.zeros((min(m - 1, n) + 1, m))
                    for l in range(m):
                        for j in range(min(m - 1, n) + 1):
                            if n >= j + 1:
                                terms_1_d[j, l] = esf_vals_d[j, l] * np.exp(
                                    -model.mu_c + ((-1)-j)*log_mu_c + sum_log_0_n[max(n, 0)]
                                    -sum_log_0_n[max(n - (j + 1), 0)] +(n - (j + 1))*log_q_d
                                    -(j + 1)*log_sum_w_predict)
                upsilon_1_d[n, :] = np.sum(terms_1_d, axis=0)
            

            # Missed detection term
            norm_const = np.dot(upsilon_0_e, cdn_predict)
            w_update = (np.dot(upsilon_1_e, cdn_predict) / norm_const) * \
                model.q_d * w_predict
            m_update = m_predict
            P_update = P_predict
            
            if m > 0:
                # Detection terms (m)
                for j in range(m):
                    w_j = (np.dot(upsilon_1_d[:, j], cdn_predict) / norm_const) * \
                        p_d * q_z[:, j] * w_predict / pdf_c
                    w_update = np.hstack([w_update, w_j])
                    m_update = np.hstack([m_update, m_filtered[:, :, j]])
                    P_update = np.dstack([P_update, P_filtered])   
            
            # Cardinality update
            cdn_update = upsilon_0_e * cdn_predict
            cdn_update /= np.sum(cdn_update)

            L_updated = len(w_update)
            upd_time += perf_counter() - t_start
            
            # Gaussian mixture management
            t_start = perf_counter()
            gm_prune(w_update, m_update, P_update, self.prune_threshold)
            L_pruned = L_updated - len(w_update)
            gm_merge(w_update, m_update, P_update, self.merge_threshold)
            L_merged = L_updated - L_pruned - len(w_update)
            gm_cap(w_update, m_update, P_update, self.max_num_of_components)
            mgm_time += perf_counter() - t_start

            # Estimates extraction
            self.extract_estimates(w_update, m_update, cdn_update, k)

            # Display diagnostics
            if self.print_flag:
                cprint(
                    ('k = {:03d}, int = {:06.2f}, crd = {:06.2f}, var = {:06.2f}, ' + 
                    'comp. updated = {:04d}, comp. pruned = {:04d}, comp. merged = {:04d}')
                        .format(
                            k, self.mu[k], self.N[k], self.var[k],
                            L_updated, L_pruned, L_merged), 
                    'cyan')

    def extract_estimates(self, w_update, m_update, cdn_update, k):
        # Save point process moments
        self.mu[k] = np.sum(w_update)

        # Estimates extraction
        cdn_map = np.argmax(cdn_update)
        N_k = min(len(w_update), cdn_map)
        cdn_mean = np.dot(np.arange(self.N_max + 1), cdn_update)
        self.N[k] = N_k
        self.var[k] = np.dot(np.arange(self.N_max + 1)**2, cdn_update) - cdn_mean ** 2
        idx_comp = np.argsort(-w_update)
        self.X[k] = m_update[:, idx_comp[:N_k]]
