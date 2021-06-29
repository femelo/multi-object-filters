# -*- coding: utf-8 -*-
# File: glmb_filter.py                                                                                       #
# Project: Multi-object Filters                                                                              #
# File Created: Wednesday, 9th June 2021 10:46:25 am                                                         #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the Generalized Labeled Multi-Bernoulli filter as proposed in:              #
#                                                                                                            #
# B.-N. Vo, B.-T. Vo, and D. Phung, "Labeled Random Finite Sets and the Bayes Multi-Target Tracking Filter," #
# IEEE Trans Signal Processing, Vol. 62, No. 24, pp. 6554-6567, 2014.                                        #
#                                                                                                            #
# BibTeX entry:                                                                                              #
# @ARTICLE{GLMB2014,                                                                                         #
#  author={B.-T. Vo and B.-N. Vo and D. Phung},                                                              #
#  journal={IEEE Transactions on Signal Processing},                                                         #
#  title={Labeled Random Finite Sets and the Bayes Multi-Target Tracking Filter},                            #
#  year={2014},                                                                                              #
#  month={Dec},                                                                                              #
#  volume={62},                                                                                              #
#  number={24},                                                                                              #
#  pages={6554-6567}}                                                                                        # 
#                                                                                                            # 
# Note 1: no lookahead PHD/CPHD allocation is implemented in this code, a simple proportional weighting      #
# scheme is used for readability.                                                                            #
# Note 2: the simple example used here is the same as in the CB-MeMBer filter code for a quick demonstration #
# and comparison purposes.                                                                                   #
# Note 3: more difficult scenarios require more components/hypotheses (thus exec time) and/or a better       #
# lookahead.                                                                                                 #
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 1:52:42 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
from scipy.stats import chi2
from time import perf_counter
from copy import deepcopy
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements import gate_measurements
from dependencies.kalman_update_multiple import kalman_update_multiple
from dependencies.k_shortest_path_any import k_shortest_wrap_pred
from dependencies.m_best_assignment_update import m_best_assignment_update
from dependencies.log_sum_exp import log_sum_exp

class GLMBComponent(object):
    def __init__(self, w=None, m=None, P=None, l=None, assoc_hist=None):
        if w is None:
            self.w = np.array([0.0])
        else:
            if isinstance(w, float):
                self.w = np.array([w])
            elif isinstance(w, np.ndarray):
                self.w = w
            else:
                self.w = np.array([0.0])
        if m is None:
            self.m = np.array([])
        else:
            if len(m.shape) == 1:
                self.m = m[:, None]
            else:
                self.m = m
        if P is None:
            self.P = np.array([[]])
        else:
            if len(P.shape) == 2:
                self.P = P[:, :, None]
            else:
                self.P = P
        if l is None:
            self.l = ()
        else:
            self.l = l
        if assoc_hist is None:
            self.assoc_hist = []
        else:
            self.assoc_hist = assoc_hist

class GLMBProcess(object):
    def __init__(self, table=None, w=None, I=None, n=None, cdn=None, hashes=None):
        # Track table for GLMB (array of structs for individual tracks)
        if table is None:
            self.table = {}
        else:
            self.table = table
        # Vector of GLMB component/hypothesis weights
        if w is None:
            self.w = np.array([1.0])
        else:
            self.w = w
        # Array of GLMB component/hypothesis labels (labels are indices/entries in track table)
        if I is None:
            self.I = {0: np.array([], dtype=int)}
        else:
            self.I = I
        # Vector of GLMB component/hypothesis cardinalities
        if n is None:
            self.n = np.array([0], dtype=int)
        else:
            self.n = n
        # Cardinality distribution of GLMB (vector of cardinality distribution probabilities)
        if cdn is None:
            self.cdn = np.array([1.0])
        else:
            self.cdn = cdn
        # Hashes for compononent/hypothesis labels
        if hashes is None:
            self.hashes = np.array([str([])], dtype=object)
        else:
            self.hashes = hashes

    def copy(self, process):
        self.table = process.table
        self.w = process.w
        self.I = process.I
        self.n = process.n
        self.cdn = process.cdn
        self.hashes = process.hashes

class GLMBFilter(object):
    def __init__(self, model, gate_flag=True):
        # Multi-object filter id
        self.id = 'GLMB'
        self.has_labels = True
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
        self.labels_map = {}
        self.label_max = 0

        # Filter parameters
        self.H_bth = 5                   # requested number of birth components/hypotheses
        self.H_sur = 3000                # requested number of surviving components/hypotheses
        self.H_upd = 3000                # requested number of updated components/hypotheses
        self.H_max = 3000                # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15       # pruning threshold for components/hypotheses

        self.max_num_of_components = 300 # limit on number of Gaussians
        self.prune_threshold = 1e-5      # pruning threshold
        self.merge_threshold = 4         # merging threshold

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
        p_update = GLMBProcess()
        p_predict = GLMBProcess()
        model = self.model

        # Run recursion
        for k in range(self.K):
            # Prediction
            t_start = perf_counter()
            p_predict.copy(p_update)
            self.predict(p_predict, k)
            self.prd_time += perf_counter() - t_start

            # Gating
            t_start = perf_counter()
            if self.gate_flag:
                m_tracks = np.zeros((model.n_x, 0))
                P_tracks = np.zeros((model.n_x, model.n_x, 0))
                for t_idx in range(len(p_predict.table)):
                    m_tracks = np.append(m_tracks, p_predict.table[t_idx].m, axis=1)
                    P_tracks = np.append(P_tracks, p_predict.table[t_idx].P, axis=2)
                Z_k, _ = gate_measurements(measurement_set.Z[k], self.gamma, model, m_tracks, P_tracks)
            else:
                Z_k = measurement_set.Z[k]
            self.gat_time += perf_counter() - t_start

            # Update
            t_start = perf_counter()
            p_update.copy(p_predict)
            self.update(p_update, Z_k, k)
            L_updated = len(p_update.w)
            self.upd_time += perf_counter() - t_start
            
            # Gaussian mixture management
            t_start = perf_counter()
            self.prune(p_update)
            L_pruned = L_updated - len(p_update.w)
            self.cap(p_update)
            L_capped = L_updated + L_pruned - len(p_update.w)
            self.mgm_time += perf_counter() - t_start

            # Estimates extraction
            self.extract_estimates(p_update, k)

            # Display diagnostics
            if self.print_flag:
                cprint(
                    ('k = {:03d}, int = {:08.5f}, crd = {:08.5f}, var = {:08.5f}, ' + 
                    'comp. updated = {:04d}, comp. pruned = {:04d}, comp. capped = {:04d}')
                        .format(
                            k, self.mu[k], self.N[k], self.var[k],
                            L_updated, L_pruned, L_capped), 
                    'cyan')

    def extract_estimates(self, p_update, k):
        model = self.model
        # Extract estimates via best cardinality,
        # then best component/hypothesis given best cardinality,
        # then best means of tracks given best component/hypothesis and cardinality
        cdn_mean = np.dot(np.arange(len(p_update.cdn)), p_update.cdn)
        self.mu[k] = cdn_mean
        self.var[k] = np.dot(np.arange(len(p_update.cdn))**2, p_update.cdn) - cdn_mean ** 2
        N_k = np.argmax(p_update.cdn)
        m_est = np.zeros((model.n_x, N_k))
        l_est = np.zeros((N_k, ), dtype=object)

        idx_cmp = np.argmax(p_update.w * (p_update.n == N_k).astype(int))
        for i in range(N_k):
            loc_idx = p_update.I[idx_cmp][i]
            idx_trk = np.argmax(p_update.table[loc_idx].w)
            m_est[:, i] = p_update.table[loc_idx].m[:, idx_trk]
            l_i = p_update.table[loc_idx].l
            if l_i in self.labels_map.keys():
                l_est[i] = self.labels_map[l_i]
            else:
                l_est[i] = self.label_max + 1
                self.labels_map[l_i] = self.label_max + 1
                self.label_max += 1

        idx = np.argsort(l_est)
        self.X[k] = m_est[:, idx]
        self.N[k] = N_k
        self.labels[k] = l_est[idx]

    def get_hash(self, I):
        if len(I) == 0:
            h = '*'
        else:
            h = '*'.join([str(item + 1) for item in sorted(I)])
            h += '*'
        return h

    def clean_prediction(self, p_predict):
        # Hash label sets, find unique ones, merge all duplicates
        num_of_components = len(p_predict.w)
        p_predict.hashes = np.zeros((num_of_components, ), dtype=object)
        for h_idx in range(num_of_components):
            p_predict.hashes[h_idx] = self.get_hash(p_predict.I[h_idx])

        u_hashes, inv_idx = np.unique(p_predict.hashes, return_inverse=True)
        n_unique = len(u_hashes)
        
        loc_p = GLMBProcess(
            table=p_predict.table, 
            w=np.zeros((n_unique, )), 
            I={},
            n=np.zeros((n_unique, ), dtype=int), 
            cdn=p_predict.cdn
        )

        for h_idx in range(len(inv_idx)):
            loc_p.w[inv_idx[h_idx]] += p_predict.w[h_idx]
            loc_p.I[inv_idx[h_idx]] = p_predict.I[h_idx]
            loc_p.n[inv_idx[h_idx]] = p_predict.n[h_idx]

        p_predict.copy(loc_p)

    def clean_update(self, p_update):
        # Flag used tracks
        num_of_components = len(p_update.w)
        used_indicator = np.zeros((len(p_update.table), )).astype(bool)
        for h_idx in range(num_of_components):
            used_indicator[p_update.I[h_idx]] = True
        
        track_count = np.sum(used_indicator.astype(int))

        # Remove unused tracks and reindex existing hypotheses/components
        new_indices = -1 * np.ones((len(p_update.table), ), dtype=int)
        new_indices[used_indicator] = np.arange(track_count)
        new_table = dict(
            [
                (new_indices[i], p_update.table[i]) 
                for i in sorted(p_update.table.keys()) if used_indicator[i]
            ]
        )

        I = {}
        hashes = []
        for h_idx in range(num_of_components):
            I[h_idx] = new_indices[p_update.I[h_idx]]
            hashes.append(self.get_hash(I[h_idx]))

        p_update.table = new_table
        p_update.I = I
        p_update.hashes = np.array(hashes, dtype=object)

    def predict(self, p_update, k):
        model = self.model
        # Generate birth hypotheses/components
        # Create birth tracks
        table_birth = {}
        for idx in range(len(self.model.r_birth)):
            table_birth[idx] = GLMBComponent(
                w=model.w1_birth[idx],
                m=model.m_birth[:, idx],
                P=model.P_birth[:, :, idx],
                l=(k, idx), assoc_hist=[])
        # Copy track table back to point process struct
        p_birth = GLMBProcess(table=table_birth)

        # Calculate best birth hypotheses/components
        neg_cost_mat = -np.log(model.r_birth / (1.0 - model.r_birth)) 
        # k-shortest path to calculate k-best births hypotheses/components  
        b_paths, neg_log_costs = k_shortest_wrap_pred(neg_cost_mat, self.H_bth)                                         
        
        # Generate corresponding birth hypotheses/components (VERIFICAR)
        len_paths = len(b_paths.keys())
        p_birth.w = np.zeros((len_paths, ))
        p_birth.n = np.zeros((len_paths, ), dtype=int)
        for idx in sorted(b_paths.keys()):
            birth_hyp_comp = b_paths[idx]
            p_birth.w[idx] = np.sum(np.log(1.0 - model.r_birth)) - neg_log_costs[idx]
            p_birth.I[idx] = np.array(birth_hyp_comp, dtype=int)
            p_birth.n[idx] = len(birth_hyp_comp)
        p_birth.w = np.exp(p_birth.w - log_sum_exp(p_birth.w)) # normalize weights

        # Extract cardinality distribution
        max_n = np.max(p_birth.n) if len(p_birth.n) > 0 else 0
        p_birth.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            p_birth.cdn[n] = np.sum(p_birth.w[p_birth.n == n]) # extract probability of n targets

        # Generate survival hypotheses/components
        table_survival = {}
        for idx in range(len(p_update.table)):
            # Create surviving tracks - via time prediction (single target CK)
            m_predict, P_predict = kalman_predict_multiple(model, p_update.table[idx].m, p_update.table[idx].P)
            table_survival[idx] = GLMBComponent(
                w=p_update.table[idx].w,
                m=m_predict,
                P=P_predict,
                l=p_update.table[idx].l, 
                assoc_hist=p_update.table[idx].assoc_hist)
        # Copy track table back to point process struct
        p_survival = GLMBProcess(table=table_survival)
            
        # Loop over posterior components/hypotheses
        r_idx = 0
        num_of_components = len(p_update.w)
        w_survival = []
        I_survival = {}
        n_survival = []
        for p_idx in range(num_of_components):
            if p_update.n[p_idx] == 0: # no target means no deaths
                w_survival.append(np.log(p_update.w[p_idx]))
                I_survival[r_idx] = p_update.I[p_idx]
                n_survival.append(p_update.n[p_idx])
                r_idx += 1
            else:
                # Calculate best survived hypotheses/components
                neg_cost_mat = -np.log(model.p_s / model.q_s) * np.ones((p_update.n[p_idx], )) 
                # k-shortest path to calculate k-best survived hypotheses/components
                K_s = int(round(self.H_sur * np.sqrt(p_update.w[p_idx]) / np.sum(np.sqrt(p_update.w))))
                s_paths, neg_log_costs = k_shortest_wrap_pred(neg_cost_mat, K_s)  

                for idx in sorted(s_paths.keys()):
                    survival_hyp_comp = np.array(s_paths[idx], dtype=int)
                    loc_w = p_update.n[p_idx] * np.log(model.q_s) \
                        + np.log(p_update.w[p_idx]) - neg_log_costs[idx]
                    w_survival.append(loc_w)
                    I_survival[r_idx] = p_update.I[p_idx][survival_hyp_comp]
                    n_survival.append(len(survival_hyp_comp))
                    r_idx += 1
        p_survival.w = np.array(w_survival)
        p_survival.I = I_survival
        p_survival.n = np.array(n_survival, dtype=int)
        p_survival.w = np.exp(p_survival.w - log_sum_exp(p_survival.w)) # normalize weights

        # Extract survival cardinality distribution
        max_n = np.max(p_survival.n) if len(p_survival.n) > 0 else 0
        p_survival.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            p_survival.cdn[n] = np.sum(p_survival.w[p_survival.n == n]) # extract probability of n targets

        # Generate predicted hypotheses/components (by convolution of birth and survive GLMBs)
        # Concatenate track tables
        table_predict = deepcopy(p_birth.table)
        next_idx = len(table_predict)
        for idx in range(len(p_survival.table)):
            table_predict[next_idx + idx] = p_survival.table[idx]
        n_birth = len(p_birth.w)
        n_survival = len(p_survival.w)
        p_predict = GLMBProcess(
            table=table_predict, 
            w=np.zeros((n_birth * n_survival, )),
            n=np.zeros((n_birth * n_survival, ), dtype=int)
            )

        # Perform convolution - just multiplication
        for b_idx in range(n_birth):
            for s_idx in range(n_survival):
                h_idx = b_idx * len(p_survival.w) + s_idx
                p_predict.w[h_idx] = p_birth.w[b_idx] * p_survival.w[s_idx]
                p_predict.I[h_idx] = np.append(p_birth.I[b_idx], len(p_birth.table) + p_survival.I[s_idx]) 
                p_predict.n[h_idx] = p_birth.n[b_idx] + p_survival.n[s_idx]
        p_predict.w /= np.sum(p_predict.w) # normalize weights

        # Extract predicted cardinality distribution
        max_n = np.max(p_predict.n) if len(p_predict.n) > 0 else 0
        p_predict.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            p_predict.cdn[n] = np.sum(p_predict.w[p_predict.n == n]) # extract probability of n targets

        # Remove duplicate entries and clean track table
        self.clean_prediction(p_predict)
        p_update.copy(p_predict)

    def update(self, p_predict, Z_k, k):
        model = self.model

        # Create updated tracks (single target Bayes update)
        # Number of measurements
        m = Z_k.shape[1]
        prd_table_len = len(p_predict.table)
        table_update = deepcopy(p_predict.table)
        # Missed detection tracks (legacy tracks)
        for t_idx in range(prd_table_len):
            table_update[t_idx].assoc_hist += [-1] # track association history (updated for missed detection)

        # Measurement updated tracks (all pairs)
        all_costs = np.zeros( (prd_table_len, m) )
        for j in range(m):
            for i in range(prd_table_len):
                # Index of predicted track i updated with measurement j is (number_predicted_tracks*j + i)
                t_idx = prd_table_len * (j + 1) + i
                q_z, m_filtered, P_filtered = kalman_update_multiple(
                    Z_k[:, j, None], p_predict.table[i].m, p_predict.table[i].P, model)
                w_ij = q_z.ravel() * p_predict.table[i].w + np.spacing(1)
                table_update[t_idx] = GLMBComponent(
                    w=w_ij / np.sum(w_ij),
                    m=m_filtered[:, :, 0],
                    P=P_filtered,
                    l=p_predict.table[i].l, 
                    assoc_hist=p_predict.table[i].assoc_hist + [j]
                    )
                all_costs[i, j] = np.sum(w_ij)

        p_update = GLMBProcess(table=table_update)

        # Component updates
        num_of_components = len(p_predict.w)
        if m == 0: # no measurements: all missed detections
            # Hypothesis/component weight
            p_update.w = -model.mu_c + p_predict.n * np.log(model.q_d) + np.log(p_predict.w)
            # Hypothesis/component tracks (via indices to track table)
            p_update.I = p_predict.I
            # Hypothesis/component cardinality
            p_update.n = p_predict.n
        else:
            # Loop over predicted components/hypotheses
            r_idx = 0
            w_update = []
            I_update = {}
            n_update = []
            for p_idx in range(num_of_components):
                if p_predict.n[p_idx] == 0: # no target means all clutter
                    loc_w = -model.mu_c + m * np.log(model.mu_c * model.pdf_c) \
                        + np.log(p_predict.w[p_idx])
                    w_update.append(loc_w)
                    I_update[r_idx] = p_predict.I[p_idx]
                    n_update.append(p_predict.n[p_idx])
                    r_idx += 1
                else: # otherwise perform update for component
                    # Calculate best updated hypotheses/components
                    neg_cost_mat = -np.log(model.p_d / model.q_d) -np.log(all_costs[p_predict.I[p_idx], :]) + np.log(model.mu_c * model.pdf_c)
                    K_u = int(round(self.H_upd * np.sqrt(p_predict.w[p_idx]) / np.sum(np.sqrt(p_predict.w))))
                    assignments, neg_log_costs = m_best_assignment_update(neg_cost_mat, K_u)

                    for h_idx in range(len(neg_log_costs)):
                        upd_hyp_comp = assignments[:, h_idx]
                        loc_w = -model.mu_c + m * np.log(model.mu_c * model.pdf_c) \
                            + p_predict.n[p_idx] * np.log(model.q_d) + np.log(p_predict.w[p_idx]) \
                            - neg_log_costs[h_idx]
                        w_update.append(loc_w)
                        I_update[r_idx] = prd_table_len * (upd_hyp_comp + 1) + p_predict.I[p_idx]
                        n_update.append(p_predict.n[p_idx])
                        r_idx += 1
            # Set updated variables
            p_update.w = np.array(w_update)
            p_update.I = I_update
            p_update.n = np.array(n_update, dtype=int)
        p_update.w = np.exp(p_update.w - log_sum_exp(p_update.w)) # normalize weights

        # Extract predicted cardinality distribution
        max_n = np.max(p_update.n) if len(p_update.n) > 0 else 0
        p_update.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            p_update.cdn[n] = np.sum(p_update.w[p_update.n == n]) # extract probability of n targets

        self.clean_update(p_update)
        p_predict.copy(p_update)

    def prune(self, process):
        # Prune components with weights lower than specified threshold
        idx_keep = np.where(process.w > self.hyp_threshold)[0]
        process.w = process.w[idx_keep]
        process.I = dict([(k, process.I[idx]) for k, idx in enumerate(idx_keep)])
        process.n = process.n[idx_keep]

        process.w /= np.sum(process.w)

        max_n = np.max(process.n)
        process.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            process.cdn[n] = np.sum(process.w[process.n == n]) # extract probability of n targets

        return

    def cap(self, process):
        # Cap total number of components to specified maximum
        if len(process.w) > self.H_max:
            idx = np.argsort(-process.w)
            idx_keep = idx[:self.H_max+1]

            process.w = process.w[idx_keep]
            process.I = dict([(k, process.I[idx]) for k, idx in enumerate(idx_keep)])
            process.n = process.n[idx_keep]

            process.w /= np.sum(process.w)

            process.cdn = np.zeros((np.max(process.n), ))
            for n in range(np.max(process.n) + 1):
                process.cdn = np.sum(process.w[process.n == n]) # extract probability of n targets
        return



