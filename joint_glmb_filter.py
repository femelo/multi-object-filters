# -*- coding: utf-8 -*-
# File: joint_glmb_filter.py                                                                                 #
# Project: Multi-object Filters                                                                              #
# File Created: Monday, 21st June 2021 5:33:33 pm                                                            #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the Generalized Labeled Multi-Bernoulli filter with joint prediction and    #
# update as proposed in:                                                                                     #
#                                                                                                            #
# B.-T. Vo, and B.-N. Vo and H. Hung, "An Efficient Implementation of the Generalized Labeled                #
# Multi-Bernoulli Filter," IEEE Trans Signal Processing, Vol. 65, No. 8, pp. 1975-1987, 2017.                #
#                                                                                                            #
# BibTeX entry:                                                                                              #
# @ARTICLE{JGLMB2017,                                                                                        #
#  author={B.-N. Vo and B.-T. Vo and H. Hung},                                                               #
#  journal={IEEE Transactions on Signal Processing},                                                         #
#  title={An Efficient Implementation of the Generalized Labeled Multi-Bernoulli Filter},                    #
#  year={2017},                                                                                              #
#  month={Apr},                                                                                              #
#  volume={65},                                                                                              #
#  number={8},                                                                                               #
#  pages={1975-1987}}                                                                                        # 
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 1:59:15 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
from scipy.stats import chi2
from time import perf_counter
from copy import deepcopy
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements_per_component import gate_measurements_per_component
from dependencies.kalman_update_multiple import kalman_update_multiple
from dependencies.kalman_update_multiple_per_component import kalman_update_multiple_per_component
from dependencies.m_best_assignment_update import m_best_assignment_gibbs_sampling
from dependencies.log_sum_exp import log_sum_exp

VAL_MIN = np.spacing(1)
LOG_VAL_MIN = np.log(VAL_MIN)
REAL_MIN = 2.0 ** -1022.0

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
        
        # Attributes for Kalman update
        self.gated_mask = None
        self.gated_indexes = None
        self.innov_vec = None
        self.sqrt_innov_cov_mat = None
        self.inv_sqrt_innov_cov_mat = None

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

class JointGLMBFilter(object):
    def __init__(self, model, gate_flag=True):
        # Multi-object filter id
        self.id = 'JGLMB'
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
        self.labels_orig = {}
        self.assoc_hist = {}
        self.hashes = {}

        # Filter parameters
        self.H_upd = 1000                # requested number of updated components/hypotheses
        self.H_max = 1000                # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15       # pruning threshold for components/hypotheses

        self.max_num_of_components = 300 # limit on number of Gaussians
        self.prune_threshold = 1e-5      # pruning threshold
        self.merge_threshold = 4         # merging threshold

        self.p_g = 0.9999999                         # gate size in percentage
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

        # Run recursion
        for k in range(self.K):
            # Prediction and update
            self.jointly_predict_and_update(p_update, measurement_set, k)

            L_updated = len(p_update.w)
            
            # Gaussian mixture management
            t_start = perf_counter()
            self.prune(p_update)
            L_pruned = L_updated - len(p_update.w)
            self.cap(p_update)
            L_capped = L_updated - L_pruned - len(p_update.w)
            self.mgm_time += perf_counter() - t_start

            # Estimates extraction
            self.extract_estimates_recursive(p_update, measurement_set, k)
            # self.extract_estimates(p_update, k)

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

    def extract_estimates_recursive(self, p_update, measurement_set, k):
        # Extract estimates via recursive estimator, where  
        # trajectories are extracted via association history, and
        # track continuity is guaranteed with a non-trivial estimator
        model = self.model
        cdn_mean = np.dot(np.arange(len(p_update.cdn)), p_update.cdn)
        self.mu[k] = cdn_mean
        self.var[k] = np.dot(np.arange(len(p_update.cdn))**2, p_update.cdn) - cdn_mean ** 2

        # Extract MAP cardinality and corresponding highest weighted component
        N_k = np.argmax(p_update.cdn)
        idx_cmp = np.argmax(p_update.w * (p_update.n == N_k).astype(int))
        loc_assoc_hist = {}
        loc_labels = {}
        for i in range(N_k):
            t_idx = p_update.I[idx_cmp][i]
            loc_assoc_hist[i] = p_update.table[t_idx].assoc_hist
            loc_labels[i] = p_update.table[t_idx].l
        
        loc_hashes = {}
        for i  in range(N_k):
            loc_hashes[i] = str(loc_labels[i][0]) + '.' + str(loc_labels[i][1])

        # Compute dead & updated & new tracks
        i_s = []
        i_n = []
        idx = []
        hashes_pool_s = []
        hashes_pool_n = []
        for i in range(len(loc_hashes)):
            if loc_hashes[i] in self.hashes.values() and not loc_hashes[i] in hashes_pool_s:
                i_s.append(i)
                hashes_pool_s.append(loc_hashes[i])
            if not loc_hashes[i] in self.hashes.values() and not loc_hashes[i] in hashes_pool_n:
                i_n.append(i)
                hashes_pool_n.append(loc_hashes[i])
        hashes_pool_n = []
        for i in range(len(self.hashes)):
            if not self.hashes[i] in loc_hashes.values() and not self.hashes[i] in hashes_pool_n:
                idx.append(i)
                hashes_pool_n.append(self.hashes[i])

        new_assoc_hist = []
        new_labels = []
        new_hashes = []
        for i in range(len(idx) + len(i_s) + len(i_n)):
            if i < len(idx):
                new_assoc_hist.append((i, self.assoc_hist[idx[i]]))
                new_labels.append((i, self.labels_orig[idx[i]]))
                new_hashes.append((i, self.hashes[idx[i]]))
            elif i >= len(idx) and i < len(idx) + len(i_s):
                new_assoc_hist.append((i, loc_assoc_hist[i_s[i - len(idx)]]))
                new_labels.append((i, loc_labels[i_s[i - len(idx)]]))
                new_hashes.append((i, loc_hashes[i_s[i - len(idx)]]))
            else:
                new_assoc_hist.append((i, loc_assoc_hist[i_n[i - len(idx) - len(i_s)]]))
                new_labels.append((i, loc_labels[i_n[i - len(idx) - len(i_s)]]))
                new_hashes.append((i, loc_hashes[i_n[i - len(idx) - len(i_s)]]))

        self.assoc_hist = dict(new_assoc_hist)
        self.labels_orig = dict(new_labels)
        self.hashes = dict(new_hashes)

        # Write out estimates in standard format
        X = dict([(k_, np.zeros((model.n_x, 0))) for k_ in range(k + 1)])
        N = dict([(k_, 0) for k_ in range(k + 1)])
        labels = dict([(k_, []) for k_ in range(k + 1)])
        for t_idx in range(len(self.assoc_hist)):
            k_s, b_idx = self.labels_orig[t_idx]
            assoc_hist = self.assoc_hist[t_idx]

            w = model.w_birth[b_idx]
            m = model.m_birth[:, b_idx, None]
            P = model.P_birth[:, :, b_idx, None]
            for i in range(len(assoc_hist)):
                m, P = kalman_predict_multiple(model, m, P)
                k_ = k_s + i
                j = assoc_hist[i]
                if j > 0:
                    q_z, m_upd, P_upd = kalman_update_multiple(measurement_set.Z[k_][:, j, None], m, P, model)
                    m = m_upd[:, :, 0]
                    P = P_upd
                    w = q_z * w + VAL_MIN
                    w /= np.sum(w)

                idx_trk = np.argmax(w)
                N[k_] += 1
                X[k_] = np.append(X[k_].reshape(model.n_x, -1), m[:, idx_trk, None], axis=1)

                if (k_s, b_idx) in self.labels_map.keys():
                    labels[k_].append(self.labels_map[(k_s, b_idx)])
                else:
                    labels[k_].append(self.label_max + 1)
                    self.labels_map[(k_s, b_idx)] = self.label_max + 1
                    self.label_max += 1
        self.N = N
        self.X = X
        self.labels = labels

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

    def jointly_predict_and_update(self, p_update, measurement_set, k):
        model = self.model
        Z_k = measurement_set.Z[k]

        # Generate next update
        # Start timer for prediction
        t_start = perf_counter()
        # Create birth tracks
        table_birth = {}
        for idx in range(len(self.model.r_birth)):
            table_birth[idx] = GLMBComponent(
                w=model.w_birth[idx],
                m=model.m_birth[:, idx],
                P=model.P_birth[:, :, idx],
                l=(k, idx), assoc_hist=[])

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

        # Concatenate track tables of birth and survival
        table_predict = deepcopy(table_birth)
        next_idx = len(table_predict)
        for idx in range(len(table_survival)):
            table_predict[next_idx + idx] = table_survival[idx]
        # Save (accumulated) prediction time
        self.prd_time += perf_counter() - t_start

        # Gating
        # Start timer for gating
        t_start = perf_counter()
        if self.gate_flag:
            m_tracks = np.zeros((model.n_x, 0))
            P_tracks = np.zeros((model.n_x, model.n_x, 0))
            for t_idx in range(len(table_predict)):
                m_tracks = np.append(m_tracks, table_predict[t_idx].m, axis=1)
                P_tracks = np.append(P_tracks, table_predict[t_idx].P, axis=2)
            Z_g, _, valid_measurements, innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat = \
                gate_measurements_per_component(
                    Z_k, 
                    self.gamma, model, 
                    m_tracks, P_tracks, truncate_innovation=False)
            for t_idx in range(len(table_predict)):
                table_predict[t_idx].gated_mask = valid_measurements[t_idx]
                table_predict[t_idx].gated_indexes = np.where(valid_measurements[t_idx])[0]
                table_predict[t_idx].innov_vec = innov_vec[t_idx]
                table_predict[t_idx].sqrt_innov_cov_mat = sqrt_innov_cov_mat[t_idx]
                table_predict[t_idx].inv_sqrt_innov_cov_mat = inv_sqrt_innov_cov_mat[t_idx]
        else:
            Z_g = Z_k
            for t_idx in range(len(table_predict)):
                table_predict[t_idx].gated_mask = np.ones((Z_g.shape[1], ), dtype=bool)
                table_predict[t_idx].gated_indexes = np.arange(Z_g.shape[1])
                table_predict[t_idx].innov_vec = None
                table_predict[t_idx].sqrt_innov_cov_mat = None
                table_predict[t_idx].inv_sqrt_innov_cov_mat = None
        # Save (accumulated) gating time
        self.gat_time += perf_counter() - t_start

        # Start timer for update
        t_start = perf_counter()
        # Copy predicted table to a point process struct
        p_predict = GLMBProcess(table=table_predict)

        # Pre-calculation of average survival/death probabilities
        avg_p_s = np.append(model.r_birth, np.zeros((len(p_update.table), )))
        for t_idx in range(len(p_update.table)):
            avg_p_s[model.L_birth + t_idx] = model.p_s
        avg_q_s = 1.0 - avg_p_s

        # Pre-calculation of average detection/missed probabilities
        avg_p_d = np.zeros((len(p_predict.table), ))
        for t_idx in range(len(p_predict.table)):
            avg_p_d[t_idx] = model.p_d
        avg_q_d = 1.0 - avg_p_d

        # Create updated tracks (single target Bayes update)
        # m = Z_g.shape[1] # number of measurements
        m = Z_k.shape[1]

        prd_table_len = len(p_predict.table)
        table_update = deepcopy(p_predict.table)
        # Missed detection tracks (legacy tracks)
        for t_idx in range(prd_table_len):
            table_update[t_idx].assoc_hist += [-1] # track association history (updated for missed detection)

        # Measurement updated tracks (all pairs)
        all_costs = np.zeros( (prd_table_len, m) )
        for i in range(prd_table_len):
            # For non-gated measurements, add a null table element
            for j in [index for index in range(m) if not index in p_predict.table[i].gated_indexes]:
                t_idx = prd_table_len * (j + 1) + i
                table_update[t_idx] = None
            # For gated measurements, compute and save the new components
            for j in p_predict.table[i].gated_indexes:
                # Index of predicted track i updated with measurement j is (number_predicted_tracks*j + i)
                t_idx = prd_table_len * (j + 1) + i
                # Update component
                if self.gate_flag:
                    q_z, m_filtered, P_filtered = kalman_update_multiple_per_component(
                        Z_k[:, j, None], p_predict.table[i].m, p_predict.table[i].P, model, 
                        [p_predict.table[i].innov_vec[:, j, None]], 
                        [p_predict.table[i].sqrt_innov_cov_mat], 
                        [p_predict.table[i].inv_sqrt_innov_cov_mat])
                else:
                    q_z, m_filtered, P_filtered = kalman_update_multiple(
                        Z_k[:, j, None], p_predict.table[i].m, p_predict.table[i].P, model)
                w_ij = q_z.ravel() * p_predict.table[i].w + VAL_MIN
                table_update[t_idx] = GLMBComponent(
                    w=w_ij / np.sum(w_ij),
                    m=m_filtered[:, :, 0],
                    P=P_filtered,
                    l=p_predict.table[i].l, 
                    assoc_hist=p_predict.table[i].assoc_hist + [j]
                    )
                all_costs[i, j] = np.sum(w_ij)

        p_next_update = GLMBProcess(table=table_update)

        # Joint cost matrix
        joint_cost_matrix = np.hstack(
            [np.diag(avg_q_s), 
            np.diag(avg_p_s * avg_q_d), 
            np.outer(avg_p_s * avg_p_d, np.ones((m, ))) * all_costs / (model.mu_c * model.pdf_c)])

        # Gated measurement mask
        gated_measurement_indexes = {}
        for t_idx in range(prd_table_len):
            gated_measurement_indexes[t_idx] = p_predict.table[t_idx].gated_indexes
                
        # Component updates
        r_idx = 0
        w_update = []
        I_update = {}
        n_update = []
        for p_idx in range(len(p_update.w)):
            #calculate best updated hypotheses/components
            n_pred = prd_table_len
            n_birth = model.L_birth
            n_exist = len(p_update.I[p_idx])
            n_track = n_birth + n_exist
            t_indices = np.append(np.arange(n_birth), n_birth + p_update.I[p_idx])
            all_indexes = np.hstack([gated_measurement_indexes[t_idx] for t_idx in range(prd_table_len)])
            m_indices = np.unique(all_indexes)
            c_indices = np.hstack([t_indices, n_pred + t_indices, 2 * n_pred + m_indices])
            cost_matrix = joint_cost_matrix[np.ix_(t_indices, c_indices)]
            neg_log_cost_matrix = np.inf * np.ones(cost_matrix.shape)
            neg_log_cost_matrix[cost_matrix > 0] = -np.log(cost_matrix[cost_matrix > 0])
            K_u = int(round(self.H_upd * np.sqrt(p_update.w[p_idx])/np.sum(np.sqrt(p_update.w))))
            assignments, neg_log_costs = m_best_assignment_gibbs_sampling(neg_log_cost_matrix, K_u)
            assignments[assignments < n_track] = -np.inf
            assignments[np.logical_and(assignments >= n_track, assignments < 2 * n_track)] = -1
            assignments[assignments >= 2 * n_track] = assignments[assignments >= 2 * n_track] - 2 * n_track
            assignments[assignments >= 0] = m_indices[assignments[assignments >= 0].astype(int)]

            # Generate corrresponding jointly predicted/updated hypotheses/components
            for h_idx in range(len(neg_log_costs)):
                upd_hyp_comp = assignments[h_idx, :]
                upd_hyp_idx = n_pred * (upd_hyp_comp + 1) + np.append(np.arange(n_birth), n_birth + p_update.I[p_idx])
                loc_w = -model.mu_c + m * np.log(model.mu_c * model.pdf_c) \
                    + np.log(p_update.w[p_idx]) - neg_log_costs[h_idx]
                w_update.append(loc_w)
                I_update[r_idx] = upd_hyp_idx[upd_hyp_idx >= 0].astype(int)
                n_update.append(np.sum((upd_hyp_idx >= 0).astype(int)))
                r_idx += 1
        
        # Set updated variables
        p_next_update.w = np.array(w_update)
        p_next_update.I = I_update
        p_next_update.n = np.array(n_update, dtype=int)
        p_next_update.w = np.exp(p_next_update.w - log_sum_exp(p_next_update.w)) # normalize weights

        # Extract predicted cardinality distribution
        max_n = np.max(p_next_update.n) if len(p_next_update.n) > 0 else 0
        p_next_update.cdn = np.zeros((max_n + 1, ))
        for n in range(max_n + 1):
            p_next_update.cdn[n] = np.sum(p_next_update.w[p_next_update.n == n]) # extract probability of n targets

        self.clean_prediction(p_next_update)
        self.clean_update(p_next_update)
        p_update.copy(p_next_update)
        # Save (accumulated) update time
        self.upd_time += perf_counter() - t_start

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
                process.cdn[n] = np.sum(process.w[process.n == n]) # extract probability of n targets
        return



