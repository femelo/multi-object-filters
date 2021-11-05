# -*- coding: utf-8 -*-
# File: lccm_filter.py                                                                                       #
# Project: Multi-object Filters                                                                              #
# File Created: Thursday, 10th June 2021 11:04:24 am                                                         #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the Linear Complexity Cumulant filter with marks based on that proposed in: #
#                                                                                                            #
# D. E. Clark and F. E. De Melo, "A Linear-Complexity Second-Order Multi-Object Filter via                   #
# Factorial Cumulants," Proc. of the 21st International Conference on Information Fusion, 2018, 1250-1259.   #
#                                                                                                            #
# BibTeX entry:                                                                                              #
# @INPROCEEDINGS{LCC2018,                                                                                    #
#  author={D. E. Clark and F. E. De Melo},                                                                   #
#  booktitle={FUSION 2018, Proceedings of the 21st International Conference on Information Fusion},          #
#  title={A Linear-Complexity Second-Order Multi-Object Filter via Factorial Cumulants},                     #
#  year={2018},                                                                                              #
#  pages={1250-1259}}                                                                                        #
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 2:05:40 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
import scipy as sp
from numpy.lib.arraysetops import unique
from copy import copy, deepcopy
from numpy.linalg.linalg import multi_dot
from scipy.stats import chi2
from time import perf_counter
from copy import copy
from termcolor import cprint
from dependencies.kalman_predict_multiple import kalman_predict_multiple
from dependencies.gate_measurements_per_component import gate_measurements_per_component
from dependencies.esf import esf
from dependencies.kalman_update_multiple_per_component import kalman_update_multiple_per_component
# from dependencies.gm_management import gm_prune_with_labels, gm_merge_with_labels, gm_cap_with_labels
from dependencies.set_birth_model import set_birth_model

VAL_MIN = np.spacing(1)

class LCCFilterWithMarks(object):
    def __init__(self, model, use_assoc_hist=True, merge_components=True):
        # Multi-object filter id
        self.id = 'LCCM'
        self.has_labels = True
        self.use_assoc_hist = use_assoc_hist
        self.merge_components = merge_components
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
        self.assoc_hist = {}

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
        if self.use_assoc_hist:
            assoc_hist_update = []
        else:
            assoc_hist_update = None
        model = self.model

        # Initial point process cumulants
        c1_update = 0.0
        c2_update = 1.0
        c1_clutter = model.mu_c
        c2_clutter = model.mu_c - model.var_c

        mu_c = model.mu_c
        pdf_c = model.pdf_c
        p_d = model.p_d
        p_s = model.p_s

        log_p_d = np.log(model.p_d)
        log_q_d = np.log(model.q_d)

        # Set initial model birth as null
        model.w_birth = np.array([])
        model.l_birth= np.array([], dtype=int)
        model.m_birth = np.array([[]])
        model.P_birth = np.array([[[]]])
        model.L_birth = 0 

        # Run recursion
        for k in range(self.K):
            # Prediction
            t_start = perf_counter()

            # Intensity prediction
            w_predict = p_s * w_update
            l_predict = l_update
            m_predict, P_predict = kalman_predict_multiple(model, m_update, P_update)
            if self.use_assoc_hist:
                assoc_hist_predict = model.L_birth * [np.array([])] + assoc_hist_update

            if len(w_predict) > 0 and model.L_birth > 0:
                w_predict = np.hstack([model.w_birth, w_predict])
                l_predict = np.hstack([model.l_birth, l_predict])
                m_predict = np.hstack([model.m_birth, m_predict])
                P_predict = np.dstack([model.P_birth, P_predict])
            elif model.L_birth > 0:
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
            if self.use_assoc_hist:
                assoc_hist_update = list(map(lambda h: np.append(h, 0), assoc_hist_predict))
            
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
                    if self.use_assoc_hist:
                        assoc_hist_update += [np.append(assoc_hist_predict[i], j + 1) for i, idx_b in enumerate(valid_idx) if idx_b]

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
            self.gm_prune(w_update, m_update, P_update, l_update, 
                self.prune_threshold, assoc_hist=assoc_hist_update)
            L_pruned = L_updated - len(w_update)
            if self.merge_components:
                self.gm_merge(w_update, m_update, P_update, l_update, 
                    self.max_num_of_components, self.merge_threshold, assoc_hist=assoc_hist_update)
                L_merged = L_updated - L_pruned - len(w_update)
            else:
                L_merged = 0
            self.gm_cap(w_update, m_update, P_update, l_update, 
                self.max_num_of_components, round(c1_update), assoc_hist=assoc_hist_update)
            L_capped = L_updated - L_pruned - L_merged - len(w_update)
            self.mgm_time += perf_counter() - t_start

            # In case all components where removed, reset cumulants
            if len(w_update) == 0:
                c1_update = 0.0
                c2_update = 1.0

            # Estimates extraction
            self.extract_estimates(w_update, m_update, l_update, c1_update, c2_update, k, assoc_hist=assoc_hist_update)

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
    
    def extract_estimates(self, w_update, m_update, l_update, c1_update, c2_update, k, assoc_hist=None):
         # Save point process moments
        self.mu[k] = c1_update
        self.var[k] = c1_update + c2_update

        unique_labels = unique(l_update)
        N_k = round(min(c1_update, len(unique_labels)))
        
        # Calculate combined weights
        if not assoc_hist is None and k > 0:
            prv_labels = self.labels[k - 1]
            sim_probs = 1.0 / (1.0 + np.exp(-2.0)) * np.ones(w_update.shape)
            for l in prv_labels:
                idx_l = l_update == l
                sim_probs[idx_l] = self.comp_sim_probabilities(
                    label=l, hypotheses=[assoc_hist[i] for i, idx_b in enumerate(idx_l) if idx_b])
            weights = w_update * sim_probs
        else:
            weights = w_update
        
        # Reorder components
        idx_comp = np.argsort(-weights)
        w_update = w_update[idx_comp]
        l_update = l_update[idx_comp]
        m_update = m_update[:, idx_comp]
        weights = weights[idx_comp]
        if not assoc_hist is None:
            assoc_hist = [assoc_hist[i] for i in idx_comp]

        # Initalize variables for estimates
        m_est = np.zeros((self.model.n_x, N_k))
        l_est = np.zeros((N_k, ), dtype=int)
        if not assoc_hist is None:
            assoc_hist_est = {}

        # Method 1: best weights per label -> best combined weights (with similarity probabilities)
        # Order set of labels according to their weights
        indexes_labels = [[i_l for i_l, idx_b in enumerate(l_update == l) if idx_b] for l in unique_labels]
        w_labels = np.array([np.sum(w_update[idx_l]) for idx_l in indexes_labels])
        idx_labels = np.argsort(-w_labels)
        indexes_labels = [indexes_labels[i] for i in idx_labels]
        w_labels = w_labels[idx_labels]
        unique_labels = unique_labels[idx_labels]

        i = 0
        N_k_ = 0
        for i_l, _ in enumerate(unique_labels):
            if i == N_k:
                break
            n_l = int(round(w_labels[i_l]))
            if n_l > 0:
                N_k_ += n_l
                indexes = indexes_labels[i_l]
                # n = 1  : takes just the best track
                # n = n_l: takes the n_l-best tracks
                n = 1
                for idx in indexes[0:n]:
                    m_est[:, i] = m_update[:, idx]
                    l_est[i] = l_update[idx]
                    i += 1
                if not assoc_hist is None:
                    # For the association history takes just the best track
                    idx = indexes[0]
                    assoc_hist_est[l_update[idx]] = assoc_hist[idx]

        # Prune the number of objects
        if i < N_k:
            m_est = m_est[:, 0:i]
            l_est = l_est[0:i]
            N_k = N_k_

        # # Method 2: best combined weights (with similarity probabilities)
        # i = 0
        # j = 0
        # while np.any(l_est == 0):
        #     if not l_update[j] in l_est:
        #         m_est[:, i] = m_update[:, j]
        #         l_est[i] = l_update[j]
        #         i += 1
        #         if not assoc_hist is None:
        #             assoc_hist_est[l_update[j]] = assoc_hist[j]
        #     j += 1
        
        idx = np.argsort(l_est)
        self.X[k] = m_est[:, idx]
        self.N[k] = N_k
        self.labels[k] = unique(l_est[idx])
        self.label_max = max(l_est.tolist() + [self.label_max])
        # Save association history
        if not assoc_hist is None:
            self.assoc_hist = assoc_hist_est

    def comp_sim_probabilities(self, label, hypotheses):
        n = len(hypotheses)
        if len(self.labels) > 0 and label in self.labels[max(self.labels.keys())]:
            h_l = self.assoc_hist[label]
            norm_h_l = np.linalg.norm(h_l)
            sim_dists = np.array(
                list(
                    map(
                        lambda h: np.maximum(np.dot(h, h_l), VAL_MIN) / np.maximum(np.linalg.norm(h) * norm_h_l, VAL_MIN), 
                        [h[:-1] for h in hypotheses]
                    )
                )
            )
            sim_probs = 1.0 / (1.0 + np.exp(-2.0 * sim_dists))
        else:
            prob = 1.0 / (1.0 + np.exp(-2.0))
            sim_probs = prob * np.ones((n, ))
        return sim_probs

    # Prune GM components with labels
    def gm_prune(self, w, x, P, l, threshold, assoc_hist=None):
        if np.all(w == 0.0):
            w[:] = np.array([])
            l[:] = np.array([], dtype=int)
            x[:] = np.array([[]])
            P[:] = np.array([[[]]])
            if not assoc_hist is None:
                assoc_hist[:] = []
            return
        
        idx = w > threshold
        
        if not np.all(idx):
            w_new = w[idx]
            l_new = l[idx]
            x_new = x[:, idx]
            P_new = P[:, :, idx]
            if not assoc_hist is None:
                assoc_hist_new = [assoc_hist[i] for i, idx_b in enumerate(idx) if idx_b]

            sum_w = np.sum(w)
            w.resize(w_new.shape, refcheck=False)
            l.resize(l_new.shape, refcheck=False)
            x.resize(x_new.shape, refcheck=False)
            P.resize(P_new.shape, refcheck=False)
            
            w[:] = w_new * ( sum_w / np.sum(w_new) )
            l[:] = l_new
            x[:] = x_new
            P[:] = P_new
            if not assoc_hist is None:
                assoc_hist[:] = assoc_hist_new

    # Merge GM components with labels
    def gm_merge(self, w, x, P, l, max_number, threshold, assoc_hist=None):
        if np.all(w == 0.0):
            w[:] = np.array([])
            l[:] = np.array([], dtype=int)
            x[:] = np.array([[]])
            P[:] = np.array([[[]]])
            if not assoc_hist is None:
                assoc_hist[:] = []
            return
        
        # Don't merge if the number of components is already below the maximum
        # if len(w) < max_number:
        #     return

        # State dimension
        n_x = x.shape[0]

        # New variables
        w_new = np.nan * np.ones(w.shape)
        x_new = np.nan * np.ones(x.shape)
        P_new = np.nan * np.ones(P.shape)
        l_new = np.nan * np.ones(l.shape)
        if not assoc_hist is None:
            assoc_hist_new = []

        #  Counter
        k = 0
        for lbl in list(set(l)):
            idx = l == lbl
            w_ = w[idx]
            l_ = l[idx]
            x_ = x[:, idx]
            P_ = P[:, :, idx]
            if not assoc_hist is None:
                assoc_hist_ = [assoc_hist[i] for i, idx_b in enumerate(idx) if idx_b]

            I = np.arange(len(l_))
            while len(I) > 0:
                if not assoc_hist is None:
                    # sim_probs = self.comp_sim_probabilities(label=lbl, hypotheses=assoc_hist_)
                    # weights = w_ * sim_probs
                    weights = w_
                else:
                    weights = w_
                j = np.argmax(weights)
                
                d_x_j = x_[:, I] - x_[:, j, None]
                P_j = P_[:, :, j]
                sqrt_P_j = np.linalg.cholesky(P_j)
                inv_sqrt_P_j = sp.linalg.solve_triangular(sqrt_P_j, np.eye(n_x), lower=True)
                dist_sq = np.sum((inv_sqrt_P_j.dot(d_x_j)) ** 2, axis=0)
                I_ = I[dist_sq <= threshold]

                sum_w = np.sum(w_[I_])
                x_bar = np.zeros((n_x, ))
                P_x_x = np.zeros((n_x, n_x))
                P_bar = np.zeros((n_x, n_x))
                for i in I_:
                    x_bar += w_[i] * x_[:, i]
                    P_x_x += w_[i] * np.outer(x_[:, i], x_[:, i])
                    P_bar += w_[i] * P_[:, :, i]
                x_bar /= sum_w
                P_x_x /= sum_w
                P_bar /= sum_w

                # Merge components
                w_new[k] = sum_w
                x_new[:, k] = x_bar
                P_new[:, :, k]= P_bar + P_x_x - np.outer(x_bar, x_bar)
                l_new[k] = lbl
                if not assoc_hist is None:
                    assoc_hist_new.append(assoc_hist_[j])
                
                I = np.array(list(set(I) - set(I_)))
                w_[I_] = -1.0
                k += 1
        
        valid_idx = np.logical_not(np.isnan(w_new))

        w_new = w_new[valid_idx]
        l_new = l_new[valid_idx]
        x_new = x_new[:, valid_idx]
        P_new = P_new[:, :, valid_idx]

        w.resize(w_new.shape, refcheck=False)
        l.resize(l_new.shape, refcheck=False)
        x.resize(x_new.shape, refcheck=False)
        P.resize(P_new.shape, refcheck=False)

        w[:] = w_new
        l[:] = l_new
        x[:] = x_new
        P[:] = P_new
        if not assoc_hist is None:
            assoc_hist[:] = assoc_hist_new
        return

    # Cap GM components with labels
    def gm_cap(self, w, x, P, l, max_number, min_number_of_labels, assoc_hist=None):
        if np.all(w == 0.0):
            w[:] = np.array([])
            l[:] = np.array([])
            x[:] = np.array([[]])
            P[:] = np.array([[[]]])
            if not assoc_hist is None:
                assoc_hist[:] = []

        # Don't cap if the number of components is already below the maximum
        if len(w) < max_number:
            return

        if not assoc_hist is None and len(self.labels) > 0:
            # prv_labels = self.labels[max(self.labels.keys())]
            # sim_probs = 1.0 / (1.0 + np.exp(-2.0)) * np.ones(w.shape)
            # for lbl in prv_labels:
            #     idx_l = l == lbl
            #     sim_probs[idx_l] = self.comp_sim_probabilities(
            #         label=lbl, hypotheses=[assoc_hist[i] for i, idx_b in enumerate(idx_l) if idx_b])
            # weights = w * sim_probs
            weights = w
        else:
            weights = w

        # It will reach this point if the number of components is bigger than the maximum allowed
        all_indexes = np.argsort(-weights)
        idx = all_indexes[:max_number]
        l_new = l[idx]
        while len(set(l_new)) < min_number_of_labels and max_number < len(idx):
            max_number += 1
            idx = all_indexes[:max_number]
            l_new = l[idx]

        w_new = copy(w[idx])
        l_new = copy(l[idx])
        x_new = copy(x[:, idx])
        P_new = copy(P[:, :, idx])
        if not assoc_hist is None:
            assoc_hist_new = [assoc_hist[i] for i in idx]

        sum_w = np.sum(w)
        w.resize(w_new.shape, refcheck=False)
        l.resize(l_new.shape, refcheck=False)
        x.resize(x_new.shape, refcheck=False)
        P.resize(P_new.shape, refcheck=False)
        
        w[:] = w_new * ( sum_w / np.sum(w_new) )
        l[:] = l_new
        x[:] = x_new
        P[:] = P_new
        if not assoc_hist is None:
            assoc_hist[:] = assoc_hist_new
