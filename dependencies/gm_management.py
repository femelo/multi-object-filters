# -*- coding: utf-8 -*-
# File: gm_management.py                                                       #
# Project: Multi-object Filters                                                #
# File Created: Tuesday, 8th June 2021 5:02:33 pm                              #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements methods for pruning, merging and capping      #
# Gaussian mixture components.
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 11:53:20 am                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
import scipy as sp
from copy import copy

# Get components
def get_components(X, c):
    if len(X) == 0:
        return np.array([[]])
    else:
        return X[c, :]

# Prune GM components
def gm_prune(w, x, P, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return
    
    idx = w > threshold
    sum_w = np.sum(w)
    w_new = copy(w[idx])
    x_new = copy(x[:, idx])
    P_new = copy(P[:, :, idx])
    w.resize(w_new.shape, refcheck=False)
    x.resize(x_new.shape, refcheck=False)
    P.resize(P_new.shape, refcheck=False)
    w[:] = w_new * ( sum_w / np.sum(w_new) )
    x[:] = x_new
    P[:] = P_new
    

# Merge GM components
def gm_merge(w, x, P, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return
    
     # State dimension
    n_x = x.shape[0]

    # New variables
    w_new = np.nan * np.ones(w.shape)
    x_new = np.nan * np.ones(x.shape)
    P_new = np.nan * np.ones(P.shape)

    n = len(w)
    I = np.arange(n)
    k = 0

    while len(I) > 0:
        j = np.argmax(w[I])

        d_x_j = x[:, I] - x[:, I[j], None]
        P_j = P[:, :, I[j]]
        sqrt_P_j = np.linalg.cholesky(P_j)

        inv_sqrt_P_j = sp.linalg.solve_triangular(sqrt_P_j, np.eye(n_x), lower=True)
        dist_sq = np.sum((inv_sqrt_P_j.dot(d_x_j)) ** 2, axis=0)
        I_ = I[dist_sq <= threshold]

        sum_w = np.sum(w[I_])
        x_bar = np.zeros((n_x, ))
        P_x_x = np.zeros((n_x, n_x))
        P_bar = np.zeros((n_x, n_x))
        for i in I_:
            x_bar += w[i] * x[:, i]
            P_x_x += w[i] * np.outer(x[:, i], x[:, i])
            P_bar += w[i] * P[:, :, i]
        x_bar /= sum_w
        P_x_x /= sum_w
        P_bar /= sum_w

        # Merge components
        w_new[k] = sum_w
        x_new[:, k] = x_bar
        P_new[:, :, k] = P_bar + P_x_x - np.outer(x_bar, x_bar)
        
        I = np.array(list(set(I) - set(I_)))
        k += 1

    valid_idx = np.logical_not(np.isnan(w_new))

    w_new = w_new[valid_idx]
    x_new = x_new[:, valid_idx]
    P_new = P_new[:, :, valid_idx]

    w.resize(w_new.shape, refcheck=False)
    x.resize(x_new.shape, refcheck=False)
    P.resize(P_new.shape, refcheck=False)

    w[:] = w_new
    x[:] = x_new
    P[:] = P_new

# Cap GM componets
def gm_cap(w, x, P, max_number):
    if len(w) > max_number:
        idx = np.argsort(-w)[:max_number]

        w_new = copy(w[idx])
        x_new = copy(x[:, idx])
        P_new = copy(P[:, :, idx])

        sum_w = np.sum(w)
        w.resize(w_new.shape, refcheck=False)
        x.resize(x_new.shape, refcheck=False)
        P.resize(P_new.shape, refcheck=False)

        w[:] = w_new * ( sum_w / np.sum(w_new) )
        x[:] = x_new
        P[:] = P_new

# Prune GM components with labels
def gm_prune_with_labels(w, x, P, l, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        l[:] = np.array([], dtype=int)
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return
    
    idx = w > threshold
    
    if not np.all(idx):
        w_new = w[idx]
        l_new = l[idx]
        x_new = x[:, idx]
        P_new = P[:, :, idx]

        sum_w = np.sum(w)
        w.resize(w_new.shape, refcheck=False)
        l.resize(l_new.shape, refcheck=False)
        x.resize(x_new.shape, refcheck=False)
        P.resize(P_new.shape, refcheck=False)
        
        w[:] = w_new * ( sum_w / np.sum(w_new) )
        l[:] = l_new
        x[:] = x_new
        P[:] = P_new

# Merge GM components with labels
def gm_merge_with_labels(w, x, P, l, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        l[:] = np.array([], dtype=int)
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return

    # State dimension
    n_x = x.shape[0]

    # New variables
    w_new = np.nan * np.ones(w.shape)
    x_new = np.nan * np.ones(x.shape)
    P_new = np.nan * np.ones(P.shape)
    l_new = np.nan * np.ones(l.shape)

    #  Counter
    k = 0

    for lbl in list(set(l)):
        idx = l == lbl
        w_ = w[idx]
        l_ = l[idx]
        x_ = x[:, idx]
        P_ = P[:, :, idx]

        I = np.arange(len(l_))
        while len(I) > 0:
            j = np.argmax(w_)
            
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

    return

# Cap GM components with labels
def gm_cap_with_labels(w, x, P, l, max_number, min_number_of_labels):
    if np.all(w == 0.0):
        w[:] = np.array([])
        l[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])

    if len(w) > max_number:
        all_indexes = np.argsort(-w)
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

        sum_w = np.sum(w)
        w.resize(w_new.shape, refcheck=False)
        l.resize(l_new.shape, refcheck=False)
        x.resize(x_new.shape, refcheck=False)
        P.resize(P_new.shape, refcheck=False)
        
        w[:] = w_new * ( sum_w / np.sum(w_new) )
        l[:] = l_new
        x[:] = x_new
        P[:] = P_new