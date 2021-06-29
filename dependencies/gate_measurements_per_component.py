# -*- coding: utf-8 -*-
# File: gate_measurements_per_component.py                                     #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/moudle implements a gating method on a per component basis.     #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 11:49:53 am                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
import scipy as sp

def gate_measurements_per_component(z, gamma_thrsh, model, m, P, truncate_innovation=True):
    z_len = z.shape[1]
    p_len = m.shape[1]
    valid_idx = np.zeros((z_len, )).astype(bool)
    valid_meas = {}
    innov_cov_mat = {}
    inv_sqrt_innov_cov_mat = {}
    sqrt_innov_cov_mat = {}
    innov_vec = {}

    for i in range(p_len):
        S_i = model.R + model.H.dot(P[:, :, i]).dot(model.H.T)
        sqrt_S_i = np.linalg.cholesky(S_i)
        inv_sqrt_S_i = sp.linalg.solve_triangular(sqrt_S_i, np.eye(S_i.shape[0]), lower=True)
        nu_j = z - model.H.dot(m[:, i, None])
        dist_sq = np.sum((inv_sqrt_S_i.dot(nu_j)) ** 2, axis=0)
        valid_meas[i] = dist_sq < gamma_thrsh
        valid_idx = np.logical_or(valid_idx, valid_meas[i])
        innov_cov_mat[i] = S_i
        sqrt_innov_cov_mat[i] = sqrt_S_i
        inv_sqrt_innov_cov_mat[i] = inv_sqrt_S_i
        innov_vec[i] = nu_j
        innov_vec[i][:, dist_sq >= gamma_thrsh] = np.nan
    
    invalid_idx = np.logical_not(valid_idx)
    z_gated = z[:, valid_idx]
    z_not_gated = z[:, invalid_idx]
    # Truncate innovation vectors
    if truncate_innovation:
        for i in range(p_len):
            innov_vec[i] = innov_vec[i][:, valid_idx]

    return z_gated, z_not_gated, valid_meas, innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat
