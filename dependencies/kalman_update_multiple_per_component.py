# -*- coding: utf-8 -*-
# File: kalman_update_multiple_per_component.py                                #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements the Kalman filter update on a per Gaussian    #
# mixture component basis.                                                     #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:13:09 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #

import numpy as np
import scipy as sp

def kalman_update_single(z, m, P, H, nu, sqrt_S, inv_sqrt_S, log_likelihood=False):
    n_z = z.shape[0]
    det_S = np.prod(np.diag(sqrt_S)) ** 2
    inv_S = inv_sqrt_S.T.dot(inv_sqrt_S)
    K = P.dot(H.T.dot(inv_S))

    m_upd = np.nan * np.ones((m.shape[0], z.shape[1]))

    valid_idx = np.all(np.isfinite(nu), axis=0)
    if log_likelihood:
        q_z_upd = np.nan * np.ones((z.shape[1], ))
        q_z_upd[valid_idx] = -0.5*n_z*np.log(2*np.pi) -0.5*np.log(det_S) -0.5*np.sum((inv_sqrt_S.dot(nu[:, valid_idx])) ** 2, axis=0)
    else:
        q_z_upd = np.nan * np.ones((z.shape[1], ))
        q_z_upd[valid_idx] = np.exp(-0.5*n_z*np.log(2*np.pi) -0.5*np.log(det_S) -0.5*np.sum((inv_sqrt_S.dot(nu[:, valid_idx])) ** 2, axis=0))
    
    m_upd[:, valid_idx] = m[:, None] + K.dot(nu[:, valid_idx])
    P_upd = P - K.dot(H.dot(P))

    return q_z_upd, m_upd, P_upd

def kalman_update_multiple_per_component(z, m, P, model, innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat, log_likelihood=False):
    p_len = m.shape[1]
    z_len = z.shape[1]

    q_z_upd = np.zeros((p_len, z_len))
    m_upd = np.zeros((model.n_x, p_len, z_len))
    P_upd = np.zeros((model.n_x, model.n_x, p_len))

    for i in range(p_len):
        q_z_i, m_i, P_i = kalman_update_single(
            z, m[:, i], P[:, :, i], model.H, 
            innov_vec[i], sqrt_innov_cov_mat[i], inv_sqrt_innov_cov_mat[i],
            log_likelihood)
        q_z_upd[i, :] = q_z_i
        m_upd[:, i, :] = m_i
        P_upd[:, :, i] = P_i

    return q_z_upd, m_upd, P_upd


