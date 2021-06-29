# -*- coding: utf-8 -*-
# File: kalman_predict_multiple.py                                             #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements the Kalman filter prediction for multiple     #
# Gaussian mixture components.                                                 #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:11:50 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #

import numpy as np

def kalman_predict_single(m, P, F, Q):
    return F.dot(m), F.dot(P).dot(F.T) + Q

def kalman_predict_multiple(model, m, P):
    p_len = m.shape[1]
    m_prd = np.zeros(m.shape)
    P_prd = np.zeros(P.shape)

    for i in range(p_len):
        m_i, P_i = kalman_predict_single(m[:, i], P[:, :, i], model.F, model.Q)
        m_prd[:, i] = m_i
        P_prd[:, :, i] = P_i

    return m_prd, P_prd