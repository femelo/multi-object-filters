# -*- coding: utf-8 -*-
# File: set_birth_model.py                                                     #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements a simple method for setting the birth model   #
# based on non-validated measurements.                                         #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:18:11 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np

def set_birth_labels(labels, L_birth):
    # This method can be used to reutilize labels that are not assigned to any object anymore
    if len(labels) == 0:
        label_max = 0
        avail_labels = np.array([])
        L_avail = 0
    else:
        label_max = max(labels)
        avail_labels = np.array(list(set(range(1, label_max + 1)) - set(labels)), dtype=int)
        L_avail = len(avail_labels)

    if L_avail >= L_birth:
        l_birth = avail_labels[:L_birth]
    else:
        if len(avail_labels) > 0:
            l_birth = np.hstack([avail_labels, np.arange(L_birth - L_avail) + label_max + 1])
        else:
            l_birth = np.arange(L_birth - L_avail) + label_max + 1
    return l_birth

def set_birth_model(model, z_not_gated, labels):
    # Get parameters
    H = model.H
    R = model.R
    F = model.F
    Q = model.Q
    n_x = model.n_x
    n_z = model.n_z

    # Get the total expected number of births
    mu_birth = model.mu_birth
    # Get the number of components
    L_birth = z_not_gated.shape[1]

    w_birth = np.zeros((L_birth, ))          # weights of Gaussian birth terms (per scan) [sum gives average rate of target birth per scan]
    m_birth = np.zeros((n_x, L_birth))       # means of Gaussian birth terms 
    B_birth = np.zeros((n_x, n_x, L_birth))  # std of Gaussian birth terms
    P_birth = np.zeros((n_x, n_x, L_birth))  # cov of Gaussian birth terms

    sigma_v = model.sigma_v_b

    P_0 = np.diag([0.0, sigma_v**2, 0.0, sigma_v**2])
    P_birth_base = F.dot(P_0 + H.T.dot(R.dot(H))).dot(F.T) + Q
    B_birth_base = np.linalg.cholesky(P_birth_base)
    if L_birth > 0:
        w_birth_base = mu_birth / L_birth
        # w_birth_base = mu_birth / model.num_of_targets
    else:
        w_birth_base = 0.0

    for l in range(L_birth):
        m_birth[:, l] = H.T.dot(z_not_gated[:, l]) # velocity initialized as (0, 0)
        P_birth[:, :, l] = P_birth_base
        B_birth[:, :, l] = B_birth_base
        w_birth[l] = w_birth_base    

    # Set new labels
    # This method can be used to reutilize labels that are not assigned to any object anymore
    # l_birth = set_birth_labels(labels, L_birth)

    # This assumes any new component is given a global unique label
    label_max = max(labels) if len(labels) > 0 else 0
    l_birth = np.arange(L_birth) + label_max + 1

    # Set output model
    model.m_birth = m_birth
    model.B_birth = B_birth
    model.P_birth = P_birth
    model.w_birth = w_birth
    model.l_birth = l_birth
    model.L_birth = L_birth

    