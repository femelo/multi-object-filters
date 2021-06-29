# -*- coding: utf-8 -*-
# File: generate_model.py                                                      #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This is a script to set the multi-object filter model parameters.            #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:26:40 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
import scipy as sp

class Model(object):
    def __init__(self, num_of_targets = 20, prob_of_detection = 0.90, clutter_rate = 10.0, 
        num_of_time_steps = 100):
        self.num_of_time_steps = num_of_time_steps
        # State and observation dimensions
        self.n_x = 4 # state vector
        self.n_z = 2 # observation vector
        
        # Kinematic model parameters (CV model)
        T = 1.0
        self.T = T # sampling period
        A_0 = np.array([[1.0, T], [0.0, 1.0]])
        # Transition matrix
        self.F = sp.linalg.block_diag(A_0, A_0)
        self.sigma_v = 2.0
        Q_0 = (self.sigma_v ** 2) * np.array([[T ** 3 / 3, T ** 2 / 2], [T ** 2 / 2, T]])
        # Process noise covariance matrix
        self.Q = sp.linalg.block_diag(Q_0, Q_0)
        self.G = np.linalg.cholesky(self.Q)
        
        # Survival process parameters
        self.p_s = 0.99
        self.q_s = 1.0 - self.p_s
       
        # Default birth process parameters (Poisson process, multiple Gaussian components)
        self.L_birth = 4
        self.r_birth = np.zeros((self.L_birth, ))
        self.w_birth = np.zeros((self.L_birth, ))
        self.w1_birth = np.ones((self.L_birth, ))
        self.l_birth = np.arange(self.L_birth) + 1
        self.m_birth = np.zeros((self.n_x, self.L_birth))
        self.B_birth = np.zeros((self.n_x, self.n_x, self.L_birth))
        self.P_birth = np.zeros((self.n_x, self.n_x, self.L_birth))
        # Average birth rate
        lambda_b = num_of_targets / self.num_of_time_steps
        sig_v_b = 1.0
        self.mu_birth = lambda_b
        self.sigma_v_b = sig_v_b
        # Birth component 1
        self.r_birth[0] = lambda_b / 4
        self.w_birth[0] = lambda_b / 4
        self.m_birth[:, 0] = np.array([-500.0, 0.0, -500.0, 0.0])
        self.B_birth[:, :, 0] = np.diag([-500.0 / 2, sig_v_b, -500.0 / 2, sig_v_b])
        self.P_birth[:, :, 0] = self.B_birth[:, :, 0].dot(self.B_birth[:, :, 0].T)
        # Birth component 2
        self.r_birth[1] = lambda_b / 4
        self.w_birth[1] = lambda_b / 4
        self.m_birth[:, 1] = np.array([-500.0, 0.0, +500.0, 0.0])
        self.B_birth[:, :, 1] = np.diag([500.0 / 2, sig_v_b, 500.0 / 2, sig_v_b])
        self.P_birth[:, :, 1] = self.B_birth[:, :, 1].dot(self.B_birth[:, :, 1].T)
        # Birth component 3
        self.r_birth[2] = lambda_b / 4
        self.w_birth[2] = lambda_b / 4
        self.m_birth[:, 2] = np.array([+500.0, 0.0, -500.0, 0.0])
        self.B_birth[:, :, 2] = np.diag([500.0 / 2, sig_v_b, 500.0 / 2, sig_v_b])
        self.P_birth[:, :, 2] = self.B_birth[:, :, 2].dot(self.B_birth[:, :, 2].T)
        # Birth component 4
        self.r_birth[3] = lambda_b / 4
        self.w_birth[3] = lambda_b / 4
        self.m_birth[:, 3] = np.array([+500.0, 0.0, +500.0, 0.0])
        self.B_birth[:, :, 3] = np.diag([500.0 / 2, sig_v_b, 500.0 / 2, sig_v_b])
        self.P_birth[:, :, 3] = self.B_birth[:, :, 3].dot(self.B_birth[:, :, 3].T)
        
        # Observation model parameters (noisy x/y only)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.D = np.diag([5.0, 5.0])
        # Observation noise matrix
        self.R = self.D.dot(self.D.T)

        # Detection process parameters
        self.p_d = prob_of_detection
        self.q_d = 1.0 - self.p_d

        # Clutter process parameters
        self.mu_c = clutter_rate
        self.var_c = clutter_rate
        self.lambda_c = clutter_rate
        self.range_c = np.array([[-1000.0, +1000.0], [-1000.0, +1000.0]])
        self.pdf_c = 1.0 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])
        self.num_of_targets = num_of_targets

def generate_model(num_of_targets, prob_of_detection, clutter_rate, num_of_time_steps):
    return Model(num_of_targets, prob_of_detection, clutter_rate, num_of_time_steps)




