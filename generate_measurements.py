# -*- coding: utf-8 -*-
# File: generate_measurements.py                                               #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This is a script to generate measurements for the demo script.               #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:26:00 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
from scipy.stats import norm, poisson

class MeasurementSet(object):
    def __init__(self):
        self.K = 0
        self.Z = {}

def rand(size=(1, )):
    prod = np.prod(size)
    return np.random.rand(prod).reshape(*size, order='F')

def randn(size=(1, )):
    return norm.ppf(rand(size))

def poissrnd(mu, size=(1, )):
    if len(size) == 1:
        return int(poisson.ppf(rand(size), mu)[0])
    else:
        return poisson.ppf(rand(size), mu).astype(int)

def generate_observation(X, model, with_noise=True):
    if len(X) == 0:
        return np.array([[]])

    # Linear observation equation (position components only)
    if with_noise:
        noise = model.D.dot(randn((model.n_z, X.shape[1])))
    else:
        noise = np.zeros((model.n_z, X.shape[1]))

    return model.H.dot(X) + noise

def generate_measurements(model, truth):
    # Instantiate measurement set
    measurement_set = MeasurementSet()
    measurement_set.K = truth.K

    # Generate measurements
    for k in range(truth.K):
        if truth.N[k] > 0:
            idx = np.random.rand(int(truth.N[k])) <= model.p_d
            # Generate observations
            Z_k = generate_observation(truth.X[k][:, idx], model, with_noise=True)
        else:
            Z_k = np.array([[]])
        N_c = poissrnd(model.lambda_c) # number of clutter points
        # Generate clutter
        if N_c > 0:
            Z_c = model.range_c[:, 0, None] + np.diag(np.ravel(model.range_c.dot(np.array([[-1.0], [1.0]])))).dot(rand((model.n_z, N_c)))
            if Z_k.shape[1] > 0:
                Z_k = np.hstack([Z_k, Z_c])
            else:
                Z_k = Z_c
        else:
            Z_c = np.array([[]])
            Z_k = Z_k
        
        # Save measurements: union of detections and clutter
        measurement_set.Z[k] = Z_k

    return measurement_set
