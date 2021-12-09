# -*- coding: utf-8 -*-
# File: result.py                                                                                            #
# Project: Multi-object Filters                                                                              #
# File Created: Friday, 5th November 2021 2:15:24 pm                                                         #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements the result class for tracking experiments.                                  #
# -----                                                                                                      #
# Last Modified: Friday, 5th November 2021 2:20:19 pm                                                        #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np

# Declare result class
class Result(object):
    def __init__(self, tracker_id, has_labels=False):
        self.id = tracker_id
        self.has_labels = has_labels
        self.label_max = 0
        self.n = np.array([])
        self.var_n = np.array([])
        self.ospa = np.array([])
        self.sq_err = np.array([])
        self.X = {}
        self.labels = {}
        self.run_time = 0.0
        self.prd_time = 0.0
        self.gat_time = 0.0
        self.upd_time = 0.0
        self.mgm_time = 0.0

# Tracker id map
TRACKER_ID_MAP = {
    1: 'PHD',
    2: 'PHDM',
    3: 'CPHD',
    4: 'DGM',
    5: 'GLMB',
    6: 'JGLMB',
    7: 'LCC',
    8: 'LCCM'
}
TRACKER_ID_REVERSE_MAP = {
    'PHD':   1,
    'PHDM':  2,
    'CPHD':  3,
    'DGM':   4,
    'GLMB':  5,
    'JGLMB': 6,
    'LCC':   7,
    'LCCM':  8
}