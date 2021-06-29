# -*- coding: utf-8 -*-
# File: log_sum_exp.py                                                         #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements a helper function for computing               #
# log(sum(exp(.))) while avoiding numerical issues (overflow/underflow).       #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:14:54 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np

def log_sum_exp(w, axis=0):
    ''' Performs log-sum-exp trick to avoid numerical underflow
        input:  w weight vector assumed already log transformed
        output: log(sum(exp(w))) 
    '''
    if np.prod(w.shape) == 0:
        return np.array(w.shape)

    max_val = np.max(w, axis=axis)
    return np.log(np.sum(np.exp(w - max_val))) + max_val
    
