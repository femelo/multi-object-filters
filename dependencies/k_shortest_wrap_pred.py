# -*- coding: utf-8 -*-
# File: k_shortest_wrap_pred.py                                                                              #
# Project: Multi-object Filters                                                                              #
# File Created: Monday, 7th June 2021 9:16:17 am                                                             #
# Author: Flávio Eler De Melo                                                                                #
# -----                                                                                                      #
# This package/module implements just the wrapper for the k-shortest path algorithm.                         #
# -----                                                                                                      #
# Last Modified: Tuesday, 29th June 2021 2:24:45 pm                                                          #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                                                  #
# -----                                                                                                      #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)                                  #
import numpy as np
from dependencies.k_shortest_path_any import k_shortest_path_any

def k_shortest_wrap_pred(r_s, k):
    if k == 0:
        paths = {}
        costs = {}
        return paths, costs
    
    n_s = len(r_s)
    i_s = np.argsort(-r_s)
    d_s = r_s[i_s]

    # Cost matrix for paths
    # !REMEMBER ZERO COST DENOTES NO ARC CONNECTION, 
    # I.E. INF COST BECAUSE OF SPARSE MATRIX INPUT FOR KSHORTEST PATH
    cost_matrix = np.zeros((n_s, n_s))

    for i in range(n_s):
        # Only allow jumps to higher numbered nodes, inf costs 
        # (equiv to zero cost for sparse representation) on lower diag 
        # prohibit reverse jumps (hence cycles)
        cost_matrix[:i, i] = d_s[i]
    
    # Extra 2 states for start and finish points
    # !REMEMBER ZERO COST DENOTES NO ARC CONNECTION, 
    # I.E. INF COST BECAUSE OF SPARSE MATRIX INPUT FOR KSHORTEST PATH
    cost_matrix_aug = np.zeros((n_s + 2, n_s + 2))
    # Must enter into one of original nodes OR
    cost_matrix_aug[0, 1:-1] = d_s
    # Exit immediately indicating no node selection 
    # (all target die) (eps used to denote zero cost or free jump, 
    # as zero is used for inf in sparse format)
    cost_matrix_aug[0, -1] = np.spacing(0)
    # Must exit at last node at no cost 
    # (eps used to denote zero cost or free jump, 
    # as zero is used for inf in sparse format)
    cost_matrix_aug[1:-1,-1] = np.spacing(0)
    # Cost for original nodes
    cost_matrix_aug[1:-1, 1:-1] = cost_matrix

    # Do k-shortest path
    paths, costs = k_shortest_path_any(cost_matrix_aug, 0, n_s + 1, k)
    
    for p in paths.keys():
        if ((len(paths[p]) == 2) and (paths[p][0] == 0) and (paths[p][1] == n_s + 1)):
            paths[p] = []
        else:
            paths[p] = [node-1 for node in paths[p][1:-1]] # Strip dummy entry and finish nodes
            paths[p] = i_s[paths[p]].tolist() # Convert index back to unsorted input

    return paths, costs
 
 