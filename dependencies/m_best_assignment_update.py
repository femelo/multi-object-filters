# -*- coding: utf-8 -*-
# File: m_best_assignment_update.py                                            #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements the Murty's m-best assignment algorithm.      #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:16:35 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
from copy import copy
from munkres import Munkres, DISALLOWED

def murty(P_0, m):
    '''
    MURTY Murty's Algorithm for m-best ranked optimal assignment problem
    based on Ba Tuong Vo's Matlab code
    by Flávio Eler De Melo, 2021 
    '''

    # Instantiate Munkres
    m_algorithm = Munkres()
    # Find the optimal and initial solution
    C_0 = 0.0
    S_0 = m_algorithm.compute(P_0.tolist())
    for row, col in S_0:
        C_0 += P_0[row, col]

    num_rows = P_0.shape[0]
    num_cols = P_0.shape[1]

    if m == 1:
        assignments = {0: S_0}
        costs = np.array([C_0])
        return assignments, costs

    # Preallocate a block of memory to hold the queue
    N = 1000  # Block size
    answer_list_P = np.zeros((num_rows, num_cols, N), dtype=object)
    answer_list_S = {}
    answer_list_C = np.nan * np.ones((N, ))

    # Initialize answer list
    answer_list_P[:, :, 0] = P_0 # problem or cost matrix
    answer_list_S[0] = S_0       # solutions or assignemnts
    answer_list_C[0] = C_0       # cost vector for problems/solutions
    answer_index_next = 1

    assignments = {}
    costs = np.zeros((m, ))

    for i in range(m):
        # If all are cleared, break the loop early 
        if np.all(np.isnan(answer_list_C)):
            # for j in range(answer_index_next-1, len(assignments)-1):
            #     del assignments[j]
            costs = costs[:answer_index_next]
            break
        
        # Grab lowest cost solution index
        idx_top = np.nanargmin(answer_list_C[:answer_index_next])
    
        # Copy the current best solution out
        assignments[i] = answer_list_S[idx_top]
        costs[i] = answer_list_C[idx_top]

        # Copy lowest cost problem to temp
        P_now = answer_list_P[:, :, idx_top].astype(object)
        S_now = answer_list_S[idx_top]
    
        # Delete the solution from the queue
        answer_list_C[idx_top] = np.nan

        for a in S_now:
            # Current assignment pair
            a_row = a[0]
            a_col = a[1]

            # Remove it and calculate new solution
            loc_P = copy(P_now)
            if a_col < num_cols - num_rows:
                loc_P[a_row, a_col] = DISALLOWED
            else:
                loc_P[a_row, num_cols-num_rows:] = DISALLOWED
            
            loc_C = 0.0
            if np.any(loc_P != DISALLOWED):
                rows_to_use = np.any(loc_P != DISALLOWED, axis=1)
                if np.all(rows_to_use):
                    loc_S = m_algorithm.compute(loc_P.tolist())
                    for row, col in loc_S:
                        loc_C += loc_P[row, col]
                else:
                    row_map = {}
                    idx = 0
                    loc_S = []
                    for r_idx in range(num_rows):
                        if rows_to_use[r_idx]:
                            row_map[idx] = r_idx
                            idx += 1
                        else:
                            loc_S.append((r_idx, -1))
                    loc_P_ = loc_P[rows_to_use, :]
                    loc_S_ = m_algorithm.compute(loc_P_.tolist())
                    # Get solutions
                    for row, col in loc_S_:
                        loc_S.append((row_map[row], col))
                        loc_C += loc_P_[row, col]
            else:
                loc_S = []
            
            # Copy to new list
            if len(loc_S) > 0:
                # If we have filled the allocated space, allocate more
                if answer_index_next > len(answer_list_C) - 1:
                    answer_list_P = np.dstack([answer_list_P, np.zeros((num_rows, num_cols, N), dtype=object)])
                    answer_list_C = np.hstack([answer_list_C, np.nan * np.ones((N, ))])

                answer_list_P[:, :, answer_index_next] = loc_P
                answer_list_S[answer_index_next] = loc_S
                answer_list_C[answer_index_next] = loc_C
                answer_index_next += 1
            
            # Enforce current assignment
            loc_value = P_now[a_row, a_col]
            P_now[a_row, :] = DISALLOWED
            P_now[:, a_col] = DISALLOWED
            P_now[a_row, a_col] = loc_value

    return assignments, costs

def m_best_assignment_update(P_0, m):
    if m == 0:
        assignments_matrix = np.array([[]])
        costs = np.array([])
        return assignments_matrix, costs

    n_1 = P_0.shape[0]
    n_2 = P_0.shape[1]

    # Padding blocks for dummy variables
    blk1 = np.full((n_1, n_1), DISALLOWED, dtype=object)
    # blk1 = np.zeros((n_1, n_1), dtype=object)
    for i in range(n_1):
        blk1[i, i] = 0.0

    # min_value = P_0.min()
    # Make costs non-negative and augment matrix
    # P_0_aug = np.hstack([P_0 - min_value, blk1])
    P_0_aug = np.hstack([P_0, blk1])

    # Murty
    assignments, costs = murty(P_0_aug, m)

    # In matrix format
    assignments_matrix = -1 * \
        np.ones((P_0.shape[0], min(m, len(assignments))), dtype=int)
    for idx in range(len(assignments)):
        a = assignments[idx]
        rows, cols = zip(*a)
        assignments_matrix[rows, idx] = cols

    # Strip dummy variables
    assignments_matrix[assignments_matrix >= n_2] = -1
    return assignments_matrix, costs

    # # Strip dummy variables
    # ass_id_map = {}
    # for a in reversed(sorted(assignments.keys())):
    #   new_assignment = [edge for edge in assignments[a] if edge[1] < n_2]
    #   if len(new_assignment) > 0:
    #     assignments[a] = new_assignment
    #   else:
    #     del assignments[a]
    #     del costs[a]
    #     if a + 1 in assignments.keys():
    #       ass_id_map[a + 1] = a

    # # Shift solutions
    # for a in reversed(sorted(ass_id_map.keys())):
    #   assignments[ass_id_map[a]] = copy(assignments[a])
    #   del assignments[a]

def sub2ind(shape, i, j):
  return np.ravel_multi_index(np.vstack([i, j]), dims=shape, order='C')

def m_best_assignment_gibbs_sampling(P_0, m):
    if np.prod(P_0.shape) == 0:
        assignments_matrix = np.array([[]])
        costs = np.array([])
        return assignments_matrix, costs

    n_1 = P_0.shape[0]
    n_2 = P_0.shape[1]

    if m > 0:
        assignments_matrix = -1 * np.ones((m, n_1))
        costs = np.zeros((m, ))
    else:
        assignments_matrix = -1 * np.ones((1, n_1))
        costs = np.zeros((1, ))

    # Use all missed detections as initial solution
    current_solution = np.arange(n_1, 2*n_1, dtype=int)
    assignments_matrix[0, :] = current_solution
    costs[0] = np.sum(P_0.ravel()[sub2ind(P_0.shape, np.arange(n_1), current_solution)])
    z_array = np.array([0.0])
    for s_idx in range(1, m):
      for v_idx in range(n_1):
        # Grab row of costs for current association variable
        row_cost = np.exp(-P_0[v_idx, :])
        # Lock out current and previous iteration step assignments 
        # except for the one in question
        inds = np.append(np.arange(v_idx), np.arange(v_idx + 1, len(current_solution)))
        row_cost[current_solution[inds]] = 0
        idx_old = np.where(row_cost > 0)[0]
        row_cost_ = row_cost[idx_old]
        histogram, _ = np.histogram(np.random.rand(), np.append(z_array, np.cumsum(row_cost_) / np.sum(row_cost_)))
        current_solution[v_idx] = idx_old[histogram.astype(bool)]
      assignments_matrix[s_idx, :] = current_solution
      costs[s_idx] = np.sum(P_0.ravel()[sub2ind(P_0.shape, np.arange(n_1), current_solution)])
    
    # Get unique assignments
    unique_assignment_matrix, inds = np.unique(assignments_matrix, return_index=True, axis=0)
    assignments_matrix = unique_assignment_matrix
    costs = costs[inds]

    return assignments_matrix, costs
