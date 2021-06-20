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