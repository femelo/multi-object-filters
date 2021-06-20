import numpy as np
from copy import copy
from munkres import DISALLOWED
from dependencies.murty import murty

def m_best_assignment_update(P_0, m):
  if m == 0:
    assignments = {}
    costs = np.array([])
    return assignments, costs

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
  assignments_matrix = -1 * np.ones((P_0.shape[0], min(m, len(assignments))), dtype=int)
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


    