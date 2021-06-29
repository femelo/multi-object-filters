# -*- coding: utf-8 -*-
# File: bfms_path_wrap.py                                                      #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This module implements the Bellman-Ford-Moore Shortest Path algorithm.       #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 11:45:59 am                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np

def initialize(G):
    ''' Transforms the sparse matrix G into the list-of-arcs form
        and intializes the shortest path parent-pointer and distance
        arrays, p and D.
        Flávio Eler De Melo, June 2021
    '''
    
    # Get arc list {u, v, duv, 1:m} from G.
    tails, heads = G.nonzero()
    n = G.shape[1]
    m = len(tails)
    weights = G.data
    
    # Shortest path tree of parent pointers
    p = np.zeros((n, )).astype(int)
    # Shortest path distances from node i = 1:n to the root of the SP tree
    D = np.inf * np.ones((n, ))

    return m, n, p, D, tails, heads, weights

def bfms_path_ot(G, r):
    '''
        Basic form of the Bellman-Ford-Moore Shortest Path algorithm
        Assumes G(N,A) is in sparse adjacency matrix form, with |N| = n, 
        |A| = m = nnz(G). It constructs a shortest path tree with root r which 
        is represented by an vector of parent 'pointers' p, along with a vector
        of shortest path lengths D.
        Complexity: O(mn)

        Unlike the original BFM algorithm, this does an optimality test on the
        SP Tree p which may greatly reduce the number of iters to convergence.
        
        WARNING: 
        This algorithm performs well on random graphs but may perform 
        badly on real problems.

        This code is inspired by the code by Derek O'Connor. 
    '''
    
    m, n, p, D, tails, heads, W = initialize(G)
    # Set the root of the SP tree (p,D)
    p[r] = -1
    D[r] = 0.0

    # Converges in <= n-1 iters if no negative cycles exist in G
    for iter in range(n - 1):
        is_optimal = True
        # O(m) for optimality test
        for arc in range(m):
            u = tails[arc]
            v = heads[arc]
            d_uv = W[arc]
            if D[v] > D[u] + d_uv:
                # SP tree not optimal: update (p, D)
                D[v] = D[u] + d_uv
                p[v] = u
                is_optimal = False
        if is_optimal:
            # SP tree is optimal
            break

    return p, D, iter

def bfms_path_wrap(ncm, source, destination):
    p, D, _ = bfms_path_ot(ncm, source)
    dist = D[destination]
    pred = p
    
    if np.isinf(dist):
        s_path = []
    else:
        s_path = [destination]
        while s_path[0] != source:
            s_path = [pred[s_path[0]]] + s_path

    return dist, s_path, pred
