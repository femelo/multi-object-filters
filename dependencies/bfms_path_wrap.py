'''
    Copyright (c) 2012, Derek O'Connor
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the distribution
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
'''
import numpy as np

def initialize(G):
    ''' Transforms the sparse matrix G into the list-of-arcs form
        and intializes the shortest path parent-pointer and distance
        arrays, p and D.
        Derek O'Connor, 21 Jan 2012
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
        Derek O'Connor, 19 Jan, 11 Sep 2012.  derekroconnor@eircom.net

        Unlike the original BFM algorithm, this does an optimality test on the
        SP Tree p which may greatly reduce the number of iters to convergence.
        USE: 
        n=10^6; G=sprand(n,n,5/n); r=1; format long g;
        tic; [p,D,iter] = BFMSpathOT(G,r);toc, disp([(1:10)' p(1:10) D(1:10)]);
        WARNING: 
        This algorithm performs well on random graphs but may perform 
        badly on real problems. 
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
    # Wrapper to convert output to MATLAB 'graphshortestpath' format (by BT Vo)
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
