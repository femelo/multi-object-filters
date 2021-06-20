import numpy as np
from scipy.sparse import coo_matrix
from termcolor import cprint
from copy import copy
from dependencies.bfms_path_wrap import bfms_path_wrap

def k_shortest_path_any(net_cost_matrix, source, destination, k_paths):
    '''
        Function k_shortest_path_any(net_cost_matrix, source, destination, k_paths) 
        returns the K first shortest paths (k_paths) from node source to node destination
        in the a network of N nodes represented by the NxN matrix netCostMatrix.
        In netCostMatrix, cost of 'inf' represents the 'absence' of a link 
        It returns 
        shortest_paths: the list of K shortest paths (in cell array 1 x K) and 
        total_costs   : costs of the K shortest paths (in array 1 x K)
        ==============================================================
        Meral Shirazipour
        This function is based on Yen's k-Shortest Path algorithm (1971)
        This function calls a slightly modified function dijkstra() 
        by Xiaodong Wang 2004.
        * netCostMatrix must have positive weights/costs
        Modified by BT VO
        Replaces dijkstra's algorithm with Derek O'Connor's Bellman-Ford-Moore implementation
        which allows negative entries in cost matrices provided there are no negative cycles
        and used in GLMB filter codes for prediction
        * netCostMatrix can have negative weights/costs
        ==============================================================
        DATE :           December 9 decembre 2009                                 
        Last Updated:    August 2 2010; January 6 2011; August 2 2011
        ----Changes April 2 2010:----
        1-previous version(9/12/2009)did not handle some exceptions which should
        have returned empty matrices for the return values (added lines 20 and 29)
        2-includes the changes proposed by Darren Rowland
        ----Changes January 6 2011:----
        1-fixed a bug reported by Babak Zafari that prevented from finding ALL
        the shortest paths in large networks
        ==============================================================
    '''

    if ((source > net_cost_matrix.shape[0] - 1) or (destination > net_cost_matrix.shape[1] - 1)):
        cprint('The source or destination node are not part of netCostMatrix', 'yellow')
        shortest_paths = {}
        total_costs = {}
    else:
        # INITIALIZATION
        k = 0
        shortest_paths = {}
        total_costs = {}
        cost, s_path, _ = bfms_path_wrap(coo_matrix(net_cost_matrix), source, destination)
        if len(s_path) > 0:
            path_id = 0
            # P is a dictionary that holds all the paths found so far
            P = {}
            P[path_id] = (s_path, cost)
            current_path_id = path_id
            len_x = 1
            X = {0: (path_id, s_path, cost)}
            S = {}
            # path_id
            S[path_id] = s_path[0] # deviation vertex is the first node initially
            
            # k = 1 is the shortest path returned by Dijkstra's algorithm:
            shortest_paths[k] = s_path
            total_costs[k] = cost

            while ((k < k_paths - 1) and (len_x != 0)):
                # Remove P from X
                for i in range(len(X.keys())):
                    if X[i][0] == current_path_id:
                        len_x -= 1
                        del X[i]
                        X_list = [(i, X[key]) for i, key in enumerate(sorted(X.keys()))]
                        X = dict(X_list)
                        break

                current_path = P[current_path_id][0]

                w = S[current_path_id]
                # Find w in (current_path, w) in set S, w was the dev vertex used to found current_path
                for i in range(len(current_path)):
                    if w == current_path[i]:
                        w_idx_in_path = i
                
                for idx_dev_vertex in range(w_idx_in_path, len(current_path)-1):
                    loc_net_cost_matrix = copy(net_cost_matrix)
                    # Remove vertices in the path before idx_dev_vertex and the incident edges
                    for i in range(idx_dev_vertex):
                        v = current_path[i]
                        loc_net_cost_matrix[v, :] = np.inf
                        loc_net_cost_matrix[:, v] = np.inf

                    # remove incident edge of v if v is in shortest_paths (K) U current_path 
                    # with similar sub_path to current_path ....
                    sp_same_subpath = {}
                    idx = 0
                    sp_same_subpath[idx] = current_path
                    for i in range(len(shortest_paths.keys())):
                        if len(shortest_paths[i]) >= idx_dev_vertex + 1:
                            if current_path[:idx_dev_vertex + 1] == shortest_paths[i][:idx_dev_vertex + 1]:
                                idx += 1
                                sp_same_subpath[idx] = shortest_paths[i]
   
                    loc_v = current_path[idx_dev_vertex]
                    for j in range(len(sp_same_subpath.keys())):
                        next = sp_same_subpath[j][idx_dev_vertex + 1]
                        loc_net_cost_matrix[loc_v, next] = np.inf

                    # Get the cost of the sub path before deviation vertex v
                    sub_path = current_path[:idx_dev_vertex + 1]
                    cost_sub_path = 0.0
                    for i in range(len(sub_path)-1):
                        cost_sub_path += net_cost_matrix[sub_path[i], sub_path[i + 1]]

                    # Call dijkstra between deviation vertex to destination node    
                    c, dev_p, _ = bfms_path_wrap(coo_matrix(loc_net_cost_matrix), current_path[idx_dev_vertex], destination)
                    if len(dev_p) > 0:
                        path_id += 1
                        P[path_id] = (sub_path[:-1] + dev_p, cost_sub_path + c)
                        S[path_id] = current_path[idx_dev_vertex]
                        len_x += 1
                        X[len_x - 1] = (path_id, P[path_id][0], P[path_id][1])
                    else:
                        pass

                # Step necessary otherwise if k is bigger than number of possible paths
                # the last results will get repeated !
                if len_x > 0:
                    shortest_x_cost = X[0][2]
                    shortest_x_id = X[0][0]
                    for i in range(1, len_x):
                        if X[i][2] < shortest_x_cost:
                            shortest_x_id = X[i][0]
                            shortest_x_cost = X[i][2]

                    current_path_id = shortest_x_id
                    k += 1
                    shortest_paths[k] = P[current_path_id][0]
                    total_costs[k] = P[current_path_id][1]
                else:
                    pass

    return shortest_paths, total_costs

