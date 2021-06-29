# -*- coding: utf-8 -*-
# File: ospa_dist.py                                                           #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This package/module implements the computation of the OSPA metric.           #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:17:25 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import numpy as np
from munkres import Munkres
def ospa_dist(X, Y, c=100.0, p=1):
    # This is the Python code for OSPA distance proposed in
    # 
    # D. Schuhmacher, B.-T. Vo, and B.-N. Vo, "A consistent metric for performance evaluation in multi-object filtering," IEEE Trans. Signal Processing, Vol. 56, No. 8 Part 1, pp. 3447--3457, 2008.
    # http://ba-ngu.vo-au.com/vo/SVV08_OSPA.pdf
    # BibTeX entry:
    # @ARTICLE{OSPA,
    # author={D. Schuhmacher and B.-T. Vo and B.-N. Vo},
    # journal={IEEE Transactions on Signal Processing},
    # title={A Consistent Metric for Performance Evaluation of Multi-Object Filters},
    # year={2008},
    # month={Aug},
    # volume={56},
    # number={8},
    # pages={3447-3457}}  
    # ---

    # Compute OSPA distance between two finite sets X and Y
    # Inputs: X,Y-   matrices of column vectors
    #        c  -   cut-off parameter
    #        p  -   p-parameter for the metric
    # Output: scalar distance between X and Y
    # Note: the Euclidean 2-norm is used as the "base" distance on the region

    if X.shape[1] == 0 and Y.shape[1] == 0:
        return 0

    if X.shape[1] == 0 or Y.shape[1] == 0:
        return c

    # Calculate sizes of the input point patterns
    n = X.shape[1]
    m = Y.shape[1]

    # Calculate cost/weight matrix for pairings - fast method with vectorization
    XX = np.kron(X, np.ones((1, m)))
    YY = np.kron(Y, np.ones((n, 1))).reshape(Y.shape[0], n*m)
    D = np.sqrt(np.sum((XX - YY) ** 2, axis=0)).reshape(n, m)
    D = np.minimum(c, D) ** p

    # Compute optimal assignment and cost using the Hungarian algorithm
    munkres = Munkres()
    assignments = munkres.compute(D.tolist())
    cost = 0.0
    for i, j in assignments:
        cost += D[i, j]

    # Calculate final distance
    dist = (((c ** p) * abs(m - n) + cost) / max(m, n)) ** (1.0 / p)

    # Output
    return dist
    
