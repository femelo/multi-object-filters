import numpy as np

def esf(Z):
    # Calculate elementary symmetric function using Mahler's recursive formula
    if len(Z) == 0:
        return 1.0

    n_z = len(Z)
    F = np.zeros((2, n_z))
    i_n = 0
    i_nm1 = 1

    for n in range(n_z):
        F[i_n, 0] = F[i_nm1, 0] + Z[n]
        for k in range(1, n + 1):
            if k == n:
                F[i_n, k] = Z[n] * F[i_nm1, k-1]
            else:
                F[i_n, k] = F[i_nm1, k] + Z[n] * F[i_nm1, k-1]
        i_n_ = i_n
        i_n = i_nm1
        i_nm1 = i_n_
  
    return np.hstack([np.array([1.0]), F[i_nm1, :]])