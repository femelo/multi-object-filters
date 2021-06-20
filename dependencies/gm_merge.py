import numpy as np
import scipy as sp

def gm_merge(w, x, P, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return
    
     # State dimension
    n_x = x.shape[0]

    # New variables
    w_new = np.nan * np.ones(w.shape)
    x_new = np.nan * np.ones(x.shape)
    P_new = np.nan * np.ones(P.shape)

    n = len(w)
    I = list(range(n))
    k = 0

    while len(I) > 0:
        j = np.argmax(w)
        
        d_x_j = x[:, I] - x[:, j, None]
        P_j = P[:, :, j]
        sqrt_P_j = np.linalg.cholesky(P_j)
        inv_sqrt_P_j = sp.linalg.solve_triangular(sqrt_P_j, np.eye(n_x), lower=True)
        dist_sq = np.sum((inv_sqrt_P_j.dot(d_x_j)) ** 2, axis=0)
        I_ = I[dist_sq <= threshold]

        sum_w = np.sum(w[I_])
        x_bar = np.zeros((n_x, ))
        P_x_x = np.zeros((n_x, n_x))
        P_bar = np.zeros((n_x, n_x))
        for i in I_:
            x_bar += w[i] * x[:, i]
            P_x_x += w[i] * x[:, i].T.dot(x[:, i])
            P_bar += w[i] * P[:, :, i]
        x_bar /= sum_w
        P_x_x /= sum_w
        P_bar /= sum_w

        # Merge components
        w_new[k] = sum_w
        x_new[:, k] = x_bar
        P_new[:, :, k]= P_bar + P_x_x - x_bar.T.dot(x_bar)
        
        I = list(set(I) - set(I_))
        k += 1

    valid_idx = np.logical_not(np.isnan(w_new))
    w[:] = w_new[valid_idx]
    x[:] = x_new[:, valid_idx]
    P[:] = P_new[:, :, valid_idx]

    return
    
