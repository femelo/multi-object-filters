import numpy as np
import scipy as sp

def gm_merge_with_labels(w, x, P, l, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        l[:] = np.array([])
        return

    # State dimension
    n_x = x.shape[0]

    # New variables
    w_new = np.nan * np.ones(w.shape)
    x_new = np.nan * np.ones(x.shape)
    P_new = np.nan * np.ones(P.shape)
    l_new = np.nan * np.ones(l.shape)

    #  Counter
    k = 0

    for lbl in list(set(l)):
        idx = l == lbl
        l_ = l[idx]
        w_ = w[idx]
        x_ = x[:, idx]
        P_ = P[:, :, idx]

        I = list(range(len(l_)))
        while len(I) > 0:
            j = np.argmax(w_)
            
            d_x_j = x_[:, I] - x_[:, j, None]
            P_j = P_[:, :, j]
            sqrt_P_j = np.linalg.cholesky(P_j)
            inv_sqrt_P_j = sp.linalg.solve_triangular(sqrt_P_j, np.eye(n_x), lower=True)
            dist_sq = np.sum((inv_sqrt_P_j.dot(d_x_j)) ** 2, axis=0)
            I_ = I[dist_sq <= threshold]

            sum_w = np.sum(w_[I_])
            x_bar = np.zeros((n_x, ))
            P_x_x = np.zeros((n_x, n_x))
            P_bar = np.zeros((n_x, n_x))
            for i in I_:
                x_bar += w_[i] * x_[:, i]
                P_x_x += w_[i] * x_[:, i].T.dot(x_[:, i])
                P_bar += w_[i] * P_[:, :, i]
            x_bar /= sum_w
            P_x_x /= sum_w
            P_bar /= sum_w

            # Merge components
            w_new[k] = sum_w
            x_new[:, k] = x_bar
            P_new[:, :, k]= P_bar + P_x_x - x_bar.T.dot(x_bar)
            l_new[k] = lbl
            
            I = list(set(I) - set(I_))
            k += 1
    
    valid_idx = np.logical_not(np.isnan(w_new))
    w[:] = w_new[valid_idx]
    x[:] = x_new[:, valid_idx]
    P[:] = P_new[:, :, valid_idx]
    l[:] = l_new[valid_idx]

    return
        
