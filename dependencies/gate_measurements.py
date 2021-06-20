
import numpy as np
import scipy as sp

def gate_measurements(z, gamma, model, m, P):
    z_len = z.shape[1]
    p_len = m.shape[1]
    valid_idx = np.zeros((z_len, )).astype(bool)

    for j in range(p_len):
        S_j = model.H.dot(P[:, :, j]).dot(model.H.T) + model.R
        sqrt_S_j = np.linalg.cholesky(S_j)
        inv_sqrt_S_j = sp.linalg.solve_triangular(sqrt_S_j, np.eye(S_j.shape[0]), lower=True)
        nu = z - model.H.dot(m[:, j, None])
        dist_sq = np.sum((inv_sqrt_S_j.dot(nu)) ** 2, axis=0)
        valid_idx = np.logical_or(valid_idx, dist_sq < gamma)
    
    invalid_idx = np.logical_not(valid_idx)
    z_gated = z[:, valid_idx]
    z_not_gated = z[:, invalid_idx]
    return z_gated, z_not_gated
