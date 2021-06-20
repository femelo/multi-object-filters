import numpy as np
import scipy as sp

def gate_measurements_per_component(z, gamma_thrsh, model, m, P):
    z_len = z.shape[1]
    p_len = m.shape[1]
    valid_idx = np.zeros((z_len, )).astype(bool)
    valid_meas = {}
    innov_cov_mat = {}
    inv_sqrt_innov_cov_mat = {}
    sqrt_innov_cov_mat = {}
    innov_vec = {}

    for j in range(p_len):
        S_j = model.R + model.H.dot(P[:, :, j]).dot(model.H.T)
        sqrt_S_j = np.linalg.cholesky(S_j)
        inv_sqrt_S_j = sp.linalg.solve_triangular(sqrt_S_j, np.eye(S_j.shape[0]), lower=True)
        nu_j = z - model.H.dot(m[:, j, None])
        dist_sq = np.sum((inv_sqrt_S_j.dot(nu_j)) ** 2, axis=0)
        valid_meas[j] = dist_sq < gamma_thrsh
        valid_idx = np.logical_or(valid_idx, valid_meas[j])
        innov_cov_mat[j] = S_j
        sqrt_innov_cov_mat[j] = sqrt_S_j
        inv_sqrt_innov_cov_mat[j] = inv_sqrt_S_j
        innov_vec[j] = nu_j
        innov_vec[j][:, dist_sq >= gamma_thrsh] = np.nan
    
    invalid_idx = np.logical_not(valid_idx)
    z_gated = z[:, valid_idx]
    z_not_gated = z[:, invalid_idx]
    # Truncate innovation vectors
    for j in range(p_len):
        innov_vec[j] = innov_vec[j][:, valid_idx]

    return z_gated, z_not_gated, valid_meas, innov_vec, sqrt_innov_cov_mat, inv_sqrt_innov_cov_mat
