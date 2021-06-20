import numpy as np
import scipy as sp

def kalman_update_single(z, m, P, H, R, log_likelihood=False):
    n_z = z.shape[0]
    nu = z - H.dot(m[:, None])
    S = H.dot(P.dot(H.T)) + R
    sqrt_S = np.linalg.cholesky(S)
    inv_sqrt_S = sp.linalg.solve_triangular(sqrt_S, np.eye(n_z), lower=True)
    inv_S = inv_sqrt_S.T.dot(inv_sqrt_S)
    det_S = np.prod(np.diag(sqrt_S)) ** 2
    K = P.dot(H.T.dot(inv_S))

    m_upd = np.nan * np.ones((m.shape[0], z.shape[1]))

    valid_idx = np.all(np.isfinite(nu), axis=0)
    if log_likelihood:
        q_z_upd = np.nan * np.ones((z.shape[1], ))
        q_z_upd[valid_idx] = -0.5*n_z*np.log(2*np.pi) -0.5*np.log(det_S) -0.5*np.sum((inv_sqrt_S.dot(nu[:, valid_idx])) ** 2, axis=0)
    else:
        q_z_upd = np.nan * np.ones((z.shape[1], ))
        q_z_upd[valid_idx] = np.exp(-0.5*n_z*np.log(2*np.pi) -0.5*np.log(det_S) -0.5*np.sum((inv_sqrt_S.dot(nu[:, valid_idx])) ** 2, axis=0))
    
    m_upd[:, valid_idx] = m[:, None] + K.dot(nu[:, valid_idx])
    P_upd = P - K.dot(H.dot(P))

    return q_z_upd, m_upd, P_upd

def kalman_update_multiple(z, m, P, model, log_likelihood=False):
    p_len = m.shape[1]
    z_len = z.shape[1]

    q_z_upd = np.zeros((p_len, z_len))
    m_upd = np.zeros((model.n_x, p_len, z_len))
    P_upd = np.zeros((model.n_x, model.n_x, p_len))

    for i in range(p_len):
        q_z_i, m_i, P_i = kalman_update_single(
            z, m[:, i], P[:, :, i], model.H, model.R,
            log_likelihood)
        q_z_upd[i, :] = q_z_i
        m_upd[:, i, :] = m_i
        P_upd[:, :, i] = P_i

    return q_z_upd, m_upd, P_upd