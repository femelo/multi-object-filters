import numpy as np

def kalman_predict_single(m, P, F, Q):
    return F.dot(m), F.dot(P).dot(F.T) + Q

def kalman_predict_multiple(model, m, P):
    p_len = m.shape[1]
    m_prd = np.zeros(m.shape)
    P_prd = np.zeros(P.shape)

    for i in range(p_len):
        m_i, P_i = kalman_predict_single(m[:, i], P[:, :, i], model.F, model.Q)
        m_prd[:, i] = m_i
        P_prd[:, :, i] = P_i

    return m_prd, P_prd