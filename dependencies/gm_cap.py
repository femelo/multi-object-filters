import numpy as np

def gm_cap(w, x, P, max_number):
    if len(w) > max_number:
        idx = np.argsort(-w)[:max_number]
        w_new = w[idx]
        w[:] = w_new * (np.sum(w) / np.sum(w_new))
        x[:] = x[:, idx]
        P[:] = P[:, :, idx]
