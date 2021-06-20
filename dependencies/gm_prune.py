import numpy as np

def gm_prune(w, x, P, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        return
    
    idx = w > threshold
    w[:] = w[idx]
    x[:] = x[:, idx]
    P[:] = P[:, :, idx]
    