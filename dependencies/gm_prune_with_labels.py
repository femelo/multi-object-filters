import numpy as np

def gm_prune_with_labels(w, x, P, l, threshold):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        l[:] = np.array([])
        return
    
    idx = w > threshold
    w[:] = w[idx]
    x[:] = x[:, idx]
    P[:] = P[:, :, idx]
    l[:] = l[idx]