import numpy as np

def gm_cap_with_labels(w, x, P, l, max_number, min_number_of_labels):
    if np.all(w == 0.0):
        w[:] = np.array([])
        x[:] = np.array([[]])
        P[:] = np.array([[[]]])
        l[:] = np.array([])

    if len(w) > max_number:
        all_indexes = np.argsort(-w)
        idx = all_indexes[:max_number]
        l_new = l[idx]
        while len(set(l_new)) < min_number_of_labels and max_number < len(idx):
            max_number += 1
            idx = all_indexes[:max_number]
            l_new = l[idx]

        w_new = w[idx]
        w[:] = w_new * (np.sum(w) / np.sum(w_new))
        x[:] = x[:, idx]
        P[:] = P[:, :, idx]
        l[:] = l_new
