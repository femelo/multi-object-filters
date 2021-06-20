import numpy as np

def get_components(X, c):
    if len(X) == 0:
        return np.array([[]])
    else:
        return X[c, :]