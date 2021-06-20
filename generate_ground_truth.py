import numpy as np
from scipy.stats import norm

class GroundTruth(object):
    def __init__(self):
        self.has_labels = True
        self.K = 0
        self.X = {}
        self.N = np.array([])
        self.L = {}
        self.labels = {}
        self.num_of_tracks = 0
        self.model = None

def rand(size=(1, )):
    prod = np.prod(size)
    return np.random.rand(prod).reshape(*size, order='F')

def randn(size=(1, )):
    return norm.ppf(rand(size))

def propagate_state(X_km1, model, with_noise=False):

    if len(X_km1) == 0:
        return np.array([[]])

    # Linear state space equation (CV model)
    if with_noise:
        noise = model.G.dot(randn((model.n_x, X_km1.shape[1])))
    else:
        noise = np.zeros(X_km1.shape)

    return model.F.dot(X_km1) + noise


def generate_ground_truth(model):
    # Instantiate ground truth
    ground_truth = GroundTruth()
    # length of data/number of scans
    ground_truth.K = model.num_of_time_steps 
    # ground truth for states of targets
    ground_truth.N = np.zeros((ground_truth.K, ))
    # ground truth for labels of targets (k,i)
    ground_truth.L = {}
    # absolute index target identities (plotting)
    ground_truth.labels = {}
    # total number of appearing tracks
    ground_truth.num_of_tracks = 0
    # State dimension
    ground_truth.model = model

    # target initial states and birth/death times
    bounds = np.array([800, 5, 800, 5])

    N = model.num_of_targets
    N_0 = round(N / 4)
    N_1 = round(N / 2) - N_0
    N_2 = round(3 * N / 4) - (N_0 + N_1)
    N_3 = N - (N_0 + N_1 + N_2)

    x_start = np.zeros((model.n_x, N + 5))
    t_birth = np.zeros((N + 5, ))
    t_death = np.zeros((N + 5, ))

    #  0
    i = 0
    for j in range(N_0):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 0
        if j <= 5:
            t_death[i] = 80 - 1
        else:
            t_death[i]  = ground_truth.K
        i += 1

    #  1
    for j in range(N_1):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 20 - 1
        t_death[i] = ground_truth.K
        i += 1

    for j in range(2):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 20 - 1
        t_death[i] = ground_truth.K
        i += 1

    #  2
    for j in range(N_2):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 40 - 1
        t_death[i] = ground_truth.K
        i += 1

    x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
    t_birth[i] = 40 - 1
    t_death[i] = ground_truth.K
    i += 1

    #  3
    for j in range(N_3):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 60 - 1
        t_death[i] = ground_truth.K
        i += 1

    for j in range(2):
        x_start[:, i] = -bounds + 2 * bounds * np.random.rand(model.n_x)
        t_birth[i] = 60 - 1
        t_death[i] = ground_truth.K
        i += 1

    num_of_births = i

    # Generate the tracks
    X = dict([(k, []) for k in range(ground_truth.K)])
    track_ids = dict([(k, []) for k in range(ground_truth.K)])
    cardinality = np.zeros((ground_truth.K, ))
    for n in range(num_of_births):
        x_k = x_start[:, n]
        t_end = min(t_death[n], ground_truth.K - 1)
        num_of_steps = int(t_end - t_birth[n] + 1)
        for k in np.linspace(t_birth[n], t_end, num=num_of_steps, endpoint=True).astype(int):
            x_km1 = x_k
            x_k = propagate_state(x_km1, model, with_noise=False)
            X[k].append(x_k[:, None])
            track_ids[k].append(n + 1)
            cardinality[k] += 1
    # Set ground truth and return
    ground_truth.X = dict([(k, np.hstack(X[k])) if len(X[k]) > 0 else (k, np.array([[]])) for k in X.keys()])
    ground_truth.labels = dict([(k, np.hstack(track_ids[k])) if len(track_ids[k]) > 0 else (k, np.array([])) for k in sorted(track_ids.keys())])
    ground_truth.N = cardinality
    ground_truth.num_of_tracks = num_of_births

    return ground_truth
