# This is a demo script for evaluation of the (linear complexity) cumulant filter with marks
# by Flávio Eler De Melo
# 12/06/2021
import sys
import os
import multiprocessing
import random
from time import perf_counter
import numpy as np
from termcolor import cprint

# Add dependencies folder
sys.path.append(os.path.abspath('./dependencies'))

# Other imports
from generate_model import generate_model
from generate_ground_truth import generate_ground_truth
from generate_measurements import generate_measurements
from dependencies.ospa_dist import ospa_dist
from generate_plots import generate_plots
from phd_filter import PHDFilter
from cphd_filter import CPHDFilter
from dg_filter import DGFilter
from glmb_filter import GLMBFilter
from lcc_filter import LCCFilter

# Declare result class
class Result(object):
    def __init__(self, tracker_id, has_labels=False):
        self.id = tracker_id
        self.has_labels = has_labels
        self.label_max = 0
        self.n = np.array([])
        self.var_n = np.array([])
        self.ospa = np.array([])
        self.sq_err = np.array([])
        self.X = {}
        self.labels = {}
        self.run_time = 0.0
        self.prd_time = 0.0
        self.gat_time = 0.0
        self.upd_time = 0.0
        self.mgm_time = 0.0

# Tracker id map
TRACKER_ID_MAP = {
    1: 'PHD',
    2: 'CPHD',
    3: 'DG',
    4: 'GLMB',
    5: 'LCC'
}

# Run tracker for a given scenario
def run_trackers(run_id, tracker_ids, model, truth, c = 100.0, p = 1):
    measurement_set = generate_measurements(model, truth)
    results_run = {}
    for tracker_id in tracker_ids:
        if tracker_id == 1:
            tracker = PHDFilter(model)
        elif tracker_id == 2:
            tracker = CPHDFilter(model)
        elif tracker_id == 3:
            tracker = DGFilter(model)
        elif tracker_id == 4:
            tracker = GLMBFilter(model)
        else:
            tracker = LCCFilter(model)

        # Run
        start_time = perf_counter()
        tracker.run(measurement_set, print_flag=True)
        run_time = perf_counter() - start_time
        
        # Gather result
        result = Result(tracker.id, tracker.has_labels)
        result.n = np.array([tracker.N[k] for k in sorted(tracker.N.keys())])
        result.var_n = np.array([tracker.var[k] for k in sorted(tracker.var.keys())])
        result.X = tracker.X
        result.labels = tracker.labels
        result.label_max = tracker.label_max
        result.run_time = run_time / measurement_set.K
        result.prd_time = tracker.prd_time / measurement_set.K
        result.gat_time = tracker.gat_time / measurement_set.K
        result.upd_time = tracker.upd_time / measurement_set.K
        result.mgm_time = tracker.mgm_time / measurement_set.K
        result.ospa = np.zeros((truth.K, ))
        for k in range(truth.K):
            result.ospa[k] = ospa_dist(truth.X[k][[0, 2], :], tracker.X[k][[0, 2], :], c=c, p=p)
        result.sq_err = (result.n - truth.N) ** 2
        
        # Set result for a given tracker
        results_run[tracker_id] = result
    cprint('Monte Carlo run complete: {:02d}'.format(run_id + 1), 'green')
    # Return measurement set and results
    return results_run, measurement_set

if __name__ == "__main__":
    # Set random seed
    np.random.seed(1)

    # Parallelize
    num_of_runs = 1
    parallelize = False

    # Scenario parameters
    num_of_targets = 10
    prob_of_detection = 0.95
    clutter_rate = 10
    num_of_time_steps = 100

    # Generate model
    model = generate_model(num_of_targets, prob_of_detection, clutter_rate, num_of_time_steps)
    truth = generate_ground_truth(model)

    # Run sequence
    # 1: PHD
    # 2: CPHD
    # 3: DG
    # 4: GLMB
    # 5: LCC
    tracker_ids = [3, 4, 5]

    measurement_sets = {}
    results = {}

    ospa_c = 100.0
    ospa_p = 1

    if parallelize:
        pool = multiprocessing.Pool(6)
        args = [(run_id, tracker_ids, model, truth, ospa_c, ospa_p) for run_id in range(num_of_runs)]
        for i, output in enumerate(pool.starmap(run_trackers, args)):
            results[i] = output[0]
            measurement_sets[i] = output[1]
        pool.close()
        pool.join()
    else:
        for i in range(num_of_runs):
            output = run_trackers(i, tracker_ids, model, truth, ospa_c, ospa_p)
            results[i] = output[0]
            measurement_sets[i] = output[1]

    # Overall results
    performance_results = []
    measurement_set_list = []
    for t_id in tracker_ids:
        result = Result(TRACKER_ID_MAP[t_id])
        # Collect results
        if num_of_runs > 1:
            n_ = np.vstack([results[i][t_id].n for i in range(num_of_runs)])
            var_n_ = np.vstack([results[i][t_id].var_n for i in range(num_of_runs)])
            ospa_ = np.vstack([results[i][t_id].ospa for i in range(num_of_runs)])
            sq_err_ = np.vstack([results[i][t_id].sq_err for i in range(num_of_runs)])
            run_time_ = np.array([results[i][t_id].run_time for i in range(num_of_runs)])
            prd_time_ = np.array([results[i][t_id].prd_time for i in range(num_of_runs)])
            gat_time_ = np.array([results[i][t_id].gat_time for i in range(num_of_runs)])
            upd_time_ = np.array([results[i][t_id].upd_time for i in range(num_of_runs)])
            mgm_time_ = np.array([results[i][t_id].mgm_time for i in range(num_of_runs)])
            # Calculate mean values
            result.n = np.nanmean(n_, axis=0)
            result.var_n = np.nanmean(var_n_, axis=0)
            result.ospa = np.nanmean(ospa_, axis=0)
            result.sq_err = np.nanmean(sq_err_, axis=0)
            result.run_time = np.nanmean(run_time_, axis=0)
            result.prd_time = np.nanmean(prd_time_, axis=0)
            result.gat_time = np.nanmean(gat_time_, axis=0)
            result.upd_time = np.nanmean(upd_time_, axis=0)
            result.mgm_time = np.nanmean(mgm_time_, axis=0)
        else:
            # Set values
            result.n = results[0][t_id].n
            result.var_n = results[0][t_id].var_n
            result.ospa = results[0][t_id].ospa
            result.sq_err = results[0][t_id].sq_err
            result.run_time = results[0][t_id].run_time
            result.prd_time = results[0][t_id].prd_time
            result.gat_time = results[0][t_id].gat_time
            result.upd_time = results[0][t_id].upd_time
            result.mgm_time = results[0][t_id].mgm_time

        if num_of_runs == 1:
            run_id = 0
        else:
            run_id = np.argmin(np.sum(ospa_, axis=1) * np.sum(sq_err_, axis=1))

        # Get exemplary run
        result.X = results[run_id][t_id].X
        result.labels = results[run_id][t_id].labels
        result.has_labels = results[run_id][t_id].has_labels
        result.label_max = results[run_id][t_id].label_max
        performance_results.append(result)
        measurement_set_list.append(measurement_sets[run_id])

        cprint('{:s} filter: {:08.5f} seconds'.format(result.id, result.run_time), 'blue')

    generate_plots(truth, measurement_set_list, performance_results, model, save_figure=True)
