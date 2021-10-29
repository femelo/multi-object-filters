#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File: demo.py                                                                #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This is a demo script for evaluation of several multi-object filters based   #
# on point processes (or random finite sets).                                  #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:20:12 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import sys
import os
import multiprocessing
from time import perf_counter
import numpy as np
from termcolor import cprint, colored
import argparse

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
from joint_glmb_filter import JointGLMBFilter
from lcc_filter import LCCFilter
from lccm_filter import LCCFilterWithMarks

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
    5: 'JGLMB',
    6: 'LCC',
    7: 'LCCM'
}
TRACKER_ID_REVERSE_MAP = {
    'PHD': 1,
    'CPHD': 2,
    'DG': 3,
    'GLMB': 4,
    'JGLMB': 5,
    'LCC': 6,
    'LCCM': 7
}

# Run tracker for a given scenario
def run_trackers(run_id, tracker_ids, model, truth, c = 100.0, p = 1, print_flag=False):
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
        elif tracker_id == 5:
            tracker = JointGLMBFilter(model)
        elif tracker_id == 6:
            tracker = LCCFilter(model)
        else:
            tracker = LCCFilterWithMarks(model)

        # Run
        start_time = perf_counter()
        tracker.run(measurement_set, print_flag=print_flag)
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
    cprint('Monte Carlo run complete: {:02d}'.format(run_id + 1), 'yellow')
    # Return measurement set and results
    return results_run, measurement_set

if __name__ == "__main__":
    # Configure parser
    argparser = argparse.ArgumentParser(
        description='Tracking using multi-object filters.')
    argparser.add_argument(
        '-f', '--filters',
        metavar='<list of filters>',
        default=['phd', 'cphd', 'lcc'],
        nargs='+',
        type=str,
        help='List of filters to run from: phd, cphd, dg, glmb, jglmb, lcc, lccm (default: phd cphd lcc)')
    argparser.add_argument(
        '-r', '--runs',
        metavar='<number of Monte Carlo runs>',
        default=1,
        type=int,
        help='Number of Monte Carlo runs (default: 1)')
    argparser.add_argument(
        '-p', '--parallelize',
        action='store_true',
        dest='parallelize',
        help='Paralellize runs')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='print_flag',
        help='Print information and debug messages')
    argparser.add_argument(
        '-t', '--number-of-time-steps',
        metavar='<number of time steps>',
        default=100,
        type=int,
        help='Number of time steps (default: 100)')
    argparser.add_argument(
        '-nt', '--number-of-targets',
        metavar='<number of targets>',
        default=10,
        type=int,
        help='Number of targets (default: 10)')
    argparser.add_argument(
        '-pd', '--probability-of-detection',
        metavar='<probability of detection>',
        default=0.98,
        type=float,
        help='Probability of detection (default: 0.98)')
    argparser.add_argument(
        '-cr', '--clutter-rate',
        metavar='<clutter rate>',
        default=10,
        type=int,
        help='Clutter rate (default: 10)')
    # Parse arguments
    args = argparser.parse_args()

    # Set random seed
    np.random.seed(1)

    # Get execution arguments
    num_of_runs = args.runs
    parallelize = args.parallelize
    print_flag = args.print_flag

    # Scenario parameters
    num_of_targets = args.number_of_targets
    prob_of_detection = args.probability_of_detection
    clutter_rate = args.clutter_rate
    num_of_time_steps = args.number_of_time_steps

    # Run sequences with chosen filters
    # 1: PHD
    # 2: CPHD
    # 3: DG
    # 4: GLMB
    # 5: JGLMB
    # 6: LCC
    # 7: LCCM
    tracker_ids = [TRACKER_ID_REVERSE_MAP[f.upper()] for f in args.filters 
        if f.upper() in TRACKER_ID_REVERSE_MAP.keys()]
    if len(tracker_ids) == 0:
        raise RuntimeError(colored('No valid filter specified.', 'red'))

    # Generate model
    model = generate_model(num_of_targets, prob_of_detection, clutter_rate, num_of_time_steps)
    truth = generate_ground_truth(model)

    measurement_sets = {}
    results = {}

    # OSPA parameters are hardcoded
    ospa_c = 100.0
    ospa_p = 1

    if parallelize:
        pool = multiprocessing.Pool(max(multiprocessing.cpu_count() - 2, 1))
        args = [(run_id, tracker_ids, model, truth, ospa_c, ospa_p, print_flag) for run_id in range(num_of_runs)]
        for i, output in enumerate(pool.starmap(run_trackers, args)):
            results[i] = output[0]
            measurement_sets[i] = output[1]
        pool.close()
        pool.join()
    else:
        for i in range(num_of_runs):
            output = run_trackers(i, tracker_ids, model, truth, ospa_c, ospa_p, print_flag)
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

    cprint('Average run times:', 'green')
    for res in performance_results:
        cprint('{:<15} : {:08.5f} seconds'.format(res.id + ' filter', res.run_time), 'green')

    generate_plots(truth, measurement_set_list, performance_results, model, save_figure=True)
    cprint('Plots saved in folder \'figures/\'.', 'yellow')
