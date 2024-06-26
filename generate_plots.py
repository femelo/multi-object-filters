#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: generate_plots.py                                                      #
# Project: Multi-object Filters                                                #
# File Created: Monday, 7th June 2021 9:16:17 am                               #
# Author: Flávio Eler De Melo                                                  #
# -----                                                                        #
# This is a script to generate plots for the performance evaluation.           #
# -----                                                                        #
# Last Modified: Tuesday, 29th June 2021 12:27:25 pm                           #
# Modified By: Flávio Eler De Melo (flavio.eler@gmail.com>)                    #
# -----                                                                        #
# License: Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0>)    #
import os
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm
import numpy as np
import argparse
from termcolor import cprint
import pickle5 as pickle
from result import Result

FIGURES_PATH = 'figures'

DPI = 300
PX = 1.0 / DPI
CM = 1.0 / 2.54

MARKER_SIZE = 5
AXIS_FONT_SIZE = 3
LABEL_FONT_SIZE = 4
LEGEND_FONT_SIZE = 4
TITLE_FONT_SIZE = 5

class TrajectorySet(object):
    def __init__(self):
        self.X = np.array([[]])
        self.t_birth = np.array([])
        self.t_death = np.array([])
        self.num_of_targets = 0

def extract_trajectories(target_set):
    assert(target_set.has_labels)
    K = target_set.K
    n_x = target_set.model.n_x
    # Variables for the trajectories
    X = np.zeros((n_x, K, target_set.num_of_tracks))
    t_birth = np.zeros((target_set.num_of_tracks, ))
    t_death = np.zeros((target_set.num_of_tracks, ))
    max_id = 0
    for k in range(K):
        labels_k = np.array(target_set.labels[k], dtype=int)
        if labels_k.shape[0] == 0:
            continue

        if target_set.X[k].shape[1] > 0:
            X[:, k, labels_k - 1] = target_set.X[k]

        if np.max(labels_k) > max_id:
            idx = labels_k > max_id
            t_birth[labels_k[idx] - 1] = k

        if labels_k.shape[0] > 0:
            max_id = np.max(labels_k)

        t_death[labels_k - 1] = k
    # Set output
    trajectory_set = TrajectorySet()
    trajectory_set.X = X
    trajectory_set.t_birth = t_birth
    trajectory_set.t_death = t_death
    trajectory_set.num_of_targets = target_set.num_of_tracks
    # Return
    return trajectory_set

def plot_tracks(ground_truth, measurement_sets, filters, model, save_figure=True):

    # Collect ground truth track trajectories
    gt_traj = extract_trajectories(ground_truth)

    limits = model.range_c
    fig = plt.figure(1, figsize=(1080*PX, 1080*PX), dpi=DPI)
    fig.tight_layout(rect=[0.0, 0.01, 1.0, 0.91])
    axis = fig.add_subplot(111)
    legend_labels = [
        'Ground truth',
        'Measurements',
        'Estimates']

    for idx, filter in enumerate(filters):
        for l_id in ['top', 'bottom', 'left', 'right']:
            axis.spines[l_id].set_linewidth(0.5)
        axis.tick_params(width=0.5)
        axis.set_xlim(limits[0, 0], limits[0, 1])
        axis.set_ylim(limits[1, 0], limits[1, 1])
        # axis.set_xticklabels(np.linspace(limits[0, 0], limits[0, 1], 10 + 1, endpoint=True), fontsize=LABEL_FONT_SIZE)
        # axis.set_yticklabels(np.linspace(limits[1, 0], limits[1, 1], 10 + 1, endpoint=True), fontsize=LABEL_FONT_SIZE)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_xlabel('Coordinate x [m]', fontsize=LABEL_FONT_SIZE)
        axis.set_ylabel('Coordinate y [m]', fontsize=LABEL_FONT_SIZE)
        axis.set_title(
            'Tracks - {:s} filter'.format(filter.id), fontsize=TITLE_FONT_SIZE)

        # Get measurement set used
        measurement_set = measurement_sets[idx]

        # Plot ground truth
        legend = legend_labels[0]
        for i in range(ground_truth.num_of_tracks):
            traj = gt_traj.X[:, np.arange(
                gt_traj.t_birth[i], gt_traj.t_death[i] + 1).astype(int), i][[0, 2], :]
            axis.plot(traj[0, :], traj[1, :], 'k-',
                      linewidth=0.50, label=legend)
            axis.plot(traj[0, 0, None], traj[1, 0, None], 'ko', markersize=np.sqrt(
                MARKER_SIZE), markerfacecolor='none', markeredgewidth=0.2, linewidth=0.1, label='_nolegend_')
            axis.plot(traj[0, -1, None], traj[1, -1, None], 'k^', markersize=np.sqrt(MARKER_SIZE),
                      markerfacecolor='none', markeredgewidth=0.2, linewidth=0.1, label='_nolegend_')
            if legend == legend_labels[0]:
                legend = '_nolegend_'

        # Plot measurements
        legend = legend_labels[1]
        for k in range(measurement_set.K):
            Z_k = measurement_set.Z[k]
            if Z_k.shape[1] > 0:
                axis.plot(Z_k[0, :], Z_k[1, :], 'x', markersize=np.sqrt(MARKER_SIZE), markeredgewidth=0.2, c=(
                    0.75, 0.75, 0.75, 0.5), linestyle='none', label=legend)
                if legend == legend_labels[1]:
                    legend = '_nolegend_'

        # Plot estimates
        if filter.has_labels:
            # num_of_colors = filter.label_max
            num_of_colors = 10
            colormap = cm.get_cmap('tab10', num_of_colors)
            colors = colormap(np.linspace(0.0, 1.0, num_of_colors))

        legend = legend_labels[2]
        started_labels = []
        for k in range(measurement_set.K):
            X_k = filter.X[k]
            if X_k.shape[1] > 0:
                if filter.has_labels:
                    labels_k = filter.labels[k]
                    for i, l_k in enumerate(labels_k):
                        if not l_k in started_labels:
                            started_labels.append(l_k) 
                            plt.text(X_k[0, i], X_k[2, i], 'T{:02d}-{:02d}'.format(l_k, k), color=colors[(l_k % num_of_colors) - 1, :], fontsize=3, alpha=1.0, label=legend)
                        # else:
                        #     plt.text(X_k[0, i], X_k[2, i], '{:02d}'.format(k), color=colors[(l_k % num_of_colors) - 1, :], fontsize=2, alpha=0.5, label=legend)
                        plt.plot(X_k[0, i], X_k[2, i], marker='.', markersize=np.sqrt(MARKER_SIZE),
                                 markerfacecolor='none', markeredgewidth=0.2, c=colors[(l_k % num_of_colors) - 1, :], alpha=0.5, label=legend)
                        if legend == legend_labels[2]:
                            legend = '_nolegend_'
                else:
                    plt.plot(X_k[0, :], X_k[2, :], marker='.', markersize=np.sqrt(MARKER_SIZE),
                             markerfacecolor='none', markeredgewidth=0.2, c='blue', alpha=0.5, linestyle='none', label=legend)
                    # for i in range(X_k.shape[1]):
                    #     plt.text(X_k[0, i], X_k[2, i], '{:02d}'.format(k), color='blue', fontsize=2, alpha=0.5, label=legend)
                    if legend == legend_labels[2]:
                        legend = '_nolegend_'
        axis.legend(loc=2, fontsize=LEGEND_FONT_SIZE)

        plt.subplots_adjust(top=0.94, bottom=0.06, right=0.94, left=0.06,
                            hspace=0, wspace=0)
        plt.margins(0, 0)

        # Draw
        plt.draw()
        plt.show(block=False)

        # Save figure
        if save_figure:
            plt.savefig(os.path.join(FIGURES_PATH, '_'.join(
                ['Tracks', filter.id, 'filter']) + '.pdf'))

        # Clear axis for the next plot
        axis.cla()

    plt.close()
    return

def plot_cardinality_performance(ground_truth, filters, save_figure=True):
    n_f = len(filters)
    # Cardinality performance
    N_max = np.max(ground_truth.N)
    limits = [0, N_max + 5]
    time_steps = np.arange(ground_truth.K)
    xticklabels = np.linspace(
        time_steps[0], time_steps[-1] + 1, 5, endpoint=True).astype(int)
    yticklabels = np.linspace(
        limits[0], limits[-1], int((limits[-1] - limits[0]) / 5), endpoint=False).astype(int)

    # num_of_colors = n_f
    num_of_colors = 10
    colormap = cm.get_cmap('tab10', num_of_colors)
    colors = colormap(np.linspace(0.0, 1.0, num_of_colors))

    legend_labels = [
        'True',
        'Mean',
        'Mean with std. dev.']

    fig = plt.figure(2, figsize=(1080*PX, 1080*PX), dpi=DPI)

    for i in range(n_f):
        axis = fig.add_subplot(int('{:d}1{:d}'.format(n_f, i + 1)))
        for l_id in ['top', 'bottom', 'left', 'right']:
            axis.spines[l_id].set_linewidth(0.5)
        axis.tick_params(width=0.5)
        axis.set_xlim(xticklabels[0], xticklabels[-1])
        axis.set_ylim(limits[0], limits[1])
        axis.set_xticks(xticklabels)
        axis.set_yticks(yticklabels)
        if i == n_f - 1:
            axis.set_xticklabels(xticklabels, fontsize=LABEL_FONT_SIZE)
            axis.set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)
        else:
            axis.set_xticklabels([], fontsize=LABEL_FONT_SIZE)
        # axis.set_title(filters[i].id + ' filter', fontsize=TITLE_FONT_SIZE)
        axis.text(0.5, 0.9, filters[i].id + ' filter', horizontalalignment='center',
                  transform=axis.transAxes, fontsize=TITLE_FONT_SIZE)
        axis.set_ylabel('Cardinality', fontsize=LABEL_FONT_SIZE)
        axis.set_yticklabels(yticklabels, fontsize=LABEL_FONT_SIZE)
        axis.grid(c='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        axis.step(time_steps, ground_truth.N, 'k',
                  linewidth=0.5, label=legend_labels[0])
        color_i = colors[(i % num_of_colors), :]
        axis.plot(time_steps, filters[i].n, c=color_i,
                  linewidth=0.5, label=legend_labels[1])
        axis.plot(time_steps, filters[i].n + np.sqrt(filters[i].var_n), c=color_i,
                  linewidth=0.5, linestyle='--', label=legend_labels[2])
        axis.plot(time_steps, filters[i].n - np.sqrt(filters[i].var_n), c=color_i,
                  linewidth=0.5, linestyle='--', label='_nolegend_')
        axis.legend(loc=2, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.91])
    plt.subplots_adjust(top=0.98, bottom=0.08, right=0.98, left=0.08,
                        hspace=0.0, wspace=0.0)
    plt.margins(0, 0)

    # Draw
    plt.draw()
    plt.show(block=False)

    # Save figure
    if save_figure:
        plt.savefig(os.path.join(FIGURES_PATH, 'Cardinality_versus_time.pdf'))

    plt.close()

def plot_ospa_performance(ground_truth, filters, save_figure=True):
    n_f = len(filters)
    time_steps = np.arange(ground_truth.K)

    xticklabels = np.linspace(
        time_steps[0], time_steps[-1] + 1, 6, endpoint=True).astype(int)
    yticklabels = np.linspace(0.0, 100.0, int(
        100 / 10) + 1, endpoint=True).astype(int)

    # Plot MOSPA versus time
    markers = ['+', 'o', 'x', '*', 's',
               '^', 'v', 'h']
    # num_of_colors = n_f
    num_of_colors = 10
    colormap = cm.get_cmap('tab10', num_of_colors)
    colors = colormap(np.linspace(0.0, 1.0, num_of_colors))

    fig = plt.figure(3, figsize=(1080*PX, 1920*PX), dpi=DPI)

    axis = fig.add_subplot(111)
    for l_id in ['top', 'bottom', 'left', 'right']:
        axis.spines[l_id].set_linewidth(0.5)
    axis.tick_params(width=0.5)
    axis.set_xlim(xticklabels[0], xticklabels[-1])
    axis.set_xlabel('Time (s)', fontsize=LABEL_FONT_SIZE)
    axis.set_xticks(xticklabels)
    axis.set_xticklabels(xticklabels, fontsize=LABEL_FONT_SIZE)
    axis.set_ylabel('Mean OSPA metric', fontsize=LABEL_FONT_SIZE)
    axis.set_yticks(yticklabels)
    axis.set_yticklabels(yticklabels, fontsize=LABEL_FONT_SIZE)
    axis.grid(c='gray', linestyle='--', linewidth=0.25, alpha=0.25)
    for i in range(n_f):
        axis.plot(time_steps, filters[i].ospa, marker=markers[i % n_f], markersize=np.sqrt(MARKER_SIZE), markeredgewidth=0.2,
                  markerfacecolor='none', c=colors[i % num_of_colors, :], linewidth=0.5, label=filters[i].id)
    axis.legend(loc=1, fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout(rect=[0.0, 0.01, 1.0, 0.99])
    plt.subplots_adjust(top=0.98, bottom=0.08, right=0.98, left=0.10,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    # Draw
    plt.draw()
    plt.show(block=False)

    # Save figure
    if save_figure:
        plt.savefig(os.path.join(FIGURES_PATH, 'MOSPA_versus_time.pdf'))

    plt.close()

# Main function
def generate_plots(ground_truth, measurement_sets, filters, model, save_figure=True):
    # Check if the figures directory exists, if not creates it
    if not os.path.isdir(FIGURES_PATH):
        os.mkdir(FIGURES_PATH)
    plot_tracks(ground_truth, measurement_sets, filters, model, save_figure)
    plot_cardinality_performance(ground_truth, filters, save_figure)
    plot_ospa_performance(ground_truth, filters, save_figure)

if __name__ == "__main__":
    # Configure parser
    argparser = argparse.ArgumentParser(
        description='Generate plots from tracking results using multi-object filters.')
    argparser.add_argument(
        'results_files',
        metavar='<input results file(s)>',
        type=str,
        nargs='+',
        help='Pickle file(s) containing results from a previous run.')
    # Parse arguments
    args = argparser.parse_args()
    results_files_ = [os.path.abspath(os.path.expanduser(r_file)) for r_file in args.results_files]
    results_files = [r_file for r_file in results_files_ if os.path.exists(r_file)]

    if len(results_files) == 0:
        cprint('No provided results file exist.', 'red')
        exit()

    # Load data from files and consolidate results
    if len(results_files) > 1:
        # Load data
        results_all = dict()
        measurement_sets_all = dict()
        for r_file in results_files:
            with open(r_file, 'rb') as f:
                truth, measurement_set_list_, performance_results_, model = pickle.load(f)
            for i, res in enumerate(performance_results_):
                if not res.id in results_all.keys():
                    results_all[res.id] = [res]
                    measurement_sets_all[res.id] = [measurement_set_list_[i]]
                else:
                    results_all[res.id].append(res)
                    measurement_sets_all[res.id].append(measurement_set_list_[i])
        # Consolidate results
        performance_results = []
        measurement_set_list = []
        tracker_ids = results_all.keys()
        for t_id in tracker_ids:
            result = Result(t_id)
            # Collect results
            num_of_runs = len(results_all[t_id])
            n_ = np.vstack([results_all[t_id][i].n for i in range(num_of_runs)])
            var_n_ = np.vstack([results_all[t_id][i].var_n for i in range(num_of_runs)])
            ospa_ = np.vstack([results_all[t_id][i].ospa for i in range(num_of_runs)])
            sq_err_ = np.vstack([results_all[t_id][i].sq_err for i in range(num_of_runs)])
            run_time_ = np.array([results_all[t_id][i].run_time for i in range(num_of_runs)])
            prd_time_ = np.array([results_all[t_id][i].prd_time for i in range(num_of_runs)])
            gat_time_ = np.array([results_all[t_id][i].gat_time for i in range(num_of_runs)])
            upd_time_ = np.array([results_all[t_id][i].upd_time for i in range(num_of_runs)])
            mgm_time_ = np.array([results_all[t_id][i].mgm_time for i in range(num_of_runs)])
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

            if num_of_runs == 1:
                run_id = 0
            else:
                run_id = np.argmin(np.sum(ospa_, axis=1) * np.sum(sq_err_, axis=1))

            # Get exemplary run
            result.X = results_all[t_id][run_id].X
            result.labels = results_all[t_id][run_id].labels
            result.has_labels = results_all[t_id][run_id].has_labels
            result.label_max = results_all[t_id][run_id].label_max
            performance_results.append(result)
            measurement_set_list.append(measurement_sets_all[t_id][run_id])
    else:
        with open(results_files[0], 'rb') as f:
            truth, measurement_set_list, performance_results, model = pickle.load(f)

    cprint('Average run times:', 'green')
    for res in performance_results:
        cprint('{:<15} : {:08.5f} seconds'.format(res.id + ' filter', res.run_time), 'green')
    
    generate_plots(truth, measurement_set_list, performance_results, model, save_figure=True)
    cprint('Plots saved in folder \'figures/\'.', 'yellow')
    
