#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse

from experiments import RMW, DMP
from plot import plot_trial
from utilities import print_title




# Generic function to simulate one of the experiments
def simulate_experiment(experiment, nb_runs, plot_perf = True, plot_trials = True):
    logs_of_all_runs = experiment.run_n_times(nb_runs)

    if plot_perf:
        experiment.plot_rat_performance(logs_of_all_runs)
    if plot_trials:
        experiment.plot_one_run(logs_of_all_runs[-1])


# Simulate the Reference Memory in the Watermaze (RMW) experiment
def simulate_RMW(nb_runs, plot_perf = True, plot_trials = True):
    print_title("Reference Memory in the Watermaze (RMW)")
    simulate_experiment(RMW(), nb_runs, plot_perf = plot_perf, plot_trials = plot_trials)


# Simulate the Delayed Matching-to-Place (DMP) experiment
def simulate_DMP(nb_runs, plot_perf = True, plot_trials = True):
    print_title("Delayed Matching-to-Place (DMP)")
    simulate_experiment(DMP(), nb_runs, plot_perf = plot_perf, plot_trials = plot_trials)


# Parse command-line arguments to set up the experiment
def parse_CLI_arguments():
    parser = argparse.ArgumentParser(description = "Simulate RMW, DMP or both experiments. Constants can be changed in 'constants.py' file!")

    def is_positive_int(value):
        int_value = int(value)
        if int_value < 0:
            raise argparse.ArgumentTypeError("The number of simulations must >= 0.")
        return int_value

    parser.add_argument("-n", dest = "nb_runs", type = is_positive_int, default = 10,
                        help = "nb. of simulations of both experiments (default is 10).")
    parser.add_argument("--rmw", dest = "nb_runs_rmw", type = is_positive_int,
                        help = "nb. of simulations of RMW experiment.")
    parser.add_argument("--dmp", dest = "nb_runs_dmp", type = is_positive_int,
                        help = "nb. of simulations of RMW experiment.")
    parser.add_argument("--no-trial-plot", dest = "plot_trials", action = "store_false",
                        help = "only plot path length (do not plot any trial).")

    return parser.parse_args()


# In case this is called as a script...
if __name__ == "__main__":
    args = parse_CLI_arguments()

    nb_runs_rmw = args.nb_runs_rmw if args.nb_runs_rmw is not None else args.nb_runs
    nb_runs_dmp = args.nb_runs_dmp if args.nb_runs_dmp is not None else args.nb_runs

    if nb_runs_rmw > 0:
        simulate_DMP(nb_runs_rmw, plot_trials = args.plot_trials)
    if nb_runs_dmp > 0:
        simulate_DMP(nb_runs_dmp, plot_trials = args.plot_trials)