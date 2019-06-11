import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from experiments import RMW, DMP
from plot import plot_trial
from utilities import print_title


# Reference Memory in the Watermaze (RMW) experiment
print_title("Reference Memory in the Watermaze (RMW)")

rmw = RMW()
logs_of_all_runs = rmw.run_n_times(50)

rmw.plot_rat_performance(logs_of_all_runs)
rmw.plot_one_run(logs_of_all_runs[-1])


# Delayed Matching-to-Place (DMP) experiment
print_title("Delayed Matching-to-Place (DMP)")

dmp = DMP()
logs_of_all_runs = dmp.run_n_times(50)

dmp.plot_rat_performance(logs_of_all_runs)
dmp.plot_one_run(logs_of_all_runs[-1])