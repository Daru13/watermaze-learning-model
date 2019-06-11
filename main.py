import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from experiments import RMW, DMP
from plot import plot_trial



# Reference Memory in the Watermaze (RMW) experiment
rmw = RMW()
logs_part_1, logs_part_2 = rmw.run_once()
rmw.plot_one_run(logs_part_1, logs_part_2)


# Delayed Matching-to-Place (DMP) experiment
dmp = DMP()
logs = dmp.run_once()
dmp.plot_one_run(logs)