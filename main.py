import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import watermaze as wm
import rat_model as rm
from plot import plot_trial



# Create a rat model
watermaze = wm.Watermaze()
rat = rm.Rat()


# Run the experiment
logs = rat.simulate_n_trials(watermaze, 50)


# Print logged data
# for index, log in enumerate(logs):
#     log_df = pd.DataFrame(log)

#     print("\n=========== Trial n°{} ===========\n".format(index))
#     print(log_df.round(decimals = 3))

#     print("\n=== Critic weights ===\n")
#     print(rat.critic.weights)

#     print("\n=== Actor weights ===\n")
#     print(rat.actor.weights)

# for i, log in enumerate(logs):
#     print("==== TRIAL {} ====".format(i))
#     print(log["error"])
#     print()

# Draw the experiment
for index, log in tqdm(enumerate(logs), desc = "Trial plots"):
    plot_trial(watermaze, rat, log,
               trial_index = index)