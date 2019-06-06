import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import watermaze as wm
import rat_model as rm
import drawing as dw



# Create a rat model
watermaze = wm.Watermaze()
rat = rm.Rat()


# Run the experiment
#log = rat.simulate_n_steps(watermaze, 500)
logs = rat.simulate_n_trials(watermaze, 5)


# Print logged data
for index, log in enumerate(logs):
    log_df = pd.DataFrame(log)

    print("\n=========== Trial n°{} ===========\n".format(index))
    print(log_df.round(decimals = 3))

    print("\n=== Critic weights ===\n")
    print(rat.critic.weights)

    print("\n=== Critic weights ===\n")
    print(rat.actor.weights)


# Draw the experiment
#dw.draw_trial(watermaze, rat, logs[ 0]) # First trial
#dw.draw_trial(watermaze, rat, logs[-1]) # Last trial

for index, log in enumerate(logs):
    dw.draw_trial(watermaze, rat, log,
                  trial_index = index)