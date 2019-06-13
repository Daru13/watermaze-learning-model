# Constants used all over the sources

# Watermaze-related constants
X_ORIGIN = 0.0                          # coordinate
Y_ORIGIN = 0.0                          # coordinate

WATERMAZE_RADIUS = 1.0                  # meters
PLATEFORM_RADIUS = 0.05                 # meters


# Rat-related constants
SWIMING_SPEED = 0.8                     # meters/second
SWIMING_MOMENTUM_RATIO = 1/3            # % of momentum in the movement

NB_PLACE_CELLS = 493                    # quantity
PLACE_CELL_STD = 0.16                   # meters

NB_ACTIONS = 8                          # quantity

LEARNING_RATE = 0.7                     # RL parameter
CRITIC_WEIGHTS_UPDATE_SCALE = 0.01      # RL parameter
ACTOR_WEIGHTS_UPDATE_SCALE = 0.1        # RL parameter


# Simulation-related constants
TIME_PER_STEP = 0.1                     # seconds/simulation step
TRIAL_TIMEOUT = 120                     # seconds


# Figure-related constants
SAVED_FIGURES_PATH = "./figures/"       # path to directory