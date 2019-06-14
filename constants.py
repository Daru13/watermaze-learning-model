# Constants used all over the sources

# Watermaze
X_ORIGIN = 0.0                          # coordinate
Y_ORIGIN = 0.0                          # coordinate

WATERMAZE_RADIUS = 1.0                  # meters
PLATEFORM_RADIUS = 0.05                 # meters


# Simulated rat
SWIMING_SPEED = 0.8                     # meters/second
ACTION_MOMENTUM_RATIO = 1/3             # percent (0-1)

TIME_PER_STEP = 0.1                     # seconds
TRIAL_TIMEOUT = 120                     # seconds


# Place cells
NB_PLACE_CELLS = 493                    # quantity
PLACE_CELL_STD = 0.16                   # meters


# Critic and Actor
TEMPORAL_DECAY = 0.95                   # RL parameter
CRITIC_LEARNING_RATE = 0.1              # RL parameter
ACTOR_LEARNING_RATE = 0.2               # RL parameter


# Figures
SAVED_FIGURES_PATH = "./figures/"       # path to directory 