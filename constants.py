# Constants

X_ORIGIN = 0.0                      # coordinate
Y_ORIGIN = 0.0                      # coordinate

WATERMAZE_RADIUS = 1.0              # meters
PLATEFORM_RADIUS = 0.05             # meters

SWIMING_SPEED = 0.8                 # meters/second
SWIMING_MOMENTUM_RATIO = 1/3        # % of momentum in the movement

NB_PLACE_CELLS = 493                # quantity
PLACE_CELL_STD = 0.16               # meters

NB_ACTIONS = 8                      # quantity

CRITIC_WEIGHTS_UPDATE_SCALE = 0.1  # RL parameter
ACTOR_WEIGHTS_UPDATE_SCALE = 1.0    # RL parameter
TIME_PER_STEP = 0.1                 # seconds/simulation step

TRIAL_TIMEOUT = 120                 # seconds