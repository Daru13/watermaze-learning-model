import numpy as np
from numpy.linalg import norm
from numpy.random import choice
from tqdm import tqdm

from constants import *
from utilities import get_random_point_in_disc




class PlaceCells:
    '''
    Set of place cells, whose centers are randomly distributed all over a watermaze.
    The activation of each cell (output) depends on the given position (input).

    '''

    centers = None

    current_activation = None
    previous_activation = None

    positions_over_watermaze = None
    activations_over_watermaze = None


    def __init__(self):
        self.set_random_centers()
        self.reset_activations()

        self.init_positions_over_watermaze()
        self.init_activations_over_watermaze()


    def reset_activations(self):
        self.current_activation = np.zeros((NB_PLACE_CELLS))
        self.previous_activation = np.zeros((NB_PLACE_CELLS))  


    def set_random_centers(self):
        def get_random_center():
            return get_random_point_in_disc((X_ORIGIN, Y_ORIGIN), WATERMAZE_RADIUS)

        self.centers = np.array(
            [get_random_center() for _ in range(NB_PLACE_CELLS)]
        )

    
    def init_positions_over_watermaze(self):
        step = 0.05

        x_coords = np.arange(X_ORIGIN - WATERMAZE_RADIUS, X_ORIGIN + WATERMAZE_RADIUS + step, step)
        y_coords = np.arange(Y_ORIGIN - WATERMAZE_RADIUS, Y_ORIGIN + WATERMAZE_RADIUS + step, step)
        
        positions = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        origin = np.array([X_ORIGIN, Y_ORIGIN])
        distances_from_origin = norm(origin - positions, axis = 1)

        self.positions_over_watermaze = positions[distances_from_origin <= WATERMAZE_RADIUS]
    

    def init_activations_over_watermaze(self):
        self.activations_over_watermaze = np.array([self.activation_at(pos) for pos in self.positions_over_watermaze])


    def activation_at(self, position):
        return np.exp(- np.square(norm(self.centers - position, axis = 1))
                        / (2 * PLACE_CELL_STD * PLACE_CELL_STD))


    def update_activations(self, new_position):
        np.copyto(self.previous_activation, self.current_activation)
        self.current_activation = self.activation_at(new_position)




class Critic:
    '''
    Critic of the actor-critic model.

    It estimates the value function from a weighted sum of the place cell activations,
    whose weights can and should be updated after each action taken by the rat.
    '''

    weights = None


    def __init__(self):
        self.reset_weights()

    
    def reset_weights(self):
        self.weights = np.zeros((NB_PLACE_CELLS))


    def estimate_values_over_watermaze(self, place_cells):
        return [np.dot(activation, self.weights) for activation in place_cells.activations_over_watermaze]


    def update_weights(self, place_cells, reward):
        previous_value_estimate = np.dot(place_cells.previous_activation, self.weights)
        current_value_estimate = np.dot(place_cells.current_activation, self.weights)

        if reward == 1:
            current_value_estimate = 0.0

        error = reward + (TEMPORAL_DECAY * current_value_estimate) - previous_value_estimate
        self.weights += CRITIC_LEARNING_RATE * (error * place_cells.previous_activation)

        return error




class Actor:
    '''
    Actor of the actor-critic model.

    It computes action probabilities from a weighted sum of the place cell activations,
    whose weights can and should be updated after each action taken by the rat.
    '''

    weights = None

    actions = [
        "top_left",
        "top",
        "top_right",
        "right",
        "bottom_right",
        "bottom",
        "bottom_left",
        "left"
    ]

    
    def __init__(self):
        self.reset_weights()

    
    def reset_weights(self):
        self.weights = np.zeros((len(self.actions), NB_PLACE_CELLS))


    def compute_action_probabilities(self, place_cells):
        activations = np.dot(self.weights, place_cells.current_activation)

        # Retrieve the maximum activations to prevent np.exp from overflowing
        # (it does not affect the final probabilities, since they are divided by the sum)
        max_activation = np.max(activations)
        softmax_activations = np.exp(2.0 * (activations - max_activation))
        
        #return softmax_activations / (softmax_activations.sum() + np.finfo(float).eps)
        return softmax_activations / softmax_activations.sum()


    def compute_action_probabilities_over_watermaze(self, place_cells):
        activations = np.array([np.dot(self.weights, activation) for activation in place_cells.activations_over_watermaze])

        max_activation = np.max(activations, axis = 1)
        softmax_activations = np.exp(2.0 * (activations.T - max_activation))
        
        return softmax_activations / softmax_activations.sum()

    
    def update_weights(self, place_cells, direction, error):
        # Only update the weights of the chosen direction
        direction_index = self.actions.index(direction)
        self.weights[direction_index, :] += ACTOR_LEARNING_RATE * (error * place_cells.previous_activation)




class Rat:
    '''
    Rat model based on the actor-critic model.

    It comprises place cells, a critic and an actor,  which are used to
    simulate steps and trials (made up of steps) and to learn weights using RL.
    '''

    place_cells = None
    critic = None
    actor = None

    current_pos = None
    previous_pos = None
    starting_pos = [
        np.array([X_ORIGIN, Y_ORIGIN + (WATERMAZE_RADIUS * 0.9)]), # Top
        np.array([X_ORIGIN, Y_ORIGIN - (WATERMAZE_RADIUS * 0.9)]), # Bottom
        np.array([X_ORIGIN + (WATERMAZE_RADIUS * 0.9), Y_ORIGIN]), # Right
        np.array([X_ORIGIN - (WATERMAZE_RADIUS * 0.9), Y_ORIGIN])  # Left
    ]

    previous_pos_diff = None
    pos_diff_by_direction = {
        "top_left":       np.array([-0.35355, 0.35355 ]),
        "top":            np.array([0.0     , 1.0     ]),
        "top_right":      np.array([0.35355 , 0.35355 ]),
        "right":          np.array([1.0     , 0.0     ]),
        "bottom_right":   np.array([0.35355 , -0.35355]),
        "bottom":         np.array([0.0     , -1.0    ]),
        "bottom_left":    np.array([-0.35355, -0.35355]),
        "left":           np.array([-1.0    , 0.0     ])
    }


    def __init__(self):
        self.place_cells = PlaceCells()
        self.critic = Critic()
        self.actor = Actor()

        self.reset_position()

    
    def reset_position(self):
        self.current_pos = self.starting_pos[choice(len(self.starting_pos))].copy()
        self.previous_pos = None

        self.previous_pos_diff = np.array([0.0, 0.0])

        # Since the currennt/previous position have been reset,
        # the precomputed place cell activations must be reset as well
        self.place_cells.reset_activations()

    
    def reset(self):
        self.reset_position()

        # Reset all the weights
        self.critic.reset_weights()
        self.actor.reset_weights()


    def is_on_plateform(self, watermaze):
        return norm(self.current_pos - watermaze.plateform.center) <= watermaze.plateform.radius


    def move_to_next_pos(self, watermaze):
        # Save the current position as the previous one
        self.previous_pos = self.current_pos

        # Pick the direction at random according to the action probabilities
        probabilities = self.actor.compute_action_probabilities(self.place_cells)
        new_direction = choice(self.actor.actions, p = probabilities)
        
        # Compute the position difference
        new_pos_diff = self.pos_diff_by_direction[new_direction] * SWIMING_SPEED * TIME_PER_STEP
        pos_diff = (ACTION_MOMENTUM_RATIO * new_pos_diff) + ((1.0 - ACTION_MOMENTUM_RATIO) * self.previous_pos_diff)

        # If the new position is beyond the watermaze wall, reverse the new direction
        if norm(self.current_pos + pos_diff) >= WATERMAZE_RADIUS:
            pos_diff *= -1.0

        # Update the positions and the place cell activations
        self.current_pos += pos_diff
        self.previous_pos_diff = pos_diff

        self.place_cells.update_activations(self.current_pos)

        # Compute the reward for this move
        reward = 1 if self.is_on_plateform(watermaze) else 0

        return new_direction, reward


    def update_weights(self, direction, reward):
        error = self.critic.update_weights(self.place_cells, reward)
        self.actor.update_weights(self.place_cells, direction, error)

        return error


    def simulate_one_step(self, watermaze):
        direction, reward = self.move_to_next_pos(watermaze)
        error = self.update_weights(direction, reward)

        return direction, reward, error

    
    def simulate_n_steps(self, watermaze, nb_steps, show_progress_bar = True):
        log = {
            "direction": [],
            "reward": [],
            "error": [],
            "position": []
        }

        # Show a progress bar (over the steps) if required
        iterator = tqdm(range(nb_steps), desc = "Steps") if show_progress_bar else range(nb_steps)

        # Simulate all the steps
        for _ in iterator:
            direction, reward, error = self.simulate_one_step(watermaze)

            log["direction"].append(direction)
            log["reward"].append(reward)
            log["error"].append(error)
            log["position"].append(self.current_pos.copy())
        
        return log

    
    def simulate_one_trial(self, watermaze):
        log = {
            "direction": [],
            "reward": [],
            "error": [],
            "position": []
        }

        # Move the rat back to its initial position,
        # and iterate for at most TRIAL_TIMEOUT seconds (for the rat)
        self.reset_position()
        
        for _ in range(int(np.round(TRIAL_TIMEOUT / TIME_PER_STEP))):
            direction, reward, error = self.simulate_one_step(watermaze)

            log["direction"].append(direction)
            log["reward"].append(reward)
            log["error"].append(error)
            log["position"].append(self.current_pos.copy())

            # Stop trial if the rat reaches the plateform (reward = 1)
            if reward == 1:
                break
        
        # Log some additional data at the end of the step
        log["critic_values"] = self.critic.estimate_values_over_watermaze(self.place_cells)
        log["action_probabilities"] = self.actor.compute_action_probabilities_over_watermaze(self.place_cells)

        return log


    def simulate_n_trials(self, watermaze, nb_trials, show_progress_bar = True):
        logs = []

        # Show a progress bar (over the trials) if required
        iterator = tqdm(range(nb_trials), desc = "Trials") if show_progress_bar else range(nb_trials)

        # Simulate all the trials
        for _ in iterator:
            log = self.simulate_one_trial(watermaze)
            logs.append(log)
        
        return logs

        