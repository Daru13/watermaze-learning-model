import numpy as np
from numpy import linalg as la
from numpy import random as rd
from tqdm import tqdm

import sys

import constants as cst
import utilities as ut



class PlaceCells:

    centers = None

    current_activation = None
    previous_activation = None


    def __init__(self):
        self.set_random_centers()
        self.reset_activations()


    def reset_activations(self):
        self.current_activation = np.zeros((cst.NB_PLACE_CELLS))
        self.previous_activation = np.zeros((cst.NB_PLACE_CELLS))  


    def set_random_centers(self):
        self.centers = np.array(
            [ut.get_random_point_in_disc((cst.X_ORIGIN, cst.Y_ORIGIN), cst.WATERMAZE_RADIUS) for _ in range(cst.NB_PLACE_CELLS)]
        )


    def activation_at(self, position):
        return np.exp(- np.square(la.norm(self.centers - position, axis = 1))
                        / (2 * cst.PLACE_CELL_STD * cst.PLACE_CELL_STD))


    def update_activations(self, new_position):
        np.copyto(self.previous_activation, self.current_activation)
        self.current_activation = self.activation_at(new_position)


    



class Critic:

    weights = np.zeros((cst.NB_PLACE_CELLS))


    #def __init__(self):

    
    def update_weights(self, place_cells, reward):
        error = reward + (cst.LEARNING_RATE * np.dot(place_cells.current_activation, self.weights) -
                np.dot(place_cells.previous_activation, self.weights))

        self.weights += error * self.weights

        #print("CRITIC WEIGHTS UPDATE")

        return error



class Actor:

    weights = {
        "top_left":       np.zeros((cst.NB_PLACE_CELLS)),
        "top":            np.zeros((cst.NB_PLACE_CELLS)),
        "top_right":      np.zeros((cst.NB_PLACE_CELLS)),
        "right":          np.zeros((cst.NB_PLACE_CELLS)),
        "bottom_right":   np.zeros((cst.NB_PLACE_CELLS)),
        "bottom":         np.zeros((cst.NB_PLACE_CELLS)),
        "bottom_left":    np.zeros((cst.NB_PLACE_CELLS)),
        "left":           np.zeros((cst.NB_PLACE_CELLS))
    }


    #def __init__(self):


    def compute_action_cells_activations(self, place_cells):
        activations = {}
        
        for direction, weights in self.weights.items():
            activations[direction] = np.dot(place_cells.current_activation, weights)

        #print("ACTIVATIONS")
        #print(activations)

        return activations


    def compute_action_probabilities(self, place_cells):
        activations = self.compute_action_cells_activations(place_cells)

        probabilities = {}
        softmax_sum = 0

        # Compute the 'softmax' activation of each direction
        for direction, activation in activations.items():
            softmax_activation = np.exp(2.0 * activation)
            
            probabilities[direction] = softmax_activation
            softmax_sum += softmax_activation

        # Divide each by the sum of them all to get probabilities
        for direction in probabilities.keys():
            probabilities[direction] /= softmax_sum

        #print("PROBABILITIES")
        #print(probabilities)

        return probabilities

    
    def update_weights(self, place_cells, direction, error):
        # Only update the weights of the chosen direction
        self.weights[direction] += error * place_cells.current_activation

        #print("ACTOR WEIGHTS UPDATE")



class Rat:

    current_pos = None
    previous_pos = None

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
        self.reset_position()

        self.place_cells = PlaceCells()

        self.critic = Critic()
        self.actor = Actor()

    
    def reset_position(self):
        self.current_pos = np.array([0.0, 0.0])
        self.previous_pos = np.array([0.0, 0.0])

        self.previous_pos_diff = np.array([0.0, 0.0])

    
    def reset(self):
        self.reset_position()
        self.place_cells.reset_activations()


    def is_on_plateform(self, watermaze):
        return la.norm(self.current_pos - watermaze.plateform.center) <= watermaze.plateform.radius


    def move_to_next_pos(self, watermaze):
        # Save the current position as the previous one
        self.previous_pos = self.current_pos

        # Get the probability of moving in each direction
        probabilities = self.actor.compute_action_probabilities(self.place_cells)

        # Pick the direction at random according to the above distribution
        new_direction = None

        random_number = rd.random()
        sum_to_random_number = 0.0

        for direction, probability in probabilities.items():
            sum_to_random_number += probability
            
            if sum_to_random_number >= random_number:
                new_direction = direction
                break
        
        # Compute the new position difference
        new_pos_diff = self.pos_diff_by_direction[new_direction] * cst.SWIMING_SPEED * cst.TIME_PER_STEP
        pos_diff = (cst.SWIMING_MOMENTUM_RATIO * new_pos_diff) + ((1.0 - cst.SWIMING_MOMENTUM_RATIO) * self.previous_pos_diff)

        # If the new position is beyond the watermaze wall, reverse the new direction
        if la.norm(self.current_pos + pos_diff) >= cst.WATERMAZE_RADIUS:
            pos_diff *= -1.0

        # Move to the new position
        self.current_pos += pos_diff
        self.previous_pos_diff = pos_diff

        #print("Moved to: {}".format(new_direction))

        # Update the place cell activations
        self.place_cells.update_activations(self.current_pos)

        # Compute the reward for this move
        reward = 1 if self.is_on_plateform(watermaze) else 0

        return new_direction, reward


    def update_weights(self, direction, reward):
        error = self.critic.update_weights(self.place_cells,
                                           reward)

        self.actor.update_weights(self.place_cells, 
                                  direction,
                                  error)

        #print("Weights updated")
        #print("Critic error: {}".format(error))

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
        iterator = tqdm(range(nb_steps)) if show_progress_bar else range(nb_steps)

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

        # Iterate for at most cst.STEP_TIMEOUT seconds
        iterator = ut.iterator_with_timeout(iter(range(sys.maxsize)), cst.TRIAL_TIMEOUT)

        # Reset the rat to make it start again (from the same point)
        self.reset()

        # Simulate all the steps
        for _ in iterator:
            direction, reward, error = self.simulate_one_step(watermaze)

            log["direction"].append(direction)
            log["reward"].append(reward)
            log["error"].append(error)
            log["position"].append(self.current_pos.copy())

            # Stop trial if reward = 1
            if reward == 1:
                break
        
        return log


    def simulate_n_trials(self, watermaze, nb_trials, show_progress_bar = True):
        # List of all the trial logs
        logs = []

        # Show a progress bar (over the trials) if required
        iterator = tqdm(range(nb_trials)) if show_progress_bar else range(nb_trials)

        # Simulate all the trials
        for _ in iterator:
            log = self.simulate_one_trial(watermaze)
            logs.append(log)
        
        return logs

        