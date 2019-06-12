import numpy as np
from tqdm import tqdm

from watermaze import Watermaze
from rat_model import Rat
from plot import plot_trial, plot_rat_performance


class RMW:

    rat = Rat()
    first_watermaze = Watermaze()
    second_watermaze = Watermaze()


    def set_new_random_plateforms(self):
        self.first_watermaze.set_random_plateform()
        self.second_watermaze.set_random_plateform()


    def run_once(self, show_progress_bar = True):
        # Reset some parameters
        self.rat.reset()
        self.set_new_random_plateforms()

        # Run trials corresponding to the first 7 days (4 trials/day)
        logs = self.rat.simulate_n_trials(self.first_watermaze, 7 * 4,
                                          show_progress_bar = show_progress_bar)

        # Run trials corresponding to the last 2 days (4 trials/day)
        logs += self.rat.simulate_n_trials(self.second_watermaze, 2 * 4,
                                          show_progress_bar = show_progress_bar)

        return logs


    def run_n_times(self, nb_times):
        logs_of_all_runs = []

        for _ in tqdm(range(nb_times)):
            logs_of_all_runs.append(self.run_once(show_progress_bar = False))
        
        return logs_of_all_runs


    def plot_one_run(self, logs):
        for index, log in tqdm(enumerate(logs), desc = "Trial plots (RMW)"):
            # Determine which watermaze corresponds to the current log
            watermaze = self.first_watermaze if index < (7 * 4) else self.second_watermaze

            # Useful indices for plot filenames
            day = 1 + (index // 4)
            daily_index = 1 + (index % 4)

            plot_trial(watermaze, self.rat, log,
                       trial_index = daily_index,
                       filename_prefix = "rmw-day-{}-trial".format(day))


    def plot_rat_performance(self, logs_of_all_runs):
        # For each run, count the number of logs of each trial (i.e. the number of rat moves)
        nb_logs_of_all_runs = np.array([[len(logs["position"]) for logs in logs_of_one_run]
                                        for logs_of_one_run in logs_of_all_runs])

        # Compute the mean number of rat moves and plot it
        mean_nb_logs = np.mean(nb_logs_of_all_runs, axis = 0)
        plot_rat_performance(mean_nb_logs, filename = "rmw-rat-performance")


class DMP:

    rat = Rat()
    watermazes = [Watermaze() for _ in range(9)]


    def set_new_random_plateforms(self):
        for watermaze in self.watermazes:
            watermaze.set_random_plateform()


    def run_once(self, show_progress_bar = True):
        # Reset some parameters
        self.rat.reset()
        self.set_new_random_plateforms()

        # Run trials for 9 days (4 trials/day)
        logs = []

        for index in range(9):
            logs += self.rat.simulate_n_trials(self.watermazes[index], 4,
                                               show_progress_bar = show_progress_bar)

        return logs


    def run_n_times(self, nb_times):
        logs_of_all_runs = []

        for _ in tqdm(range(nb_times)):
            logs_of_all_runs.append(self.run_once(show_progress_bar = False))
        
        return logs_of_all_runs

    
    def plot_one_run(self, logs):
        for index, log in tqdm(enumerate(logs), desc = "Trial plots (DMP)"):
            # Useful indices for plot filenames
            day = 1 + (index // 4)
            daily_index = 1 + (index % 4)

            plot_trial(self.watermazes[day - 1], self.rat, log,
                    trial_index = daily_index,
                    filename_prefix = "dmp-day-{}-trial".format(day))


    def plot_rat_performance(self, logs_of_all_runs):
        # For each run, count the number of logs of each trial (i.e. the number of rat moves)
        nb_logs_of_all_runs = np.array([[len(logs["position"]) for logs in logs_of_one_run]
                                        for logs_of_one_run in logs_of_all_runs])

        # Compute the mean number of rat moves and plot it
        mean_nb_logs = np.mean(nb_logs_of_all_runs, axis = 0)
        plot_rat_performance(mean_nb_logs, filename = "dmp-rat-performance")