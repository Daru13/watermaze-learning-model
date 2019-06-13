import numpy as np
from tqdm import tqdm

from watermaze import Watermaze
from rat import Rat
from figures import TrialFigure, RatPerformanceFigure




class Experiment:
    '''
    Generic experiment class. It must be extended to be used.
    '''

    rat = None

    # It must be defined by concrete child classes!
    rat_perf_filename = None


    def __init__(self):
        self.rat = Rat()


    def run_n_times(self, nb_times):
        logs_of_all_runs = []

        for _ in tqdm(range(nb_times)):
            logs_of_all_runs.append(self.run_once(show_progress_bar = False))
        
        return logs_of_all_runs


    def plot_rat_performance(self, logs_of_all_runs):
        RatPerformanceFigure(logs_of_all_runs).save_and_close(self.rat_perf_filename + ".png")




class RMW(Experiment):
    '''
    Reference Memory in the Watermaze (RMW) experiment.
    '''
    
    first_watermaze = None
    second_watermaze = None


    def __init__(self):
        super().__init__()
        self.rat_perf_filename = "rmw-rat-performance"

        self.first_watermaze = Watermaze()
        self.second_watermaze = Watermaze()


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

            day = 1 + (index // 4)
            daily_index = 1 + (index % 4)

            filename = "rmw-day-{}-trial-{}".format(day, daily_index)
            TrialFigure(watermaze, self.rat, log).save_and_close(filename + ".png")




class DMP(Experiment):
    '''
    Delayed Matching-to-Place (DMP) experiment.
    '''

    watermazes = None


    def __init__(self):
        super().__init__()
        self.rat_perf_filename = "dmp-rat-performance"

        self.watermazes = [Watermaze() for _ in range(9)]
        

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

    
    def plot_one_run(self, logs):
        for index, log in tqdm(enumerate(logs), desc = "Trial plots (DMP)"):
            day = 1 + (index // 4)
            daily_index = 1 + (index % 4)

            filename = "dmp-day-{}-trial-{}".format(day, daily_index)
            TrialFigure(self.watermazes[day - 1], self.rat, log).save_and_close(filename + ".png")