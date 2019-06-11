from tqdm import tqdm

from watermaze import Watermaze
from rat_model import Rat
from plot import plot_trial


class Experiment:

    rat = Rat()



class RMW (Experiment):

    first_watermaze = Watermaze()
    second_watermaze = Watermaze()


    def set_new_random_plateforms(self):
        self.first_watermaze.set_random_plateform()
        self.second_watermaze.set_random_plateform()


    def run_once(self):
        # Move plateforms to random positions
        self.set_new_random_plateforms()

        # Run trials corresponding to the first 7 days (4 trials/day)
        logs_part_1 = self.rat.simulate_n_trials(self.first_watermaze, 7 * 4)

        # Run trials corresponding to the last 2 days (4 trials/day)
        logs_part_2 = self.rat.simulate_n_trials(self.second_watermaze, 2 * 4)

        return logs_part_1, logs_part_2


    def plot_one_run(self, logs_part_1, logs_part_2):
        for index, log in tqdm(enumerate(logs_part_1), desc = "Trial plots (days 1-7)"):
            # Useful indices for plot filenames
            day = 1 + (index // 4)
            daily_index = 1 + (index % 4)

            plot_trial(self.first_watermaze, self.rat, log,
                       trial_index = daily_index,
                       filename_prefix = "day-{}-trial-".format(day))

        for index, log in tqdm(enumerate(logs_part_2), desc = "Trial plots (days 8-9)"):
            # Useful indices for plot filenames
            day = 8 + (index // 4)
            daily_index = 8 + (index % 4)

            plot_trial(self.second_watermaze, self.rat, log,
                       trial_index = daily_index,
                       filename_prefix = "rmw-day-{}-trial".format(day))


class DMP (Experiment):

    watermazes = [Watermaze() for _ in range(9)]


    def set_new_random_plateforms(self):
        for watermaze in self.watermazes:
            watermaze.set_random_plateform()


    def run_once(self):
        # Move plateforms to random positions
        self.set_new_random_plateforms()

        # Run trials for 9 days (4 trials/day)
        return [self.rat.simulate_n_trials(self.watermazes[index], 4) for index in range(9)]

    
    def plot_one_run(self, logs_per_day):
        for day_index, day_logs in tqdm(enumerate(logs_per_day), desc = "Trial plots (days 1-9)"):
            for index, log in enumerate(day_logs):
                # Useful indices for plot filenames
                day = 1 + day_index
                daily_index = 1 + index

                plot_trial(self.watermazes[index], self.rat, log,
                        trial_index = daily_index,
                        filename_prefix = "dmp-day-{}-trial".format(day))