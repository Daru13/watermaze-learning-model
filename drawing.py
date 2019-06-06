import numpy as np 
import matplotlib.pyplot as plt

import constants as cst


def draw_watermaze(axis, watermaze):
    circle = plt.Circle((watermaze.center[0], watermaze.center[1]),
                        watermaze.radius,
                        color = "#AAAAFF",
                        zorder = 0)

    axis.add_artist(circle)


def draw_plateform(axis, plateform):
    circle = plt.Circle((plateform.center[0], plateform.center[1]),
                        plateform.radius,
                        color = "#FF2222",
                        zorder = 10)

    axis.add_artist(circle)


def draw_place_cells(axis, place_cells):
    cell_centers_x = place_cells.centers[:, 0]
    cell_centers_y = place_cells.centers[:, 1]

    axis.scatter(cell_centers_x, cell_centers_y, zorder = 5)


def draw_rat_positions(axis, positions):
    positions_x = [pos[0] for pos in positions]
    positions_y = [pos[1] for pos in positions]

    axis.scatter(positions_x, positions_y,
                c = np.arange(len(positions)),
                cmap = "autumn",
                marker = ".",
                zorder = 20)

    axis.plot(positions_x, positions_y,
              color = "black",
              zorder = 25)


def draw_trial(watermaze, rat, log, trial_index = None, save_as_img = True, show_figure = False):
    figure, axis = plt.subplots(figsize = (10, 10))

    if trial_index is not None:
        axis.set_title("Trial {}".format(trial_index))

    draw_watermaze(axis, watermaze)
    draw_place_cells(axis, rat.place_cells)
    draw_plateform(axis, watermaze.plateform)
    draw_rat_positions(axis, log["position"])

    if save_as_img:
        figure.savefig("figures/trial-{}.png".format(trial_index))

    plt.show()
