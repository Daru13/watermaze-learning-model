import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *


def create_trial_figure():
    figure = plt.figure(figsize = (15, 5))

    grid = plt.GridSpec(1, 3, wspace = 0, hspace = 0)

    watermaze_axis = figure.add_subplot(grid[0, 0], adjustable = "box")
    errors_axis = figure.add_subplot(grid[0, 1], adjustable = "box")
    values_axis = figure.add_subplot(grid[0, 2], adjustable = "box", projection = "3d")

    return figure, watermaze_axis, errors_axis, values_axis


def draw_watermaze(axis, watermaze):
    background_circle = plt.Circle((watermaze.center[0], watermaze.center[1]), watermaze.radius,
                                   color = "#CEDEF2", zorder = 0)
    border_circle = plt.Circle((watermaze.center[0], watermaze.center[1]), watermaze.radius,
                               color = "black", linewidth = 1, fill = False, zorder = 1)

    axis.add_artist(background_circle)
    axis.add_artist(border_circle)


def draw_plateform(axis, plateform):
    background_circle = plt.Circle((plateform.center[0], plateform.center[1]), plateform.radius,
                                   color = "white", zorder = 10)
    border_circle = plt.Circle((plateform.center[0], plateform.center[1]), plateform.radius,
                               color = "black", linewidth = 1, fill = False, zorder = 11)

    axis.add_artist(background_circle)
    axis.add_artist(border_circle)


def draw_place_cells(axis, place_cells):
    cell_centers_x = place_cells.centers[:, 0]
    cell_centers_y = place_cells.centers[:, 1]

    axis.scatter(cell_centers_x, cell_centers_y, zorder = 5)


def draw_rat_positions(axis, positions):
    axis.set_title("Rat positions", fontsize = 12)
    axis.set_xlim(X_ORIGIN - (WATERMAZE_RADIUS * 1.1), X_ORIGIN + (WATERMAZE_RADIUS * 1.1))
    axis.set_ylim(Y_ORIGIN - (WATERMAZE_RADIUS * 1.1), Y_ORIGIN + (WATERMAZE_RADIUS * 1.1))
    axis.set_axis_off()

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


def draw_critic_errors(axis, errors):
    axis.set_title("Critic error", fontsize = 12)
    axis.set_ylim(0, 1)

    axis.plot(np.absolute(errors))


def draw_actor_preferred_directions(axis, watermaze, rat, action_probabilities):
    axis.set_title("Preferred directions", fontsize = 12)
    axis.set_xlim(X_ORIGIN - (WATERMAZE_RADIUS * 1.1), X_ORIGIN + (WATERMAZE_RADIUS * 1.1))
    axis.set_ylim(Y_ORIGIN - (WATERMAZE_RADIUS * 1.1), Y_ORIGIN + (WATERMAZE_RADIUS * 1.1))
    axis.set_axis_off()

    draw_watermaze(axis, watermaze)

    # Draws arrows showing the preferred directions all over the watermaze
    positions_x = rat.place_cells.positions_over_watermaze[:, 0]
    positions_y = rat.place_cells.positions_over_watermaze[:, 1]

    most_probable_directions_indices = np.argmax(action_probabilities.T, axis = 1)
    most_probable_directions = [rat.actor.actions[i] for i in most_probable_directions_indices]

    arrow_directions_x = np.array([rat.pos_diff_by_direction[direction][0] for direction in most_probable_directions])
    arrow_directions_y = np.array([rat.pos_diff_by_direction[direction][1] for direction in most_probable_directions])

    most_probable_directions_probabilities = action_probabilities[most_probable_directions_indices][0]
    arrow_directions_x *= most_probable_directions_probabilities
    arrow_directions_y *= most_probable_directions_probabilities

    indices_to_keep = np.arange(0, len(positions_x), 8)
    axis.quiver(positions_x[indices_to_keep], positions_y[indices_to_keep],
                arrow_directions_x[indices_to_keep], arrow_directions_y[indices_to_keep])

    draw_plateform(axis, watermaze.plateform)


def draw_critic_values(axis, positions, critic_values):
    axis.set_title("Value function estimate", fontsize = 12, pad = 20)
    axis.set_zlim(0, 1)
    axis.set_xticks([X_ORIGIN - WATERMAZE_RADIUS, X_ORIGIN, X_ORIGIN + WATERMAZE_RADIUS])
    axis.set_yticks([Y_ORIGIN - WATERMAZE_RADIUS, Y_ORIGIN, Y_ORIGIN + WATERMAZE_RADIUS])

    positions_x = positions[:, 0]
    positions_y = positions[:, 1]

    axis.plot_trisurf(positions_x, positions_y, np.absolute(critic_values),
                      linewidth = 0.2,
                      edgecolor = "black",
                      cmap = "RdBu")


def plot_trial(watermaze, rat, log, trial_index = None, save_as_img = True, filename_prefix = "trial", show_figure = False):
    figure, watermaze_axis, errors_axis, values_axis = create_trial_figure()

    draw_watermaze(watermaze_axis, watermaze)
    #draw_place_cells(watermaze_axis, rat.place_cells)
    draw_plateform(watermaze_axis, watermaze.plateform)
    draw_rat_positions(watermaze_axis, log["position"])

    #draw_critic_errors(errors_axis, log["error"])
    draw_actor_preferred_directions(errors_axis, watermaze, rat, log["action_probabilities"])

    draw_critic_values(values_axis, rat.place_cells.positions_over_watermaze, log["critic_values"])

    if save_as_img:
        figure.tight_layout()
        figure.savefig("figures/{}-{}.png".format(filename_prefix, trial_index))

    if show_figure:
        plt.show()

    plt.close(fig = figure)


def create_rat_performance_figure():
    figure, axis = plt.subplots(figsize = (15, 9))

    # Set various axis parameters
    axis.set_title("Performance of the rat", fontsize = 22, pad = 10)
    axis.set_xlabel("Trials", fontsize = 16, labelpad = 20)
    axis.set_ylabel("Path length (m)", fontsize = 16, labelpad = 10)
    axis.set_ylim(0, TRIAL_TIMEOUT)
    axis.tick_params(length = 0, labelsize = 12)

    axis.yaxis.grid()

    # Group the labels by day using X labels
    axis.xaxis.set_major_locator(plt.FixedLocator([2.5 + (i * 4) for i in range(0, 10)]))
    axis.xaxis.set_major_formatter(plt.FixedFormatter(["Day " + str(i) for i in range(1, 10)]))

    return figure, axis
    

def plot_rat_performance(mean_nb_logs, filename = "rat-performance"):
    figure, axis = create_rat_performance_figure()

    # Plot the length of the path of each trial (day by day)
    path_lengths = mean_nb_logs * TIME_PER_STEP * SWIMING_SPEED

    for day_index in range(9):
        # Compute some parametrs depending on the day index
        trial_indices = np.arange((day_index * 4), (day_index * 4) + 4)
        plot_color = "#002340" if day_index % 2 == 0 else "#0068BF"
        background_color = "#FFFFFF" if day_index % 2 == 0 else "#E9EFF5"

        # Add a background color to the plots of the day
        axis.axvspan(0.5 + day_index * 4, 0.5 + (day_index + 1) * 4,
                     facecolor = background_color)

        # Plot the path lengths
        axis.plot(trial_indices + 1, path_lengths[trial_indices],
                  color = plot_color, marker = "o")

    # Save and close the figure
    figure.savefig("figures/{}.png".format(filename))
    plt.close(fig = figure)