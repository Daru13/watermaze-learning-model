import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import constants as cst


def create_trial_figure():
    figure = plt.figure(figsize = (15, 9))

    grid = plt.GridSpec(2, 5, wspace = 0.5, hspace = 0.5)

    watermaze_axis = figure.add_subplot(grid[:, :3])
    errors_axis = figure.add_subplot(grid[0, 3:])
    values_axis = figure.add_subplot(grid[1, 3:], projection = "3d")

    return figure, watermaze_axis, errors_axis, values_axis


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
    axis.set_title("Rat positions")

    axis.set_xlim(cst.X_ORIGIN - cst.WATERMAZE_RADIUS, cst.X_ORIGIN + cst.WATERMAZE_RADIUS)
    axis.set_ylim(cst.Y_ORIGIN - cst.WATERMAZE_RADIUS, cst.Y_ORIGIN + cst.WATERMAZE_RADIUS)

    axis.set_xticks([cst.X_ORIGIN - cst.WATERMAZE_RADIUS, cst.X_ORIGIN, cst.X_ORIGIN + cst.WATERMAZE_RADIUS])
    axis.set_yticks([cst.Y_ORIGIN - cst.WATERMAZE_RADIUS, cst.Y_ORIGIN, cst.Y_ORIGIN + cst.WATERMAZE_RADIUS])

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
    axis.set_title("Critic error")
    axis.set_ylim(0, 1)

    axis.plot(np.absolute(errors))


def draw_actor_preferred_directions(axis, rat, action_probabilities):
    axis.set_title("Preferred directions")
    axis.set_xticks([cst.X_ORIGIN - cst.WATERMAZE_RADIUS, cst.X_ORIGIN, cst.X_ORIGIN + cst.WATERMAZE_RADIUS])
    axis.set_yticks([cst.Y_ORIGIN - cst.WATERMAZE_RADIUS, cst.Y_ORIGIN, cst.Y_ORIGIN + cst.WATERMAZE_RADIUS])

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


def draw_critic_values(axis, positions, critic_values):
    axis.set_title("Value function estimate")
    axis.set_zlim(0, 1)
    axis.set_xticks([cst.X_ORIGIN - cst.WATERMAZE_RADIUS, cst.X_ORIGIN, cst.X_ORIGIN + cst.WATERMAZE_RADIUS])
    axis.set_yticks([cst.Y_ORIGIN - cst.WATERMAZE_RADIUS, cst.Y_ORIGIN, cst.Y_ORIGIN + cst.WATERMAZE_RADIUS])

    positions_x = positions[:, 0]
    positions_y = positions[:, 1]

    axis.plot_trisurf(positions_x, positions_y, np.absolute(critic_values),
                      linewidth = 0,
                      cmap = "RdBu")


def plot_trial(watermaze, rat, log, trial_index = None, save_as_img = True, show_figure = False):
    figure, watermaze_axis, errors_axis, values_axis = create_trial_figure()

    draw_watermaze(watermaze_axis, watermaze)
    #draw_place_cells(watermaze_axis, rat.place_cells)
    draw_plateform(watermaze_axis, watermaze.plateform)
    draw_rat_positions(watermaze_axis, log["position"])

    #draw_critic_errors(errors_axis, log["error"])
    draw_actor_preferred_directions(errors_axis, rat, log["action_probabilities"])

    draw_critic_values(values_axis, rat.place_cells.positions_over_watermaze, log["critic_values"])

    if save_as_img:
        figure.savefig("figures/trial-{}.png".format(trial_index))

    if show_figure:
        plt.show()

    plt.close(fig = figure)