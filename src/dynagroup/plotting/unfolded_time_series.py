import matplotlib.pyplot as plt
import numpy as np

from dynagroup.types import NumpyArray3D


def _get_x_and_y_coords(x_entity, period):
    T = np.shape(x_entity)[0]
    x_coords = np.zeros(T)
    y_coords = np.zeros(T)
    for t in range(T):
        timestep_adjustment = t / period
        # timestep_adjustment = 0
        x_coords[t] = x_entity[t, 0] + timestep_adjustment
        y_coords[t] = x_entity[t, 1]
    return x_coords, y_coords


def plot_unfolded_time_series(xs: NumpyArray3D, period_to_use=5):
    """
    Plot planar time series, x \in R^2, where we plot:
        * x_2 on the "y axis"
        * x_1 + time on the "x axis"

    In other words, we "unfold" the time series over time. The purpose
    of this is to let us visualize 2dim time series.

    Arguments:
        xs: has shape (T, J, D)

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    J = np.shape(xs)[1]

    plt.figure(figsize=(8, 9))

    for j in range(J):
        plt.subplot(J, 1, j + 1)
        x_entity = xs[:, j, :]
        x_coords, y_coords = _get_x_and_y_coords(x_entity, period_to_use)
        plt.ylabel(r"$x_2$")
        plt.xlabel(rf"$x_1$ + time/{period_to_use}")
        plt.plot(x_coords, y_coords)
        plt.title(f"Entity {j+1}")

    plt.tight_layout()
    plt.show()
