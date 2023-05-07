from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from dynagroup.supra.changepoints import SeriesAndChangePoints
from dynagroup.supra.directions import LABELS_OF_DIRECTIONS, RADIANS_OF_DIRECTIONS


COLOR_CYCLE = ["#4286f4", "#f44174"]


def plot_changepoint_dict(
    changepoints_dict: Dict[str, SeriesAndChangePoints],
    changepoint_color="k",
    changepoint_linewidth=3,
    changepoint_linestyle="--",
    changepoint_alpha=1.0,
    **kwargs
):
    """Display time series and change points provided in alternating colors.
    The following matplotlib subplots options is set by
    default, but can be changed when calling  this function:

    - figure size `figsize`, defaults to `(10, 2 * n_features)`.

    Args:
        changepoints_dict
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axis_array) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """

    n_series = len(changepoints_dict)

    n_cols = 2
    n_rows = int(np.ceil(n_series / n_cols))

    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 1.3 * n_rows),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # create plots
    fig, axis_array = plt.subplots(n_rows, n_cols, sharex=True, **matplotlib_options)

    if n_rows == 1:
        axis_array = axis_array.reshape(1, -1)
    if n_cols == 1:
        axis_array = axis_array.reshape(-1, 1)

    # check that dict all has the same number of samples; if not, raise error.
    for i, (data_name, series_and_cps) in enumerate(changepoints_dict.items()):
        series, changepoints = series_and_cps.series, series_and_cps.changepoints
        n_samples = len(series)

        col, row = divmod(i, n_rows)
        ax = axis_array[row, col]

        ax.plot(range(n_samples), series)
        for changepoint in changepoints:
            ax.axvline(
                x=changepoint - 0.5,
                color=changepoint_color,
                linewidth=changepoint_linewidth,
                linestyle=changepoint_linestyle,
                alpha=changepoint_alpha,
            )

        # Below uses alternating colors... we'll save this for HMM, so as to not falsely suggest correspondences.
        #
        # from itertools import cycle
        # from ruptures.utils import pairwise
        #  color_cycle = cycle(COLOR_CYCLE)
        #
        #  markers = [0] + changepoints
        #  alpha = 0.2  # transparency of the colored background

        # for (start, end), col in zip(pairwise(markers), color_cycle):
        #     # # `axvspan` adds a vertical span (rectangle) across the Axes.
        #     # ax.axvspan(xmin=max(0, start - 0.5), xmax=end - 0.5, facecolor=col, alpha=alpha)

        ax.set_ylabel(data_name)

        if "heading" in data_name:
            ax.set_yticks(RADIANS_OF_DIRECTIONS)
            ax.set_yticklabels(LABELS_OF_DIRECTIONS)
    plt.tight_layout()
    return fig, axis_array
