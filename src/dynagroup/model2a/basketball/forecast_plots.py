from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.model2a.basketball.court import (
    COURT_AXIS_UNNORM,
    COURT_IMAGE,
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
)
from dynagroup.model2a.basketball.forecast_analysis import Metrics
from dynagroup.types import NumpyArray4D, NumpyArray5D


def plot_team_forecasts(
    forecasts_1: NumpyArray5D,
    forecasts_2: NumpyArray5D,
    ground_truth: NumpyArray4D,
    metrics_1: Metrics,
    metrics_2: Metrics,
    e: int,
    s_1: int,
    s_2: int,
    show_plot: bool = False,
    save_dir: Optional[str] = None,
    filename_prefix: str = "",
    figsize: Optional[Tuple[int]] = (20, 4),
):
    """
    Arguments:
        forecasts_1: Has shape (E,S,T_forecast,J,D)
            Forecasts for method 1.
        forecasts_2: Has shape (E,S,T_forecast,J,D)
            Forecasts for method 2.
        ground_truth: Has shape (E,T_forecast,J,D)
        metrics_1: Metrics for method 1.
        metrics_2: Metrics for method 2.
    """

    (E, S, T_forecast, J, D) = np.shape(forecasts_1)

    MSE_1 = metrics_1.CLE__MSE_ES[e, s_1]
    MSE_2 = metrics_2.CLE__MSE_ES[e, s_2]

    fig, axes = plt.subplots(ncols=5, figsize=figsize)

    common_x_min = X_MIN_COURT
    common_x_max = X_MAX_COURT
    common_y_min = Y_MIN_COURT
    common_y_max = Y_MAX_COURT

    for j in range(5):
        axes[j].imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
        axes[j].scatter(
            forecasts_1[e, s_1, :, j, 0],
            forecasts_1[e, s_1, :, j, 1],
            c=[i for i in range(T_forecast)],
            cmap="cool",
            alpha=1.0,
            zorder=1,
        )
        axes[j].scatter(
            forecasts_2[e, s_2, :, j, 0],
            forecasts_2[e, s_2, :, j, 1],
            c=[i for i in range(T_forecast)],
            cmap="Wistia",
            alpha=1.0,
            zorder=1,
        )
        axes[j].scatter(
            ground_truth[e, :, j, 0],
            ground_truth[e, :, j, 1],
            c=[i for i in range(T_forecast)],
            cmap="cool",
            marker="x",
            alpha=0.50,
            zorder=2,
        )

        common_x_min = np.min(
            [
                np.full(T_forecast, common_x_min),
                forecasts_1[e, s_1, :, j, 0],
                forecasts_2[e, s_2, :, j, 0],
                ground_truth[e, :, j, 0],
            ]
        )
        common_x_max = np.max(
            [
                np.full(T_forecast, common_x_max),
                forecasts_1[e, s_1, :, j, 0],
                forecasts_2[e, s_2, :, j, 0],
                ground_truth[e, :, j, 0],
            ]
        )
        common_y_min = np.min(
            [
                np.full(T_forecast, common_y_min),
                forecasts_1[e, s_1, :, j, 1],
                forecasts_2[e, s_2, :, j, 1],
                ground_truth[e, :, j, 1],
            ]
        )
        common_y_max = np.max(
            [
                np.full(T_forecast, common_y_max),
                forecasts_1[e, s_1, :, j, 1],
                forecasts_2[e, s_2, :, j, 1],
                ground_truth[e, :, j, 1],
            ]
        )

    for ax in axes:
        ax.set_xlim((common_x_min, common_x_max))
        ax.set_ylim((common_y_min, common_y_max))

    plt.suptitle(f"MSE_1 : {MSE_1:.02f}, MSE_2: {MSE_2:.02f}")
    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_dir:
        fig.savefig(save_dir + f"{filename_prefix}_sims_{s_1}_and_{s_2}.pdf")

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())
