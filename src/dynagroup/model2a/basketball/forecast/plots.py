from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from dynagroup.model2a.basketball.court import (
    COURT_AXIS_UNNORM,
    COURT_IMAGE,
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
)
from dynagroup.types import NumpyArray1D, NumpyArray4D, NumpyArray5D


# Create custom colormaps starting partway through the colormaps
PURPLES = plt.get_cmap("Purples")
REDS = plt.get_cmap("Reds")
GREENS = plt.get_cmap("Greens")
GREYS = plt.get_cmap("Greys")

PURPLES_PARTWAY = LinearSegmentedColormap.from_list("Blues_Partway", PURPLES(np.linspace(0.5, 1, 256)))
REDS_PARTWAY = LinearSegmentedColormap.from_list("Blues_Partway", REDS(np.linspace(0.5, 1, 256)))
GREENS_PARTWAY = LinearSegmentedColormap.from_list("Blues_Partway", GREENS(np.linspace(0.5, 1, 256)))
GREYS_PARTWAY = LinearSegmentedColormap.from_list("Blues_Partway", GREYS(np.linspace(0.5, 1, 256)))

# CMAP_LIST_FOR_SAMPLES = ["Blues", "Reds", "Greens"]
# CMAP_FOR_TRUTH = "Greys"

CMAP_LIST_FOR_SAMPLES = [PURPLES_PARTWAY, REDS_PARTWAY, GREENS_PARTWAY]
CMAP_FOR_TRUTH = GREYS_PARTWAY


def get_plot_lims_for_column_and_row(
    model_list,
    forecasts_dict,
    ground_truth,
    s_matrix,
    e,
) -> Tuple[NumpyArray1D, NumpyArray1D, NumpyArray1D, NumpyArray1D]:
    M = len(model_list)
    J = 5
    x_mins = np.zeros((J))
    x_maxes = np.zeros((J))
    y_mins = np.zeros((M))
    y_maxes = np.zeros((M))

    (E, S, T_forecast, J, D) = np.shape(list(forecasts_dict.values())[0])

    for j in range(5):
        common_x_min = X_MIN_COURT
        common_x_max = X_MAX_COURT
        ### TODO: I shouldn't hardcode this temporary override used for my specific example
        common_x_max = 70
        for m, model in enumerate(model_list):
            common_x_min = np.min(
                np.hstack(
                    (
                        np.full(T_forecast, common_x_min)[:, None],
                        forecasts_dict[model][e, s_matrix[m], :, j, 0].T,
                        ground_truth[e, :, j, 0][:, None],
                    )
                )
            )
            common_x_max = np.max(
                np.hstack(
                    (
                        np.full(T_forecast, common_x_max)[:, None],
                        forecasts_dict[model][e, s_matrix[m], :, j, 0].T,
                        ground_truth[e, :, j, 0][:, None],
                    )
                )
            )
        x_mins[j] = common_x_min
        x_maxes[j] = common_x_max

    common_y_min = Y_MIN_COURT
    common_y_max = Y_MAX_COURT

    for m, model in enumerate(model_list):
        for j in range(5):
            common_y_min = np.min(
                np.hstack(
                    (
                        np.full(T_forecast, common_y_min)[:, None],
                        forecasts_dict[model][e, s_matrix[m], :, j, 1].T,
                        ground_truth[e, :, j, 1][:, None],
                    )
                )
            )
            common_y_max = np.max(
                np.hstack(
                    (
                        np.full(T_forecast, common_y_max)[:, None],
                        forecasts_dict[model][e, s_matrix[m], :, j, 1].T,
                        ground_truth[e, :, j, 1][:, None],
                    )
                )
            )

    for m, model in enumerate(model_list):
        y_mins[m] = common_y_min
        y_maxes[m] = common_y_max
    return x_mins, x_maxes, y_mins, y_maxes


def plot_team_forecasts_giving_multiple_samples_for_one_model_and_one_player(
    forecasts: NumpyArray5D,
    ground_truth: NumpyArray4D,
    e: int,
    j: int,
    s_list: List[int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    show_plot: bool = False,
    save_dir: Optional[str] = None,
    basename_before_extension: str = "",
    figsize: Optional[Tuple[int]] = (8, 6),
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

    (E, S, T_forecast, J, D) = np.shape(forecasts)

    # MSE_1 = metrics_1.CLE__MSE_ES[e, s_1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)

    # plot a few sampled forecasts
    for i, s in enumerate(s_list):
        ax.scatter(
            forecasts[e, s, :, j, 0],
            forecasts[e, s, :, j, 1],
            c=[i for i in range(T_forecast)],
            cmap=CMAP_LIST_FOR_SAMPLES[i],
            alpha=1.0,
            s=200,
            zorder=1,
        )
    # plot ground truth
    ax.scatter(
        ground_truth[e, :, j, 0],
        ground_truth[e, :, j, 1],
        c=[i for i in range(T_forecast)],
        cmap=CMAP_FOR_TRUTH,
        marker="x",
        alpha=1.0,
        s=200,
        zorder=2,
    )

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    # Turn off ticks on both x and y axes
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)

    # Turn off tick labels on both x and y axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # plt.suptitle(f"MSE_1 : {MSE_1:.02f}, MSE_2: {MSE_2:.02f}")
    plt.tight_layout()

    if show_plot:
        plt.show()

    if save_dir:
        fig.savefig(save_dir + f"{basename_before_extension}.pdf", bbox_inches="tight", pad_inches=0)

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())
