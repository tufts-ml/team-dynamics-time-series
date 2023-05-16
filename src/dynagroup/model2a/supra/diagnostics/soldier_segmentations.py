from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dynagroup.model2a.circle.directions import (
    LABELS_OF_DIRECTIONS,
    RADIANS_OF_DIRECTIONS,
)
from dynagroup.plotting.paneled_series import plot_time_series_with_regime_panels
from dynagroup.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


### Define a custom color palette
ENTITY_REGIME_COLOR_NAMES = [
    "maroon",
    "baby blue",
    "magenta",
    "teal",
    "lime",
    "cyan",
    "olive",
    "silver",
]

ENTITY_REGIME_COLORS = sns.xkcd_palette(ENTITY_REGIME_COLOR_NAMES)
sns.set_style("white")
sns.set_context("talk")


def compute_likely_soldier_regimes(
    VEZ_expected_regimes: NumpyArray3D,
) -> NumpyArray2D:
    # TODO: Viterbi would be better. Here we take the variational MAP
    T, J, K = np.shape(VEZ_expected_regimes)
    z_hats = np.zeros((T, J), dtype=np.int32)
    for j in range(J):
        z_hats[:, j] = np.argmax(VEZ_expected_regimes[:, j], axis=1)
    return z_hats


def polar_plot_the_soldier_headings_with_learned_segmentations(
    squad_angles: NumpyArray2D,
    clock_times: NumpyArray1D,
    likely_soldier_regimes: NumpyArray2D,
    save_dir: Optional[str] = None,
    basename_prefix: Optional[str] = "",
    show_plot: bool = True,
):
    """
    Arguments:
        likely_soldier_regimes: the return value of compute_likely_soldier_regimes.
    """
    T, J = np.shape(squad_angles)
    for j in range(J):
        print(f"Entity {j}")
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.scatter(
            squad_angles[:, j],
            clock_times,
            color=[ENTITY_REGIME_COLORS[z] for z in likely_soldier_regimes[:, j]],
        )
        ax.set_yticklabels([])
        # Set the tick locations and labels
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
        ax.set_xticklabels(["0 (W)", "45", "90 (S)", "135", "180 (E)", "225", "270 (N)", "315"])

        if save_dir is not None:
            fig.savefig(
                save_dir + f"{basename_prefix}_soldier_{j}_segmented_headings_on_circle.pdf"
            )
        if show_plot:
            plt.show()


def panel_plot_the_soldier_headings_with_learned_segmentations(
    squad_angles: NumpyArray2D,
    clock_times: NumpyArray1D,
    likely_soldier_regimes: NumpyArray2D,
    save_dir: Optional[str] = None,
    basename_prefix: Optional[str] = "",
    show_plot: bool = True,
):
    T, J = np.shape(squad_angles)

    for j in range(J):
        fig, ax = plot_time_series_with_regime_panels(
            squad_angles[:, j],
            likely_soldier_regimes[:, j],
            clock_times,
            COLORS=ENTITY_REGIME_COLORS,
        )
        ax.set_yticks(RADIANS_OF_DIRECTIONS)
        ax.set_yticklabels(LABELS_OF_DIRECTIONS)
        if save_dir is not None:
            fig.savefig(
                save_dir + f"{basename_prefix}_soldier_{j}_segmented_headings_on_line_segment.pdf"
            )
        if show_plot:
            plt.show()
