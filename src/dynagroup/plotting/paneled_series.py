import copy
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ruptures.utils import pairwise

from dynagroup.types import NumpyArray1D, NumpyArray2D


SOME_COLOR_NAMES = ["windows blue", "red", "amber", "faded green", "dusty purple", "orange"]
SOME_COLORS = sns.xkcd_palette(SOME_COLOR_NAMES)
sns.set_style("white")
sns.set_context("talk")

MARKERS = list(matplotlib.markers.MarkerStyle.markers.keys())


def _make_segments(fitted_regime_sequence: NumpyArray1D) -> NumpyArray1D:
    """
    Find changepoints, and if needed, append [0] at the beginning and the final timepoint at the end

    Arguments:
        fitted_regime_sequence, an array of length (T,) whose t-th entry is in {1,...,K}
    """
    changepoints = list(np.where(fitted_regime_sequence[1:] != fitted_regime_sequence[:-1])[0] + 1)
    segments = changepoints
    if 0 not in segments:
        segments = [0] + segments
    if len(fitted_regime_sequence) not in segments:
        segments = segments + [len(fitted_regime_sequence)]
    return segments


# TODO: Handle this earlier up in the pipeline
def _relabel_regime_sequence_to_remove_unused_regimes(
    fitted_regime_sequence: NumpyArray1D,
) -> NumpyArray1D:
    num_regimes_nominally = max(fitted_regime_sequence) + 1
    regimes_used = set(fitted_regime_sequence)

    relabeled_regime_sequence = copy.copy(fitted_regime_sequence)
    for k in range(num_regimes_nominally):
        if k not in regimes_used:
            relabeled_regime_sequence[relabeled_regime_sequence >= k] -= 1

    return relabeled_regime_sequence


def plot_time_series_with_regime_panels(
    series: Union[NumpyArray1D, NumpyArray2D],
    fitted_regime_sequence: NumpyArray1D,
    time_labels: Optional[NumpyArray1D] = None,
    dim_labels: Optional[List[str]] = None,
    COLORS=None,
    regime_color_index_offset: Optional[int] = None,
    axis_to_write_on: Optional[matplotlib.axes._axes.Axes] = None,
    fig_to_write_on=None,
    figsize: Optional[Tuple[int]] = (8, 6),
    add_x_label: Optional[bool] = True,
    x_lim: Optional[Tuple[int]] = None,
    y_lim: Optional[Tuple[int]] = None,
    suppress_panels: bool = False,
    **kwargs
):
    """
    Display time series with regimes provided in background colors.

    Arguments:
        series: array of shape (T,D)
        dim_labels: List of D strings, one for each dimension
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axis_array) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """

    if COLORS is None:
        COLORS = SOME_COLORS

    if fig_to_write_on is None:
        # let's set a sensible defaut size for the subplots
        matplotlib_options = {"figsize": figsize}
        # add/update the options given by the user
        matplotlib_options.update(kwargs)

    # remove unused regime sequences
    fitted_regime_sequence_relabeled = _relabel_regime_sequence_to_remove_unused_regimes(
        fitted_regime_sequence
    )

    n_samples = np.shape(series)[0]
    n_dims = np.shape(series)[1] if series.ndim == 2 else 1

    # construct dimension labels
    if dim_labels is None:
        dim_labels = [None] * n_dims

    # create plots
    if axis_to_write_on is None:
        fig, ax = plt.subplots(1, 1, **matplotlib_options)
    else:
        fig = fig_to_write_on
        ax = axis_to_write_on

    for d in range(n_dims):
        dim_color = "black" if d % 2 == 0 else "navy"
        series_to_plot = series[:, d] if series.ndim == 2 else series
        ax.plot(
            range(n_samples),
            series_to_plot,
            color=dim_color,
            marker=MARKERS[d],
            markersize=10,
            label=dim_labels[d],
        )

    if time_labels is not None:
        ticks = np.linspace(0, n_samples - 1, 6, dtype=int)
        ax.set_xticks(ticks, time_labels[ticks])

    # Below uses alternating colors... we'll save this for HMM, so as to not falsely suggest correspondences.
    segments = _make_segments(fitted_regime_sequence_relabeled)
    alpha = 0.2  # transparency of the colored background

    if not suppress_panels:
        for start, end in pairwise(segments):
            k = fitted_regime_sequence_relabeled[start]
            # `axvspan` adds a vertical span (rectangle) across the Axes.
            ax.axvspan(
                xmin=max(0, start - 0.5),
                xmax=end - 0.5,
                facecolor=COLORS[k + regime_color_index_offset],
                alpha=alpha,
            )

    if dim_labels[0] is not None:
        plt.legend(loc="best")

    if add_x_label:
        ax.set_xlabel("Time step")

    if x_lim is not None:
        ax.set_ylim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    # ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.tight_layout()

    return fig, ax
