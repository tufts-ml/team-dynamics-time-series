import copy
from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ruptures.utils import pairwise

from dynagroup.types import NumpyArray1D


color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")


def _make_markers(fitted_regime_sequence: NumpyArray1D) -> NumpyArray1D:
    """
    Find changepoints, and if needed, append [0] at the beginning and the final timepoint at the end

    Arguments:
        fitted_regime_sequence, an array of length (T,) whose t-th entry is in {1,...,K}
    """
    changepoints = list(np.where(fitted_regime_sequence[1:] != fitted_regime_sequence[:-1])[0] + 1)
    markers = changepoints
    if 0 not in markers:
        markers = [0] + markers
    if len(fitted_regime_sequence) not in markers:
        markers = markers + [len(fitted_regime_sequence)]
    return markers


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
    series: NumpyArray1D,
    fitted_regime_sequence: NumpyArray1D,
    time_labels: Optional[NumpyArray1D] = None,
    **kwargs
):
    """
    Display time series with regimes provided in background colors.

    Arguments:
        **kwargs : all additional keyword arguments are passed to the plt.subplots call.

    Returns:
        tuple: (figure, axis_array) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.
    """

    # let's set a sensible defaut size for the subplots
    matplotlib_options = {
        "figsize": (10, 8),  # figure size
    }
    # add/update the options given by the user
    matplotlib_options.update(kwargs)

    # remove unused regime sequences
    fitted_regime_sequence_relabeled = _relabel_regime_sequence_to_remove_unused_regimes(
        fitted_regime_sequence
    )

    # create plots
    fig, ax = plt.subplots(1, 1, **matplotlib_options)

    n_samples = len(series)
    ax.plot(range(n_samples), series, color="black")

    if time_labels is not None:
        ticks = list(range(0, n_samples, int(n_samples / 8))) + [n_samples - 1]
        ax.set_xticks(ticks, time_labels[ticks])

    # Below uses alternating colors... we'll save this for HMM, so as to not falsely suggest correspondences.
    markers = _make_markers(fitted_regime_sequence_relabeled)
    alpha = 0.2  # transparency of the colored background

    for start, end in pairwise(markers):
        k = fitted_regime_sequence_relabeled[start]
        # `axvspan` adds a vertical span (rectangle) across the Axes.
        ax.axvspan(xmin=max(0, start - 0.5), xmax=end - 0.5, facecolor=colors[k], alpha=alpha)

    plt.tight_layout()
    return fig, ax
