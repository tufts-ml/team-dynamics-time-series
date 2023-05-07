import numpy as np
from matplotlib import cm, pyplot as plt
from ruptures.utils import pairwise

from dynagroup.types import NumpyArray1D


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


def plot_time_series_with_regime_panels(
    series: NumpyArray1D, fitted_regime_sequence: NumpyArray1D, **kwargs
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

    # create an array of n colors from the colormap
    K = len(set(fitted_regime_sequence))
    cmap = cm.get_cmap("cool")  # choose a colormap
    colors = cmap(np.linspace(0, 1, K))

    # create plots
    fig, ax = plt.subplots(1, 1, **matplotlib_options)

    n_samples = len(series)
    ax.plot(range(n_samples), series, color="black")

    # Below uses alternating colors... we'll save this for HMM, so as to not falsely suggest correspondences.
    markers = _make_markers(fitted_regime_sequence)
    alpha = 0.2  # transparency of the colored background

    for start, end in pairwise(markers):
        k = fitted_regime_sequence[start]
        # `axvspan` adds a vertical span (rectangle) across the Axes.
        ax.axvspan(xmin=max(0, start - 0.5), xmax=end - 0.5, facecolor=colors[k], alpha=alpha)

    plt.tight_layout()
    return fig, ax
