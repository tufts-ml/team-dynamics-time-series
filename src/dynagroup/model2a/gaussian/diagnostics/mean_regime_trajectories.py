from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.types import NumpyArray2D, NumpyArray3D, NumpyArray4D


"""
We plot deterministic trajectories with time evolution as color.
We can plot these using the true, initialized, or learned parameters.

Usage:
    xs = get_deterministic_trajectories(params_true.CSP.As, params_true.CSP.bs, num_time_samples=100)
    plot_deterministic_trajectories(xs, "true")

    xs = get_deterministic_trajectories(params_learned.CSP.As, params_learned.CSP.bs, num_time_samples=100)
    plot_deterministic_trajectories(xs, "learned")
"""


def get_deterministic_trajectories(
    As: NumpyArray4D,
    bs: NumpyArray3D,
    num_time_samples: int,
    x_init: Optional[NumpyArray2D] = None,
) -> NumpyArray4D:
    """
    Arguments:
        As: has shape J x K x D x D
        bs: has shape J x K x D
        x_init: has shape J x D

    Returns:
        xs: has shape (J,K, num_time_samples, D)
    """
    J, K, D, _ = np.shape(As)
    xs = np.zeros((J, K, num_time_samples, D))
    Ts = [i for i in range(1, num_time_samples)]

    if x_init is None:
        x_init = np.zeros((J, D))

    for j in range(J):
        for k in range(K):
            A = As[j, k]
            b = bs[j, k]

            xs[j, k, 0] = x_init[j]
            for t in Ts:
                xs[j, k, t] = A @ xs[j, k, t - 1] + b
    return xs


def plot_deterministic_trajectories(
    xs: NumpyArray4D,
    filename_prefix: str = "",
    title_postfix: str = "",
    state_entity_regimes_in_subplots: bool = True,
    save_dir: Optional[str] = None,
) -> None:
    """
    Arguments:
        xs: has shape (J,K, num_time_samples, D)
            The return value of get_deterministic_trajectories
    """
    J, K, num_time_samples, D = np.shape(xs)

    if D != 2:
        raise ValueError("We assume that the trajectories live in the plane.")

    if state_entity_regimes_in_subplots:
        make_entity_regime_str = lambda k: f", Regime {k}"
    else:
        make_entity_regime_str = lambda k: f""

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(J, K + 1, width_ratios=[1] * K + [0.15], wspace=0.1, hspace=0.3)

    Ts = [i for i in range(1, num_time_samples)]
    for j in range(J):
        # Plot the trajectories on the first two subplots
        k = 0
        ax0 = fig.add_subplot(gs[j, k])
        im0 = ax0.scatter(xs[j, k, 1:, 0], xs[j, k, 1:, 1], c=Ts, cmap="cool", alpha=1.0)
        ax0.set_ylabel("y")
        ax0.set_title(f"Entity {j}" + make_entity_regime_str(k))
        if j != J - 1:
            ax0.tick_params(axis="x", bottom=False, labelbottom=False)
        if j == J - 1:
            ax0.set_xlabel("x")

        for k in range(1, K):
            ax = fig.add_subplot(gs[j, k], sharex=ax0, sharey=ax0)
            ax.scatter(
                xs[j, k, 1:, 0],
                xs[j, k, 1:, 1],
                c=Ts,
                cmap="cool",
                alpha=1.0,
            )
            ax.set_title(f"Entity {j}" + make_entity_regime_str(k))
            # I wanted to do the below, but it also affected ax0.  So instead I use the tick_params method.
            # ax1.set_yticks([])
            # ax1.set_yticklabels([])
            ax.tick_params(axis="y", bottom=False, labelleft=False)
            if j != J - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if j == J - 1:
                ax.set_xlabel("x")

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[0, K])
    cbar = fig.colorbar(im0, cax=ax_cbar)

    # Add padding to the right of the colorbar
    fig.subplots_adjust(right=0.9)

    # Add title to colorbar
    cbar.set_label("Timesteps", rotation=90)

    # add suptitle
    plt.suptitle(f"{filename_prefix} trajectory colored by timestep {title_postfix}", fontsize=16)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir + f"{filename_prefix}_mean_regime_trajectories_{title_postfix}.pdf")

    plt.show()
