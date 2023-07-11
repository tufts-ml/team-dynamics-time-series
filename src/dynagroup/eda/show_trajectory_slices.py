from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D


def plot_trajectory_slices(
    continuous_states: JaxNumpyArray3D,
    t_0: int,
    T_slice_max: int,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    save_dir: Optional[str] = None,
    show_plot: Optional[bool] = False,
    figsize: Optional[Tuple[int]] = (4, 6),
) -> None:
    J = np.shape(continuous_states)[1]
    for j in range(J):
        plot_trajectory_slice(
            continuous_states[:, j],
            t_0,
            T_slice_max,
            entity_idx=j,
            x_lim=x_lim,
            y_lim=y_lim,
            save_dir=save_dir,
            show_plot=show_plot,
            figsize=figsize,
        )


def plot_trajectory_slice(
    continuous_states_for_entity: JaxNumpyArray2D,
    t_0: int,
    T_slice_max: int,
    entity_idx: int,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    save_dir: Optional[str] = None,
    show_plot: Optional[bool] = False,
    figsize: Optional[Tuple[int]] = (4, 6),
    title_prefix: Optional[str] = "truth_clip",
) -> None:
    """
    Arguments:
        continuous_states_for_entity: jnp.array with shape (T, D)
        T_slice_max: The attempted number of timesteps in the slice that we work with for fitting and forecasting.
            The actual T_slice could be less if there are not enough timesteps remaining in the sample.
    """

    T = np.shape(continuous_states_for_entity)[0]
    t_end = np.min((t_0 + T_slice_max, T))

    print(f"Plotting the truth (clip) for entity {entity_idx}.")
    plt.close("all")
    fig = plt.figure(figsize=figsize)
    plt.scatter(
        continuous_states_for_entity[t_0:t_end, 0],
        continuous_states_for_entity[t_0:t_end, 1],
        c=[i for i in range(t_0, t_end)],
        cmap="cool",
        alpha=1.0,
    )
    if y_lim:
        plt.ylim(y_lim)
    if x_lim:
        plt.xlim(x_lim)
    if save_dir is not None:
        fig.savefig(save_dir + title_prefix + f"_entity_{entity_idx}.pdf")
    if show_plot:
        plt.show()
