from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.types import JaxNumpyArray3D


def plot_trajectory_slices(
    continuous_states: JaxNumpyArray3D,
    t_0: int,
    T_slice_max: int,
    entity_idxs: Optional[List[int]] = None,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    save_dir: Optional[str] = None,
    show_plot: Optional[bool] = False,
) -> None:
    """
    Arguments:
        continuous_states: jnp.array with shape (T, J, D)
        T_slice_max: The attempted number of timesteps in the slice that we work with for fitting and forecasting.
            The actual T_slice could be less if there are not enough timesteps remaining in the sample.
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
    """

    T, J, _ = np.shape(continuous_states)
    if entity_idxs is None:
        entity_idxs = [j for j in range(J)]

    for j in entity_idxs:
        t_end = np.min((t_0 + T_slice_max, T))

        print(f"Plotting the truth (clip) for entity {j}.")
        plt.close("all")
        fig = plt.figure(figsize=(4, 6))
        plt.scatter(
            continuous_states[t_0:t_end, j, 0],
            continuous_states[t_0:t_end, j, 1],
            c=[i for i in range(t_0, t_end)],
            cmap="cool",
            alpha=1.0,
        )
        if y_lim:
            plt.ylim(y_lim)
        if x_lim:
            plt.xlim(x_lim)
        if save_dir is not None:
            fig.savefig(save_dir + f"truth_clip_entity_{j}.pdf")
        if show_plot:
            plt.show()
