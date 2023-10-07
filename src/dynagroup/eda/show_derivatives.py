from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from dynagroup.sample_weights import (
    make_sample_weights_which_mask_the_initial_timestep_for_each_event,
)
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D, NumpyArray1D


def plot_discrete_derivatives(
    continuous_states: JaxNumpyArray3D,
    example_end_times: Optional[NumpyArray1D] = None,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Plotting the discrete derivatives can be useful because during pre-initialization
    of the bottom-level rAR-HMM, we define the emissions parameters for each entity
    state in terms of velocities (discrete derivativs)
    """
    T, J, D = np.shape(continuous_states)

    if D != 2:
        raise NotImplementedError("This function currently assumes that the data has dimension 2.")

    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        continuous_states,
        example_end_times,
        use_continuous_states,
    )

    continuous_state_diffs = continuous_states[1:, :, :] - continuous_states[:-1, :, :]
    sample_weight_diffs = sample_weights[1:, :] * sample_weights[:-1, :]

    for j in range(J):
        xs = [x for (t, x) in enumerate(continuous_state_diffs[:, j, 0]) if sample_weight_diffs[t, j] == True]
        ys = [y for (t, y) in enumerate(continuous_state_diffs[:, j, 1]) if sample_weight_diffs[t, j] == True]

        # Create a hexbin plot
        plt.figure(figsize=(10, 8))
        plt.hexbin(xs, ys, gridsize=50, cmap="viridis", bins="log", mincnt=1)
        plt.colorbar(label="Log Count")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Hexbin plot of discrete derivatives for entity {j}")

        if show_plot:
            plt.show()
        if save_dir is not None:
            plt.savefig(save_dir + f"{basename_prefix}_hexbin_plot_of_discrete_derivatives_for_entity_{j}.pdf")
        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())
