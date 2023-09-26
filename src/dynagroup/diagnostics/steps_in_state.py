from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.sample_weights import (
    make_sample_weights_which_mask_the_initial_timestep_for_each_event,
)
from dynagroup.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


def plot_steps_assigned_to_state(
    outcomes_jk: NumpyArray2D,
    predictors_jk: NumpyArray2D,
    j: int,
    k: int,
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Relative to the "kmeans" diagnostics plots:
        PRO: These show not only the discrete derivative, but also the location of origin.
        CON: There is a separate plot for each state.

    Arguments:
        outcomes_jk:  The outcomes for the j-th entity that were assigned to the k-th state.
        predictors_jk: The observations immediately before the outcomes.
        j: entity index, used for title
        k: state index, used for title
    """
    # Create a figure and axis
    plt.close("all")
    fig, ax = plt.subplots()

    # Add arrows between pairs of points
    for i in range(len(outcomes_jk)):
        ax.annotate(
            "",
            xytext=predictors_jk[i],
            xy=outcomes_jk[i],
            arrowprops=dict(arrowstyle="->", color="r"),
        )

    plt.title(f"Steps for entity {j} in state {k}.")
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(
            save_dir + f"{basename_prefix}_steps_assigned_to_state_for_entity_{j}_state_{k}.pdf"
        )


def plot_steps_within_examples_assigned_to_each_entity_state(
    continuous_states: NumpyArray3D,
    continuous_state_labels: NumpyArray2D,
    example_end_times: Optional[NumpyArray1D],
    use_continuous_states: Optional[NumpyArray2D],
    K: int,
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Arguments:
        continuous_states: has shape (T,J,D)
        continuous_state_labels:  has shape (T,J).
            Gives a label to the continuous state for each (t,j).
        example_end_times: optional, has shape (E+1,)
            Provides `example` boundaries, which allows us to interpret a time series of shape (T,J,:)
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across i.i.d "examples".
            An example boundary might be induced by a large time gap between timesteps, and/or a discontinuity in the continuous states x.

            If there are E examples, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th example ended.
            So to get the timesteps for the e-th example, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].

        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is True if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step (on STP, ETP, or CSP), nor the VES step.
            We leave the VEZ steps as is, though.
        K: number of possible labels
    """

    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        continuous_states,
        example_end_times,
        use_continuous_states,
    )

    J = np.shape(continuous_states)[1]
    for j in range(J):
        for k in range(K):
            timesteps_assigned_to_jk_regardless_of_sample_weight = np.where(
                continuous_state_labels[:, j] == k
            )[0]
            timesteps_with_nonzero_sample_weight = np.where(sample_weights[:, j])[0]
            # Find timesteps for entity j assigned to label k which also have nonzero sample weight.
            timesteps_assigned_to_jk = np.sort(
                list(
                    set(timesteps_assigned_to_jk_regardless_of_sample_weight)
                    & set(timesteps_with_nonzero_sample_weight)
                )
            )
            if len(timesteps_assigned_to_jk) > 0:
                responses_jk = np.array(continuous_states[timesteps_assigned_to_jk, j])
                predictors_jk = np.array(continuous_states[timesteps_assigned_to_jk - 1, j])
            else:
                responses_jk, predictors_jk = [], []
            plot_steps_assigned_to_state(
                responses_jk, predictors_jk, j, k, save_dir, show_plot, basename_prefix
            )
