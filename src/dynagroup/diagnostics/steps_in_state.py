from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.sample_weights import (
    make_sample_weights_which_mask_the_initial_timestep_for_each_event,
)
from dynagroup.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


###
# Structs
###
@dataclass
class EntityDataInState:
    """
    Gives data for entity j=1,...,J when in entity state k=1,...,K

    The responses_jk and predictors_jk both have shape (T_*, D),
    where T_* depends on how many observations were assigned to state k for entity j.

    Note that the --responses-- are what are assigned to the entity states;
    the predictors are simply the data from the timestep immediately preceding those responses.
    """

    responses_jk: NumpyArray2D
    predictors_jk: NumpyArray2D


###
# Functions
###


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
        plt.savefig(save_dir + f"{basename_prefix}_steps_assigned_to_state_for_entity_{j}_state_{k}.pdf")

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())


def get_entity_data_within_examples_and_assigned_to_entity_state(
    continuous_states: NumpyArray3D,
    continuous_state_labels: NumpyArray2D,
    example_end_times: Optional[NumpyArray1D],
    use_continuous_states: Optional[NumpyArray2D],
    j: int,
    k: int,
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
        j: entity index
        k: entity state index
    """

    # TODO: It is redundant to compute the sample_weights (for all J,K) each time we call this
    # (for some j,k). Is there some better way to structure this?  Note however that the runtime
    # is fast, at least on the Cleveland Starters basketball data, so this consideration might not
    # be important.
    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        continuous_states,
        example_end_times,
        use_continuous_states,
    )

    timesteps_assigned_to_jk_regardless_of_sample_weight = np.where(continuous_state_labels[:, j] == k)[0]
    timesteps_with_nonzero_sample_weight = np.where(sample_weights[:, j])[0]
    # Find timesteps for entity j assigned to label k which also have nonzero sample weight.
    timesteps_assigned_to_jk = np.sort(
        list(set(timesteps_assigned_to_jk_regardless_of_sample_weight) & set(timesteps_with_nonzero_sample_weight))
    )
    if len(timesteps_assigned_to_jk) > 0:
        responses_jk = np.array(continuous_states[timesteps_assigned_to_jk, j])
        predictors_jk = np.array(continuous_states[timesteps_assigned_to_jk - 1, j])
    else:
        responses_jk, predictors_jk = np.array([]), np.array([])
    return EntityDataInState(responses_jk, predictors_jk)


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

    J = np.shape(continuous_states)[1]
    for j in range(J):
        for k in range(K):
            entity_data_in_state = get_entity_data_within_examples_and_assigned_to_entity_state(
                continuous_states,
                continuous_state_labels,
                example_end_times,
                use_continuous_states,
                j,
                k,
            )
            plot_steps_assigned_to_state(
                entity_data_in_state.responses_jk,
                entity_data_in_state.predictors_jk,
                j,
                k,
                save_dir,
                show_plot,
                basename_prefix,
            )


def get_mean_discrete_derivatives_for_each_entity_state(
    continuous_states: NumpyArray3D,
    continuous_state_labels: NumpyArray2D,
    example_end_times: Optional[NumpyArray1D],
    use_continuous_states: Optional[NumpyArray2D],
    K: int,
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

    _, J, D = np.shape(continuous_states)
    mean_steps_in_state = np.zeros((J, K, D))
    for j in range(J):
        for k in range(K):
            entity_data_in_state = get_entity_data_within_examples_and_assigned_to_entity_state(
                continuous_states,
                continuous_state_labels,
                example_end_times,
                use_continuous_states,
                j,
                k,
            )
            diffs_jk = entity_data_in_state.responses_jk - entity_data_in_state.predictors_jk
            mean_steps_in_state[j, k] = np.mean(diffs_jk, 0)
    return mean_steps_in_state
