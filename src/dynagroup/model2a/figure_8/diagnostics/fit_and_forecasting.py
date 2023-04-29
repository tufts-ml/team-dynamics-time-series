from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.sampler import sample_team_dynamics
from dynagroup.types import JaxNumpyArray3D


###
# HELPERS
###


def find_last_index_in_interval_where_array_value_is_close_to_desired_point(
    array: np.array,
    desired_point: np.array,
    starting_index: int,
    ending_index: int,
) -> float:
    closeness_threshold = 0.15

    for t in reversed(range(starting_index, ending_index)):
        if np.linalg.norm(array[t] - desired_point) < closeness_threshold:
            return t
    return np.nan


###
# MAIN
###


def plot_fit_and_forecast_on_slice(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    T_slice: int,
    model: Model,
    forecast_seeds: List[int],
    save_dir: str,
    entity_idxs: Optional[List[int]] = None,
) -> None:
    """
    Arguments:
        continuous_states: jnp.array with shape (T, J, D)
        T_slice: The number of timesteps in the slice that we work with for fitting and forecasting
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
    """

    ###
    # Upfront info
    ###
    T, J, _ = np.shape(continuous_states)
    if entity_idxs is None:
        entity_idxs = [j for j in range(J)]

    ###
    # START PROCESSING
    ###
    for j in entity_idxs:
        ###
        # Derived info
        ###
        sample_entity = continuous_states[:, j]  # TxD
        DIMS = dims_from_params(params)
        D, K = DIMS.D, DIMS.K
        if DIMS.L == 2:
            tag = f"HSDM_entity_{j}"
        elif DIMS.L == 1:
            tag = f"flat_SDM_entity_{j}"

        ###
        # Find starting point
        ###
        # I want to work with when we transition from up to down (timesteps 100-200)
        starting_x_of_interest = np.array([1, 1])
        t_0 = find_last_index_in_interval_where_array_value_is_close_to_desired_point(
            sample_entity,
            starting_x_of_interest,
            starting_index=0,
            ending_index=100,
        )
        x_0 = sample_entity[t_0]

        ###
        # Plotting the truth
        ###

        print("Plotting the truth (whole)")
        plt.close("all")
        fig1 = plt.figure(figsize=(4, 6))
        im = plt.scatter(
            continuous_states[:, j, 0],
            continuous_states[:, j, 1],
            c=[i for i in range(T)],
            cmap="cool",
            alpha=1.0,
        )
        fig1.savefig(save_dir + f"truth_whole_entity_{j}.pdf")

        fig2 = plt.figure(figsize=(2, 6))
        cax = fig2.add_subplot()
        cbar = fig1.colorbar(im, cax=cax)
        cbar.set_label("Timesteps", rotation=90)
        plt.tight_layout()
        fig2.savefig(save_dir + "colorbar_whole.pdf")

        print("Plotting the truth (clip)")
        plt.close("all")
        fig = plt.figure(figsize=(4, 6))
        plt.scatter(
            continuous_states[t_0 : t_0 + T_slice, j, 0],
            continuous_states[t_0 : t_0 + T_slice, j, 1],
            c=[i for i in range(t_0, t_0 + T_slice)],
            cmap="cool",
            alpha=1.0,
        )
        fig.savefig(save_dir + f"truth_clip_entity_{j}.pdf")

        ###
        # Function: compute fit via posterior means.
        ###

        A_j = params.CSP.As[j]
        b_j = params.CSP.bs[j]

        x_means = np.zeros((T_slice, D))
        x_means[0] = x_0
        times_of_interest = [t for t in range(t_0 + 1, t_0 + T_slice)]
        for i, time_of_interest in enumerate(times_of_interest):
            for k in range(K):
                x_means_KD = A_j @ x_means[i] + b_j
                prob_entity_regimes_K = VEZ_summaries.expected_regimes[time_of_interest, j]
                x_means[i + 1] = np.einsum("kd, k -> d", x_means_KD, prob_entity_regimes_K)

        plt.clf()
        fig = plt.figure(figsize=(4, 6))
        plt.scatter(
            x_means[:, 0],
            x_means[:, 1],
            c=[i for i in range(T_slice)],
            cmap="cool",
            alpha=1.0,
        )
        fig.savefig(save_dir + f"{tag}_fit.pdf")

        ###
        # Function: compute via simulation
        ###

        DIMS = dims_from_params(params)
        fixed_init_continuous_states = np.tile(x_0, (DIMS.J, 1))
        fixed_init_entity_regimes = np.argmax(VEZ_summaries.expected_regimes[t_0], axis=1)
        fixed_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)[t_0 : t_0 + T_slice]

        for forecast_seed in forecast_seeds:
            print(f"Plotting the forecast with the model using forecast_seed {forecast_seed}.")
            sample_ahead = sample_team_dynamics(
                params,
                T_slice,
                model,
                seed=forecast_seed,
                fixed_system_regimes=fixed_system_regimes,
                fixed_init_entity_regimes=fixed_init_entity_regimes,
                fixed_init_continuous_states=fixed_init_continuous_states,
            )

            plt.clf()
            fig1 = plt.figure(figsize=(4, 6))
            im = plt.scatter(
                sample_ahead.xs[:, j, 0],
                sample_ahead.xs[:, j, 1],
                c=[i for i in range(t_0, t_0 + T_slice)],
                cmap="cool",
                alpha=1.0,
            )
            plt.ylim(-2.5, 2.5)
            fig1.savefig(save_dir + f"{tag}_sample_ahead_forecast_seed_{forecast_seed}.pdf")

        fig2 = plt.figure(figsize=(2, 6))
        cax = fig2.add_subplot()
        cbar = fig1.colorbar(im, cax=cax)
        cbar.set_label("Timesteps", rotation=90)
        plt.tight_layout()
        fig2.savefig(save_dir + "colorbar_clip.pdf")
