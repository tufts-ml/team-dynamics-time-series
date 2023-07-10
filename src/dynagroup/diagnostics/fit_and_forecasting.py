import warnings
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from dynagroup.eda.show_trajectory_slices import plot_trajectory_slice
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.sampler import sample_team_dynamics
from dynagroup.types import JaxNumpyArray3D, NumpyArray1D, NumpyArray2D


def evaluate_posterior_mean_and_forward_simulation_on_slice(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    T_slice_max: int,
    model: Model,
    forecast_seeds: List[int],
    save_dir: str,
    entity_idxs: Optional[List[int]],
    find_t0_for_entity_sample: Callable,
    use_continuous_states: NumpyArray2D,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    filename_prefix: Optional[str] = "",
    figsize: Optional[Tuple[int]] = (8, 4),
) -> Tuple[NumpyArray1D, NumpyArray1D]:
    """

    By posterior mean and forward simulation, we mean:
        - posterior mean: Compute the posterior mean over a certain segment of time.
            (This is useful for evaluating model fit, assuming that the model was trained in such as way that
            the model saw observations for the period of time over which the posterior mean was computed.)
        - forward simulation: Forward sample from the model, given an initial trajectory and known
            future system states. (This is useful for partial forecasting, assuming the model
            was trained in such a way that the model learned system states without using info
            from a heldout entity.)

    Arguments:
        continuous_states: jnp.array with shape (T, J, D)
        T_slice_max: The attempted number of timesteps in the slice that we work with for fitting and forecasting.
            The actual T_slice could be less if there are not enough timesteps remaining in the sample.
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
        find_t0_for_entity_sample: Function converting entity sample (in (T,D)) to initial time for forecasting
        use_continuous_states: boolean array of shape (T,J). If true, the continuous states were observed
            (not masked) during initialization and inference.

    Returns:
        MSEs_posterior_mean, MSEs_forward_sim.  Each is an array of size (J,) that gives the performance for each of the
            J entities.
    """

    ###
    # Upfront info
    ###
    T, J, _ = np.shape(continuous_states)
    if entity_idxs is None:
        entity_idxs = [j for j in range(J)]

    ####
    # First plot the trajectory slices:
    ###

    for j in entity_idxs:
        t_0 = find_t0_for_entity_sample(continuous_states[:, j])
        plot_trajectory_slice(
            continuous_states[:, j],
            t_0,
            T_slice_max,
            j,
            x_lim,
            y_lim,
            save_dir,
            figsize=figsize,
        )

    ###
    # Now iterate over entities
    ###
    MSEs_posterior_mean = np.zeros(J)
    MSEs_forward_sim = np.zeros(J)

    for j in entity_idxs:
        ###
        # Find starting and ending point for entity
        ###
        sample_entity = continuous_states[:, j]  # TxD
        x_0 = sample_entity[t_0]
        t_end = np.min((t_0 + T_slice_max, T))
        T_slice = t_end - t_0

        ###
        # Determine if the observations were masked
        ###
        if use_continuous_states is None:
            had_masking = False
        else:
            had_masking = False in use_continuous_states[t_0:t_end, j]

        ###
        # Construct entity-specific filename tags
        ###
        DIMS = dims_from_params(params)
        D = DIMS.D
        if DIMS.L > 1:
            tag = f"{filename_prefix}_HSDM_entity_{j}_was_masked_{had_masking}"
        elif DIMS.L == 1:
            tag = f"{filename_prefix}_flat_SDM_entity_{j}_was_masked_{had_masking}"

        ###
        # Plotting the truth for the whole entity.
        ###
        print("Plotting the truth (whole)")
        plt.close("all")
        fig1 = plt.figure(figsize=figsize)
        im = plt.scatter(
            continuous_states[:, j, 0],
            continuous_states[:, j, 1],
            c=[i for i in range(T)],
            cmap="cool",
            alpha=1.0,
        )
        if y_lim:
            plt.ylim(y_lim)
        if x_lim:
            plt.xlim(x_lim)
        fig1.savefig(save_dir + f"truth_whole_entity_{j}.pdf")

        ###
        # Compute posterior means (useful for evaluting model fit)
        ###

        # Rk: It would also be possible to compute fit via sampling from `prob_entity_regimes_K`.
        A_j = params.CSP.As[j]
        b_j = params.CSP.bs[j]

        x_means = np.zeros((T_slice, D))
        x_means[0] = x_0
        times_of_interest = [t for t in range(t_0 + 1, t_end)]
        for i, time_of_interest in enumerate(times_of_interest):
            x_means_KD = A_j @ x_means[i] + b_j
            prob_entity_regimes_K = VEZ_summaries.expected_regimes[time_of_interest, j]
            x_means[i + 1] = np.einsum("kd, k -> d", x_means_KD, prob_entity_regimes_K)
            print(f"At time {i} the most likely regime is {np.argmax(prob_entity_regimes_K)}")

        # MSE
        ground_truth = continuous_states[t_0:t_end, j]
        MSE_fit = np.mean((ground_truth - x_means) ** 2)
        MSEs_posterior_mean[j] = MSE_fit

        # plot
        fig = plt.figure(figsize=figsize)
        plt.scatter(
            x_means[:, 0],
            x_means[:, 1],
            c=[i for i in range(T_slice)],
            cmap="cool",
            alpha=1.0,
        )
        plt.scatter(
            ground_truth[:, 0],
            ground_truth[:, 1],
            c=[i for i in range(T_slice)],
            cmap="cool",
            marker="x",
            alpha=0.25,
        )
        if y_lim:
            plt.ylim(y_lim)
        if x_lim:
            plt.xlim(x_lim)

        plt.title(f"MSE: {MSE_fit:.05f}")
        fig.savefig(save_dir + f"fit_via_posterior_means_{tag}.pdf")

        ###
        # Compute forward simulations (useful for evaluating partial forecasts)
        ###

        warnings.warn(
            f"The current implementation of (partial) forecasting assumes the interactions between entities "
            "are determined top-down by knowing the system regime.  Is this true for the current model?"
        )
        DIMS = dims_from_params(params)

        # Rk: Our implementation of foreward sampling is a hack.  We really want to sample the j-th entity's
        # trajectory given the system regimes.  But we don't have an API for that. So instead we just
        # initialize all J entities with the x_0^(j) of interest, and pull out the j-th sampled trajectories.
        # This implementaitonal strategy assumes that there is no recurrent feedback from the j'-th entities
        # to the j-th enity other than via the system regime.  Is that true?

        # Rk: The forecasting can be harder to get correct than the fitting because in forecasting we need
        # to know IN ADVANCE the entity-level transitions (z_{t+1}^^j | z_t^^j, ...), whereas in fitting
        # we already have good insight into these from the future data.
        # A good model would be able to predict the above entity-level transitions in advance based
        # on all the continuous states, x_t^^{1:J}.
        fixed_init_continuous_states = np.tile(x_0, (DIMS.J, 1))
        fixed_init_entity_regimes = np.argmax(VEZ_summaries.expected_regimes[t_0], axis=1)
        fixed_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)[t_0:t_end]

        for forecast_seed in forecast_seeds:
            print(
                f"Plotting the forward simulation for entity {j} using forecast_seed {forecast_seed}."
            )
            sample_ahead = sample_team_dynamics(
                params,
                T_slice,
                model,
                seed=forecast_seed,
                fixed_system_regimes=fixed_system_regimes,
                fixed_init_entity_regimes=fixed_init_entity_regimes,
                fixed_init_continuous_states=fixed_init_continuous_states,
            )

            # MSE
            ground_truth = continuous_states[t_0:t_end, j]
            MSE_forward_sim = np.mean((ground_truth - sample_ahead.xs[:, j]) ** 2)
            MSEs_forward_sim[j] = MSE_forward_sim

            # plot
            fig1 = plt.figure(figsize=figsize)
            im = plt.scatter(
                sample_ahead.xs[:, j, 0],
                sample_ahead.xs[:, j, 1],
                c=[i for i in range(t_0, t_end)],
                cmap="cool",
                alpha=1.0,
            )
            plt.scatter(
                ground_truth[:, 0],
                ground_truth[:, 1],
                c=[i for i in range(T_slice)],
                cmap="cool",
                marker="x",
                alpha=0.25,
            )
            if y_lim:
                plt.ylim(y_lim)
            if x_lim:
                plt.xlim(x_lim)
            plt.title(f"MSE: {MSE_forward_sim:.05f}. Had masking: {had_masking}")
            fig1.savefig(save_dir + f"forecast_{tag}_seed_{forecast_seed}.pdf")

        fig2 = plt.figure(figsize=(2, 6))
        cax = fig2.add_subplot()
        cbar = fig1.colorbar(im, cax=cax)
        cbar.set_label("Timesteps", rotation=90)
        plt.tight_layout()
        fig2.savefig(save_dir + "colorbar_clip.pdf")
    return MSEs_posterior_mean, MSEs_forward_sim
