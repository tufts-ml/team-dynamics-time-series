import os
import warnings
from enum import Enum
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynagroup.eda.show_trajectory_slices import plot_trajectory_slice
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.model2a.basketball.court import (
    COURT_AXIS_UNNORM,
    COURT_IMAGE,
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
    unnormalize_coords,
)
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.sampler import sample_team_dynamics
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D, NumpyArray1D, NumpyArray2D


###
# STRUCTS
###
class ForecastType(Enum):
    PARTIAL = 1
    COMPLETE = 2


###
# FUNCTIONS
###
def _had_masking_bool(
    use_continuous_states: NumpyArray2D,
    j: int,
    t_0: int,
    t_end: int,
) -> bool:
    ###
    # Determine if the observations were masked
    ###
    if use_continuous_states is None:
        return False
    else:
        return False in use_continuous_states[t_0:t_end, j]


def _get_filename_tag(
    use_continuous_states: NumpyArray2D,
    j: int,
    t_0: int,
    t_end: int,
    filename_prefix: str,
    L: int,
) -> str:
    """
    Arguments:
        use_continuous_states: boolean array with shape (T,J) specifying whether a given continuous state
            should be used during inference.
        j: index of entity currently under investigation
        L: Number of system-level regimes
    """

    had_masking = _had_masking_bool(use_continuous_states, j, t_0, t_end)

    ###
    # Construct entity-specific filename tags
    ###
    if L > 1:
        tag = f"{filename_prefix}_HSDM_entity_{j}_was_masked_{had_masking}"
    elif L == 1:
        tag = f"{filename_prefix}_flat_SDM_entity_{j}_was_masked_{had_masking}"
    return tag


def evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    forward_simulation_seeds: List[int],
    save_dir: str,
    use_continuous_states: NumpyArray2D,
    entity_idxs: Optional[List[int]],
    find_forward_sim_t0_for_entity_sample: Callable,
    max_forward_sim_window: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    find_posterior_mean_t0_for_entity_sample: Optional[Callable] = None,
    max_posterior_mean_window: Optional[int] = None,
    filename_prefix: Optional[str] = "",
    figsize: Optional[Tuple[int]] = (8, 4),
    forecast_type: ForecastType = ForecastType.PARTIAL,
) -> Tuple[NumpyArray1D, NumpyArray2D, NumpyArray1D]:
    """
    A helper function for write_model_evaluation_via_posterior_mean_and_forward_simulation_on_slice

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
        max_forward_sim_window: The attempted number of timesteps in the slice that we work with for forward simulating.
            The actual window size could be less if there are not enough timesteps remaining in the sample.
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
        find_forward_sim_t0_for_entity_sample: Function converting entity sample (in (T,D)) to initial time for forecasting
        use_continuous_states: boolean array of shape (T,J). If true, the continuous states were observed
            (not masked) during initialization and inference.

    Returns:
        MSEs_posterior_mean: An array of size (J,) that describes the model performance for each of the
            J entities over a time period requested for the posterior mean.
        MSEs_forward_sims: An array of size (J,S)  that describes the model performance for each of the
            J entities for each of S simulations over a time period requested for the forward sims.
            The value is NaN if the entity was not masked.
        MSEs_velocity_baseline: An array of size (J,) that describes the model performance for each of the
            J entities over the same time period as requested for the forward sims.
            The value is NaN if the entity was not masked.
    """
    # TODO: Rewrite this function so it builds off the `forecasts` module, which has more up-to-date code.
    # That code is better factored and returns nice objects for Forecast_Collection_For_Example, Forecast_MSEs_For_Example, etc.
    # This code also may have a bug such that the first timestep of fixed_velocity matches the truth,
    # but the first timestep of forward_simulations does not.  Ideally, we would just destroy this code, and
    # force all callers to use the new code.

    # Another possible problme with this code: the system state and entity states
    # knows what is happening in the future. See the error message below.  I think we CAN’T do complete forecasting
    # using the old code, where we “pick up where we left off” in processing a long time series.
    # Possibly this problem is handled though if we ablate all inputs (all entities) to the system state.
    if forecast_type != ForecastType.PARTIAL:
        raise ValueError(
            f"We can't do complete forecasting with this strategy; the VES summary and VEZ summaries at timestep t use "
            f"information from the future. "
        )

    ###
    # Constants
    ###
    X_LIM_COURT = (X_MIN_COURT, X_MAX_COURT)
    Y_LIM_COURT = (Y_MIN_COURT, Y_MAX_COURT)

    ###
    # Upfront info
    ###
    DIMS = dims_from_params(params)
    T, J, _ = np.shape(continuous_states)
    if entity_idxs is None:
        entity_idxs = [j for j in range(J)]
    if find_posterior_mean_t0_for_entity_sample is None:
        find_posterior_mean_t0_for_entity_sample = find_forward_sim_t0_for_entity_sample
    if max_posterior_mean_window is None:
        max_posterior_mean_window = max_forward_sim_window

    ###
    # Iterate over entities
    ###
    S = len(forward_simulation_seeds)
    MSEs_posterior_mean = np.zeros(J)
    MSEs_forward_sims = np.full((J, S), np.nan)  # value is NaN if entity was not masked
    MSEs_velocity_baseline = np.full(J, np.nan)  # value is NaN if entity was not masked.

    for j in entity_idxs:
        ###
        # Plotting the truth for the whole entity.
        ###

        print("Plotting the truth (whole)")
        plt.close("all")
        fig1 = plt.figure(figsize=figsize)
        plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
        continuous_states_j_unnorm = unnormalize_coords(continuous_states[:, j])
        im = plt.scatter(
            continuous_states_j_unnorm[:, 0],
            continuous_states_j_unnorm[:, 1],
            c=[i for i in range(T)],
            cmap="cool",
            alpha=1.0,
            zorder=1,  # Set zorder higher to draw the scatter plot over the image
        )
        plt.ylim(Y_LIM_COURT)
        plt.xlim(X_LIM_COURT)
        fig1.savefig(save_dir + f"truth_whole_entity_{j}.pdf")

        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())

        ###
        # Find starting and ending point for slice for entity
        ###
        sample_entity = continuous_states[:, j]  # TxD
        t_0_forward_sim = find_forward_sim_t0_for_entity_sample(continuous_states[:, j])
        x_0_forward_sim = sample_entity[t_0_forward_sim]
        t_end_forward_sim = np.min((t_0_forward_sim + max_forward_sim_window, T))
        T_slice_forward_sim = t_end_forward_sim - t_0_forward_sim

        t_0_posterior_mean = find_posterior_mean_t0_for_entity_sample(continuous_states[:, j])
        x_0_posterior_mean = sample_entity[t_0_posterior_mean]
        t_end_posterior_mean = np.min((t_0_posterior_mean + max_posterior_mean_window, T))
        T_slice_posterior_mean = t_end_posterior_mean - t_0_posterior_mean

        ###
        # Plot true trajectory slices
        ###
        plot_trajectory_slice(
            unnormalize_coords(continuous_states[:, j]),
            t_0_forward_sim,
            max_forward_sim_window,
            j,
            X_LIM_COURT,
            Y_LIM_COURT,
            save_dir,
            figsize=figsize,
            title_prefix="truth_for_clip",
        )

        ###
        # Compute posterior means (useful for evaluting model fit)
        ###

        # Rk: It would also be possible to compute fit via sampling from `prob_entity_regimes_K`.
        A_j = params.CSP.As[j]
        b_j = params.CSP.bs[j]

        x_means = np.zeros((T_slice_posterior_mean, DIMS.D))
        x_means[0] = x_0_posterior_mean
        times_of_interest = [t for t in range(t_0_posterior_mean + 1, t_end_posterior_mean)]
        for i, time_of_interest in enumerate(times_of_interest):
            x_means_KD = A_j @ x_means[i] + b_j
            prob_entity_regimes_K = VEZ_summaries.expected_regimes[time_of_interest, j]
            x_means[i + 1] = np.einsum("kd, k -> d", x_means_KD, prob_entity_regimes_K)
            print(f"At time {i} the most likely regime is {np.argmax(prob_entity_regimes_K)}")

        # Unnorm
        x_means_unnorm = unnormalize_coords(x_means)
        ground_truth_posterior_mean = continuous_states[t_0_posterior_mean:t_end_posterior_mean, j]
        ground_truth_posterior_mean_unnorm = unnormalize_coords(ground_truth_posterior_mean)

        # MSE
        MSE_posterior_mean = np.mean((ground_truth_posterior_mean_unnorm - x_means_unnorm) ** 2)
        MSEs_posterior_mean[j] = MSE_posterior_mean

        # plot
        fig = plt.figure(figsize=figsize)
        plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
        plt.scatter(
            x_means_unnorm[:, 0],
            x_means_unnorm[:, 1],
            c=[i for i in range(T_slice_posterior_mean)],
            cmap="cool",
            alpha=1.0,
            zorder=2,
        )
        plt.scatter(
            ground_truth_posterior_mean_unnorm[:, 0],
            ground_truth_posterior_mean_unnorm[:, 1],
            c=[i for i in range(T_slice_posterior_mean)],
            cmap="cool",
            marker="x",
            alpha=0.25,
            zorder=1,
        )
        plt.ylim(Y_LIM_COURT)
        plt.xlim(X_LIM_COURT)
        plt.title(f"MSE: {MSE_posterior_mean:.05f}")
        tag_posterior_mean = _get_filename_tag(
            use_continuous_states,
            j,
            t_0_posterior_mean,
            t_end_posterior_mean,
            filename_prefix,
            DIMS.L,
        )
        fig.savefig(save_dir + f"fit_via_posterior_mean_{tag_posterior_mean}_MSE_{MSE_posterior_mean:.03f}.pdf")
        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())

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
        fixed_init_continuous_states = np.tile(x_0_forward_sim, (DIMS.J, 1))
        fixed_init_entity_regimes = np.argmax(VEZ_summaries.expected_regimes[t_0_forward_sim], axis=1)
        fixed_init_system_regime = np.argmax(VES_summary.expected_regimes, axis=1)[t_0_forward_sim]
        fixed_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)[t_0_forward_sim:t_end_forward_sim]

        had_masking = _had_masking_bool(use_continuous_states, j, t_0_forward_sim, t_end_forward_sim)

        warnings.warn("Forward simulations assume that there are NO system covariates. Is this correct?")
        for s, forward_simulation_seed in enumerate(forward_simulation_seeds):
            print(
                f"Plotting the forward simulation for entity {j} using forward_simulation_seed {forward_simulation_seed}."
            )

            if forecast_type == ForecastType.PARTIAL:
                sample_ahead = sample_team_dynamics(
                    params,
                    T_slice_forward_sim,
                    model,
                    seed=forward_simulation_seed,
                    fixed_system_regimes=fixed_system_regimes,
                    fixed_init_entity_regimes=fixed_init_entity_regimes,
                    fixed_init_continuous_states=fixed_init_continuous_states,
                    system_covariates=system_covariates,
                )
            elif forecast_type == ForecastType.COMPLETE:
                sample_ahead = sample_team_dynamics(
                    params,
                    T_slice_forward_sim,
                    model,
                    seed=forward_simulation_seed,
                    fixed_init_system_regime=fixed_init_system_regime,
                    fixed_init_entity_regimes=fixed_init_entity_regimes,
                    fixed_init_continuous_states=fixed_init_continuous_states,
                    system_covariates=system_covariates,
                )
            else:
                raise ValueError(f"I don't understand forecast type {forecast_type}.")

            # Unnorm
            sample_ahead_xs_j_unnorm = unnormalize_coords(sample_ahead.xs[:, j])
            ground_truth_forward_sim = continuous_states[t_0_forward_sim:t_end_forward_sim, j]
            ground_truth_forward_sim_unnorm = unnormalize_coords(ground_truth_forward_sim)

            # MSE
            MSE_forward_sim = np.mean((ground_truth_forward_sim_unnorm - sample_ahead_xs_j_unnorm) ** 2)
            if had_masking:
                MSEs_forward_sims[j, s] = MSE_forward_sim
            # Rk: `MSE_forward_sim` mixes entities with seen vs unseen data in the forecasting window.
            # Main distinction is whether the VES step on q(s_t) incorporated info the relevant entity-level states
            # q(z_t^^j)'s or not.  There's also a difference in which information was used in the M-step, but for sufficiently
            # long and regular time series, this probably wouldn't play a big role.

            # plot
            fig1 = plt.figure(figsize=figsize)
            plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)

            im = plt.scatter(
                sample_ahead_xs_j_unnorm[:, 0],
                sample_ahead_xs_j_unnorm[:, 1],
                c=[i for i in range(t_0_forward_sim, t_end_forward_sim)],
                cmap="cool",
                alpha=1.0,
                zorder=2,
            )
            plt.scatter(
                ground_truth_forward_sim_unnorm[:, 0],
                ground_truth_forward_sim_unnorm[:, 1],
                c=[i for i in range(T_slice_forward_sim)],
                cmap="cool",
                marker="x",
                alpha=0.25,
                zorder=1,
            )
            plt.xlim(
                np.min([X_LIM_COURT[0], np.min(sample_ahead_xs_j_unnorm[:, 0])]),
                np.max([X_LIM_COURT[1], np.max(sample_ahead_xs_j_unnorm[:, 0])]),
            )
            plt.ylim(
                np.min([Y_LIM_COURT[0], np.min(sample_ahead_xs_j_unnorm[:, 1])]),
                np.max([Y_LIM_COURT[1], np.max(sample_ahead_xs_j_unnorm[:, 1])]),
            )
            tag_forward_sim = _get_filename_tag(
                use_continuous_states,
                j,
                t_0_forward_sim,
                t_end_forward_sim,
                filename_prefix,
                DIMS.L,
            )
            plt.title(f"MSE: {MSE_forward_sim:.05f}.")
            fig1.savefig(
                save_dir
                + f"forward_simulation_{tag_forward_sim}_seed_{forward_simulation_seed}_MSE_{MSE_forward_sim:.03f}.pdf"
            )

            # An attempt to avoid inadventently retaining figures which consume too much memory.
            # References:
            # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
            # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
            plt.close(plt.gcf())

        ###
        # Compute velocity baseline
        ###
        # RK: We reuse the forward sim params for the velocity baseline
        discrete_derivative = continuous_states[t_0_forward_sim, j] - continuous_states[t_0_forward_sim - 1, j]
        velocity_baseline = np.zeros((T_slice_forward_sim, 2))
        velocity_baseline[0] = continuous_states[t_0_forward_sim, j]
        for t in range(1, T_slice_forward_sim):
            velocity_baseline[t] = velocity_baseline[t - 1] + discrete_derivative

        # Unnorm
        velocity_baseline_unnorm = unnormalize_coords(velocity_baseline)
        ground_truth_forward_sim_unnorm = unnormalize_coords(ground_truth_forward_sim)

        # MSE
        MSE_velocity_baseline = np.mean((ground_truth_forward_sim_unnorm - velocity_baseline_unnorm) ** 2)
        if had_masking:
            MSEs_velocity_baseline[j] = MSE_velocity_baseline

        # Plot
        fig = plt.figure(figsize=figsize)
        plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
        plt.scatter(
            velocity_baseline_unnorm[:, 0],
            velocity_baseline_unnorm[:, 1],
            c=[i for i in range(T_slice_forward_sim)],
            cmap="cool",
            alpha=1.0,
            zorder=1,
        )
        plt.scatter(
            ground_truth_forward_sim_unnorm[:, 0],
            ground_truth_forward_sim_unnorm[:, 1],
            c=[i for i in range(T_slice_forward_sim)],
            cmap="cool",
            marker="x",
            alpha=0.25,
            zorder=2,
        )
        plt.xlim(
            np.min([X_LIM_COURT[0], np.min(velocity_baseline_unnorm[:, 0])]),
            np.max([X_LIM_COURT[1], np.max(velocity_baseline_unnorm[:, 0])]),
        )
        plt.ylim(
            np.min([Y_LIM_COURT[0], np.min(velocity_baseline_unnorm[:, 1])]),
            np.max([Y_LIM_COURT[1], np.max(velocity_baseline_unnorm[:, 1])]),
        )

        plt.title(f"MSE: {MSE_velocity_baseline:.05f}")
        tag_velocity_baseline = _get_filename_tag(
            use_continuous_states,
            j,
            t_0_forward_sim,
            t_end_forward_sim,
            filename_prefix,
            DIMS.L,
        )
        fig.savefig(
            save_dir + f"forecast_via_velocity_baseline_{tag_velocity_baseline}_MSE_{MSE_velocity_baseline:.03f}.pdf"
        )

        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())

    fig2 = plt.figure(figsize=(2, 6))
    cax = fig2.add_subplot()
    cbar = fig1.colorbar(im, cax=cax)
    cbar.set_label("Timesteps", rotation=90)
    plt.tight_layout()
    fig2.savefig(save_dir + "colorbar_clip.pdf")

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())

    return MSEs_posterior_mean, MSEs_forward_sims, MSEs_velocity_baseline


def write_model_evaluation_via_posterior_mean_and_forward_simulation_on_slice(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    forward_simulation_seeds: List[int],
    save_dir: str,
    use_continuous_states: NumpyArray2D,
    entity_idxs: Optional[List[int]],
    find_forward_sim_t0_for_entity_sample: Callable,
    max_forward_sim_window: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    find_posterior_mean_t0_for_entity_sample: Optional[Callable] = None,
    max_posterior_mean_window: Optional[int] = None,
    filename_prefix: Optional[str] = "",
    figsize: Optional[Tuple[int]] = (8, 4),
    verbose: Optional[bool] = True,
    forecast_type: ForecastType = ForecastType.PARTIAL,
) -> Tuple[NumpyArray1D, NumpyArray2D, NumpyArray1D]:
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
        max_forward_sim_window: The attempted number of timesteps in the slice that we work with for forward simulating.
            The actual window size could be less if there are not enough timesteps remaining in the sample.
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
        find_forward_sim_t0_for_entity_sample: Function converting entity sample (in (T,D)) to initial time for forecasting
        use_continuous_states: boolean array of shape (T,J). If true, the continuous states were observed
            (not masked) during initialization and inference.

    Returns:
        MSEs_posterior_mean: An array of size (J,) that describes the model performance for each of the
            J entities over a time period requested for the posterior mean.
        MSEs_forward_sims: An array of size (J,S)  that describes the model performance for each of the
            J entities for each of S simulations over a time period requested for the forward sims.
            The value is NaN if the entity was not masked.
        MSEs_velocity_baseline: An array of size (J,) that describes the model performance for each of the
            J entities over the same time period as requested for the forward sims.
            The value is NaN if the entity was not masked.
    """
    # Rk: `MMSE_forward_sim` mixes entities with seen vs unseen data in the forecasting window.
    # Main distinction is whether the VES step on q(s_t) incorporated info the relevant entity-level states
    # q(z_t^^j)'s or not.  There's also a difference in which information was used in the M-step, but for sufficiently
    # long and regular time series, this probably wouldn't play a big role.

    (
        MSEs_posterior_mean,
        MSEs_forward_sims,
        MSEs_velocity_baseline,
        _,
    ) = evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice(
        continuous_states,
        params,
        VES_summary,
        VEZ_summaries,
        model,
        forward_simulation_seeds,
        save_dir,
        use_continuous_states,
        entity_idxs,
        find_forward_sim_t0_for_entity_sample,
        max_forward_sim_window,
        system_covariates,
        find_posterior_mean_t0_for_entity_sample,
        max_posterior_mean_window,
        filename_prefix,
        figsize,
        forecast_type,
    )

    MMSE_posterior_mean = np.mean(MSEs_posterior_mean)
    MMSE_forward_sim, mean_median_MSE_forward_sim = np.nanmean(MSEs_forward_sims), np.nanmean(
        np.nanmedian(MSEs_forward_sims, 1)
    )
    MMSE_velocity_baseline = np.nanmean(MSEs_velocity_baseline)
    if verbose:
        print(
            f"After initialization, the mean (across entities) MSE for posterior mean is {MMSE_posterior_mean:.03f}."
            f"\nThe mean (across entities and sims) MSE for forward sim is is {MMSE_forward_sim:.03f}."
            f"\nThe mean (across entities) median (across simulations) MSEs for forward sim is {mean_median_MSE_forward_sim:.03f}."
            f"\nThe mean (across entities and sims) MSE for velocity baseline is {MMSE_velocity_baseline:.03f}."
        )
    df_eval = pd.DataFrame(
        {
            "Mean MSE for posterior mean": [MMSE_posterior_mean],
            "SD MSE for posterior mean": [np.std(MSEs_posterior_mean)],
            "Mean MSE for forward sim": [MMSE_forward_sim],
            "SD MSE for forward sim": [np.nanstd(MSEs_forward_sims)],
            "Mean Median MSE for forward sim": [mean_median_MSE_forward_sim],
            "Mean MSE for velocity baseline": [MMSE_velocity_baseline],
            "SD MSE for velocity baseline": [np.nanstd(MSEs_velocity_baseline)],
        }
    )
    df_eval.to_csv(os.path.join(save_dir, f"performance_MSEs_{filename_prefix}.csv"))
    return MSEs_posterior_mean, MSEs_forward_sims, MSEs_velocity_baseline
