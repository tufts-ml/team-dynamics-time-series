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
    if L > 1:
        tag = f"{filename_prefix}_HSDM_entity_{j}_was_masked_{had_masking}"
    elif L == 1:
        tag = f"{filename_prefix}_flat_SDM_entity_{j}_was_masked_{had_masking}"
    return tag


def evaluate_posterior_mean_and_forward_simulation_on_slice(
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
    find_posterior_mean_t0_for_entity_sample: Optional[Callable] = None,
    max_posterior_mean_window: Optional[int] = None,
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
        max_forward_sim_window: The attempted number of timesteps in the slice that we work with for forward simulating.
            The actual window size could be less if there are not enough timesteps remaining in the sample.
        entity_idxs:  If None, we plot results for all entities.  Else we just plot results for the entity
            indices provided.
        find_forward_sim_t0_for_entity_sample: Function converting entity sample (in (T,D)) to initial time for forecasting
        use_continuous_states: boolean array of shape (T,J). If true, the continuous states were observed
            (not masked) during initialization and inference.

    Returns:
        MSEs_posterior_mean, MSEs_forward_sim.  Each is an array of size (J,) that gives the performance for each of the
            J entities.
    """

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
    MSEs_forward_sim = np.zeros((J, S))

    for j in entity_idxs:
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
            continuous_states[:, j],
            t_0_forward_sim,
            max_forward_sim_window,
            j,
            x_lim,
            y_lim,
            save_dir,
            figsize=figsize,
            title_prefix="truth_for_forward_sim_clip",
        )

        plot_trajectory_slice(
            continuous_states[:, j],
            t_0_posterior_mean,
            max_posterior_mean_window,
            j,
            x_lim,
            y_lim,
            save_dir,
            figsize=figsize,
            title_prefix="truth_for_posterior_mean_clip",
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

        # MSE
        ground_truth_posterior_mean = continuous_states[t_0_posterior_mean:t_end_posterior_mean, j]
        MSE_posterior_mean = np.mean((ground_truth_posterior_mean - x_means) ** 2)
        MSEs_posterior_mean[j] = MSE_posterior_mean

        # plot
        fig = plt.figure(figsize=figsize)
        plt.scatter(
            x_means[:, 0],
            x_means[:, 1],
            c=[i for i in range(T_slice_posterior_mean)],
            cmap="cool",
            alpha=1.0,
        )
        plt.scatter(
            ground_truth_posterior_mean[:, 0],
            ground_truth_posterior_mean[:, 1],
            c=[i for i in range(T_slice_posterior_mean)],
            cmap="cool",
            marker="x",
            alpha=0.25,
        )
        if y_lim:
            plt.ylim(y_lim)
        if x_lim:
            plt.xlim(x_lim)

        plt.title(f"MSE: {MSE_posterior_mean:.05f}")
        tag_posterior_mean = _get_filename_tag(
            use_continuous_states,
            j,
            t_0_posterior_mean,
            t_end_posterior_mean,
            filename_prefix,
            DIMS.L,
        )
        fig.savefig(save_dir + f"fit_via_posterior_mean_{tag_posterior_mean}.pdf")

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
        fixed_init_entity_regimes = np.argmax(
            VEZ_summaries.expected_regimes[t_0_forward_sim], axis=1
        )
        fixed_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)[
            t_0_forward_sim:t_end_forward_sim
        ]

        for s, forward_simulation_seed in enumerate(forward_simulation_seeds):
            print(
                f"Plotting the forward simulation for entity {j} using forward_simulation_seed {forward_simulation_seed}."
            )
            sample_ahead = sample_team_dynamics(
                params,
                T_slice_forward_sim,
                model,
                seed=forward_simulation_seed,
                fixed_system_regimes=fixed_system_regimes,
                fixed_init_entity_regimes=fixed_init_entity_regimes,
                fixed_init_continuous_states=fixed_init_continuous_states,
            )

            # MSE
            ground_truth_forward_sim = continuous_states[t_0_forward_sim:t_end_forward_sim, j]
            MSE_forward_sim = np.mean((ground_truth_forward_sim - sample_ahead.xs[:, j]) ** 2)
            MSEs_forward_sim[j, s] = MSE_forward_sim
            # Rk: `MSE_forward_sim` mixes entities with seen vs unseen data in the forecasting window.
            # Main distinction is whether the VES step on q(s_t) incorporated info the relevant entity-level states
            # q(z_t^^j)'s or not.  There's also a difference in which information was used in the M-step, but for sufficiently
            # long and regular time series, this probably wouldn't play a big role.

            # plot
            fig1 = plt.figure(figsize=figsize)
            im = plt.scatter(
                sample_ahead.xs[:, j, 0],
                sample_ahead.xs[:, j, 1],
                c=[i for i in range(t_0_forward_sim, t_end_forward_sim)],
                cmap="cool",
                alpha=1.0,
            )
            plt.scatter(
                ground_truth_forward_sim[:, 0],
                ground_truth_forward_sim[:, 1],
                c=[i for i in range(T_slice_forward_sim)],
                cmap="cool",
                marker="x",
                alpha=0.25,
            )
            if y_lim:
                plt.ylim(y_lim)
            if x_lim:
                plt.xlim(x_lim)
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
                + f"forward_simulation_{tag_forward_sim}_seed_{forward_simulation_seed}.pdf"
            )

        fig2 = plt.figure(figsize=(2, 6))
        cax = fig2.add_subplot()
        cbar = fig1.colorbar(im, cax=cax)
        cbar.set_label("Timesteps", rotation=90)
        plt.tight_layout()
        fig2.savefig(save_dir + "colorbar_clip.pdf")
    return MSEs_posterior_mean, np.mean(MSEs_forward_sim, axis=1)
