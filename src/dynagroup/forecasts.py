import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from jax import vmap

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.io import ensure_dir
from dynagroup.model import Model
from dynagroup.model2a.basketball.court import (
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
    unnormalize_coords,
)
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.sampler import get_multiple_samples_of_team_dynamics
from dynagroup.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
)


###
# CONSTANTS
###

# TODO: Move this to `court` file.  It's reused both here and in animate.
# TODO: Can I make a scheme where I plot the court in the NORMALIZED coords,
# so that I don't have to unnormalize all the time?!
COURT_AXIS_UNNORM = [X_MIN_COURT, X_MAX_COURT, Y_MIN_COURT, Y_MAX_COURT]
COURT_IMAGE = mpimg.imread("image/nba_court_T.png")


###
# Structs
###


@dataclass
class Forecast_Collection_For_Example:
    """
    Attributes:
        forward_simulations: array with shape (S, T_forecast, J, D), where S is the number of forecasts
        fixed_velocity: array with shape (T_forecast, J, D)
        ground_truth: array with shape (T_forecast, J, D)
        raw: bool.
            If True, we use raw (unnormalized) basketball coords in [0,94]x[0,50] rather than [0,1]x[0,1]

    """

    forward_simulations: NumpyArray4D
    fixed_velocity: NumpyArray3D
    ground_truth: NumpyArray3D
    raw_coords: bool


@dataclass
class Forecast_MSEs_For_Example:
    """
    Attributes:
        forward_simulations: array of shape (S, J), where S is the number of forecasts,
            giving the MSE of the forward simulations of our model with respect to ground truth.
            Here the mean is taken over the timesteps T and dims D.
        fixed_velocity: array with shape (J,)
            giving the MSE of the fixed velocity baseline with respect to ground truth.
            Here the mean is taken over the timesteps T and dims D.
        raw: bool.
            If True, we use raw (unnormalized) basketball coords in [0,94]x[0,50] rather than [0,1]x[0,1]
    """

    forward_simulation: NumpyArray2D
    fixed_velocity: NumpyArray1D
    raw_coords: bool


@dataclass
class Summary_Of_Forecast_MSEs_From_One_Example:
    mean_forward_simulation_over_entities_per_sample: List[float]
    mean_forward_simulation: float
    mean_fixed_velocity: float
    median_forward_simulation: float
    median_fixed_velocity: float
    mean_forward_simulation_CLE_only: float
    mean_fixed_velocity_CLE_only: float


###
# Make forecasts
###


def make_forecast_collection_for_one_example(
    continuous_states: Union[JaxNumpyArray2D, JaxNumpyArray3D],
    params_learned: AllParameters_JAX,
    model: Model,
    VEZ_summaries_from_context_window: HMM_Posterior_Summaries_JAX,
    VES_summary_from_context_window: HMM_Posterior_Summary_JAX,
    T_context: int,
    T_forecast: int,
    n_forecasts_from_our_model: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    use_raw_coords: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> Forecast_Collection_For_Example:
    """
    Makes (complete, not partial) forecasts for our model against baselines.

    Arguments:
        use_raw_coords: bool.  If True, we use raw (unnormalized) basketball coords
            in [0,94]x[0,50] rather than [0,1]x[0,1]
    """
    npr.seed(seed)

    ###
    # Upfront stuff
    ###

    DIMS = dims_from_params(params_learned)
    T_forecast_plus_initialization_timestep_to_discard = T_forecast + 1

    continuous_states_during_context_window = continuous_states[:T_context]
    continuous_states_during_forecast_window = continuous_states[T_context : T_context + T_forecast]
    discrete_derivative = continuous_states[T_context + 1] - continuous_states[T_context]

    ###
    # Ground truth
    ###
    ground_truth_in_normalized_coords = continuous_states_during_forecast_window  # (forecast_window, J, D)

    ###
    # Forward simulations from our HSRDM model
    ###

    forward_simulations_in_normalized_coords = np.zeros((n_forecasts_from_our_model, T_forecast, DIMS.J, DIMS.D))

    ### Sample s and z's for initialization, based on the VES and VEZ steps.

    probs = np.asarray(VES_summary_from_context_window.expected_regimes[-1]).astype("float64")
    probs /= np.sum(probs)
    fixed_init_system_regime = npr.choice(range(DIMS.L), p=probs)

    fixed_init_entity_regimes = np.zeros(DIMS.J, dtype=int)
    for j in range(DIMS.J):
        probs = np.asarray(VEZ_summaries_from_context_window.expected_regimes[-1, j]).astype("float64")
        probs /= np.sum(probs)
        fixed_init_entity_regimes[j] = npr.choice(range(DIMS.K), p=probs)

    fixed_init_continuous_states = continuous_states_during_context_window[-1]

    forward_samples_with_init_at_beginning = get_multiple_samples_of_team_dynamics(
        n_forecasts_from_our_model,
        params_learned,
        T_forecast_plus_initialization_timestep_to_discard,
        model,
        seed=seed,
        fixed_init_system_regime=fixed_init_system_regime,
        fixed_init_entity_regimes=fixed_init_entity_regimes,
        fixed_init_continuous_states=fixed_init_continuous_states,
        system_covariates=system_covariates,
    )
    for s in range(n_forecasts_from_our_model):
        forward_simulations_in_normalized_coords[s] = forward_samples_with_init_at_beginning[s].xs[1:]
        # ` forward_simulations_in_normalized_coords` has shape (n_forecasts_from_our_model, forecast_window, J, D)

    ###
    # Compute velocity baseline
    ###

    velocity_baseline_in_normalized_coords_with_init_at_beginning = np.zeros(
        (T_forecast_plus_initialization_timestep_to_discard, DIMS.J, DIMS.D)
    )
    velocity_baseline_in_normalized_coords_with_init_at_beginning[0] = continuous_states_during_context_window[-1]
    for t in range(1, T_forecast_plus_initialization_timestep_to_discard):
        velocity_baseline_in_normalized_coords_with_init_at_beginning[t] = (
            velocity_baseline_in_normalized_coords_with_init_at_beginning[t - 1] + discrete_derivative
        )
    velocity_baseline_in_normalized_coords = velocity_baseline_in_normalized_coords_with_init_at_beginning[1:]

    ###
    # Prepare return object
    ###
    if use_raw_coords:
        forward_simulations = np.array(
            [
                unnormalize_coords(forward_simulations_in_normalized_coords[s])
                for s in range(len(forward_simulations_in_normalized_coords))
            ]
        )
        ground_truth = unnormalize_coords(ground_truth_in_normalized_coords)
        velocity_baseline = unnormalize_coords(velocity_baseline_in_normalized_coords)
        raw_coords = True
    else:
        forward_simulations = forward_simulations_in_normalized_coords
        ground_truth = ground_truth_in_normalized_coords
        velocity_baseline = velocity_baseline_in_normalized_coords
        raw_coords = False

    # ### Convert to float64 so that we can write dataclass to json
    # forward_simulations=np.array(forward_simulations, dtype=np.float64)
    # velocity_baseline=np.array(velocity_baseline, dtype=np.float64)
    # ground_truth=np.array(ground_truth, dtype=np.float64)

    return Forecast_Collection_For_Example(forward_simulations, velocity_baseline, ground_truth, raw_coords)


###
# Save forecasts
###
def save_forecasts(
    forecasts: Forecast_Collection_For_Example, save_dir: str, forecast_description: str, example_description: str
):
    save_subdir = os.path.join(save_dir, f"forecasts_{forecast_description}", example_description)
    ensure_dir(save_subdir)
    np.save(os.path.join(save_subdir, "forward_simulations.npy"), forecasts.forward_simulations)
    np.save(os.path.join(save_subdir, "fixed_velocity.npy"), forecasts.fixed_velocity)
    np.save(os.path.join(save_subdir, "ground_truth.npy"), forecasts.ground_truth)


###
# Make Forecasting MSEs
###


def MSEs_from_forecasts(forecasts: Forecast_Collection_For_Example):
    def _compute_mse_over_matrix(matrix_estimated: NumpyArray2D, matrix_true: NumpyArray2D) -> float:
        """
        Arguments:
            matrix_estimated: e.g. has shape (T,D)
            matrix_true:e.g. has shape (T,D)
        """
        return np.mean((matrix_estimated - matrix_true) ** 2)

    ### Set up vectorization functions
    compute_MSEs_under_one_forecast_JAX = vmap(_compute_mse_over_matrix, 1)  # (t,j,d), (t,j,d) ->j
    compute_MSEs_under_one_forecast = lambda x, y: np.array(compute_MSEs_under_one_forecast_JAX(x, y))

    # #TODO: Is there a way to do this using vmap?
    # compute_mse_under_many_forecasts_JAX = vmap(compute_mse_over_matrix, (0,2))  #(s,t,j,d), (t,j,d) ->(s,j)
    # compute_mse_under_many_forecasts = lambda x,y: np.array(compute_mse_under_many_forecasts_JAX(x,y))
    def _compute_MSEs_under_many_forecasts(many_forecasts, ground_truth):
        S, T, J, D = np.shape(many_forecasts)
        MSEs = np.zeros((S, J))
        for s in range(S):
            MSEs[s] = compute_MSEs_under_one_forecast(many_forecasts[s], ground_truth)
        return MSEs

    ### Compute MSEs
    velocity_MSEs = compute_MSEs_under_one_forecast(forecasts.fixed_velocity, forecasts.ground_truth)
    forward_simulation_MSEs = _compute_MSEs_under_many_forecasts(forecasts.forward_simulations, forecasts.ground_truth)

    return Forecast_MSEs_For_Example(forward_simulation_MSEs, velocity_MSEs, forecasts.raw_coords)


def summarize_forecast_MSEs_from_one_example(
    forecast_MSEs: Forecast_MSEs_For_Example,
) -> Summary_Of_Forecast_MSEs_From_One_Example:
    """
    Here the MSE summary characterizes a single example.

    it is taken across (S,J) elements for the forward simulations,
    and (J,) elements for the fixed velocity, where
        S : number of samples in forward simulation
        J : number of entities
    """
    return Summary_Of_Forecast_MSEs_From_One_Example(
        np.mean(forecast_MSEs.forward_simulation, 1),
        np.mean(forecast_MSEs.forward_simulation),
        np.mean(forecast_MSEs.fixed_velocity),
        np.median(forecast_MSEs.forward_simulation),
        np.median(forecast_MSEs.fixed_velocity),
        np.mean(forecast_MSEs.forward_simulation[:, :5]),
        np.mean(forecast_MSEs.fixed_velocity[:5]),
    )


###
# Plot forecasts
###


def plot_forecasts(
    forecasts: Forecast_Collection_For_Example,
    forecasting_MSEs: Forecast_MSEs_For_Example,
    save_dir: str,
    filename_prefix: str = "",
    figsize: Optional[Tuple[int]] = (8, 4),
):
    S, T_forecast, J, D = np.shape(forecasts.forward_simulations)

    for j in range(J):
        ### Fixed velocity plots
        fig = plt.figure(figsize=figsize)
        plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
        plt.scatter(
            forecasts.fixed_velocity[:, j, 0],
            forecasts.fixed_velocity[:, j, 1],
            c=[i for i in range(T_forecast)],
            cmap="cool",
            alpha=1.0,
            zorder=1,
        )
        plt.scatter(
            forecasts.ground_truth[:, j, 0],
            forecasts.ground_truth[:, j, 1],
            c=[i for i in range(T_forecast)],
            cmap="cool",
            marker="x",
            alpha=0.25,
            zorder=2,
        )
        plt.xlim(
            np.min([X_MIN_COURT, np.min(forecasts.fixed_velocity[:, j, 0])]),
            np.max([X_MAX_COURT, np.max(forecasts.fixed_velocity[:, j, 0])]),
        )
        plt.ylim(
            np.min([Y_MIN_COURT, np.min(forecasts.fixed_velocity[:, j, 1])]),
            np.max([Y_MAX_COURT, np.max(forecasts.fixed_velocity[:, j, 1])]),
        )

        plt.title(f"MSE: {forecasting_MSEs.fixed_velocity[j]:.03f}")
        fig.savefig(
            save_dir + f"{filename_prefix}_entity_{j}_fixed_velocity_MSE_{forecasting_MSEs.fixed_velocity[j]:.03f}.pdf"
        )
        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())

        for s in range(S):
            ### Forward simulation
            fig = plt.figure(figsize=figsize)
            plt.imshow(COURT_IMAGE, extent=COURT_AXIS_UNNORM, zorder=0)
            plt.scatter(
                forecasts.forward_simulations[s, :, j, 0],
                forecasts.forward_simulations[s, :, j, 1],
                c=[i for i in range(T_forecast)],
                cmap="cool",
                alpha=1.0,
                zorder=1,
            )
            plt.scatter(
                forecasts.ground_truth[:, j, 0],
                forecasts.ground_truth[:, j, 1],
                c=[i for i in range(T_forecast)],
                cmap="cool",
                marker="x",
                alpha=0.25,
                zorder=2,
            )
            plt.xlim(
                np.min([X_MIN_COURT, np.min(forecasts.forward_simulations[s, :, j, 0])]),
                np.max([X_MAX_COURT, np.max(forecasts.forward_simulations[s, :, j, 0])]),
            )
            plt.ylim(
                np.min([Y_MIN_COURT, np.min(forecasts.forward_simulations[s, :, j, 1])]),
                np.max([Y_MAX_COURT, np.max(forecasts.forward_simulations[s, :, j, 1])]),
            )
            plt.title(f"MSE: {forecasting_MSEs.forward_simulation[s,j]:.03f}")
            fig.savefig(
                save_dir
                + f"{filename_prefix}_entity_{j}_forward_sim_MSE_{forecasting_MSEs.forward_simulation[s,j]:.03f}_sim_{s}.pdf"
            )
            # An attempt to avoid inadventently retaining figures which consume too much memory.
            # References:
            # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
            # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
            plt.close(plt.gcf())
