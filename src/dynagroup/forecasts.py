from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.random as npr
from jax import vmap

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.model2a.basketball.court import unnormalize_coords
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.sampler import sample_team_dynamics
from dynagroup.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)


@dataclass
class Forecasts:
    """
    Attributes:
        forward_simulations: array with shape (S, T_forecast, J, D), where S is the number of forecasts
        fixed_velocity: array with shape (T_forecast, J, D)
        ground_truth: array with shape (T_forecast, J, D)
        raw: bool.
            If True, we use raw (unnormalized) basketball coords in [0,94]x[0,50] rather than [0,1]x[0,1]

    """

    forward_simulations: NumpyArray3D
    fixed_velocity: NumpyArray3D
    ground_truth: NumpyArray3D
    raw_coords: bool


@dataclass
class Forecast_MSEs:
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


def make_complete_forecasts_for_our_model_and_baselines(
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
    verbose: bool = True,
) -> Forecasts:
    """
    Arguments:
        use_raw_coords: bool.  If True, we use raw (unnormalized) basketball coords
            in [0,94]x[0,50] rather than [0,1]x[0,1]
    """
    ###
    # Upfront stuff
    ###

    DIMS = dims_from_params(params_learned)
    continuous_states_during_context_window = continuous_states[:T_context]
    continuous_states_during_forecast_window = continuous_states[T_context : T_context + T_forecast]
    discrete_derivative = continuous_states[T_context + 1] - continuous_states[T_context]

    # MSE_forecasts=np.zeros(n_forecasts_from_our_model)

    ###
    # Forward simulations from our HSRDM model
    ###

    forward_simulations_unnormalized = np.zeros(
        (n_forecasts_from_our_model, T_forecast, DIMS.J, DIMS.D)
    )

    ### Sample s and z's for initialization, based on the VES and VEZ steps.

    fixed_init_system_regime = npr.choice(
        range(DIMS.L), p=VES_summary_from_context_window.expected_regimes[-1]
    )
    fixed_init_entity_regimes = np.zeros(DIMS.J, dtype=int)
    for j in range(DIMS.J):
        probs = np.asarray(VEZ_summaries_from_context_window.expected_regimes[-1, j]).astype(
            "float64"
        )
        probs /= np.sum(probs)
        fixed_init_entity_regimes[j] = npr.choice(range(DIMS.K), p=probs)
    fixed_init_continuous_states = continuous_states_during_context_window[-1]

    T_forecast_plus_initialization_timestep_to_discard = T_forecast + 1

    for forecast_seed in range(n_forecasts_from_our_model):
        if verbose:
            print(
                f"Now running forward sims for seed {forecast_seed+1}/{n_forecasts_from_our_model}.",
                end="\r",
            )

        forward_sample_with_init_at_beginning = sample_team_dynamics(
            params_learned,
            T_forecast_plus_initialization_timestep_to_discard,
            model,
            seed=forecast_seed,
            fixed_init_system_regime=fixed_init_system_regime,
            fixed_init_entity_regimes=fixed_init_entity_regimes,
            fixed_init_continuous_states=fixed_init_continuous_states,
            system_covariates=system_covariates,
        )
        forward_simulations_unnormalized[forecast_seed] = forward_sample_with_init_at_beginning.xs[
            1:
        ]  # (forecast_window, J, D)
        ground_truth_in_normalized_coords = (
            continuous_states_during_forecast_window  # (forecast_window, J, D)
        )

    ###
    # Compute velocity baseline
    ###

    velocity_baseline_in_normalized_coords = np.zeros((T_forecast, DIMS.J, DIMS.D))
    velocity_baseline_in_normalized_coords[0] = continuous_states_during_forecast_window[0]
    for t in range(1, T_forecast):
        velocity_baseline_in_normalized_coords[t] = (
            velocity_baseline_in_normalized_coords[t - 1] + discrete_derivative
        )

    ###
    # Prepare return object
    ###
    if use_raw_coords:
        forward_simulations = np.array(
            [
                unnormalize_coords(forward_simulations_unnormalized[s])
                for s in range(len(forward_simulations_unnormalized))
            ]
        )
        ground_truth = unnormalize_coords(ground_truth_in_normalized_coords)
        velocity_baseline = unnormalize_coords(velocity_baseline_in_normalized_coords)
        raw_coords = True
    else:
        forward_simulations = forward_simulations_unnormalized
        ground_truth = ground_truth_in_normalized_coords
        velocity_baseline = velocity_baseline_in_normalized_coords
        raw_coords = False

    return Forecasts(forward_simulations, velocity_baseline, ground_truth, raw_coords)


def compute_mse_over_matrix(matrix_estimated: NumpyArray2D, matrix_true: NumpyArray2D) -> float:
    """
    Arguments:
        matrix_estimated: e.g. has shape (T,D)
        matrix_true:e.g. has shape (T,D)
    """
    return np.mean((matrix_estimated - matrix_true) ** 2)


def MSEs_from_forecasts(forecasts: Forecasts):
    ###
    # Set up vectorization functions
    ###
    compute_MSEs_under_one_forecast_JAX = vmap(compute_mse_over_matrix, 1)  # (t,j,d), (t,j,d) ->j
    compute_MSEs_under_one_forecast = lambda x, y: np.array(
        compute_MSEs_under_one_forecast_JAX(x, y)
    )

    # #TODO: Is there a way to do this using vmap?
    # compute_mse_under_many_forecasts_JAX = vmap(compute_mse_over_matrix, (0,2))  #(s,t,j,d), (t,j,d) ->(s,j)
    # compute_mse_under_many_forecasts = lambda x,y: np.array(compute_mse_under_many_forecasts_JAX(x,y))
    def compute_MSEs_under_many_forecasts(many_forecasts, ground_truth):
        S, T, J, D = np.shape(many_forecasts)
        MSEs = np.zeros((S, J))
        for s in range(S):
            MSEs[s] = compute_MSEs_under_one_forecast(many_forecasts[s], ground_truth)
        return MSEs

    ###
    # Compute MSEs
    ###
    velocity_MSEs = compute_MSEs_under_one_forecast(
        forecasts.fixed_velocity, forecasts.ground_truth
    )
    forward_simulation_MSEs = compute_MSEs_under_many_forecasts(
        forecasts.forward_simulations, forecasts.ground_truth
    )

    return Forecast_MSEs(forward_simulation_MSEs, velocity_MSEs, forecasts.raw_coords)
