from typing import Optional, Union

import numpy as np

from dynagroup.forecasts import (
    Forecast_MSEs,
    MSEs_from_forecasts,
    make_complete_forecasts_for_our_model_and_baselines,
)
from dynagroup.model import Model
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D
from dynagroup.vi.E_step import run_VES_step_JAX, run_VEZ_step_JAX


def get_forecasting_MSEs_on_test_set(
    continuous_states: Union[JaxNumpyArray2D, JaxNumpyArray3D],
    params_learned: AllParameters_JAX,
    model: Model,
    T_context: int,
    T_forecast: int,
    n_cavi_iterations: int,
    n_forecasts: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    seed: int = 0,
) -> Forecast_MSEs:
    """
    We partition the T timesteps in continuous_states into 3 cells:
        context window, forecasting window, unused window.

    For the context window, we run the E-steps (VEZ, VES) of CAVI.
    For the forecasting window, we make forecasts.


    Arguments:
        T_context: Number of samples for context period
        T_forecast: Number of samples for forecasting period
    """

    # TODO: Revise this so that it can return raw forecasts, which can then
    # separately get converted into MSEs and/or plots, as desired

    ###
    # Upfront computations
    ###
    event_end_times = None
    use_continuous_states = None

    T_test, J = np.shape(continuous_states)[:2]

    if T_context >= T_test:
        raise ValueError(
            f"The number of timesteps desired for context {T_context} must be less than "
            f"the number of timesteps in the provided test set {T_test}. "
        )
    if event_end_times is None:
        event_end_times = np.array([-1, T_context])

    ###
    # Initialize the E-step for the context period
    ###

    print("Now initializing E step for the context portion of the test set...")
    DIMS = dims_from_params(params_learned)
    continuous_states_during_context_window = continuous_states[:T_context]
    results_init = smart_initialize_model_2a(
        DIMS,
        continuous_states_during_context_window,
        event_end_times,
        model,
        preinitialization_strategy_for_CSP=PreInitialization_Strategy_For_CSP.DERIVATIVE,
        num_em_iterations_for_bottom_half=5,
        num_em_iterations_for_top_half=20,
        seed=seed,
        system_covariates=system_covariates,
        params_frozen=params_learned,
        verbose=False,
    )
    print("...Done.")
    # TODO: Add assertion or unit test that parameters indeed haven't changed.
    # To aid with this, we may want to define a notion of equality on the parameters class.

    ###
    # Run E-steps for the context period
    ###
    VES_summary, VEZ_summaries = results_init.ES_summary, results_init.EZ_summaries

    for i in range(n_cavi_iterations):
        print(f"Running CAVI (E-step-only) iteration {i+1}/{n_cavi_iterations}.", end="\r")
        VES_summary_from_context_window = run_VES_step_JAX(
            params_learned.STP,
            params_learned.ETP,
            params_learned.IP,
            continuous_states_during_context_window,
            VEZ_summaries,
            model,
            event_end_times,
            system_covariates,
            use_continuous_states,
        )

        VEZ_summaries_from_context_window = run_VEZ_step_JAX(
            params_learned.CSP,
            params_learned.ETP,
            params_learned.IP,
            continuous_states_during_context_window,
            VES_summary.expected_regimes,
            model,
            event_end_times,
        )

    ###
    # Forecasting
    ###
    forecasts = make_complete_forecasts_for_our_model_and_baselines(
        continuous_states,
        params_learned,
        model,
        VEZ_summaries_from_context_window,
        VES_summary_from_context_window,
        T_context,
        T_forecast,
        n_forecasts_from_our_model=n_forecasts,
        system_covariates=system_covariates,
        use_raw_coords=True,
    )
    MSEs = MSEs_from_forecasts(forecasts)
    return MSEs
