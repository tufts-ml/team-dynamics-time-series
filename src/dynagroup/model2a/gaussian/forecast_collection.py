from typing import Optional, Union

import numpy as np

from dynagroup.forecast_collection import (
    Forecast_Collection_For_Example,
    make_forecast_collection_for_one_example,
)
from dynagroup.model import Model
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D
from dynagroup.vi.E_step import run_VES_step_JAX, run_VEZ_step_JAX


def get_forecast_collection_on_test_set_example(
    continuous_states_for_one_example: Union[JaxNumpyArray2D, JaxNumpyArray3D],
    params_learned: AllParameters_JAX,
    model: Model,
    T_context: int,
    T_forecast: int,
    n_cavi_iterations: int,
    n_forecasts: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    seed: int = 0,
) -> Forecast_Collection_For_Example:
    """
    We partition the T timesteps in continuous_states into 3 cells:
        context window, forecasting window, unused window.

    For the context window, we run the E-steps (VEZ, VES) of CAVI.
    For the forecasting window, we make forecasts.

    Warning:
        The function assumes that the `continuous_states` argument does NOT
        straddle an exmaple boundary.

    Arguments:
        T_context: Number of timesteps for context period
        T_forecast: Number of timesteps for forecasting period
    """

    # TODO: Expand this to make forecasts with non-Gaussian emissions

    # TODO: Revise this so that it can return raw forecasts, which can then
    # separately get converted into MSEs and/or plots, as desired

    ###
    # Upfront computations
    ###
    example_end_times = None

    T_test, J = np.shape(continuous_states_for_one_example)[:2]

    if T_context >= T_test:
        raise ValueError(
            f"The number of timesteps desired for context {T_context} must be less than "
            f"the number of timesteps in the provided test set {T_test}. "
        )
    if example_end_times is None:
        example_end_times = np.array([-1, T_context])

    ###
    # Initialize the E-step for the context period
    ###

    print("Now initializing E step for the context portion of the test set...")
    DIMS = dims_from_params(params_learned)
    continuous_states_during_context_window = continuous_states_for_one_example[:T_context]
    results_init = smart_initialize_model_2a(
        DIMS,
        continuous_states_during_context_window,
        example_end_times,
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
    use_continuous_states_during_context_window = None
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
            example_end_times,
            system_covariates,
            use_continuous_states_during_context_window,
        )

        VEZ_summaries_from_context_window = run_VEZ_step_JAX(
            params_learned.CSP,
            params_learned.ETP,
            params_learned.IP,
            continuous_states_during_context_window,
            VES_summary.expected_regimes,
            model,
            example_end_times,
        )
    print("")

    ###
    # Forecasting
    ###
    forecast_collection = make_forecast_collection_for_one_example(
        continuous_states_for_one_example,
        params_learned,
        model,
        VEZ_summaries_from_context_window,
        VES_summary_from_context_window,
        T_context,
        T_forecast,
        n_forecasts_from_our_model=n_forecasts,
        system_covariates=system_covariates,
        use_raw_coords=True,
        seed=seed,
    )
    return forecast_collection
