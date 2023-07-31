from typing import Optional, Tuple, Union

import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D
from dynagroup.vi.E_step import run_VES_step_JAX, run_VEZ_step_JAX


def run_forecasts_on_test_set(
    continuous_states: Union[JaxNumpyArray2D, JaxNumpyArray3D],
    params_learned: AllParameters_JAX,
    model: Model,
    T_context: int,
    T_forecast: int,
    n_cavi_iterations: int,
    n_forecasts: int,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[HMM_Posterior_Summary_JAX, HMM_Posterior_Summaries_JAX, AllParameters_JAX]:
    """
    We partition the T timesteps in continuous_states into 3 cells:
        context window, forecasting window, unused window.

    For the context window, we run the E-steps (VEZ, VES) of CAVI.
    For the forecasting window, we make forecasts.


    Arguments:
        T_context: Number of samples for context period
        T_forecast: Number of samples for forecasting period
    """

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
    continuous_states_during_context_period = continuous_states[:T_context]

    if event_end_times is None:
        event_end_times = np.array([-1, T_context])

    if system_covariates is None:
        # TODO: Check that M_s=0 as well; if not there is an inconsistency in the implied desire of the caller.
        system_covariates = np.zeros((T_context, 0))

    ###
    # Initialize the E-step for the context period
    ###

    print("Now initializing E step for the context portion of the test set...")
    dims = dims_from_params(params_learned)
    results_init = smart_initialize_model_2a(
        dims,
        continuous_states_during_context_period,
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
        VES_summary = run_VES_step_JAX(
            params_learned.STP,
            params_learned.ETP,
            params_learned.IP,
            continuous_states_during_context_period,
            VEZ_summaries,
            model,
            event_end_times,
            system_covariates,
            use_continuous_states,
        )

        VEZ_summaries = run_VEZ_step_JAX(
            params_learned.CSP,
            params_learned.ETP,
            params_learned.IP,
            continuous_states_during_context_period,
            VES_summary.expected_regimes,
            model,
            event_end_times,
        )


###
# MAIN
###

# I run demo_inference_with_animations first, and then I try running the body of the code
# with the arguments below.

continuous_states = xs
model = model_basketball
T_context = 20
n_cavi_iterations = 5

# defaults set by function args.
seed = 0
verbose = True
