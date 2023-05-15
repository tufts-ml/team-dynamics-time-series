from typing import Optional, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynagroup.hmm_posterior import HMM_Posterior_Summaries_JAX
from dynagroup.initialize import (
    InitializationResults,
    RawInitializationResults,
    ResultsFromBottomHalfInit,
    ResultsFromTopHalfInit,
    initialization_results_from_raw_initialization_results,
)
from dynagroup.model import Model
from dynagroup.model2a.gaussian.initialize import (
    make_data_free_initialization_of_EP_JAX,
    make_data_free_initialization_of_ETP_JAX,
)
from dynagroup.params import (
    ContinuousStateParameters_VonMises_JAX,
    Dims,
    InitializationParameters_VonMises_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import JaxNumpyArray2D, NumpyArray1D, NumpyArray2D, NumpyArray3D
from dynagroup.util import make_fixed_sticky_tpm_JAX
from dynagroup.vi.E_step import run_VES_step_JAX
from dynagroup.vi.M_step_and_ELBO import (
    run_M_step_for_ETP_via_gradient_descent,
    run_M_step_for_STP_via_gradient_descent,
)
from dynagroup.von_mises.inference.arhmm import run_EM_for_von_mises_arhmm


###
# PARAMETER INITS
###


# TODO: Integrate this with the parameter inits from figure 8,
# move to a single param_init module!


def make_data_free_initialization_of_STP_JAX(
    DIMS: Dims,
    method_for_Upsilon: str,
    fixed_self_transition_prob: float,
    seed: int,
) -> SystemTransitionParameters_JAX:
    """
    method_for_Psis : zeros or rnorm
    """
    key = jr.PRNGKey(seed)
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_s = DIMS.L, DIMS.J, DIMS.K, DIMS.M_s

    Gammas = jnp.zeros((J, L, K))

    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.asarray(tpm)

    if method_for_Upsilon == "rnorm":
        Upsilon = jr.normal(key, (L, M_s))
    elif method_for_Upsilon == "zeros":
        Upsilon = jnp.zeros((L, M_s))
    else:
        raise ValueError("What is the method for Upsilon?")
    return SystemTransitionParameters_JAX(Gammas, Upsilon, Pi)


def make_data_free_initialization_of_IP_JAX(
    DIMS,
) -> InitializationParameters_VonMises_JAX:
    pi_system = np.ones(DIMS.L) / DIMS.L
    pi_entities = np.ones((DIMS.J, DIMS.K)) / DIMS.K
    locs = jnp.zeros((DIMS.J, DIMS.K))
    kappas = jnp.ones((DIMS.J, DIMS.K))
    return InitializationParameters_VonMises_JAX(pi_system, pi_entities, locs, kappas)


###
# AR-HMM (BOTTOM  HALF)
###


def fit_ARHMM_to_bottom_half_of_model(
    team_angles: Union[NumpyArray2D, JaxNumpyArray2D],
    num_regimes: int,
    num_EM_iterations: int,
    init_self_transition_prob: float,
    init_changepoint_penalty: float,
    init_min_segment_size: int,
    fix_ar_kappa_to_unity_rather_than_estimate: bool = False,
    verbose: bool = True,
    parallelize_the_CSP_M_step_across_regimes: bool = False,
) -> ResultsFromBottomHalfInit:
    """
    We assume the transitions are governed by an ordinary tpm.
    We ignore the system-level toggles.
    """

    # Rk: We convert jax arrays to arrays, otherwise this is slow as heck.
    # not sure why. is it because of indexing team_angles[:,j] in the function call?
    # or because we're using np in the code body?
    team_angles = np.asarray(team_angles)

    T, J = np.shape(team_angles)
    K = num_regimes

    # Pre-initialize info for HMM_Posterior_Summaries (for J entities)
    # RK: we need to construct this for the collection from the individual ones.
    # TODO: we can grab the log normalizers or entropies later if needed.
    # Currently `run_EM_for_von_mises_params` does not return these
    expected_regimes = np.zeros((T, J, K))
    expected_joints = np.zeros((T - 1, J, K, K))
    log_normalizers = np.zeros(J)
    entropies = np.zeros(J)

    # Pre-initialize info for the CSP Params
    ar_coefs = np.zeros((J, K))
    drifts = np.zeros((J, K))
    kappas = np.zeros((J, K))

    for j in range(J):
        print(f"\n\n---- Now initializing bottom-level model for entity {j+1}/{J}.")
        # maybe use the transitions to smart initialize the top half of the model?!?!
        # if so we might need to update `ResultsFromBottomHalfInit` to be able to include these.
        # alternatively, this is simple to recompute from EZ_summaries.

        # TODO: I'll just assume that eventually running this will produce non-nan results.
        (
            posterior_summary,
            emissions_params_by_regime_learned,
            transitions,
        ) = run_EM_for_von_mises_arhmm(
            team_angles[:, j],
            num_regimes,
            num_EM_iterations,
            init_self_transition_prob,
            init_changepoint_penalty,
            init_min_segment_size,
            fix_ar_kappa_to_unity_rather_than_estimate=fix_ar_kappa_to_unity_rather_than_estimate,
            parallelize_the_M_step=parallelize_the_CSP_M_step_across_regimes,
        )

        # update attributes for HMM posterior summaries
        expected_regimes[:, j, :] = posterior_summary.expected_regimes
        expected_joints[:, j, :, :] = posterior_summary.expected_joints
        log_normalizers[j] = posterior_summary.log_normalizer
        entropies[j] = posterior_summary.entropy

        # update attributes for CSP_JAX
        # TODO: Should I store these as VonMisesParams instead of CSP?
        # The relabeling might get confusing.. I think it depends on what happens
        # to them downstream.
        for k in range(K):
            ar_coefs[j, k] = emissions_params_by_regime_learned[k].ar_coef
            drifts[j, k] = emissions_params_by_regime_learned[k].drift
            kappas[j, k] = emissions_params_by_regime_learned[k].kappa

    # TODO: Can grab the entropies later if needed
    EZ_summaries = HMM_Posterior_Summaries_JAX(
        expected_regimes,
        expected_joints,
        log_normalizers,
        entropies,
    )
    CSP_JAX = ContinuousStateParameters_VonMises_JAX(ar_coefs, drifts, kappas)

    # TODO: Can grab the record of most likely state later if needed.
    return ResultsFromBottomHalfInit(CSP_JAX, EZ_summaries, record_of_most_likely_states=None)


###
# AR-HMM (TOP  HALF)
###


def fit_ARHMM_to_top_half_of_model(
    group_angles: Union[NumpyArray2D, NumpyArray3D],
    DIMS: Dims,
    EZ_summaries: HMM_Posterior_Summaries_JAX,
    system_covariates: NumpyArray2D,
    model: Model,
    IP_JAX: InitializationParameters_VonMises_JAX,
    num_EM_iterations: int,
    num_M_step_iterations_for_ETP_gradient_descent: int,
    num_M_step_iterations_for_STP_gradient_descent: int,
    seed: int = 0,
    verbose: bool = True,
    event_end_times: Optional[NumpyArray1D] = None,
) -> ResultsFromTopHalfInit:
    """
    Arguments:
        group_angles: array of shape (T,J) or (T,J,1)
    """

    ###
    # Setup
    ###

    T, J = np.shape(group_angles)[:2]

    # force there to be a third array dimension for the D
    group_angles = group_angles.reshape((T, J, -1))

    # make event_end_times have the right structure
    if event_end_times is None:
        event_end_times = np.array([-1, T])

    record_of_most_likely_states = np.zeros((T, num_EM_iterations))

    ### Initialization
    ETP_JAX = make_data_free_initialization_of_ETP_JAX(
        DIMS=DIMS,
        method_for_Psis="zeros",
        seed=seed,
    )  # Ps is (J, L, K, K)

    STP_JAX = make_data_free_initialization_of_STP_JAX(
        DIMS,
        method_for_Upsilon="rnorm",
        fixed_self_transition_prob=0.95,
        seed=seed,
    )

    # zhats = np.argmax(results_bottom.EZ_summaries.expected_regimes, axis=2) # zhats is (J,T) with each entry in {1,..K} (but zero-indexed)
    # TODO: Is there a better way to init the skip-level recurrence matrices?

    print("\n--- Now running AR-HMM on top half of Model 2a. ---")

    for iteration in range(num_EM_iterations):
        print(
            f"Now running EM iteration {iteration+1}/{num_EM_iterations} for AR-HMM on top half of Model 2a."
        )

        ###  E-step
        ES_summary = run_VES_step_JAX(
            STP_JAX,
            ETP_JAX,
            IP_JAX,
            group_angles,
            EZ_summaries,
            model,
            event_end_times,
            system_covariates,
        )

        ### BELOW THIS HAS NOT YET BEEN UPDATED
        record_of_most_likely_states[:, iteration] = np.argmax(ES_summary.expected_regimes, axis=1)

        ### M-step (ETP)

        # TODO: In its current construction, we could do this in closed form. Do it.
        ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
            ETP_JAX,
            ES_summary,
            EZ_summaries,
            group_angles,
            iteration,
            num_M_step_iterations_for_ETP_gradient_descent,
            model,
            event_end_times,
            verbose=verbose,
        )

        ### M-step (STP)

        # TODO: Incorporate the system transition prior
        system_transition_prior = None

        STP_JAX = run_M_step_for_STP_via_gradient_descent(
            STP_JAX,
            ES_summary,
            system_transition_prior,
            iteration,
            num_M_step_iterations_for_STP_gradient_descent,
            model,
            event_end_times,
            system_covariates,
        )

    return ResultsFromTopHalfInit(STP_JAX, ETP_JAX, ES_summary, record_of_most_likely_states)


##
# MAIN
###


def smart_initialize_model_2a_for_circles(
    DIMS: Dims,
    group_angles: Union[NumpyArray2D, NumpyArray3D],
    system_covariates: NumpyArray2D,
    model: Model,
    bottom_half_self_transition_prob: float = 0.995,
    bottom_half_changepoint_penalty: float = 10.0,
    bottom_half_min_segment_size: int = 10,
    bottom_half_num_EM_iterations: int = 3,
    top_half_num_EM_iterations: int = 20,
    seed: int = 0,
    fix_ar_kappa_to_unity_rather_than_estimate: bool = False,
    parallelize_the_CSP_M_step_for_the_bottom_half_model: bool = False,
) -> InitializationResults:
    """
    Remarks:
        1) Note that the regime labeling (for entities and system) can differ from the truth, and also even
            from entity to entity!  For example, for the Figure 8 experiment, we could have
            0=bottom circle, 1=top circle for one entity, and flipped for the other.
        2) Currently, the only thing random here is the
        3) The initialization is actually specific for the Figure 8 experiment.  It's ALMOST good for Model 2a generally,
            but isn't QUITE general enough.  The main things missing are:
            a) smart initialization for covariates [Figure 8 experiment has no covariates]
            b) y-level observations, rather than x-level observations.
    """

    ### TODO: Make smart initialization better. E.g.
    # 1) Run init x times, pick the one with the best ELBO.
    # 2) Find a way to do smarter init for the skip-level recurrence parameters
    # 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

    ###
    # Fit Bottom-level HMM
    ###

    results_bottom = fit_ARHMM_to_bottom_half_of_model(
        group_angles,
        DIMS.K,
        bottom_half_num_EM_iterations,
        bottom_half_self_transition_prob,
        bottom_half_changepoint_penalty,
        bottom_half_min_segment_size,
        fix_ar_kappa_to_unity_rather_than_estimate=fix_ar_kappa_to_unity_rather_than_estimate,
        parallelize_the_CSP_M_step_across_regimes=parallelize_the_CSP_M_step_for_the_bottom_half_model,
    )

    ###
    # Run top half of model
    ###
    IP_JAX = make_data_free_initialization_of_IP_JAX(DIMS)

    num_M_step_iterations_for_ETP_gradient_descent = 10
    num_M_step_iterations_for_STP_gradient_descent = 10

    results_top = fit_ARHMM_to_top_half_of_model(
        group_angles,
        DIMS,
        results_bottom.EZ_summaries,
        system_covariates,
        model,
        IP_JAX,
        top_half_num_EM_iterations,
        num_M_step_iterations_for_ETP_gradient_descent,
        num_M_step_iterations_for_STP_gradient_descent,
        seed=seed,
    )

    ###
    # Construct return value
    ###

    # EP_JAX is a placeholder; not used for circle data (or any variant of Model 2a).
    EP_JAX = make_data_free_initialization_of_EP_JAX(DIMS)
    results_raw = RawInitializationResults(results_bottom, results_top, IP_JAX, EP_JAX)
    return initialization_results_from_raw_initialization_results(results_raw)
