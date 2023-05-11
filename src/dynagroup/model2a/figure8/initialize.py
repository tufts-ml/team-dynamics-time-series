import warnings
from typing import Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sklearn.cluster import KMeans

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    compute_hmm_posterior_summaries_JAX,
)
from dynagroup.initialize import (
    InitializationResults,
    RawInitializationResults,
    ResultsFromBottomHalfInit,
    ResultsFromTopHalfInit,
    initialization_results_from_raw_initialization_results,
)
from dynagroup.model import Model
from dynagroup.model2a.figure8.model_factors import (
    compute_log_entity_transition_probability_matrices_JAX,
)
from dynagroup.params import (
    ContinuousStateParameters_Gaussian_JAX,
    Dims,
    EmissionsParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_Gaussian_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import JaxNumpyArray3D, NumpyArray3D
from dynagroup.util import make_fixed_sticky_tpm_JAX
from dynagroup.vi.E_step import (
    compute_log_continuous_state_emissions_JAX,
    run_VES_step_JAX,
)
from dynagroup.vi.M_step_and_ELBO import (
    run_M_step_for_CSP_in_closed_form__Gaussian_case,
    run_M_step_for_ETP_via_gradient_descent,
    run_M_step_for_STP_in_closed_form,
)


###
# PARAM INITS
###

# TODO: Combine with helpers from init
# TODO: Make Enum: "random", "fixed", "tpm_only", etc.


def make_data_free_initialization_of_IP_JAX(DIMS) -> InitializationParameters_Gaussian_JAX:
    pi_system = np.ones(DIMS.L) / DIMS.L
    pi_entities = np.ones((DIMS.J, DIMS.K)) / DIMS.K
    mu_0s = jnp.zeros((DIMS.J, DIMS.K, DIMS.D))
    Sigma_0s = jnp.tile(jnp.eye(DIMS.D), (DIMS.J, DIMS.K, 1, 1))
    return InitializationParameters_Gaussian_JAX(pi_system, pi_entities, mu_0s, Sigma_0s)


def make_tpm_only_initialization_of_STP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> SystemTransitionParameters_JAX:
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_s = DIMS.L, DIMS.J, DIMS.K, DIMS.M_s
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.log(tpm)
    Gammas = jnp.zeros((J, L, K))  # Gammas must be zero for no feedback.
    Upsilon = jnp.zeros((L, M_s))
    return SystemTransitionParameters_JAX(Gammas, Upsilon, Pi)


def make_data_free_initialization_of_STP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> SystemTransitionParameters_JAX:
    return make_tpm_only_initialization_of_STP_JAX(DIMS, fixed_self_transition_prob)


def make_data_free_initialization_of_ETP_JAX(
    DIMS: Dims,
    method_for_Psis: str,
    seed: int,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    method_for_Psis : zeros or rnorm
    """
    key = jr.PRNGKey(seed)
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_e, D_t = DIMS.L, DIMS.J, DIMS.K, DIMS.M_e, DIMS.D_t
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(0.95, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    if method_for_Psis == "rnorm":
        Psis = jr.normal(key, (J, L, K, D_t))
    elif method_for_Psis == "zeros":
        Psis = jnp.zeros((J, L, K, D_t))
    else:
        raise ValueError("What is the method for Psis?")
    Omegas = jnp.zeros((J, L, K, M_e))
    return EntityTransitionParameters_MetaSwitch_JAX(Psis, Omegas, Ps)


def make_tpm_only_initialization_of_ETP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> EntityTransitionParameters_MetaSwitch_JAX:
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_e, D_t = DIMS.L, DIMS.J, DIMS.K, DIMS.M_e, DIMS.D_t
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    Psis = jnp.zeros((J, L, K, D_t))
    Omegas = jnp.zeros((J, L, K, M_e))
    return EntityTransitionParameters_MetaSwitch_JAX(Psis, Omegas, Ps)


def make_kmeans_initialization_of_CSP_JAX(
    DIMS: Dims, continuous_states: JaxNumpyArray3D
) -> ContinuousStateParameters_Gaussian_JAX:
    """
    We fit the bias terms (CSP.bs) using the cluster centers from a K-means algorithm.
    The state matrices (CSP.As) are initialized at 0.
    The state noise covariances (CSP.Qs) are initialized to identity martrices.
    """

    (
        J,
        K,
        D,
    ) = (
        DIMS.J,
        DIMS.K,
        DIMS.D,
    )

    As = jnp.zeros((J, K, D, D))
    Qs = jnp.tile(jnp.eye(D)[None, None, :, :], (J, K, 1, 1))
    bs = np.zeros((J, K, D))

    # We initialize bs using cluster centers of k-means
    continuous_states = jnp.asarray(continuous_states)
    for j in range(J):
        with warnings.catch_warnings():
            # sklearn gives the warning below, but I don't know how to execute on the advice, currently.
            #   FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4.
            #   Set the value of `n_init` explicitly to suppress the warning
            warnings.simplefilter(action="ignore", category=FutureWarning)
            km = KMeans(K).fit(continuous_states[:, j, :])
            bs[j, :, :] = jnp.array(km.cluster_centers_)
    bs = jnp.asarray(bs)

    return ContinuousStateParameters_Gaussian_JAX(As, bs, Qs)


def make_data_free_initialization_of_EP_JAX(
    DIMS: Dims,
) -> EmissionsParameters_JAX:
    J, D, N = DIMS.J, DIMS.D, DIMS.N

    Cs = jnp.zeros((J, N, D))
    ds = jnp.zeros((J, N))
    Rs = jnp.tile(jnp.eye(N)[None, :, :], (J, 1, 1))

    return EmissionsParameters_JAX(Cs, ds, Rs)


###
# AR-HMM (BOTTOM  HALF)
###


def fit_ARHMM_to_bottom_half_of_model(
    continuous_states: JaxNumpyArray3D,
    CSP_JAX: ContinuousStateParameters_Gaussian_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_Gaussian_JAX,
    model: Model,
    num_EM_iterations: int,
) -> ResultsFromBottomHalfInit:
    """
    We assume the transitions are governed by an ordinary tpm.
    We ignore the system-level toggles.
    """
    T = len(continuous_states)
    J, L, K, _ = np.shape(ETP_JAX.Ps)

    record_of_most_likely_states = np.zeros((T, J, num_EM_iterations))

    print("\n--- Now running AR-HMM on bottom half of Model 2a. ---")
    for i in range(num_EM_iterations):
        print(
            f"Now running EM iteration {i+1}/{num_EM_iterations} for AR-HMM on bottom half of Model 2a."
        )

        ###
        # Get ingredients for HMM.
        ###
        log_state_emissions = compute_log_continuous_state_emissions_JAX(
            CSP_JAX, IP_JAX, continuous_states, model
        )
        log_transition_matrices = compute_log_entity_transition_probability_matrices_JAX(
            ETP_JAX,
            continuous_states[:-1],
            model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
        )
        # TODO: The averaging should make sense, because for the purposes of initialziation, we have constructed the entity-level transition probs to be identical
        # for the system-level regimes
        log_transition_matrices_averaged_over_system_regimes = jnp.mean(
            log_transition_matrices, axis=2
        )

        ###
        # E-step.
        ###
        EZ_summaries = compute_hmm_posterior_summaries_JAX(
            log_transition_matrices_averaged_over_system_regimes,
            log_state_emissions,
            IP_JAX.pi_entities,
        )
        for j in range(J):
            record_of_most_likely_states[:, j, i] = np.argmax(
                EZ_summaries.expected_regimes[:, j, :], axis=1
            )

        ###
        # M-step (for ETP, represented as a transition probability matrix)
        ###

        # We need it to have shape (J,L,K,K).  So just do it with (J,K,K), then tile it over L.
        tpms = np.zeros((J, K, K))
        for j in range(J):
            tpms[j] = np.sum(EZ_summaries.expected_joints[2:, j], axis=0) / np.sum(
                EZ_summaries.expected_regimes[:-1, j], axis=0
            )
        Ps_new = jnp.tile(jnp.log(tpms[:, None, :, :]), (1, L, 1, 1))
        ETP_JAX = EntityTransitionParameters_MetaSwitch_JAX(ETP_JAX.Psis, ETP_JAX.Omegas, Ps_new)

        ###
        # M-step (for CSP)
        ###

        CSP_JAX = run_M_step_for_CSP_in_closed_form__Gaussian_case(EZ_summaries, continuous_states)
    return ResultsFromBottomHalfInit(CSP_JAX, EZ_summaries, record_of_most_likely_states)


###
# AR-HMM (TOP HALF)
###


def fit_ARHMM_to_top_half_of_model(
    continuous_states: NumpyArray3D,
    STP_JAX: SystemTransitionParameters_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_Gaussian_JAX,
    EZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    num_EM_iterations: int,
    num_M_step_iterations_for_ETP_gradient_descent: int,
    verbose: bool = True,
) -> ResultsFromTopHalfInit:
    T = len(continuous_states)
    record_of_most_likely_states = np.zeros((T, num_EM_iterations))

    print("\n--- Now running AR-HMM on top half of Model 2a. ---")

    for i in range(num_EM_iterations):
        print(
            f"Now running EM iteration {i+1}/{num_EM_iterations} for AR-HMM on top half of Model 2a."
        )

        ###  E-step
        ES_summary = run_VES_step_JAX(
            STP_JAX,
            ETP_JAX,
            IP_JAX,
            continuous_states,
            EZ_summaries,
            model,
        )

        record_of_most_likely_states[:, i] = np.argmax(ES_summary.expected_regimes, axis=1)

        ### M-step (ETP)
        ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
            ETP_JAX,
            ES_summary,
            EZ_summaries,
            continuous_states,
            i,
            num_M_step_iterations_for_ETP_gradient_descent,
            model,
            verbose,
        )

        ### M-step (STP)
        STP_JAX = run_M_step_for_STP_in_closed_form(STP_JAX, ES_summary)
    return ResultsFromTopHalfInit(STP_JAX, ETP_JAX, ES_summary, record_of_most_likely_states)


###
# MAIN
###


def smart_initialize_model_2a(
    DIMS: Dims,
    continuous_states: Union[NumpyArray3D, JaxNumpyArray3D],
    model: Model,
    seed: int = 0,
    verbose: bool = True,
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
    # 2) Find a way to do smarter init for the recurrence parameters
    # 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

    continuous_states = jnp.asarray(continuous_states)

    ###
    # Initialize Params
    ####

    # TODO: Support fixed or random draws from prior for As, Qs.
    CSP_JAX = make_kmeans_initialization_of_CSP_JAX(DIMS, continuous_states)
    # TODO: Support fixed or random draws from prior.
    ETP_JAX = make_tpm_only_initialization_of_ETP_JAX(DIMS, fixed_self_transition_prob=0.90)
    # TODO: Support fixed or random draws from prior.
    IP_JAX = make_data_free_initialization_of_IP_JAX(DIMS)
    # EP_JAX is a placeholder; not used for Figure 8.
    EP_JAX = make_data_free_initialization_of_EP_JAX(DIMS)

    ###
    # Fit Bottom-level HMM
    ###
    results_bottom = fit_ARHMM_to_bottom_half_of_model(
        continuous_states,
        CSP_JAX,
        ETP_JAX,
        IP_JAX,
        model,
        num_EM_iterations=5,
    )

    ###
    # Top-level HMM
    ###

    ### Initialization
    ETP_JAX = make_data_free_initialization_of_ETP_JAX(
        DIMS, method_for_Psis="rnorm", seed=seed
    )  # Psis is (J, L, K, D_t)
    STP_JAX = make_tpm_only_initialization_of_STP_JAX(DIMS, fixed_self_transition_prob=0.95)
    # zhats = np.argmax(results_bottom.EZ_summaries.expected_regimes, axis=2) # zhats is (J,T) with each entry in {1,..K} (but zero-indexed)
    # TODO: Is there a better way to init the recurrence matrices?

    ### run HMM
    num_EM_iterations = 20
    num_M_step_iterations_for_ETP_gradient_descent = 5
    results_top = fit_ARHMM_to_top_half_of_model(
        continuous_states,
        STP_JAX,
        ETP_JAX,
        IP_JAX,
        results_bottom.EZ_summaries,
        model,
        num_EM_iterations,
        num_M_step_iterations_for_ETP_gradient_descent,
        verbose=verbose,
    )
    results_raw = RawInitializationResults(results_bottom, results_top, IP_JAX, EP_JAX)
    return initialization_results_from_raw_initialization_results(results_raw)
