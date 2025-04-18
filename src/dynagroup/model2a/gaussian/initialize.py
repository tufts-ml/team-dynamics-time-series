import warnings
from enum import Enum
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from dynagroup.diagnostics.kmeans import plot_kmeans_on_2d_data
from dynagroup.diagnostics.steps_in_state import plot_steps_assigned_to_state
from dynagroup.examples import example_end_times_are_proper
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    compute_closed_form_M_step_on_posterior_summaries,
)
from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.model2a.marching_band.data.run_sim import system_regimes_gt
from dynagroup.initialize import (
    InitializationResults,
    RawInitializationResults,
    ResultsFromBottomHalfInit,
    ResultsFromTopHalfInit,
    initialization_results_from_raw_initialization_results,
)
from dynagroup.model import Model
from dynagroup.params import (
    AllParameters_JAX,
    ContinuousStateParameters_Gaussian_JAX,
    Dims,
    EmissionsParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_Gaussian_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.sample_weights import (
    make_sample_weights_which_mask_the_initial_timestep_for_each_event,
)
from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)
from dynagroup.util import make_fixed_sticky_tpm_JAX
from dynagroup.vi.E_step import run_VES_step_JAX, run_VEZ_step_JAX
from dynagroup.vi.M_step_and_ELBO import (
    M_Step_Toggle_Value,
    run_M_step_for_CSP_in_closed_form__Gaussian_case,
    run_M_step_for_ETP_via_gradient_descent,
    run_M_step_for_IP,
    run_M_step_for_STP_in_closed_form,
    run_M_step_for_STP_via_gradient_descent,
)


###
# PARAM INITS
###

# TODO: Integrate all these param ints with those from "circles" into a single param_init module!


# TODO: Combine with helpers from init
# TODO: Make Enum: "random", "fixed", "tpm_only", etc.


def make_data_free_preinitialization_of_IP_JAX(DIMS, shared_variance=1.0) -> InitializationParameters_Gaussian_JAX:
    pi_system = np.ones(DIMS.L) / DIMS.L
    pi_entities = np.ones((DIMS.J, DIMS.K)) / DIMS.K
    mu_0s = jnp.zeros((DIMS.J, DIMS.K, DIMS.D))
    Sigma_0s = jnp.tile(shared_variance * jnp.eye(DIMS.D), (DIMS.J, DIMS.K, 1, 1))
    return InitializationParameters_Gaussian_JAX(pi_system, pi_entities, mu_0s, Sigma_0s)


def make_tpm_only_preinitialization_of_STP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> SystemTransitionParameters_JAX:
    # TODO: Support fixed or random draws from prior.
    L, J, K, D_s = DIMS.L, DIMS.J, DIMS.K, DIMS.D_s
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.log(tpm)
    Gammas = jnp.zeros((J, L, K))  # Gammas must be zero for no feedback.
    Upsilon = jnp.zeros((L, D_s))
    return SystemTransitionParameters_JAX(Gammas, Upsilon, Pi)


def make_data_free_preinitialization_of_STP_JAX(
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
    L, J, K, D_s = DIMS.L, DIMS.J, DIMS.K, DIMS.D_s

    Gammas = jnp.zeros((J, L, K))

    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=L)
    Pi = jnp.log(tpm)

    if method_for_Upsilon == "rnorm":
        Upsilon = jr.normal(key, (L, D_s))
    elif method_for_Upsilon == "zeros":
        Upsilon = jnp.zeros((L, D_s))
    else:
        raise ValueError("What is the method for Upsilon?")
    return SystemTransitionParameters_JAX(Gammas, Upsilon, Pi)


def make_data_free_preinitialization_of_ETP_JAX(
    DIMS: Dims,
    method_for_Psis: str,
    seed: int,
    fixed_self_transition_prob: float = 0.90,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    method_for_Psis : zeros or rnorm
    """
    key = jr.PRNGKey(seed)
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_e, D_e = DIMS.L, DIMS.J, DIMS.K, DIMS.M_e, DIMS.D_e
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    if method_for_Psis == "rnorm":
        Psis = jr.normal(key, (J, L, K, D_e))
    elif method_for_Psis == "zeros":
        Psis = jnp.zeros((J, L, K, D_e))
    else:
        raise ValueError("What is the method for Psis?")
    Omegas = jnp.zeros((J, L, K, M_e))
    return EntityTransitionParameters_MetaSwitch_JAX(Psis, Omegas, Ps)


def make_tpm_only_preinitialization_of_ETP_JAX(
    DIMS: Dims, fixed_self_transition_prob: float
) -> EntityTransitionParameters_MetaSwitch_JAX:
    # TODO: Support fixed or random draws from prior.
    L, J, K, M_e, D_e = DIMS.L, DIMS.J, DIMS.K, DIMS.M_e, DIMS.D_e
    # make a tpm
    tpm = make_fixed_sticky_tpm_JAX(fixed_self_transition_prob, num_states=K)
    Ps = jnp.tile(np.log(tpm), (J, L, 1, 1))
    Psis = jnp.zeros((J, L, K, D_e))
    Omegas = jnp.zeros((J, L, K, M_e))
    return EntityTransitionParameters_MetaSwitch_JAX(Psis, Omegas, Ps)


class PreInitialization_Strategy_For_CSP(Enum):
    LOCATION = 1
    DERIVATIVE = 2


def make_kmeans_preinitialization_of_CSP_JAX(
    DIMS: Dims,
    continuous_states: JaxNumpyArray3D,
    strategy: PreInitialization_Strategy_For_CSP,
    example_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    plotbose: bool = False,
) -> Tuple[ContinuousStateParameters_Gaussian_JAX, sklearn.cluster._kmeans.KMeans]:
    """
    We assign the continuous_states to K regimes by applying k-means to either the locations (values)
        or velocities (discrete derivatives) of the continuous_states.
    We then initialize CSP parameters by running separate vector autoregressions within each cluster/regime:
        - We find regime-specific state matrix (CSP.As) and biases (CSP.bs) by applying a (multi-outcome) linear regression
            to predict the next continuous_state from the previous continuous_state.
        - We estimate the regime-specific covariance matrices (CSP.Qs) from the residuals of the above linear regresssion.

    Arguments:
        strategy: an Enum which determines whether we apply k-means to the locations (values)
            or velocities (discrete derivatives) of the continuous_states.
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is 1 if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step.
        plotbose: verbose in plotting
    """
    if verbose:
        print("Now performing k-means pre-initialization of CSP parameters.")

    ### Up-front computations
    continuous_states = jnp.asarray(continuous_states)
    T, J, D = np.shape(continuous_states)
    K = DIMS.K

    ### Make sample weights (as a combo of `use_continuous_states`` and `example_end_times`)
    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        continuous_states,
        example_end_times,
        use_continuous_states,
    )

    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.tile(np.eye(D)[None, None, :, :], (J, K, 1, 1))

    ### Find cluster memberships based on locations (values) or velocities (discrete derivatives) of continuous states
    continuous_state_diffs = continuous_states[1:, :, :] - continuous_states[:-1, :, :]

    if strategy == PreInitialization_Strategy_For_CSP.LOCATION:
        data_for_kmeans = continuous_states
        weights_for_kmeans = sample_weights
    elif strategy == PreInitialization_Strategy_For_CSP.DERIVATIVE:
        data_for_kmeans = continuous_state_diffs
        weights_for_kmeans = sample_weights[
            1:, :
        ]  # if response is initial time step for event, then give the pair of obs zero weight.
    else:
        raise ValueError(f"I don't understand the requested preinitialization strategy for CSP, {strategy}.")

    kms = [None] * J
    for j in range(J):
        with warnings.catch_warnings():
            # sklearn gives the warning below, but I don't know how to execute on the advice, currently.
            #   FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4.
            #   Set the value of `n_init` explicitly to suppress the warning
            warnings.simplefilter(action="ignore", category=FutureWarning)
            kms[j] = KMeans(K, random_state=120).fit(data_for_kmeans[:, j, :], sample_weight=weights_for_kmeans[:, j])

    # ### Plot K-means fits
    if plotbose:
        plot_kmeans_on_2d_data(
            data_for_kmeans,
            weights_for_kmeans,
            kms,
            save_dir,
        )

    ### Initialize parameters by running separate vector autoregressions within each cluster.

    # For each state, initialize CSP via a (multivariate-outcome) linear regression
    # finding weights A,b by predicting the next continuous_state from the previous continuous_state.
    # We then estimate the regime-specific covariance matrices (Q’s) from the residuals.

    # TODO: Parallelize this for speed
    for j in range(J):
        for k in range(K):
            ### find which samples to use
            # we only use samples which are in the cluster currently under consideration IF they got weighted
            # non-zero in the k-means calculation
            samples_are_in_cluster_jk = kms[j].labels_ == k
            bools_use_pair_for_cluster_jk = samples_are_in_cluster_jk * weights_for_kmeans[:, j]

            if strategy == PreInitialization_Strategy_For_CSP.LOCATION:
                outcome_indices_jk = np.where(bools_use_pair_for_cluster_jk)[0]
                predictor_indices_jk = outcome_indices_jk - 1
            elif strategy == PreInitialization_Strategy_For_CSP.DERIVATIVE:
                outcome_indices_jk = np.where(bools_use_pair_for_cluster_jk)[0] + 1
                predictor_indices_jk = outcome_indices_jk - 1
            else:
                raise ValueError(f"I don't understand the requested preinitialization strategy for CSP, {strategy}.")

            outcomes_jk = continuous_states[outcome_indices_jk, j, :]
            predictors_jk = continuous_states[predictor_indices_jk, j, :]
            if plotbose:
                plot_steps_assigned_to_state(outcomes_jk, predictors_jk, j, k, save_dir, basename_prefix="init_kmeans")

            ### run vector autoregression
            lr = LinearRegression(fit_intercept=True)
            lr.fit(predictors_jk, outcomes_jk)
            As[j, k] = lr.coef_ 
            bs[j, k] = lr.intercept_
            expectations_jk = (As[j, k] @ predictors_jk.T).T + bs[j, k]
            residuals_jk = outcomes_jk - expectations_jk
            # tied_residuals_j = np.concatenate((tied_residuals_j, residuals_jk))
            Qs[j, k] = np.cov(residuals_jk, rowvar=False)

    As = jnp.asarray(As)
    bs = jnp.asarray(bs)
    Qs = jnp.asarray(Qs)
    return ContinuousStateParameters_Gaussian_JAX(As, bs, Qs), kms


def make_data_free_preinitialization_of_EP_JAX(
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


def fit_rARHMM_to_bottom_half_of_model(
    continuous_states: JaxNumpyArray3D,
    example_end_times: Optional[JaxNumpyArray1D],
    CSP_JAX: ContinuousStateParameters_Gaussian_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_Gaussian_JAX,
    model: Model,
    num_EM_iterations: int,
    treat_ETP_params_as_tpm: bool = False,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    params_frozen: Optional[AllParameters_JAX] = None,
    verbose: bool = True,
) -> ResultsFromBottomHalfInit:
    """
    We assume the transitions are governed by an ordinary tpm.
    We ignore the system-level toggles.

    Arguments:
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element is True if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step (on STP, ETP, or CSP), nor the VES step.
            We leave the VEZ steps as is, though.  Note that This means that the ELBO is wrong.
    """
    ### TODO: We assume that (0,j) was always observed! If not, raise a ValueError.

    T = len(continuous_states)
    J, L, K, _ = np.shape(ETP_JAX.Ps)

    record_of_most_likely_states = np.zeros((T, J, num_EM_iterations), dtype=int)

    if verbose:
        print("\n--- Now running AR-HMM on bottom half of Model 2a. ---")
    for i in range(num_EM_iterations):
        if verbose:
            print(f"Now running EM iteration {i+1}/{num_EM_iterations} for AR-HMM on bottom half of Model 2a.")

        ###
        # E-step
        ###

        VES_expected_regimes__uniform = np.ones((T, L)) / L
        VES_expected_regimes__good = system_regimes_gt(10, [1227, 2840, 6128, 7392, 9553, 9680])

        EZ_summaries = run_VEZ_step_JAX(
            CSP_JAX,
            ETP_JAX,
            IP_JAX,
            continuous_states,
            VES_expected_regimes__uniform, 
            model,
            example_end_times,
        )

        for j in range(J):
            record_of_most_likely_states[:, j, i] = np.array(
                np.argmax(EZ_summaries.expected_regimes[:, j, :], axis=1), dtype=int
            )

        ###
        # M-step (for ETP, represented as a transition probability matrix)
        ###

        if params_frozen:
            ETP_JAX = params_frozen.ETP
            CSP_JAX = params_frozen.CSP
        else:
            ###
            # M-step (for ETP)
            ###

            if treat_ETP_params_as_tpm:
                # We need it to have shape (J,L,K,K).  So just do it with (J,K,K), then tile it over L.
                # TODO: This is rewriting the logic of "compute_closed_form_M_step."  Be sure that that can
                # work when we have J tpms, and then
                tpms = compute_closed_form_M_step_on_posterior_summaries(
                    EZ_summaries,
                    use_continuous_states,
                    example_end_times,
                )
                Ps_new = jnp.tile(jnp.log(tpms[:, None, :, :]), (1, L, 1, 1))
                ETP_JAX = EntityTransitionParameters_MetaSwitch_JAX(ETP_JAX.Psis, ETP_JAX.Omegas, Ps_new)

            else:
                ### New way: update ETP_JAX by using gradient descent
                num_M_step_iterations_for_ETP_gradient_descent = 5
                ES_summary_uniform = HMM_Posterior_Summary_JAX(
                    expected_regimes=VES_expected_regimes__uniform,  
                    expected_joints=jnp.ones((T - 1, L, L)) / L,
                    log_normalizer=jnp.nan,
                )
                ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
                    ETP_JAX,
                    ES_summary_uniform,
                    EZ_summaries,
                    continuous_states,
                    i,
                    num_M_step_iterations_for_ETP_gradient_descent,
                    model,
                    example_end_times,
                    use_continuous_states,
                    verbose,
                )

            ###
            # # M-step (for CSP)
            # ###
            CSP_JAX = run_M_step_for_CSP_in_closed_form__Gaussian_case(
                EZ_summaries.expected_regimes,
                continuous_states,
                example_end_times,
                use_continuous_states,
            )
    return ResultsFromBottomHalfInit(CSP_JAX, EZ_summaries, record_of_most_likely_states, ETP_JAX)


###
# AR-HMM (TOP HALF)
###


def fit_ARHMM_to_top_half_of_model(
    continuous_states: NumpyArray3D,
    system_covariates: Optional[NumpyArray2D],
    example_end_times: Optional[JaxNumpyArray1D],
    STP_JAX: SystemTransitionParameters_JAX,
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    IP_JAX: InitializationParameters_Gaussian_JAX,
    EZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    num_EM_iterations: int,
    num_M_step_iterations_for_ETP_gradient_descent: int,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    params_frozen: Optional[AllParameters_JAX] = None,
    verbose: bool = True,
) -> ResultsFromTopHalfInit:
    """
    Arguments:
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is True if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step (on STP, ETP, or CSP), nor the VES step.
            We leave the VEZ steps as is, though.

    """
    T = len(continuous_states)
    record_of_most_likely_states = np.zeros((T, num_EM_iterations))

    if system_covariates is None:
        # TODO: Check that D_s=0 as well; if not there is an inconsistency in the implied desire of the caller.
        system_covariates = np.zeros((T, 0))

    if verbose:
        print("\n--- Now running AR-HMM on top half of Model 2a. ---")

    for iteration in range(num_EM_iterations):
        if verbose:
            print(f"Now running EM iteration {iteration+1}/{num_EM_iterations} for AR-HMM on top half of Model 2a.")

        ###
        # E-step
        ###

        # TODO: Handle system covariates properly in this function.
        ES_summary = run_VES_step_JAX(
            STP_JAX,
            ETP_JAX,
            IP_JAX,
            continuous_states,
            EZ_summaries,
            model,
            example_end_times,
            system_covariates,
            use_continuous_states=use_continuous_states,
        )
        

        record_of_most_likely_states[:, iteration] = np.array(np.argmax(ES_summary.expected_regimes, axis=1), dtype=int)


       
        ###
        # M-step
        ###

        if params_frozen:
            ETP_JAX = params_frozen.ETP
            STP_JAX = params_frozen.STP
        else:
            ### M-step (ETP)
            ETP_JAX = run_M_step_for_ETP_via_gradient_descent(
                ETP_JAX,
                ES_summary,
                EZ_summaries,
                continuous_states,
                iteration,
                num_M_step_iterations_for_ETP_gradient_descent,
                model,
                example_end_times,
                use_continuous_states,
                verbose,
            )

            # TODO: Incorporate the system transition prior.  Can I can do this is closed form?
            system_transition_prior = None

            ### M-step (STP)
            num_system_states = np.shape(STP_JAX.Pi)[0]
            if num_system_states == 1:
                # TODO: I had written earlier that the VES step has already taken care of the `use_continuous_states` mask.
                # But I might want to double check that.
                STP_JAX = run_M_step_for_STP_in_closed_form(STP_JAX, ES_summary, example_end_times)
            else:
                NUM_M_STEP_ITERATIONS_FOR_STP_GRADIENT_DESCENT = 5
                STP_JAX = run_M_step_for_STP_via_gradient_descent(
                    STP_JAX,
                    ES_summary,
                    system_transition_prior,
                    iteration,
                    NUM_M_STEP_ITERATIONS_FOR_STP_GRADIENT_DESCENT,
                    model,
                    example_end_times,
                    system_covariates,
                    continuous_states,
                    verbose,
                )
    
    return ResultsFromTopHalfInit(STP_JAX, ETP_JAX, ES_summary, record_of_most_likely_states)


###
# MAIN
###


def smart_initialize_model_2a(
    DIMS: Dims,
    continuous_states: Union[NumpyArray3D, JaxNumpyArray3D],
    example_end_times: Optional[NumpyArray1D],
    model: Model,
    preinitialization_strategy_for_CSP: PreInitialization_Strategy_For_CSP,
    num_em_iterations_for_bottom_half: int = 5,
    num_em_iterations_for_top_half: int = 20,
    seed: int = 120,
    system_covariates: Optional[NumpyArray2D] = None,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    save_dir: Optional[str] = None,
    treat_ETP_params_as_tpm_during_bottom_half_inference: bool = True,
    params_frozen: Optional[AllParameters_JAX] = None,
    verbose: bool = True,
    plotbose: bool = False,
) -> InitializationResults:
    """
    Arguments:
        example_end_times: optional, has shape (E+1,)
            Provides `example` boundaries, which allows us to interpret a time series of shape (T,J,:)
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across i.i.d "examples".
            An example boundary might be induced by a large time gap between timesteps, and/or a discontinuity in the continuous states x.

            If there are E examples, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th example ended.
            So to get the timesteps for the e-th example, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].

        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is True if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step (on STP, ETP, or CSP), nor the VES step.
            We leave the VEZ steps as is, though.

        treat_ETP_params_as_tpm_during_bottom_half_inference: If True, the ETP (entity transition parameters) parameters
            will be treated as a transition probability matrix during the initialization step of the bottom-half rAR-HMMs.
            This is True by default for backwards compatibility.  I think the original motivation for this was that the ETP
            parameters would get overriden anyhow during the top-half ARHMM initialization.  Still, it might be useful to get a better
            estimate due to the fact that the top-half ARHMM initialization starts with a VES step, which requires parameter
            values for ETP.

        params_frozen: Typically this will be None.  However, non-None values can be useful
            for test set inference, when we need to run the E-step on the (context) portion of new data.

        plotbose: plot version of verbose.  Currently this just toggles whether or not we create the expensive
            set of plots where we show the steps (discrete derivatives) assigned to each entity state.

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

    if example_end_times is None:
        T = len(continuous_states)
        example_end_times = np.array([-1, T])

    if not example_end_times_are_proper(example_end_times, len(continuous_states)):
        raise ValueError(
            f"Event end times do not have the proper format. Consult the `events` module "
            f"and try again.  Event_end_times MUST begin with -1 and end with T, the length "
            f"of the grand time series."
        )

    continuous_states = jnp.asarray(continuous_states)

    ###
    # Initialize Params
    ####

    if params_frozen:
        CSP_JAX = params_frozen.CSP
        ETP_JAX = params_frozen.ETP
        IP_JAX = params_frozen.IP
        EP_JAX = params_frozen.EP

    else:
        # TODO: Support fixed or random draws from prior for As, Qs.
        CSP_JAX, kms = make_kmeans_preinitialization_of_CSP_JAX(
            DIMS,
            continuous_states,
            preinitialization_strategy_for_CSP,
            example_end_times,
            use_continuous_states,
            save_dir,
            verbose,
            plotbose,
        )
        # TODO: Support fixed or random draws from prior.
        ETP_JAX = make_data_free_preinitialization_of_ETP_JAX(
            DIMS, method_for_Psis="rnorm", fixed_self_transition_prob=0.90, seed=seed
        )  # Psis is (J, L, K, D_e)
        # TODO: Support fixed or random draws from prior.
        IP_JAX = make_data_free_preinitialization_of_IP_JAX(DIMS)
        # EP_JAX is a placeholder; not used for Figure 8.
        EP_JAX = make_data_free_preinitialization_of_EP_JAX(DIMS)

    ###
    # Fit Bottom-level HMM
    ###
    results_bottom = fit_rARHMM_to_bottom_half_of_model(
        continuous_states,
        example_end_times,
        CSP_JAX,
        ETP_JAX,
        IP_JAX,
        model,
        num_em_iterations_for_bottom_half,
        treat_ETP_params_as_tpm_during_bottom_half_inference,
        use_continuous_states,
        params_frozen,
        verbose,
    )
    # zhats = np.argmax(
    #    results_bottom.EZ_summaries.expected_regimes, axis=2
    # )  # zhats is (J,T) with each entry in {1,..K} (but zero-indexed)

    ###
    # Top-level HMM
    ###

    if params_frozen:
        STP_JAX = params_frozen.STP
    else:
        ### Initialization
        STP_JAX = make_data_free_preinitialization_of_STP_JAX(
            DIMS,
            method_for_Upsilon="rnorm",
            fixed_self_transition_prob=0.95,
            seed=seed,
        )
        # TODO: Is there a better way to init the recurrence matrices and covariances matrices in STP_JAX and ETP_JAX than randomly?
  
    ### run HMM
    num_M_step_iterations_for_ETP_gradient_descent = 5

    # num_M_step_iterations_for_STP_gradient_descent is specified within `fit_ARHMM_to_top_half_of_model`, since
    # we only run gradient descent when there are system covariates

    results_top = fit_ARHMM_to_top_half_of_model(
        continuous_states,
        system_covariates,
        example_end_times,
        STP_JAX,
        results_bottom.ETP,
        IP_JAX,
        results_bottom.EZ_summaries,
        model,
        num_em_iterations_for_top_half,
        num_M_step_iterations_for_ETP_gradient_descent,
        use_continuous_states,
        params_frozen,
        verbose=verbose,
    )

    ### Update Initialization Params
    if params_frozen:
        IP_JAX = params_frozen.IP
    else:
        IP_JAX = run_M_step_for_IP(
            IP_JAX,
            M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN,
            results_top.ES_summary,
            results_bottom.EZ_summaries,
            continuous_states,
            example_end_times,
        )


    results_raw = RawInitializationResults(results_bottom, results_top, IP_JAX, EP_JAX)
    return initialization_results_from_raw_initialization_results(results_raw, params_frozen)
