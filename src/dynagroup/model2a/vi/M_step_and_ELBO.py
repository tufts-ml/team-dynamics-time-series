import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from dynamax.utils.optimize import run_gradient_descent
from jax.scipy.stats import multivariate_normal as mvn_JAX
from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    HMM_Posterior_Summary_NUMPY,
)
from dynagroup.model2a.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)
from dynagroup.model2a.vi.dims import (
    variational_dims_from_summaries,
    variational_dims_from_summaries_JAX,
)
from dynagroup.model2a.vi.prior import SystemTransitionPrior_JAX
from dynagroup.params import (
    AllParameters_JAX,
    CSP_with_unconstrained_covariances_from_ordinary_CSP,
    ContinuousStateParameters_JAX,
    ContinuousStateParameters_WithUnconstrainedCovariances_JAX,
    ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch,
    EntityTransitionParameters_JAX,
    EntityTransitionParameters_MetaSwitch,
    EntityTransitionParameters_MetaSwitch_JAX,
    EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    InitializationParameters_JAX,
    STP_with_unconstrained_tpms_from_ordinary_STP,
    SystemTransitionParameters_JAX,
    SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    ordinary_CSP_from_CSP_with_unconstrained_covariances,
    ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms,
    ordinary_STP_from_STP_with_unconstrained_tpms,
)
from dynagroup.sticky import (
    evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant,
)
from dynagroup.types import JaxNumpyArray3D, JaxNumpyArray5D, NumpyArray3D
from dynagroup.util import (
    normalize_log_potentials_by_axis_JAX,
    normalize_potentials_by_axis_JAX,
)


###
# ELBO
###


@jdc.pytree_dataclass
class ELBO_Decomposed:
    energy: float
    entropy: float
    elbo: float


def compute_energy_from_init(
    IP: InitializationParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    continuous_states: JaxNumpyArray3D,
) -> float:
    energy_init_system = jnp.sum(VES_summary.expected_regimes[0] * jnp.log(IP.pi_system))

    J, K = np.shape(IP.mu_0s)[:2]
    energy_init_entities = 0.0
    for j in range(J):
        energy_init_entities += jnp.sum(
            VEZ_summaries.expected_regimes[0, j] * jnp.log(IP.pi_entities[j])
        )

    energy_init_continuous_states = 0.0
    for j in range(J):
        for k in range(K):
            energy_init_continuous_states += VEZ_summaries.expected_regimes[
                0, j, k
            ] * mvn_JAX.logpdf(continuous_states[0, j], IP.mu_0s[j, k], IP.Sigma_0s[j, k])

    return energy_init_system + energy_init_entities + energy_init_continuous_states


def compute_energy(
    STP: SystemTransitionParameters_JAX,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    CSP: ContinuousStateParameters_JAX,
    IP: InitializationParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> float:
    energy_init = compute_energy_from_init(IP, VES_summary, VEZ_summaries, continuous_states)
    energy_post_init_negated_and_divided_by_num_timesteps = 0.0
    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_objective_for_system_transition_parameters_JAX(
            STP, VES_summary, system_transition_prior
        )
    )
    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_objective_for_entity_transition_parameters_JAX(
            ETP,
            continuous_states,
            VES_summary,
            VEZ_summaries,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
        )
    )

    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_objective_for_continuous_state_parameters_after_initial_timestep_JAX(
            CSP,
            continuous_states,
            VEZ_summaries,
        )
    )

    T = np.shape(continuous_states)[0]
    energy_post_init = -energy_post_init_negated_and_divided_by_num_timesteps * T

    return energy_init + energy_post_init


def compute_entropy(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> float:
    return VES_summary.entropy + jnp.sum(VEZ_summaries.entropies)


def compute_elbo_decomposed(
    all_params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> ELBO_Decomposed:
    energy = compute_energy(
        all_params.STP,
        all_params.ETP,
        all_params.CSP,
        all_params.IP,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    entropy = compute_entropy(VES_summary, VEZ_summaries)
    elbo = energy + entropy
    return ELBO_Decomposed(energy, entropy, elbo)


###
# M-step Toggles
###


class M_Step_Toggle_Value(Enum):
    OFF = 1
    GRADIENT_DESCENT = 2
    CLOSED_FORM = 3


@dataclass
class M_Step_Toggles:
    STP: M_Step_Toggle_Value
    ETP: M_Step_Toggle_Value
    CSP: M_Step_Toggle_Value
    IP: M_Step_Toggle_Value


def M_step_toggles_from_strings(
    STP_toggle: str,
    ETP_toggle: str,
    CSP_toggle: str,
    IP_toggle: str,
) -> M_Step_Toggles:
    """
    Describes what kind of M-step should be done for each subclass of parameters:
        gradient-descent, closed-form, or off.

    As of 4/20/23, supported values are:
        STP: Closed-form, gradient decent, or off
        ETP: Gradient decent, or off
        CSP: Closed-form, gradient decent, or off (but gradient descent doesn't work very well)
        IP: Closed-form or off
    """
    return M_Step_Toggles(
        STP=M_Step_Toggle_Value[STP_toggle.upper()],
        ETP=M_Step_Toggle_Value[ETP_toggle.upper()],
        CSP=M_Step_Toggle_Value[CSP_toggle.upper()],
        IP=M_Step_Toggle_Value[IP_toggle.upper()],
    )


###
# The M-step
###


def run_M_step_for_init_params_JAX(
    IP: InitializationParameters_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    continuous_states: JaxNumpyArray3D,
) -> InitializationParameters_JAX:
    EPSILON = 1e-3
    # These are set to be the values that minimize the cross-entropy, plus some noise
    pi_system = normalize_potentials_by_axis_JAX(VES_summary.expected_regimes[0] + EPSILON, axis=0)
    pi_entities = normalize_potentials_by_axis_JAX(
        VEZ_summaries.expected_regimes[0] + EPSILON, axis=1
    )

    K = jnp.shape(pi_entities)[1]

    # set mu_0s to be equal to observed x's.
    mu_0s = jnp.tile(continuous_states[0][:, None, :], (1, K, 1))
    # keep Sigma_0s to tbe the same as initialized... not clear how to learn these
    return InitializationParameters_JAX(pi_system, pi_entities, mu_0s, IP.Sigma_0s)


def compute_variational_posterior_on_entity_transitions(
    VES_summary: HMM_Posterior_Summary_NUMPY,
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
):
    """
    Returns:
        np.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the VARIATIONAL
            probability of the the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives Q(z_{t+1}^j = k',  z_t^j =k) Q(s_{t+1}=l).
            This gives a probability distribution over all triplets (l,k,k').
    """
    DIMS = variational_dims_from_summaries(VES_summary, VEZ_summaries)

    variational_probs = np.ones((DIMS.T - 1, DIMS.J, DIMS.L, DIMS.K, DIMS.K))
    for j in range(DIMS.J):
        for l in range(DIMS.L):
            for k in range(DIMS.K):
                for k_prime in range(DIMS.K):
                    variational_probs[:, j, l, k, k_prime] *= VES_summary.expected_regimes[1:, l]
                    variational_probs[:, j, l, k, k_prime] *= VEZ_summaries[j].expected_joints[
                        :, k, k_prime
                    ]

    # TODO: write test confirming that I am intrepreting K and K prime in the right order
    # when extracting the info from VEZ_summaries.

    # TODO: write test confirming that this gives a valid probability distribution over all triplets (l,k,k')
    # We should have np.sum(variational_probs[t,j])==1 for all t,j.
    return variational_probs


def compute_variational_posterior_on_entity_transitions_JAX(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> JaxNumpyArray5D:
    """
    Returns:
        np.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the VARIATIONAL
            probability of the the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives Q(z_{t+1}^j = k',  z_t^j =k) Q(s_{t+1}=l).
            This gives a probability distribution over all triplets (l,k,k').
    """

    # TODO: write test confirming that this gives a valid probability distribution over all triplets (l,k,k')
    # We should have np.sum(variational_probs[t,j])==1 for all t,j.

    # VES_summary.expected_regimes has shape (T,L)
    # VEZ_summaries.expected_joints has shape (T-1,J,K,K)
    return (
        VES_summary.expected_regimes[1:, None, :, None, None]
        * VEZ_summaries.expected_joints[:, :, None, :, :]
    )


def compute_expected_log_entity_transitions(
    continuous_states: NumpyArray3D,
    ETP: EntityTransitionParameters_MetaSwitch,
    VES_summary: HMM_Posterior_Summary_NUMPY,
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> float:
    """
    Arguments:
        continuous_states: has shape (T, J, D)
    """
    variational_probs = compute_variational_posterior_on_entity_transitions(
        VES_summary, VEZ_summaries
    )
    log_transition_matrices = compute_log_entity_transition_probability_matrices(
        ETP,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    )
    return np.sum(variational_probs * log_transition_matrices)


def compute_expected_log_entity_transitions_JAX(
    continuous_states: JaxNumpyArray3D,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> float:
    """
    Arguments:
        continuous_states: has shape (T, J, D)
    """
    variational_probs = compute_variational_posterior_on_entity_transitions_JAX(
        VES_summary, VEZ_summaries
    )
    log_transition_matrices = compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        continuous_states[:-1],
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    )
    return jnp.sum(variational_probs * log_transition_matrices)


def compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> float:
    variational_probs = VEZ_summaries.expected_regimes[1:]
    log_continuous_state_dynamics = (
        compute_log_continuous_state_emissions_after_initial_timestep_JAX(
            CSP,
            continuous_states,
        )
    )
    return jnp.sum(variational_probs * log_continuous_state_dynamics)


def compute_expected_log_system_transitions_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
) -> float:
    """
    Arguments:
        continuous_states: has shape (T, J, D)
    """
    # `variational_probs` has shape (T-1,L,L); entry (t,l,l') gives q(s_{t+1}=l', s_t=1)
    variational_probs = VES_summary.expected_joints
    # ` log_transition_matrices` has shape (T-1,L,L)
    log_transition_matrices = compute_log_system_transition_probability_matrices_JAX(
        STP, T=np.shape(variational_probs)[0] + 1
    )

    return jnp.sum(variational_probs * log_transition_matrices)


def compute_objective_for_entity_transition_parameters(
    continuous_states: NumpyArray3D,
    ETP: EntityTransitionParameters_MetaSwitch,
    VES_summary: HMM_Posterior_Summary_NUMPY,
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> float:
    expected_log_transitions = compute_expected_log_entity_transitions(
        continuous_states,
        ETP,
        VES_summary,
        VEZ_summaries,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    )
    log_prior = 0.0  # TODO: Add prior?
    energy_non_constant = expected_log_transitions + log_prior

    # Normalize and negate for minimization
    DIMS = variational_dims_from_summaries(VES_summary, VEZ_summaries)
    return -energy_non_constant / DIMS.T


def compute_objective_for_entity_transition_parameters_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    continuous_states: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> float:
    # TODO: Combine with `compute_objective_for_system_transition_parameters_JAX` ?
    expected_log_transitions = compute_expected_log_entity_transitions_JAX(
        continuous_states,
        ETP,
        VES_summary,
        VEZ_summaries,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    log_prior = 0.0  # TODO: Add prior?
    energy = expected_log_transitions + log_prior

    # Normalize and negate for minimization
    DIMS = variational_dims_from_summaries_JAX(VES_summary, VEZ_summaries)
    return -energy / DIMS.T


def compute_objective_for_entity_transition_parameters_with_unconstrained_tpms_JAX(
    ETP_WUC: EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    continuous_states: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> float:
    ETP = ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(ETP_WUC)
    return compute_objective_for_entity_transition_parameters_JAX(
        ETP,
        continuous_states,
        VES_summary,
        VEZ_summaries,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )


def compute_objective_for_system_transition_parameters_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
) -> float:
    # TODO: Combine with `compute_objective_for_entity_transition_parameters_JAX` ?
    expected_log_transitions = compute_expected_log_system_transitions_JAX(STP, VES_summary)
    if system_transition_prior is not None:
        log_prior = evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
            normalize_log_potentials_by_axis_JAX(STP.Pi, axis=1),
            system_transition_prior.alpha,
            system_transition_prior.kappa,
        )
    else:
        log_prior = 0.0
    energy_non_constant = expected_log_transitions + log_prior
    T = jnp.shape(VES_summary.expected_regimes)[0]
    return -energy_non_constant / T


def compute_objective_for_system_transition_parameters_with_unconstrained_tpms_JAX(
    STP_WUC: SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
) -> float:
    STP = ordinary_STP_from_STP_with_unconstrained_tpms(STP_WUC)
    return compute_objective_for_system_transition_parameters_JAX(
        STP,
        VES_summary,
        system_transition_prior,
    )


def compute_objective_for_continuous_state_parameters_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> float:
    expected_log_state_dynamics = (
        compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
            CSP,
            continuous_states,
            VEZ_summaries,
        )
    )
    log_prior = 0.0
    energy = expected_log_state_dynamics + log_prior
    T = len(continuous_states)
    return -energy / T


def compute_objective_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX(
    CSP_WUC: ContinuousStateParameters_WithUnconstrainedCovariances_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> float:
    CSP = ordinary_CSP_from_CSP_with_unconstrained_covariances(CSP_WUC)
    return compute_objective_for_continuous_state_parameters_after_initial_timestep_JAX(
        CSP,
        continuous_states,
        VEZ_summaries,
    )


def run_M_step_in_closed_form_for_continuous_state_params_JAX(
    VEZ_summaries: HMM_Posterior_Summaries_JAX, continuous_states: JaxNumpyArray3D
) -> ContinuousStateParameters_JAX:
    """
    The M-step for CSP for this model is just the solution for a vector auto-regression (VAR) model
    with sample weights given by the expected entity-level regimes.

    x_t^j | x_{t-1}^j, z_t^j=k ~ N(A_j^k x_{t-1}^j + b_j^k, Q_j^k)

    so to get the parameters for the (j,k)-th entity and entity-regime,
    we weight each sample by the q(z_t^j=k).
    """

    _, J, K = np.shape(VEZ_summaries.expected_regimes)
    D = np.shape(continuous_states)[2]
    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.zeros((J, K, D, D))

    for j in range(J):
        xs = np.asarray(continuous_states[:, j, :])
        for k in range(K):
            weights = np.asarray(VEZ_summaries.expected_regimes[:, j, k])  # TxJxK
            responses = xs[1:]
            predictors = add_constant(xs[:-1], prepend=False)
            wls_model = WLS(responses, predictors, hasconst=True, weights=weights[1:])
            results = wls_model.fit()
            # WLS returns parameters where the d-th column gives the weights for predicting d-th element of response vector.
            # So we need to transpose to get a state transition matrix
            As[j, k] = results.params[:-1].T
            bs[j, k] = results.params[-1]
            residuals = results.resid * weights[1:][:, None]
            Qs[j, k] = np.cov(residuals.T)
    return ContinuousStateParameters_JAX(jnp.asarray(As), jnp.asarray(bs), jnp.asarray(Qs))


def run_M_step_for_ETP_via_gradient_descent(
    ETP: EntityTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Optional[
        Callable
    ] = None,
    verbose: bool = True,
) -> EntityTransitionParameters_JAX:
    ### Do gradient descent on unconstrained parameters.
    ETP_WUC = ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch(ETP)

    cost_function_ETP = functools.partial(
        compute_objective_for_entity_transition_parameters_with_unconstrained_tpms_JAX,
        continuous_states=continuous_states,
        VES_summary=VES_summary,
        VEZ_summaries=VEZ_summaries,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX=transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )

    # We reset the optimizer state to None before each run of the optimizer (which is ADAM)
    # because we want to reset the EWMA now that we have new contextual information from the other substeps of
    # variational EM (as provided via the frozen arguments in the partial function representation of cost_function_ETP).
    optimizer_state_for_entity_transitions = None
    (
        ETP_WUC_new,
        optimizer_state_for_entity_transitions,
        losses_for_entity_transitions,
    ) = run_gradient_descent(
        cost_function_ETP,
        ETP_WUC,
        optimizer_state=optimizer_state_for_entity_transitions,
        num_mstep_iters=num_M_step_iters,
    )

    if verbose:
        print(
            f"For iteration {iteration+1} of the M-step with entity transitions, First 5 Losses are {losses_for_entity_transitions[:5]}. Last 5 losses are {losses_for_entity_transitions[-5:]}"
        )

    return ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(ETP_WUC_new)


def run_M_step_for_ETP(
    all_params: AllParameters_JAX,
    M_step_toggles_ETP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Optional[
        Callable
    ] = None,
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_ETP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for ETP, as requested.")
        return all_params
    elif M_step_toggles_ETP == M_Step_Toggle_Value.CLOSED_FORM:
        raise ValueError(
            "Closed-form solution to the M-step for Entity Transition parameters is not available."
        )
    elif M_step_toggles_ETP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        ### Do gradient descent on unconstrained parameters.
        ETP_new = run_M_step_for_ETP_via_gradient_descent(
            all_params.ETP,
            VES_summary,
            VEZ_summaries,
            continuous_states,
            iteration,
            num_M_step_iters,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
            verbose,
        )
    else:
        raise ValueError("I do not know what to do with ETP for the M-step.")

    all_params = AllParameters_JAX(
        all_params.STP, ETP_new, all_params.CSP, all_params.EP, all_params.IP
    )

    elbo_decomposed = compute_elbo_decomposed(
        all_params,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    if verbose:
        print(
            f"After ETP-M step on iteration {iteration+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
        )
    return all_params


def run_M_step_for_STP_in_closed_form(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
) -> SystemTransitionParameters_JAX:
    # Previously, we had
    #     STP_gives_a_TPM = not STP.Gammas.any() and not STP.Upsilon.any() and STP.Pi.any()
    # But in the special case where the number of system regimes L=1,
    # then the tpm is [[1]], and the log of that is [[0]], so the check fails.
    # Thus, we change the condition to simply
    #     STP_gives_a_TPM = not STP.Gammas.any() and not STP.Upsilon.any()
    STP_gives_a_TPM = not STP.Gammas.any() and not STP.Upsilon.any()
    if not STP_gives_a_TPM:
        raise ValueError("Closed-form M step is available for STP only if STP gives a TPM.")
    warnings.warn(
        "Running closed-form M-step for STP.  Note that this ignores the prior specification."
    )
    L = np.shape(VES_summary.expected_joints)[1]
    exp_Pi = np.zeros((L, L))
    for l in range(L):
        for l_prime in range(L):
            exp_Pi[l, l_prime] = np.sum(
                VES_summary.expected_joints[2:, l, l_prime], axis=0
            ) / np.sum(VES_summary.expected_regimes[:-1, l], axis=0)
    Pi_new = jnp.asarray(np.log(exp_Pi))
    return SystemTransitionParameters_JAX(STP.Gammas, STP.Upsilon, Pi_new)


def run_M_step_for_STP(
    all_params: AllParameters_JAX,
    M_step_toggles_STP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Optional[
        Callable
    ] = None,
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_STP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for STP, as requested.")
        return all_params
    elif M_step_toggles_STP == M_Step_Toggle_Value.CLOSED_FORM:
        STP_new = run_M_step_for_STP_in_closed_form(all_params.STP, VES_summary)
    elif M_step_toggles_STP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        ### Do gradient descent on unconstrained parameters.
        STP_WUC = STP_with_unconstrained_tpms_from_ordinary_STP(all_params.STP)
        cost_function_STP = functools.partial(
            compute_objective_for_system_transition_parameters_with_unconstrained_tpms_JAX,
            VES_summary=VES_summary,
            system_transition_prior=system_transition_prior,
        )

        # We reset the optimizer state to None before each run of the optimizer (which is ADAM)
        # because we want to reset the EWMA now that we have new contextual information (here, VES_summary).
        optimizer_state_for_system_transitions = None
        (
            STP_WUC_new,
            optimizer_state_for_system_transitions,
            losses_for_system_transitions,
        ) = run_gradient_descent(
            cost_function_STP,
            STP_WUC,
            optimizer_state=optimizer_state_for_system_transitions,
            num_mstep_iters=num_M_step_iters,
        )

        STP_new = ordinary_STP_from_STP_with_unconstrained_tpms(STP_WUC_new)

        if verbose:
            print(
                f"For iteration {iteration+1} of the M-step with system transitions, First 5 Losses are {losses_for_system_transitions[:5]}. Last 5 losses are {losses_for_system_transitions[-5:]}"
            )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with system transition parameters."
        )

    all_params = AllParameters_JAX(
        STP_new, all_params.ETP, all_params.CSP, all_params.EP, all_params.IP
    )

    elbo_decomposed = compute_elbo_decomposed(
        all_params,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    if verbose:
        print(
            f"After STP-M step on iteration {iteration+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
        )
    return all_params


def run_M_step_for_CSP(
    all_params: AllParameters_JAX,
    M_step_toggles_CSP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Optional[
        Callable
    ] = None,
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_CSP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for CSP, as requested.")
        return all_params
    elif M_step_toggles_CSP == M_Step_Toggle_Value.CLOSED_FORM:
        CSP_new = run_M_step_in_closed_form_for_continuous_state_params_JAX(
            VEZ_summaries, continuous_states
        )
    elif M_step_toggles_CSP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        warnings.warn(
            f"Learning the CSP parameters by gradient descent.  Performance seems to be worse than with the closed-form approach. "
            f"We should do the analogue to ETP, STP steps which used tensorflow.probability to convert simplex-valued parameters to unconstrained rep and back."
            f"that is, we should rely upon tensorflow.probability to convert covariance parameters to unconstrained representation and back."
        )

        CSP_WUC = CSP_with_unconstrained_covariances_from_ordinary_CSP(all_params.CSP)

        cost_function_CSP = functools.partial(
            compute_objective_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX,
            continuous_states=continuous_states,
            VEZ_summaries=VEZ_summaries,
        )

        optimizer_state_for_state_dynamics = None
        (
            CSP_WUC_new,
            optimizer_state_for_state_dynamics,
            losses_for_state_dynamics,
        ) = run_gradient_descent(
            cost_function_CSP,
            CSP_WUC,
            optimizer_state=optimizer_state_for_state_dynamics,
            num_mstep_iters=num_M_step_iters,
        )

        CSP_new = ordinary_CSP_from_CSP_with_unconstrained_covariances(CSP_WUC_new)

        print(
            f"For iteration {iteration+1} of the M-step with continuous state dynamics, First 5 Losses are {losses_for_state_dynamics[:5]}. Last 5 losses are {losses_for_state_dynamics[-5:]}"
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with continuous state parameters."
        )

    all_params = AllParameters_JAX(
        all_params.STP, all_params.ETP, CSP_new, all_params.EP, all_params.IP
    )

    elbo_decomposed = compute_elbo_decomposed(
        all_params,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    if verbose:
        print(
            f"After CSP-M step on iteration {iteration+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
        )
    return all_params


def run_M_step_for_IP(
    all_params: AllParameters_JAX,
    M_step_toggles_IP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Optional[
        Callable
    ] = None,
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_IP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for IP, as requested.")
        return all_params
    elif M_step_toggles_IP == M_Step_Toggle_Value.CLOSED_FORM:
        IP_new = run_M_step_for_init_params_JAX(
            all_params.IP,
            VEZ_summaries,
            VES_summary,
            continuous_states,
        )
    elif M_step_toggles_IP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        raise ValueError(
            f"Learning the IP parameters by gradient descent is not currently supported.  Try closed-form"
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with continuous state parameters."
        )

    all_params = AllParameters_JAX(
        all_params.STP, all_params.ETP, all_params.CSP, all_params.EP, IP_new
    )

    elbo_decomposed = compute_elbo_decomposed(
        all_params,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    if verbose:
        print(
            f"After IP-M step on iteration {iteration+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
        )
    return all_params
