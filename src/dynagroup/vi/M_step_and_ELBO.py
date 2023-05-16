import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from dynamax.utils.optimize import run_gradient_descent
from joblib import Parallel, delayed
from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant

from dynagroup.events import (
    eligible_transitions_to_next,
    get_initialization_times,
    get_non_initialization_times,
    only_one_event,
)
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    compute_closed_form_M_step,
)
from dynagroup.model import Model
from dynagroup.params import (
    AllParameters_JAX,
    CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian,
    ContinuousStateParameters_Gaussian_JAX,
    ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
    ContinuousStateParameters_JAX,
    ContinuousStateParameters_VonMises_JAX,
    ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch,
    EntityTransitionParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    InitializationParameters_Gaussian_JAX,
    InitializationParameters_JAX,
    InitializationParameters_VonMises_JAX,
    STP_with_unconstrained_tpms_from_ordinary_STP,
    SystemTransitionParameters_JAX,
    SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances,
    ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms,
    ordinary_STP_from_STP_with_unconstrained_tpms,
)
from dynagroup.sticky import (
    evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant,
)
from dynagroup.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
    NumpyArray1D,
    NumpyArray3D,
)
from dynagroup.util import (
    normalize_log_potentials_by_axis_JAX,
    normalize_potentials_by_axis_JAX,
)
from dynagroup.vi.prior import SystemTransitionPrior_JAX
from dynagroup.von_mises.inference.ar import (
    VonMisesModelType,
    estimate_von_mises_params,
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
    model: Model,
    event_end_times: NumpyArray1D,
) -> float:
    init_times = get_initialization_times(event_end_times)

    expected_system_init_probs = jnp.sum(
        VES_summary.expected_regimes[init_times], axis=0
    )  # shape (L,)
    energy_init_system = jnp.sum(expected_system_init_probs * jnp.log(IP.pi_system))

    J, K = np.shape(IP.pi_entities)
    energy_init_entities = 0.0
    for j in range(J):
        expected_entity_init_probs = jnp.sum(
            VEZ_summaries.expected_regimes[init_times, j], axis=0
        )  # shape (K,)
        energy_init_entities += jnp.sum(expected_entity_init_probs * jnp.log(IP.pi_entities[j]))

    energy_init_continuous_states = 0.0
    for j in range(J):
        for k in range(K):
            for t_init in init_times:
                log_pdfs_at_some_init_time = (
                    model.compute_log_initial_continuous_state_emissions_JAX(
                        IP, continuous_states[t_init]
                    )
                )
                energy_init_continuous_states += (
                    VEZ_summaries.expected_regimes[t_init, j, k] * log_pdfs_at_some_init_time[j, k]
                )

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
    model: Model,
    event_end_times: JaxNumpyArray2D,
    system_covariates: Optional[JaxNumpyArray2D],
) -> float:
    energy_init = compute_energy_from_init(
        IP, VES_summary, VEZ_summaries, continuous_states, model, event_end_times
    )
    energy_post_init_negated_and_divided_by_num_timesteps = 0.0
    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_cost_for_system_transition_parameters_JAX(
            STP,
            VES_summary,
            system_transition_prior,
            model,
            event_end_times,
            system_covariates,
        )
    )
    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_cost_for_entity_transition_parameters_JAX(
            ETP,
            continuous_states,
            VES_summary,
            VEZ_summaries,
            model,
            event_end_times,
        )
    )

    energy_post_init_negated_and_divided_by_num_timesteps += (
        compute_cost_for_continuous_state_parameters_after_initial_timestep_JAX(
            CSP,
            continuous_states,
            VEZ_summaries,
            model,
            event_end_times,
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
    model: Model,
    event_end_times: JaxNumpyArray2D,
    system_covariates: Optional[JaxNumpyArray2D],
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
        model,
        event_end_times,
        system_covariates,
    )
    entropy = compute_entropy(VES_summary, VEZ_summaries)
    elbo = energy + entropy
    return ELBO_Decomposed(energy, entropy, elbo)


###
# M-step Toggles
###


class M_Step_Toggle_Value(Enum):
    # TODO: The variations on closed form (GAUSSIAN, VON_MISES, etc.)
    # Should probably just be offloaded to an Inference class (similar to Model class)
    # but where we provide specific functions for closed-form inference, if available.
    OFF = 1
    GRADIENT_DESCENT = 2
    CLOSED_FORM_TPM = 3
    CLOSED_FORM_GAUSSIAN = 4
    CLOSED_FORM_VON_MISES = 5


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
# Compute costs
###

# These costs are used for two things:
#   1) For parameter optimization
#   2) For computing the ELBO (the negative cost gives that model subcomponent's
#       contribution to the expected log likelihood)


def compute_variational_posterior_on_regime_triplets_JAX(
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


def compute_expected_log_entity_transitions_JAX(
    continuous_states: JaxNumpyArray3D,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    Arguments:
        continuous_states: has shape (T, J, D)
    """

    T, J = np.shape(continuous_states)[:2]

    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    variational_probs = compute_variational_posterior_on_regime_triplets_JAX(
        VES_summary, VEZ_summaries
    )
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        continuous_states[:-1],
        model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    log_transition_matrices_weighted = (
        log_transition_matrices
        * use_continuous_states[1:, :, None, None, None]
        * eligible_transitions_to_next(event_end_times)[:, None, None, None, None]
    )

    return jnp.sum(variational_probs * log_transition_matrices_weighted)


def compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: JaxNumpyArray2D,
) -> float:
    non_initialization_times = get_non_initialization_times(event_end_times)
    non_initialization_times_shifted_one_index_lower = non_initialization_times - 1

    variational_probs = VEZ_summaries.expected_regimes[non_initialization_times]
    log_continuous_state_dynamics_after_time_zero = (
        model.compute_log_continuous_state_emissions_after_initial_timestep_JAX(
            CSP,
            continuous_states,
        )
    )
    # log_continuous_state_dynamics is (T-1,J,K)

    log_continuous_state_dynamics_weighted = (
        log_continuous_state_dynamics_after_time_zero[
            non_initialization_times_shifted_one_index_lower
        ]
        * use_continuous_states[non_initialization_times, :, None]
    )
    return jnp.sum(variational_probs * log_continuous_state_dynamics_weighted)


def compute_expected_log_system_transitions_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D],
) -> float:
    """
    Arguments:
        system_covariates: has shape (T, M_d)
    """
    # `variational_probs` has shape (T-1,L,L); entry (t,l,l') gives q(s_{t+1}=l', s_t=1)
    variational_probs = VES_summary.expected_joints
    T_minus_1 = np.shape(variational_probs)[0]

    # ` log_transition_matrices` has shape (T-1,L,L)
    log_transition_matrices = model.compute_log_system_transition_probability_matrices_JAX(
        STP,
        T_minus_1,
        system_covariates=system_covariates,
    )
    return jnp.sum(
        variational_probs
        * log_transition_matrices
        * eligible_transitions_to_next(event_end_times)[:, None, None]
    )


def compute_cost_for_entity_transition_parameters_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    continuous_states: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.

    Arguments:
        use_continuous_states: Defaults to None (which means all states are used) because
            when we compute the ELBO, we always assume a full dataset.  This is simply because
            I haven't had the time yet to dig into ssm.messages to handle partial observations
            when doing forward backward.
    """
    T, J = np.shape(continuous_states)[:2]
    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    # TODO: Combine with `compute_cost_for_system_transition_parameters_JAX` ?
    expected_log_transitions = compute_expected_log_entity_transitions_JAX(
        continuous_states,
        ETP,
        VES_summary,
        VEZ_summaries,
        model,
        event_end_times,
        use_continuous_states,
    )
    log_prior = 0.0  # TODO: Add prior?
    energy = expected_log_transitions + log_prior

    return -energy / jnp.sum(use_continuous_states)


def compute_cost_for_entity_transition_parameters_with_unconstrained_tpms_JAX(
    ETP_WUC: EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    continuous_states: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.
    """
    ETP = ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(ETP_WUC)
    return compute_cost_for_entity_transition_parameters_JAX(
        ETP,
        continuous_states,
        VES_summary,
        VEZ_summaries,
        model,
        event_end_times,
        use_continuous_states=use_continuous_states,
    )


def compute_cost_for_system_transition_parameters_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    event_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D],
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.
    """
    # TODO: Combine with `compute_cost_for_entity_transition_parameters_JAX` ?
    expected_log_transitions = compute_expected_log_system_transitions_JAX(
        STP,
        VES_summary,
        model,
        event_end_times,
        system_covariates,
    )
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


def compute_cost_for_system_transition_parameters_with_unconstrained_tpms_JAX(
    STP_WUC: SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    event_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D],
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.
    """
    STP = ordinary_STP_from_STP_with_unconstrained_tpms(STP_WUC)
    return compute_cost_for_system_transition_parameters_JAX(
        STP,
        VES_summary,
        system_transition_prior,
        model,
        event_end_times,
        system_covariates,
    )


def compute_cost_for_continuous_state_parameters_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.

    Arguments:
        use_continuous_states: Defaults to None (which means all states are used) because
            when we compute the ELBO, we always assume a full dataset.  This is simply because
            I haven't had the time yet to dig into ssm.messages to handle partial observations
            when doing forward backward.
    """

    T, J = np.shape(continuous_states)[:2]
    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    expected_log_state_dynamics = (
        compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
            CSP,
            continuous_states,
            VEZ_summaries,
            model,
            event_end_times,
            use_continuous_states,
        )
    )
    log_prior = 0.0
    energy = expected_log_state_dynamics + log_prior
    return -energy / jnp.sum(use_continuous_states)


def compute_cost_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX(
    CSP_WUC: ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: JaxNumpyArray2D,
) -> float:
    """
    The cost function is the negative of the energy, where the energy is the
        expected log likelihood + log prior

    Note that we only need the parts of the log likelihood and log prior that are
    relevant to these particular parameters.
    """
    CSP = ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(CSP_WUC)
    return compute_cost_for_continuous_state_parameters_after_initial_timestep_JAX(
        CSP,
        continuous_states,
        VEZ_summaries,
        model,
        event_end_times,
        use_continuous_states,
    )


###
# Run M-steps
###


def run_M_step_for_CSP_in_closed_form__Gaussian_case(
    VEZ_expected_regimes: JaxNumpyArray3D,
    continuous_states: JaxNumpyArray3D,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> ContinuousStateParameters_Gaussian_JAX:
    """
    The M-step for CSP for this model is just the solution for a vector auto-regression (VAR) model
    with sample weights given by the expected entity-level regimes.

    x_t^j | x_{t-1}^j, z_t^j=k ~ N(A_j^k x_{t-1}^j + b_j^k, Q_j^k)

    so to get the parameters for the (j,k)-th entity and entity-regime,
    we weight each sample by the q(z_t^j=k).

    Arguments:
        VEZ_expected_regimes: (T,J,K) array, the expected_regimes attribute from
            the HMM_Posterior_Summaries_JAX class.
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is True if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
            that info to do the M-step.
    """

    # Upfront
    T, J, K = np.shape(VEZ_expected_regimes)
    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    # Preprocess based on the possible existence of event segmentation times
    times_to_use = slice(None)
    if not only_one_event(event_end_times, T):
        times_to_use = get_non_initialization_times(event_end_times)

    VEZ_expected_regimes_to_use = VEZ_expected_regimes[times_to_use]
    continuous_states_to_use = continuous_states[times_to_use]
    use_continuous_states_to_use = (
        use_continuous_states[times_to_use] if use_continuous_states is not None else None
    )

    # Run the closed-form M-step.
    D = np.shape(continuous_states_to_use)[2]

    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.zeros((J, K, D, D))

    for j in range(J):
        xs = np.asarray(continuous_states_to_use[:, j, :])
        for k in range(K):
            weights = np.asarray(
                VEZ_expected_regimes_to_use[:, j, k] * use_continuous_states_to_use[:, j]
            )  # TxJxK
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
    return ContinuousStateParameters_Gaussian_JAX(jnp.asarray(As), jnp.asarray(bs), jnp.asarray(Qs))


# TODO: Consider moving `run_M_step_for_VonMises_CSP_in_closed_form`
#  out of the general M-step module (which should be only for general gradient descent)


def run_M_step_for_CSP_in_closed_form__VonMises_case(
    VEZ_expected_regimes: JaxNumpyArray3D,
    group_angles: JaxNumpyArray2D,
    all_params: AllParameters_JAX,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D],
    parallelize: bool = True,
) -> ContinuousStateParameters_VonMises_JAX:
    """
    Arguments:
        VEZ_expected_regimes: (T,J,K) array, the expected_regimes attribute from
            the HMM_Posterior_Summaries_JAX class.

        group_angles:
            Array of shape (T,J)

    Remarks:
        Unlike with the Gaussian model, we need to initialize the M-step (from the previous
        value of the parameters).
    """
    if (use_continuous_states is not None) and (False in use_continuous_states):
        raise NotImplementedError(
            f"This function has not yet been expanded to handle the case where a subset of continuous "
            f"states are not used."
        )

    if not only_one_event(event_end_times, T=len(group_angles)):
        raise NotImplementedError(
            f"This function has not yet been expanded to handle the case where the time series is "
            f"spliced into separate events.  For guidance, see how this was handled in the Gaussian case."
        )
    J = np.shape(group_angles)[1]
    K = np.shape(VEZ_expected_regimes)[-1]

    # TODO: Pick a shape for group_angles up front and enforce it!
    # I keep switching around from function to function
    group_angles = np.squeeze(group_angles)

    ar_coefs = np.zeros((J, K))
    drifts = np.zeros((J, K))
    kappas = np.zeros((J, K))

    def estimate_params(j, k):
        emissions_params = estimate_von_mises_params(
            np.asarray(group_angles[:, j]),
            VonMisesModelType.AUTOREGRESSION,
            all_params.CSP.ar_coefs[j, k],
            all_params.CSP.drifts[j, k],
            sample_weights=np.asarray(VEZ_expected_regimes[:, j, k]),
            suppress_warnings=True,
        )
        return emissions_params.ar_coef, emissions_params.drift, emissions_params.kappa

    if not parallelize:
        for j in range(J):
            for k in range(K):
                emissions_params = estimate_params(j, k)
                ar_coefs[j, k] = emissions_params.ar_coef
                drifts[j, k] = emissions_params.drift
                kappas[j, k] = emissions_params.kappa

    else:
        # Parallelize the loop using joblib
        results = Parallel(n_jobs=-1)(
            delayed(estimate_params)(j, k) for j in range(J) for k in range(K)
        )

        # Unpack the results
        ar_coefs = np.empty((J, K))
        drifts = np.empty((J, K))
        kappas = np.empty((J, K))
        for i, (ar_coef, drift, kappa) in enumerate(results):
            j = i // K
            k = i % K
            ar_coefs[j, k] = ar_coef
            drifts[j, k] = drift
            kappas[j, k] = kappa

    return ContinuousStateParameters_VonMises_JAX(
        jnp.asarray(ar_coefs), jnp.asarray(drifts), jnp.asarray(kappas)
    )


def run_M_step_for_ETP_via_gradient_descent(
    ETP: EntityTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    verbose: bool = True,
) -> EntityTransitionParameters_JAX:
    """
    Arguments:
        event_end_times: optional, has shape (E+1,)
            An `event` takes an ordinary sampled group time series of shape (T,J,:) and interprets it as (T_grand,J,:),
            where T_grand is the sum of the number of timesteps across i.i.d "events".  An event might induce a large
            time gap between timesteps, and a discontinuity in the continuous states x.

            If there are E events, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th eveent ended.
            So to get the timesteps for the e-th event, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].
    """
    T, J = np.shape(continuous_states)[:2]
    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    ### Do gradient descent on unconstrained parameters.
    ETP_WUC = ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch(ETP)

    cost_function_ETP = functools.partial(
        compute_cost_for_entity_transition_parameters_with_unconstrained_tpms_JAX,
        continuous_states=continuous_states,
        VES_summary=VES_summary,
        VEZ_summaries=VEZ_summaries,
        model=model,
        event_end_times=event_end_times,
        use_continuous_states=use_continuous_states,
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
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_ETP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for ETP, as requested.")
        return all_params
    elif M_step_toggles_ETP == M_Step_Toggle_Value.CLOSED_FORM_TPM:
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
            model,
            event_end_times,
            use_continuous_states,
            verbose,
        )
    else:
        raise ValueError("I do not know what to do with ETP for the M-step.")

    all_params = AllParameters_JAX(
        all_params.STP, ETP_new, all_params.CSP, all_params.EP, all_params.IP
    )

    return all_params


def run_M_step_for_STP_in_closed_form(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    event_end_times: NumpyArray1D,
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
    exp_Pi = compute_closed_form_M_step(VES_summary, event_end_times=event_end_times)
    Pi_new = jnp.asarray(np.log(exp_Pi))
    return SystemTransitionParameters_JAX(STP.Gammas, STP.Upsilon, Pi_new)


def run_M_step_for_STP_via_gradient_descent(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    event_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D],
    verbose: bool = True,
) -> SystemTransitionParameters_JAX:
    ### Do gradient descent on unconstrained parameters.
    STP_WUC = STP_with_unconstrained_tpms_from_ordinary_STP(STP)
    cost_function_STP = functools.partial(
        compute_cost_for_system_transition_parameters_with_unconstrained_tpms_JAX,
        VES_summary=VES_summary,
        system_transition_prior=system_transition_prior,
        model=model,
        event_end_times=event_end_times,
        system_covariates=system_covariates,
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
    return STP_new


def run_M_step_for_STP(
    all_params: AllParameters_JAX,
    M_step_toggles_STP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    event_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D],
    verbose: bool = True,
) -> AllParameters_JAX:
    if M_step_toggles_STP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for STP, as requested.")
        return all_params
    elif M_step_toggles_STP == M_Step_Toggle_Value.CLOSED_FORM_TPM:
        STP_new = run_M_step_for_STP_in_closed_form(all_params.STP, VES_summary, event_end_times)
    elif M_step_toggles_STP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        STP_new = run_M_step_for_STP_via_gradient_descent(
            all_params.STP,
            VES_summary,
            system_transition_prior,
            iteration,
            num_M_step_iters,
            model,
            event_end_times,
            system_covariates,
            verbose,
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with system transition parameters."
        )

    all_params = AllParameters_JAX(
        STP_new, all_params.ETP, all_params.CSP, all_params.EP, all_params.IP
    )
    return all_params


def run_M_step_for_CSP(
    all_params: AllParameters_JAX,
    M_step_toggles_CSP: M_Step_Toggle_Value,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    continuous_states: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    event_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D],
) -> AllParameters_JAX:
    if M_step_toggles_CSP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for CSP, as requested.")
        return all_params
    elif M_step_toggles_CSP == M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN:
        CSP_new = run_M_step_for_CSP_in_closed_form__Gaussian_case(
            VEZ_summaries.expected_regimes,
            continuous_states,
            event_end_times,
            use_continuous_states,
        )
    elif M_step_toggles_CSP == M_Step_Toggle_Value.CLOSED_FORM_VON_MISES:
        CSP_new = run_M_step_for_CSP_in_closed_form__VonMises_case(
            VEZ_summaries.expected_regimes,
            continuous_states,
            all_params,
            event_end_times,
            use_continuous_states,
        )
    elif M_step_toggles_CSP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        warnings.warn(
            f"Learning the CSP parameters by gradient descent.  Performance seems to be worse than with the closed-form approach. "
            f"We should do the analogue to ETP, STP steps which used tensorflow.probability to convert simplex-valued parameters to unconstrained rep and back."
            f"that is, we should rely upon tensorflow.probability to convert covariance parameters to unconstrained representation and back."
        )

        CSP_WUC = CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian(
            all_params.CSP
        )

        cost_function_CSP = functools.partial(
            compute_cost_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX,
            continuous_states=continuous_states,
            VEZ_summaries=VEZ_summaries,
            model=model,
            event_end_times=event_end_times,
            use_continuous_states=use_continuous_states,
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

        CSP_new = ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(
            CSP_WUC_new
        )

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
    return all_params


def run_M_step_for_IP_in_closed_form__Gaussian_case(
    IP: InitializationParameters_Gaussian_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    continuous_states: JaxNumpyArray3D,
    event_end_times: NumpyArray1D,
) -> InitializationParameters_Gaussian_JAX:
    """
    Arguments:
        VEZ_expected_regimes: (T,J,K) array, the expected_regimes attribute from
            the HMM_Posterior_Summaries_JAX class.
    """

    init_times = get_initialization_times(event_end_times)

    EPSILON = 1e-3
    # These are set to be the values that minimize the cross-entropy, plus some noise
    expected_system_regime_init_probs = jnp.mean(VES_summary.expected_regimes[init_times], axis=0)
    pi_system = normalize_potentials_by_axis_JAX(
        expected_system_regime_init_probs + EPSILON, axis=0
    )

    expected_entity_regime_init_probs = jnp.mean(VEZ_summaries.expected_regimes[init_times], axis=0)
    pi_entities = normalize_potentials_by_axis_JAX(
        expected_entity_regime_init_probs + EPSILON, axis=1
    )

    J, K = jnp.shape(pi_entities)

    # set mu_0s to be equal to observed x's.
    empirical_continuous_state_init_means = jnp.mean(continuous_states[init_times], axis=0)  # (J,D)
    # TODO: We are assuming that the initial means are identical across the K regimes.  No reason for this.
    # Take the (expected-regime-)weighted mean above instead of the arithmetic mean.
    mu_0s = jnp.tile(empirical_continuous_state_init_means[:, None, :], (1, K, 1))

    empirical_continuous_state_init_vars = jnp.var(continuous_states[init_times], axis=0)  # (J,D)
    CUTOFF_NUM_OF_INIT_EXAMPLES_TO_USE_ML_ESTIMATE_OF_INIT_VARIANCES = 5
    if len(init_times) < CUTOFF_NUM_OF_INIT_EXAMPLES_TO_USE_ML_ESTIMATE_OF_INIT_VARIANCES:
        # if len(init_idxs)=1, keep Sigma_0s to tbe the same as initialized... not clear how to learn these
        # although could do a Bayesian update (of the prior) even with only one observation.
        Sigma_0s = IP.Sigma_0s

    else:
        D = np.shape(IP.Sigma_0s)[-1]
        Sigma_0s = np.zeros((J, K, D, D))
        # TODO: Vectorize this
        for j in range(J):
            cov_empirical_across_examples = empirical_continuous_state_init_vars[j] * np.eye(D)
            for k in range(K):
                # TODO: We are currently forcing the init covs to be diagonal.  There's no reason for this at all -
                # just implementational haste.  Go back and do it correctly
                #
                # TODO: We are assuming that the initial covs are identical across the K regimes.  No reason for this.
                # Take the (expected-regime-)weighted mean above, instead of the arithmetic mean.
                Sigma_0s[j, k] = cov_empirical_across_examples
    return InitializationParameters_Gaussian_JAX(pi_system, pi_entities, mu_0s, jnp.array(Sigma_0s))


# TODO: Combine with `run_M_step_for_IP_in_closed_form__Gaussian_case`.
# It's the same logic, up to having a loc and scale parameter, and dealing
# with the difference in dimensionality
def run_M_step_for_IP_in_closed_form__VonMises_case(
    IP: InitializationParameters_VonMises_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    group_angles: Union[JaxNumpyArray2D, JaxNumpyArray3D],
    event_end_times: NumpyArray1D,
) -> InitializationParameters_VonMises_JAX:
    if not only_one_event(event_end_times, T=len(group_angles)):
        raise NotImplementedError(
            f"Haven't yet implemented M-step for init params in von mises case when there are "
            f"multiple events."
        )
    EPSILON = 1e-3
    # These are set to be the values that minimize the cross-entropy, plus some noise
    pi_system = normalize_potentials_by_axis_JAX(VES_summary.expected_regimes[0] + EPSILON, axis=0)
    pi_entities = normalize_potentials_by_axis_JAX(
        VEZ_summaries.expected_regimes[0] + EPSILON, axis=1
    )

    K = jnp.shape(pi_entities)[1]

    # set mu_0s to be equal to observed x's.
    # TODO: Pick a consistent shape for group_angles throughout a circle package; (T,J,1) or (T,J)
    group_angles = jnp.squeeze(group_angles)
    locs = jnp.tile(group_angles[0][:, None], (1, K))
    # keep Sigma_0s to tbe the same as initialized... not clear how to learn these
    return InitializationParameters_VonMises_JAX(pi_system, pi_entities, locs, IP.kappas)


def run_M_step_for_IP(
    IP: InitializationParameters_JAX,
    M_step_toggles_IP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    continuous_states: NumpyArray3D,
    event_end_times: NumpyArray1D,
) -> InitializationParameters_JAX:
    if M_step_toggles_IP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for IP, as requested.")
        return IP
    elif M_step_toggles_IP == M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN:
        IP_new = run_M_step_for_IP_in_closed_form__Gaussian_case(
            IP,
            VEZ_summaries,
            VES_summary,
            continuous_states,
            event_end_times,
        )
    elif M_step_toggles_IP == M_Step_Toggle_Value.CLOSED_FORM_VON_MISES:
        IP_new = run_M_step_for_IP_in_closed_form__VonMises_case(
            IP,
            VEZ_summaries,
            VES_summary,
            continuous_states,
            event_end_times,
        )
    elif M_step_toggles_IP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        raise ValueError(
            f"Learning the IP parameters by gradient descent is not currently supported.  Try closed-form"
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with continuous state parameters."
        )

    return IP_new
