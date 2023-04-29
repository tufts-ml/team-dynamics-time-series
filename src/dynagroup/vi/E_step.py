import jax.numpy as jnp
import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    compute_hmm_posterior_summaries_JAX,
    compute_hmm_posterior_summary_JAX,
)
from dynagroup.model import Model
from dynagroup.params import (
    ContinuousStateParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D, JaxNumpyArray4D


###
# VES Step
###


def compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    variationally_expected_joints_for_entity_regimes: JaxNumpyArray4D,
    continuous_states_JAX: JaxNumpyArray3D,
    model: Model,
) -> JaxNumpyArray3D:
    """
    Compute expected log transition probability matrices, where the expectations are taken with
    respect to the variational joint distribution over the entity-level regimes.

    Arguments:
         variationally_expected_joints_for_entity_regimes : np.array of size (T-1,J,K,K) whose (t,j,k,k')-th element gives
            the probability distribution for entity j over all pairwise options
            (z^j_{t+1}=k', z^j_t=k).
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Returns:
        np.array of shape (T,J,L), whose (t,j,l)-th element gives the (unnormalized) (autoregressive)
            (categorical) emissions density at time t for entity j under system regime l

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        continuous_states_JAX[:-1],
        model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )
    expected_log_transition_matrices = jnp.einsum(
        "tjlkd, tjkd -> tjl",
        log_transition_matrices,
        variationally_expected_joints_for_entity_regimes,
    )
    # TODO: Write unit test that the einsum is performing as expected.
    # Just check it on the (tjl)-th element.
    # e.g.
    # t,j,l = 0,0,0
    # A= log_transition_matrices[t,j,l]
    # B= expected_joints[t,j]
    # assert expected_log_transition_matrices[t,j,l] = np.sum(A * B)
    return expected_log_transition_matrices


def run_VES_step_JAX(
    STP: SystemTransitionParameters_JAX,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    IP: InitializationParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
) -> HMM_Posterior_Summary_JAX:
    """
    Overview:
        Compute the VES step for a group dynamics model, and return results in the
        form a HMM_Posterior_Summary

    Details:
        The VES step for a group dynamics model can be
        seen as the posterior of a HMM with adjusted parameters (due to taking expectations
        over the other random variables.)  First, we construct the initial distribution
        (actually a function arg), the transitions, and the emissions.
        Then, we feed this information into a generic forward-backward algo,
        namely the `hmm_expected_states` function from Linderman's ssm repo.
        Finally, we return the HMM_Posterior_Summary.

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D.  These can be sampled for the full model (where x's are not observed)
            or observed (if we have a switching
            AR-HMM i.e. a meta-switching recurrent AR model)
        variationally_expected_joints_for_entity_regimes:
            np.array of size (T-1,J,K,K) whose (t,j,k,k')-th element gives
            the probability distribution for entity j over all pairwise options
            (z^j_{t+1}=k', z^j_t=k).

            May come from a list of J HMM_Posterior_Summary instances created by the VEZ step.
        variationally_expected_initial_entity_regimes: np.array of size (J,K)
            The j-th row must live on the simplex for all j=1,...,J.
            May come from a list of J HMM_Posterior_Summary instances created by the VEZ step.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.


    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    T = np.shape(continuous_states)[0]
    L = len(IP.pi_system)

    # `transitions` is (T-1) x L x L
    log_transitions = model.compute_log_system_transition_probability_matrices_JAX(STP, T - 1)

    # `inital_log_emission` has shape (L,)
    initial_log_emission_for_each_system_regime = jnp.sum(
        VEZ_summaries.expected_regimes[0] * np.log(IP.pi_entities)
    )

    initial_log_emission = jnp.repeat(initial_log_emission_for_each_system_regime, L)

    # `log_emissions_for_each_entity_after_initial_time` is (T-1) x J x L.. We need to collapse the J; emissions are independent over J.
    log_emissions_for_each_entity_after_initial_time = (
        compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes_JAX(
            ETP,
            VEZ_summaries.expected_joints,
            continuous_states,
            model,
        )
    )

    # `log_emissions_after_initial_time` is (T-1) x L.
    log_emissions_after_initial_time = jnp.sum(
        log_emissions_for_each_entity_after_initial_time, axis=1
    )

    # 'log_emissions' is TxL
    log_emissions = jnp.vstack((initial_log_emission, log_emissions_after_initial_time))

    return compute_hmm_posterior_summary_JAX(
        log_transitions,
        log_emissions,
        IP.pi_system,
    )


###
# VEZ Step
###


def compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    variationally_expected_system_regimes: JaxNumpyArray2D,
    continuous_states: JaxNumpyArray3D,
    model: Model,
) -> JaxNumpyArray3D:
    """
    Compute expected log transition probability matrices, where the expectations are taken with
    respect to the variational distribution over the system-level regimes.

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D
        variationally_expected_system_regimes : np.array of size (T,L) whose (t,l)-th element gives
            the probability distribution of system regime l, q(s_t=l).

            May come from a HMM_Posterior_Summary instance created by the VES step.

    Returns:
        np.array of size (T-1,J,K,K) whose (t,j,k,k')-th element gives
            the probability distribution for entity j over all pairwise options
            (z^j_{t+1}=k', z^j_t=k).

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        continuous_states[:-1],
        model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    )

    # TODO: This isn't working yet; needs initialization
    expected_log_transition_matrices = jnp.einsum(
        "tjlkd, tl -> tjkd",
        log_transition_matrices,
        variationally_expected_system_regimes[1:],
    )

    # TODO: Write unit test that the einsum is performing as expected.
    # Just check it on the (tjl)-th element.
    # e.g.
    # t,j = 1,0  # note t needs to start at t=1 or later.
    # A= log_transition_matrices[t,j]
    # B= variationally_expected_system_regimes[t]
    # assert np.allclose(expected_log_transition_matrices[t,j], (A.T @ B).T)
    return expected_log_transition_matrices


def run_VEZ_step_JAX(
    CSP: ContinuousStateParameters_JAX,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    IP: InitializationParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    variationally_expected_system_regimes: JaxNumpyArray2D,
    model: Model,
) -> HMM_Posterior_Summaries_JAX:
    """
    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D.  These can be sampled for the full model (where x's are not observed)
            or observed (if we have a switching
            AR-HMM i.e. a meta-switching recurrent AR model)
        variationally_expected_system_regimes : np.array of size (T,L) whose (t,l)-th element gives
            the probability distribution of system regime l, q(s_t=l).

            May come from a HMM_Posterior_Summary instance created by the VES step.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    T, J, _ = jnp.shape(continuous_states)

    # `transitions` has shape (T-1,J,K,K)
    log_transitions = (
        compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes_JAX(
            ETP,
            variationally_expected_system_regimes,
            continuous_states,
            model,
        )
    )
    # REMARK 1: Transitions are unnormalized! However, it shouldn't matter.  The emissions are not normalized
    # either, due to the expectations.  Indeed, the updates in the notes are all written up to a constant
    # of proportionality.  I did a test and using unnormalized transitions changes the log_normalizer
    # but not anything else.   So I think it's okay -- assuming that we're not using the log normalizer
    # for anything critical.  Perhaps check this more carefully, though.
    #
    # REMARK 2: When I initialized q(z) to ground truth, transitions(t,j,k,:) sum to (close to 1).
    # But when I initialized q(z) uniformly, I got some sums very far from 1.  There may be an
    # overflow/underflow issue.

    # TODO: Below this is where I left off.  It's copy pasta'd!

    # log_state_emissions has shape (T,J,K)
    log_state_emissions = model.compute_log_continuous_state_emissions_JAX(
        CSP,
        IP,
        continuous_states,
        model.compute_log_continuous_state_emissions_at_initial_timestep_JAX,
        model.compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    )
    return compute_hmm_posterior_summaries_JAX(log_transitions, log_state_emissions, IP.pi_entities)
