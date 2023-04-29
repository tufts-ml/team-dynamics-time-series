from typing import Callable, List

import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summary_NUMPY,
    compute_hmm_posterior_summaries_NUMPY,
    compute_hmm_posterior_summary_NUMPY,
)
from dynagroup.model2a.figure_8.model_factors import (
    compute_log_continuous_state_emissions,
    compute_log_entity_transition_probability_matrices,
    compute_log_system_transition_probability_matrices,
)
from dynagroup.params import (
    ContinuousStateParameters,
    EntityTransitionParameters_MetaSwitch,
    InitializationParameters,
    SystemTransitionParameters,
)
from dynagroup.types import NumpyArray1D, NumpyArray2D, NumpyArray3D, NumpyArray4D


###
# VES Step
###


def compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes(
    ETP: EntityTransitionParameters_MetaSwitch,
    variationally_expected_joints_for_entity_regimes: NumpyArray4D,
    continuous_states: NumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> NumpyArray3D:
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
    log_transition_matrices = compute_log_entity_transition_probability_matrices(
        ETP,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    )
    expected_log_transition_matrices = np.einsum(
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


def run_VES_step_NUMPY(
    STP: SystemTransitionParameters,
    ETP: EntityTransitionParameters_MetaSwitch,
    continuous_states: NumpyArray3D,
    variationally_expected_joints_for_entity_regimes: NumpyArray4D,
    variationally_expected_initial_entity_regimes: NumpyArray2D,
    init_dist_over_system_regimes: NumpyArray1D,
    init_dists_over_entity_regimes: NumpyArray2D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> HMM_Posterior_Summary_NUMPY:
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
        init_dist_over_system_regimes: np.array of size (L,)
            Must live on simplex.
        init_dists_over_entity_regimes: np.array of size (J,K)
            The j-th row must live on the simplex for all j=1,...,J.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.


    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    T = np.shape(continuous_states)[0]
    L = len(init_dist_over_system_regimes)

    # `transitions` is (T-1) x L x L
    log_transitions = compute_log_system_transition_probability_matrices(STP, T)

    # `log_emissions_for_each_entity_after_initial_time` is T x J x L.. We need to collapse the J; emissions are independent over J.
    log_emissions_for_each_entity_after_initial_time = (
        compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes(
            ETP,
            variationally_expected_joints_for_entity_regimes,
            continuous_states,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
        )
    )
    # 'log_emissions' is TxL
    log_emissions = np.zeros((T, L))
    initial_log_emission = np.sum(
        variationally_expected_initial_entity_regimes * np.log(init_dists_over_entity_regimes)
    )
    log_emissions[0, :] = np.repeat(initial_log_emission, L)
    log_emissions[1:, :] = np.sum(log_emissions_for_each_entity_after_initial_time, axis=1)
    return compute_hmm_posterior_summary_NUMPY(
        log_transitions, log_emissions, init_dist_over_system_regimes
    )


###
# VEZ Step
###


def compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes(
    ETP: EntityTransitionParameters_MetaSwitch,
    variationally_expected_system_regimes: NumpyArray2D,
    continuous_states: NumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> NumpyArray3D:
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
    log_transition_matrices = compute_log_entity_transition_probability_matrices(
        ETP,
        continuous_states,
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    )

    # TODO: This isn't working yet; needs initialization
    expected_log_transition_matrices = np.einsum(
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


def run_VEZ_step_NUMPY(
    CSP: ContinuousStateParameters,
    ETP: EntityTransitionParameters_MetaSwitch,
    IP: InitializationParameters,
    continuous_states: NumpyArray3D,
    variationally_expected_system_regimes: NumpyArray2D,
    init_dists_over_entity_regimes: NumpyArray2D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> List[HMM_Posterior_Summary_NUMPY]:
    """
    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D.  These can be sampled for the full model (where x's are not observed)
            or observed (if we have a switching
            AR-HMM i.e. a meta-switching recurrent AR model)
        variationally_expected_system_regimes : np.array of size (T,L) whose (t,l)-th element gives
            the probability distribution of system regime l, q(s_t=l).

            May come from a HMM_Posterior_Summary instance created by the VES step.
        init_dists_over_entity_regimes: np.array of size (J,K)
            The j-th row must live on the simplex for all j=1,...,J.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    T, J, _ = np.shape(continuous_states)

    # `transitions` has shape (T-1,J,K,K)
    log_transitions = (
        compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes(
            ETP,
            variationally_expected_system_regimes,
            continuous_states,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
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
    log_state_emissions = compute_log_continuous_state_emissions(CSP, IP, continuous_states)

    return compute_hmm_posterior_summaries_NUMPY(
        log_transitions, log_state_emissions, init_dists_over_entity_regimes
    )
