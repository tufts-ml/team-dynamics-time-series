from typing import Callable, List, Tuple

import numpy as np

from dynagroup.hmm_posterior import HMM_Posterior_Summary_NUMPY
from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.params import AllParameters, dims_from_params
from dynagroup.sampler import Sample
from dynagroup.types import NumpyArray2D, NumpyArray4D
from dynagroup.vi.vi_numpy.E_step_numpy import run_VES_step_NUMPY, run_VEZ_step_NUMPY


"""
We do not use the numpy version of CAVI in practice, but the jax version instead
so that we can use autodiff for gradient descent.

However, we maintain this legacy function here for the purposes of cross-checking and
possibly unit testing.
"""


###
# Helpers for converting between types
###


def variationally_expected_joints_for_entity_regimes_from_VEZ_summaries(
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
) -> NumpyArray4D:
    """
    A helper function which can be used to convert the return value of run_VEZ_step
    into inputs needed for run_VES_step.

    Returns:
        np.array of shape (T-1, J, K, K)
    """
    J = len(VEZ_summaries)
    expected_joints_by_entity = [vs.expected_joints for vs in VEZ_summaries]
    T_minus_1, K, _ = np.shape(expected_joints_by_entity[0])
    expected_entity_regime_joints = np.zeros((T_minus_1, J, K, K))
    for j in range(J):
        expected_entity_regime_joints[:, j] = expected_joints_by_entity[j]
    return expected_entity_regime_joints


def variationally_expected_initial_entity_regimes_from_VEZ_summaries(
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
) -> NumpyArray2D:
    """
    A helper function which can be used to convert the return value of run_VEZ_step
    into inputs needed for run_VES_step.

    Returns:
        np.array of shape (J, K)
    """
    J = len(VEZ_summaries)
    expected_regimes_by_entity = [vs.expected_regimes for vs in VEZ_summaries]
    T, K = np.shape(expected_regimes_by_entity[0])
    expected_initial_entity_regimes = np.zeros((J, K))
    for j in range(J):
        expected_initial_entity_regimes[j, :] = expected_regimes_by_entity[j][0]
    return expected_initial_entity_regimes


###
# Helper functions (for top-level VI function)
###


def generate_expected_joints_uniformly(T: int, J: int, K: int):
    """
    Returns matrix of shape (T-1,J,K,K) which is uniform across all pairwise options
    (z^j_{t-1}=k, z^j_t=k')
    """
    expected_joints = np.zeros((T - 1, J, K, K))
    block = np.ones((K, K)) / (K**2)
    for t in range(T - 1):
        for j in range(J):
            expected_joints[t, j, :, :] = block
    return expected_joints


def generate_expected_state_regimes_uniformly(T: int, L: int):
    """
    Returns matrix of shape (T-1,L) which is uniform across all options
    (s_t=l)
    """
    expected_regimes = np.zeros((T, L))
    block = np.ones(L) / L
    for t in range(T):
        expected_regimes[t, :] = block
    return expected_regimes


###
# MAIN
###


def run_CAVI_with_numpy(
    sample: Sample,
    n_iterations: int,
    all_params: AllParameters,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
    run_M_step: bool = True,
    verbose: bool = True,
) -> Tuple[HMM_Posterior_Summary_NUMPY, List[HMM_Posterior_Summary_NUMPY], AllParameters]:
    """
    Arguments:
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    ###
    # UP-FRONT INFO
    ###

    # TODO: need to add M-step so that we can LEARN params
    DIMS = dims_from_params(all_params)
    J, K, L = DIMS.J, DIMS.K, DIMS.L
    # Not uses: D, N = DIMS.D, DIMS.N
    T = len(sample.s)

    ###
    # INITIALIZATION
    ###

    ### Uniform initialization of q(Z) - should be bad!

    # TODO: Need to smart initialize the expected joints; see Linderman
    variationally_expected_joints_for_entity_regimes = generate_expected_joints_uniformly(T, J, K)
    variationally_expected_initial_entity_regimes = np.ones((J, K)) / K
    init_dist_over_system_regimes = np.ones(L) / L
    init_dists_over_entity_regimes = np.ones((J, K)) / K

    ###
    # CAVI
    ###
    for i in range(n_iterations):
        print(f"\n ---- Now running iteration {i+1} ----")

        # TODO: how to update init dists?!?! Need parameter updates I guess?

        VES_summary = run_VES_step_NUMPY(
            all_params.STP,
            all_params.ETP,
            sample.xs,
            variationally_expected_joints_for_entity_regimes,
            variationally_expected_initial_entity_regimes,
            init_dist_over_system_regimes,
            init_dists_over_entity_regimes,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
        )

        most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
        pct_correct_system = compute_regime_labeling_accuracy(most_likely_system_regimes, sample.s)

        if verbose:
            print(
                f"\nVES step's log normalizer for entity regimes when we use uniform inits for q(Z): {VES_summary.log_normalizer:.02f}"
            )
            print(
                f"Percent correct classifications for system segmentations {pct_correct_system:.02f}"
            )

        VEZ_summaries = run_VEZ_step_NUMPY(
            all_params.CSP,
            all_params.ETP,
            all_params.IP,
            sample.xs,
            VES_summary.expected_regimes,
            init_dists_over_entity_regimes,
            transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
        )

        pct_corrects_entities = np.empty(J)
        for j in range(J):
            most_likely_system_regimes = np.argmax(VEZ_summaries[j].expected_regimes, axis=1)
            pct_corrects_entities[j] = compute_regime_labeling_accuracy(
                most_likely_system_regimes, sample.zs[:, j]
            )

        log_normalizers = np.array([VEZ_summaries[j].log_normalizer for j in range(J)])

        if verbose:
            print(
                f"\nVEZ step's log normalizer by entities for continuous state emissions when we use VES inits for q(S): {log_normalizers}"
            )
            print(
                f"Percent correct classifications for entity-level segmentations {pct_corrects_entities}"
            )

        # setup for next VES step
        variationally_expected_joints_for_entity_regimes = (
            variationally_expected_joints_for_entity_regimes_from_VEZ_summaries(VEZ_summaries)
        )
        variationally_expected_initial_entity_regimes = (
            variationally_expected_initial_entity_regimes_from_VEZ_summaries(
                VEZ_summaries,
            )
        )

        if run_M_step:
            raise NotImplementedError

    return VES_summary, VEZ_summaries, all_params
