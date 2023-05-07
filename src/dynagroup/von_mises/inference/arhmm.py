from typing import List, Tuple

import numpy as np
from scipy.stats import vonmises


np.set_printoptions(precision=3, suppress=True)

import ruptures as rpt

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summary_NUMPY,
    compute_closed_form_M_step,
    compute_hmm_posterior_summary_NUMPY,
)
from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.util import make_fixed_sticky_tpm
from dynagroup.von_mises.inference.ar import (
    VonMisesModelType,
    VonMisesParams,
    estimate_von_mises_params,
)


###
# Helpers
###


def _get_indices_for_largest_values_of_distance_matrix(dist_matrix):
    """
    Find indices of the largest C values of a CxC distance matrix.
    """
    C = len(dist_matrix)

    # Flatten the distance matrix into a 1D array
    flat_dist = dist_matrix.flatten()
    # Get the indices that would sort the flattened array in descending order
    sorted_indices = np.argsort(flat_dist)[::-1]
    # Get the top K indices from the sorted indices
    top_K_indices = sorted_indices[:C]
    # Convert the flattened indices back into 2D indices
    row_indices = top_K_indices // dist_matrix.shape[0]
    col_indices = top_K_indices % dist_matrix.shape[1]
    # Return the 2D indices as a tuple
    return (row_indices, col_indices)


###
# Main
###


def compute_log_emissions(angles: NumpyArray1D, params_by_regime: List[VonMisesParams]):
    T, K = len(angles), len(params_by_regime)
    # TODO: What should the initial emission be?!!?
    log_emissions = np.zeros((T, K))
    for k in range(K):
        params = params_by_regime[k]

        # TODO: How to handle initial observations
        expectations = params.drift + params.ar_coef * angles[:-1]
        log_emissions[1:, k] = vonmises.logpdf(angles[1:], params.kappa, loc=expectations)
    return log_emissions


def estimate_von_mises_autoregression_params_separately_over_changepoint_segments(
    angles: NumpyArray1D, changepoint_penalty: float = 10.0
) -> List[VonMisesParams]:
    algo = rpt.Pelt(model="rbf").fit(angles)
    changepoints = algo.predict(pen=changepoint_penalty)

    markers = np.insert(changepoints, 0, 0)

    # and estimate ARHMM to initialize params
    C = len(changepoints)
    emissions_params_by_changepoint_clip = [None] * C
    for c in range(C):
        start, end = markers[c], markers[c + 1]
        angles_clip = angles[start:end]
        params = estimate_von_mises_params(
            angles_clip,
            VonMisesModelType.AUTOREGRESSION,
            suppress_warnings=True,
        )
        emissions_params_by_changepoint_clip[c] = params

    return emissions_params_by_changepoint_clip


def smart_initialize_emission_params_by_regime_for_von_mises_arhmm(
    angles: NumpyArray1D,
    num_regimes: int,
    changepoint_penalty: float = 10.0,
    verbose: float = True,
) -> List[VonMisesParams]:
    params_by_changepoint_segment = (
        estimate_von_mises_autoregression_params_separately_over_changepoint_segments(
            angles, changepoint_penalty
        )
    )
    num_changepoint_segments = len(params_by_changepoint_segment)
    if verbose:
        print(
            f"We found {num_changepoint_segments} changepoint segments for initializing emission params of {num_regimes} regimes."
        )

    # TODO: Perhaps automate this manual work with a try/catch? If we don't find enough changepoints, then lower the number
    # of changepoints? In fact, should we TRY to get the # changepoint and # regimes to match? (I think no, because I think
    # we expect regimes to recur..

    if num_changepoint_segments < num_regimes:
        raise ValueError(
            f"Problem initializing the emissions for a von mises AR-HMM. "
            f"We initialize via changepoints on the angles.  We found {num_changepoint_segments} changepoint segments, but "
            f"want to have {num_regimes} regimes.  Try reducing the changepoint penalty, {changepoint_penalty:.02f}."
        )

    ### TODO: What strategies should we use for moving from changepoints to regimes?
    #   1. Maybe take the two most different parameter collections across all changepoints and use that for init?!
    #   2. Maybe take consecutive ones?
    #   3. Maybe take random ones?

    param_values = [None] * num_changepoint_segments
    for c in range(num_changepoint_segments):
        param_values[c] = np.array(list(params_by_changepoint_segment[c].__dict__.values()))

    distance_matrix = np.zeros((num_changepoint_segments, num_changepoint_segments))
    for c in range(num_changepoint_segments):
        for c_prime in range(num_changepoint_segments):
            if c_prime > c:
                distance_matrix[c, c_prime] = np.linalg.norm(
                    param_values[c] - param_values[c_prime]
                )

    # find the index of the maximum element in the distance matrix
    sorted_list_of_indices = _get_indices_for_largest_values_of_distance_matrix(distance_matrix)
    changepoint_segments_to_use = []
    for segment_pair in sorted_list_of_indices:
        for i in [0, 1]:
            if segment_pair[i] not in changepoint_segments_to_use:
                changepoint_segments_to_use.append(segment_pair[i])

            if len(changepoint_segments_to_use) < num_regimes:
                break

    params_by_regime = [None] * num_regimes
    for k in range(num_regimes):
        c = changepoint_segments_to_use[k]
        params_by_regime[k] = params_by_changepoint_segment[c]

    return params_by_regime


def run_EM_for_von_mises_arhmm(
    angles: NumpyArray1D,
    K: int,
    self_transition_prob_init: float,
    num_EM_iterations: float,
    verbose: bool = True,
) -> Tuple[HMM_Posterior_Summary_NUMPY, List[VonMisesParams], NumpyArray2D]:
    """
    Remarks
        1) Not currently learning the initial regime distribution. We are presetting the initial
        autoregressive emissions to 1 for each regime, and therefore letting the choice of regime
        be determined by whatever happens in the following observations.
    """
    ###
    # Initialization
    ###

    T = len(angles)

    emissions_params_by_regime_learned = (
        smart_initialize_emission_params_by_regime_for_von_mises_arhmm(
            angles,
            K,
            changepoint_penalty=10.0,
        )
    )
    log_emissions = compute_log_emissions(angles, emissions_params_by_regime_learned)
    init_dist_over_regimes = np.array([1] * K) / K  # or, more likely to be focused at start?
    tpm = make_fixed_sticky_tpm(self_transition_prob_init, num_states=K)
    transitions = np.tile(tpm[None, :, :], (T - 1, 1, 1))
    log_transitions = np.log(transitions)

    ###
    # Inference
    ###

    for i in range(num_EM_iterations):
        print(f"\n----Now running EM iteration {i}")

        ###
        # E-step
        ###
        posterior_summary = compute_hmm_posterior_summary_NUMPY(
            log_transitions, log_emissions, init_dist_over_regimes
        )

        ###
        # M-step
        ###

        ### Emissions M-step
        for k in range(K):
            if verbose:
                print(
                    f"\nFor regime {k}, the params before learning were {emissions_params_by_regime_learned[k]}."
                )
            emissions_params_by_regime_learned[k] = estimate_von_mises_params(
                angles,
                VonMisesModelType.AUTOREGRESSION,
                sample_weights=posterior_summary.expected_regimes[:, k],
                suppress_warnings=True,
            )
            if verbose:
                print(
                    f"For regime {k}, the params after learning were {emissions_params_by_regime_learned[k]}."
                )

        ### Transitions M-step
        tpm = compute_closed_form_M_step(posterior_summary)
        print(f"new tpm: {tpm}")
        transitions = np.tile(tpm[None, :, :], (T - 1, 1, 1))
        log_transitions = np.log(transitions)

        ### Initialization M-step
        # Skip for now
    return posterior_summary, emissions_params_by_regime_learned, transitions
