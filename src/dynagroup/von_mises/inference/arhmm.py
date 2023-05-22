import warnings
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
    enforce_von_mises_params_to_have_kappa_at_least_unity,
    estimate_von_mises_params,
)
from dynagroup.von_mises.patches import try_changepoint_initialization_n_times


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
    angles: NumpyArray1D,
    changepoint_penalty: float = 10.0,
    min_segment_size: int = 0,
    verbose=True,
) -> List[VonMisesParams]:
    """
    Arguments:
        min_segment_size: The minimum size for a changeppoint segment to be used for computing
            von mises AR parameters.
    """
    algo = rpt.Pelt(model="rbf").fit(angles)
    changepoints = algo.predict(pen=changepoint_penalty)

    markers = np.insert(changepoints, 0, 0)

    # and estimate ARHMM to initialize params
    C = len(changepoints)
    emissions_params_for_changepoint_clips_which_are_sufficiently_long_and_have_ar_coef_unit_interval = (
        []
    )
    for c in range(C):
        if verbose:
            print(f"Now estimating von mises parameters on segment found by changepoint detector.")
        start, end = markers[c], markers[c + 1]
        size = end - start
        if size >= min_segment_size:
            angles_clip = angles[start:end]
            params = estimate_von_mises_params(
                angles_clip,
                VonMisesModelType.AUTOREGRESSION,
                suppress_warnings=True,
            )
        if np.abs(params.ar_coef) <= 1.0:
            emissions_params_for_changepoint_clips_which_are_sufficiently_long_and_have_ar_coef_unit_interval.append(
                params
            )
    return emissions_params_for_changepoint_clips_which_are_sufficiently_long_and_have_ar_coef_unit_interval


def smart_initialize_emission_params_by_regime_for_von_mises_arhmm(
    angles: NumpyArray1D,
    num_regimes: int,
    changepoint_penalty: float = 10.0,
    min_segment_size: int = 10,
    verbose: float = True,
) -> List[VonMisesParams]:
    # get params by changepoint segment
    params_by_changepoint_segment = (
        estimate_von_mises_autoregression_params_separately_over_changepoint_segments(
            angles, changepoint_penalty, min_segment_size
        )
    )
    # enforce kappas at least unity
    params_by_changepoint_segment = [
        enforce_von_mises_params_to_have_kappa_at_least_unity(vmp)
        for vmp in params_by_changepoint_segment
    ]

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
    changepoint_pairs = np.array(sorted_list_of_indices).T  # num changepoints x 2

    for changepoint_pair in changepoint_pairs:
        for i in [0, 1]:
            if changepoint_pair[i] not in changepoint_segments_to_use:
                changepoint_segments_to_use.append(changepoint_pair[i])

            if len(changepoint_segments_to_use) > num_regimes:
                break

    if len(changepoint_segments_to_use) < num_regimes:
        # My _get_indices_for_largest_values_of_distance_matrix sometimes doesn't return enough pairs.
        # If it doesn't, just use the first `num_regimes` regimes, as a hack for now.
        # TODO: Improve all this later.
        warnings.warn(
            "Resulting to hack for which changepoint segments to use. Fix implementation."
        )
        changepoint_segments_to_use = [i for i in range(num_regimes)]

    params_by_regime = [None] * num_regimes
    for k in range(num_regimes):
        c = changepoint_segments_to_use[k]
        params_by_regime[k] = params_by_changepoint_segment[c]

    return params_by_regime


from joblib import Parallel, delayed


def run_EM_for_von_mises_arhmm(
    angles: NumpyArray1D,
    K: int,
    num_EM_iterations: float,
    init_self_transition_prob: float,
    init_changepoint_penalty: float = 10.0,
    init_min_segment_size: int = 10,
    fix_ar_kappa_to_unity_rather_than_estimate: bool = False,
    verbose: bool = True,
    parallelize_the_M_step: bool = False,
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

    emissions_params_by_regime_learned = try_changepoint_initialization_n_times(
        smart_initialize_emission_params_by_regime_for_von_mises_arhmm, n=5
    )(
        angles,
        K,
        changepoint_penalty=init_changepoint_penalty,
        min_segment_size=init_min_segment_size,
    )
    log_emissions = compute_log_emissions(angles, emissions_params_by_regime_learned)
    init_dist_over_regimes = np.array([1] * K) / K  # or, more likely to be focused at start?
    tpm = make_fixed_sticky_tpm(init_self_transition_prob, num_states=K)
    transitions = np.tile(tpm[None, :, :], (T - 1, 1, 1))
    log_transitions = np.log(transitions)

    ###
    # Inference
    ###

    for i in range(num_EM_iterations):
        print(f"\n----Now running EM iteration {i+1}/{num_EM_iterations}.")

        ###
        # E-step
        ###
        posterior_summary = compute_hmm_posterior_summary_NUMPY(
            log_transitions, log_emissions, init_dist_over_regimes
        )
        if np.isnan(posterior_summary.log_normalizer):
            breakpoint()

        ###
        # M-step
        ###
        ### Emissions M-step (in parallel)
        def _estimate_von_mises_params_on_one_regime(k):
            return estimate_von_mises_params(
                angles,
                VonMisesModelType.AUTOREGRESSION,
                emissions_params_by_regime_learned[k].ar_coef,
                emissions_params_by_regime_learned[k].drift,
                sample_weights=posterior_summary.expected_regimes[:, k],
                suppress_warnings=False,
                fix_ar_kappa_to_unity_rather_than_estimate=fix_ar_kappa_to_unity_rather_than_estimate,
            )

        if verbose:
            for k in range(K):
                print(
                    f"\nFor regime {k}, the params before learning were {emissions_params_by_regime_learned[k]}."
                )

        if parallelize_the_M_step:
            # Parallelize the loop using joblib
            results = Parallel(n_jobs=-1)(
                delayed(_estimate_von_mises_params_on_one_regime)(k) for k in range(K)
            )
            for k, result in enumerate(results):
                emissions_params_by_regime_learned[k] = result
        else:
            for k in range(K):
                emissions_params_by_regime_learned[k] = _estimate_von_mises_params_on_one_regime(k)

        if verbose:
            for k in range(K):
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
