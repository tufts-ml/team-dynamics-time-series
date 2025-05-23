import copy
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import zscore

from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import (
    Event,
    player_coords_from_moments,
)
from dynagroup.types import NumpyArray1D
from dynagroup.util import construct_a_new_list_after_removing_multiple_items


###
# Structs
###
@dataclass
class Example_Boundary_Constants:
    """
    Stores information useful for determining that an example boundary exists.
    Given a nominal sampling rate (in Hz), we compute the time we expect (in ms) between samples.
    This is stored as `expected_time_in_ms_between_samples`.

    However, it is normal for samples to have a bit of noise around that value.
    So in practice, we conclude that two timesteps belong to the same example if

        observed time (in ms) between samples \in [lower threshold, upper threhold]
    """

    expected_time_in_ms_between_samples: float
    wall_clock_diff_lower_threshold: float
    wall_clock_diff_upper_threshold: float


def example_boundary_constants_from_sampling_rate_Hz(
    sampling_rate_Hz: float,
) -> Example_Boundary_Constants:
    expected_time_in_ms_between_samples = 1000 / sampling_rate_Hz
    wall_clock_diff_lower_threshold = expected_time_in_ms_between_samples * 0.8
    wall_clock_diff_upper_threshold = expected_time_in_ms_between_samples * 1.2
    return Example_Boundary_Constants(
        expected_time_in_ms_between_samples,
        wall_clock_diff_lower_threshold,
        wall_clock_diff_upper_threshold,
    )


###
# Main Functions
###


def clean_events_of_moments_with_too_small_intervals(
    events: List[Event],
    sampling_rate_Hz: float,
    verbose: bool = True,
) -> List[Event]:
    EBC = example_boundary_constants_from_sampling_rate_Hz(sampling_rate_Hz)

    events_cleaned = copy.deepcopy(events)

    prev_wall_clock = -np.inf

    for event_idx, event in enumerate(events):
        moment_idxs_to_remove = []

        # Get indices of new events, as defiend by wall clock diffs that are too big.
        for moment_idx, moment in enumerate(event.moments):
            curr_wall_clock = moment.wall_clock
            wall_clock_diff = curr_wall_clock - prev_wall_clock

            if wall_clock_diff < EBC.wall_clock_diff_lower_threshold:
                if verbose:
                    print(
                        f"For event idx {event_idx}, flagging for removal a moment whose wall clock diff was {wall_clock_diff:.02f}"
                    )
                moment_idxs_to_remove.append(moment_idx)

            prev_wall_clock = curr_wall_clock

        # Remove moments whose wall clock diffs that are too big.
        new_moments = construct_a_new_list_after_removing_multiple_items(event.moments, moment_idxs_to_remove)
        events_cleaned[event_idx].moments = new_moments

    return events_cleaned


def get_example_stop_idxs(
    events: List[Event],
    sampling_rate_Hz: float,
    verbose: bool = True,
) -> List[int]:
    example_stop_idxs_from__large_time_deltas = get_example_stop_idxs__based_on_large_time_deltas(
        events, sampling_rate_Hz, verbose
    )
    example_stop_idxs_from__anomalous_step_sizes = get_example_stop_idxs__based_on_anomalous_step_sizes(events)
    return sorted(set(example_stop_idxs_from__large_time_deltas).union(example_stop_idxs_from__anomalous_step_sizes))


def get_example_stop_idxs__based_on_large_time_deltas(
    events: List[Event],
    sampling_rate_Hz: float,
    verbose: bool = True,
) -> List[int]:
    EBC = example_boundary_constants_from_sampling_rate_Hz(sampling_rate_Hz)

    example_end_times = []
    T_curr = 0
    prev_wall_clock = -np.inf

    for event in events:
        for moment in event.moments:
            curr_wall_clock = moment.wall_clock
            wall_clock_diff = curr_wall_clock - prev_wall_clock

            if wall_clock_diff > EBC.wall_clock_diff_upper_threshold:
                if verbose:
                    print(f"Constructing new event; wall_clock_diff between moments was {wall_clock_diff:.02f}")
                example_end_times.append(T_curr - 1)
            prev_wall_clock = curr_wall_clock
            T_curr += 1

    # Then append the last timestep
    last_timestep = T_curr
    example_end_times.append(last_timestep)
    return example_end_times


def get_example_stop_idxs__based_on_anomalous_step_sizes(
    events: List[Event],
) -> List[int]:
    Z_SCORE_CUTOFF_FOR_STEP_SIZE_ON_COURT = 4.0
    idxs_with_unusually_large_step_sizes = set()
    moments_processed = [moment for event in events for moment in event.moments]
    player_coords_unnormalized = player_coords_from_moments(moments_processed)

    player_coord_diffs_unnormalized = player_coords_unnormalized[1:, :, :] - player_coords_unnormalized[:-1, :, :]
    J = np.shape(player_coords_unnormalized)[1]
    for j in range(J):
        z_scores = zscore(player_coord_diffs_unnormalized[:, j])
        abs_z_score_maxes = np.max(np.abs(z_scores), 1)
        idxs_with_unusually_large_step_sizes_for_one_player = np.where(
            abs_z_score_maxes > Z_SCORE_CUTOFF_FOR_STEP_SIZE_ON_COURT
        )[0].tolist()
        idxs_with_unusually_large_step_sizes = idxs_with_unusually_large_step_sizes.union(
            idxs_with_unusually_large_step_sizes_for_one_player
        )
    return sorted(idxs_with_unusually_large_step_sizes)


def get_play_start_stop_idxs(events: List[Event]) -> List[Tuple[int]]:
    # TODO: It's probably weird that we represent examples with stop idxs and
    # plays with stop-start idxs.  Align these representations?

    event_start_stop_idxs = []
    num_moments_so_far = 0

    for event in events:
        event_first_moment = num_moments_so_far
        num_moments = len(event.moments)
        num_moments_so_far += num_moments
        event_last_moment = num_moments_so_far
        event_start_stop_idxs.extend([(event_first_moment, event_last_moment)])
    return event_start_stop_idxs


###
# Helper Functions
###

### The two functions below are used to convert `example_end_times` into pairs
### of indices designating the beginning and end of an event.


def get_start_and_stop_timestep_idxs_from_event_idx__using_one_indexing(
    event_stop_idxs: NumpyArray1D, event_idx_in_one_indexing: int
) -> Tuple[int]:
    """
    Arguments:
        event_stop_idxs: whose usage is very well documented in `run_CAVI_with_JAX`
    """
    start_idx = event_stop_idxs[(event_idx_in_one_indexing - 1)] + 1
    stop_idx = event_stop_idxs[event_idx_in_one_indexing]
    return (start_idx, stop_idx)


def get_start_and_stop_timestep_idxs_from_event_idx(
    event_stop_idxs: NumpyArray1D, event_idx_in_zero_indexing: int
) -> Tuple[int]:
    """
    Arguments:
        event_stop_idxs: whose usage is very well documented in `run_CAVI_with_JAX`
    """
    return get_start_and_stop_timestep_idxs_from_event_idx__using_one_indexing(
        event_stop_idxs, event_idx_in_zero_indexing + 1
    )
