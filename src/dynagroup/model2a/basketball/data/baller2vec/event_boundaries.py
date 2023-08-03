from typing import List, Tuple

import numpy as np

from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import Event


def get_stop_idxs_for_inferred_events_from_provided_events(events: List[Event]) -> List[int]:
    """
    Try to figure out where the breaks are in our dataset over timesteps
    by finding huge shifts in coordinates.
    """

    MEAN_DIST_THRESH = 22.36
    # How was `MEAN_DIST_THRESH` determined? 20 percent of the [0,100]x[0,50]
    # court in both directions would be a diff of [20,10], which has a (Euclidean) dist of 22.36
    # We assume that if the AVERAGE player moved more than this in a single timestep (1/5 of a second)
    # that this is impossible.   Since  a basketball court is around 100feet x 50 feet, a movement of 22.36 in one
    # timestep translates to about 110 feet per second (PER player), which is much
    # faster than Usain Bolt's world record of 33.80 feet per second for 100 meters.

    event_end_times = []
    TT = -1
    prev_xs_raw = np.zeros(10)
    prev_ys_raw = np.zeros(10)
    for event in events:
        for moment in event.moments:
            curr_xs_raw = moment.player_xs
            curr_ys_raw = moment.player_ys

            diff_xs_raw = curr_xs_raw - prev_xs_raw
            diff_ys_raw = curr_ys_raw - prev_ys_raw

            mean_dist_raw = np.mean(np.sqrt(diff_xs_raw**2 + diff_ys_raw**2))
            # print(f"mean_dist_raw:{ mean_dist_raw:.02f}")
            prev_xs_raw = curr_xs_raw
            prev_ys_raw = curr_ys_raw

            if mean_dist_raw >= MEAN_DIST_THRESH:
                event_end_times.append(TT)
            TT += 1

    # then append the last time step
    # TODO: recall why we do that.
    last_timestep = TT + 1
    event_end_times.append(last_timestep)
    return event_end_times


def get_start_stop_idxs_from_provided_events(events: List[Event]) -> List[Tuple[int]]:
    event_start_stop_idxs = []
    num_moments_so_far = 0

    for event in events:
        event_first_moment = num_moments_so_far
        num_moments = len(event.moments)
        num_moments_so_far += num_moments
        event_last_moment = num_moments_so_far
        event_start_stop_idxs.extend([(event_first_moment, event_last_moment)])
    return event_start_stop_idxs
