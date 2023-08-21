from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    clean_events_of_moments_with_too_small_intervals,
    get_example_stop_idxs,
)


def test_get_example_stop_idxs(event_with_large_temporal_gap_between_first_two_moments):
    """
    Arguments:
        event_with_large_temporal_gap_between_first_two_moments: This event has a large temporal gap between
        moments 0 and 1.  Here we use it to test the construction of example stop idxs.
    """
    events = [event_with_large_temporal_gap_between_first_two_moments]
    example_stop_idxs = get_example_stop_idxs(events, sampling_rate_Hz=5.0)

    example_stop_idxs_expected = [-1, 0, 76]

    assert example_stop_idxs == example_stop_idxs_expected


def test_clean_events_of_moments_with_too_small_intervals(
    two_events_with_unusually_small_temporal_gap_between_them,
):
    events = two_events_with_unusually_small_temporal_gap_between_them
    events_cleaned = clean_events_of_moments_with_too_small_intervals(events, sampling_rate_Hz=5.0)
    num_moments_before_cleaning = len(events[1].moments)
    num_moments_after_cleaning = len(events_cleaned[1].moments)
    assert num_moments_after_cleaning < num_moments_before_cleaning
