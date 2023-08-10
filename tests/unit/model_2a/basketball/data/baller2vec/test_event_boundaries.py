from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    clean_events_of_moments_with_too_small_intervals_and_return_example_stop_idxs,
)


def test_clean_events_of_moments_with_too_small_intervals_and_return_example_stop_idxs(event):
    """
    Arguments:
        event: This event has a large temporal gap between moments 0 and 1.
            Here we use it to test the construction of example end times.
    """
    events = [event]

    (
        _,
        example_end_times,
    ) = clean_events_of_moments_with_too_small_intervals_and_return_example_stop_idxs(
        events, sampling_rate_Hz=5
    )

    example_end_times_expected = [-1, 0, 76]

    assert example_end_times == example_end_times_expected
