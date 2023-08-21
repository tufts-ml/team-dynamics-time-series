import numpy as np
import pytest

from dynagroup.model2a.basketball.data.baller2vec.TOR_vs_CHA import (
    get_basketball_data_for_TOR_vs_CHA,
)
from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import (
    get_event_in_baller2vec_format,
    load_event_label_decoder_from_pydict_info_path,
    load_player_data_from_pydict_info_path,
)


@pytest.fixture
def event_with_large_temporal_gap_between_first_two_moments():
    """
    This event has a large temporal gap between moments 0 and 1.
    It can be used to test the construction of example_end_times.
    """
    path_to_game_data = (
        "data/basketball/baller2vec_format/preprocessed/CLE_starters/games/0021500601_X.npy"
    )
    path_to_event_label_data = (
        "data/basketball/baller2vec_format/preprocessed/CLE_starters/games/0021500601_y.npy"
    )
    path_to_baller2vec_info = (
        "data/basketball/baller2vec_format/preprocessed/CLE_starters/info/baller2vec_info.pydict"
    )

    game_data = np.load(path_to_game_data)
    event_label_data = np.load(path_to_event_label_data)
    event_label_dict = load_event_label_decoder_from_pydict_info_path(path_to_baller2vec_info)
    player_data = load_player_data_from_pydict_info_path(path_to_baller2vec_info)

    event_idx = 4
    sampling_rate_Hz = 5
    verbose = False

    event = get_event_in_baller2vec_format(
        event_idx,
        game_data,
        event_label_data,
        event_label_dict,
        player_data,
        sampling_rate_Hz=sampling_rate_Hz,
        verbose=verbose,
    )
    return event


@pytest.fixture
def two_events_with_unusually_small_temporal_gap_between_them():
    """
    The stream of events has an unsually SMALL temporal gap between the first moment of the second
    event and the last moment of the first event.

    That is,
        events[1].moment[0].wall_clock - events[0].moment[-1].wall_clock
    is unusually small.  In particular, it is 80 ms, rather than the ~200 ms expected
    from sampling at 5 Hz.

    This can be used to test `clean_events_of_moments_with_too_small_intervals`
    """
    path_to_game_data = "data/basketball/baller2vec_format/preprocessed/TOR_vs_CHA/0021500492_X.npy"
    path_to_event_label_data = (
        "data/basketball/baller2vec_format/preprocessed/TOR_vs_CHA/0021500492_y.npy"
    )
    path_to_baller2vec_info = (
        "data/basketball/baller2vec_format/preprocessed/TOR_vs_CHA/baller2vec_config.pydict"
    )

    game_data = np.load(path_to_game_data)
    event_label_data = np.load(path_to_event_label_data)
    event_label_dict = load_event_label_decoder_from_pydict_info_path(path_to_baller2vec_info)
    player_data = load_player_data_from_pydict_info_path(path_to_baller2vec_info)

    event_idx = 1
    sampling_rate_Hz = 5
    verbose = False

    events = []
    for event_idx in [0, 1]:
        event = get_event_in_baller2vec_format(
            event_idx,
            game_data,
            event_label_data,
            event_label_dict,
            player_data,
            sampling_rate_Hz=sampling_rate_Hz,
            verbose=verbose,
        )
        events.append(event)

    return events


@pytest.fixture()
def basketball_data():
    return get_basketball_data_for_TOR_vs_CHA(
        event_idxs=None,
        sampling_rate_Hz=5,
        discard_nonstandard_hoop_sides=False,
        verbose=False,
    )
