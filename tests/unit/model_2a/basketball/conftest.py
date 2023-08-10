import numpy as np
import pytest

from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import (
    get_event_in_baller2vec_format,
    load_event_label_decoder_from_pydict_info_path,
    load_player_data_from_pydict_info_path,
)


@pytest.fixture
def event():
    """
    This event has a large temporal gap between moments 0 and 1.
    It can be used to test the construction of examples.
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
