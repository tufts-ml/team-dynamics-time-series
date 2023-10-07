import glob
import os

import numpy as np
import pytest

from dynagroup.model2a.basketball.data.baller2vec.CLE_starters_dataset import (
    CLE_POTENTIAL_STARTER_NAMES_2_ENTITY_IDXS,
    GAME_AND_EVENT_LABEL_DATA_DIR,
    PATH_TO_BALLER2VEC_INFO,
    PLAYER_NAMES_IN_DATASET_2_POSITIONS,
)
from dynagroup.model2a.basketball.data.baller2vec.data import (
    construct_basketball_data_from_single_game_file,
)


@pytest.fixture
def game():
    """
    load in a CLE basketball game
    """

    # Get a list of all files in the directory that end with *_X.npy or *_y.npy
    X_paths = glob.glob(os.path.join(GAME_AND_EVENT_LABEL_DATA_DIR, "*_X.npy"))
    y_paths = glob.glob(os.path.join(GAME_AND_EVENT_LABEL_DATA_DIR, "*_y.npy"))

    # Choose a game to analyze
    path_to_game_data = X_paths[0]
    path_to_event_label_data = y_paths[0]

    return construct_basketball_data_from_single_game_file(
        path_to_game_data,
        path_to_event_label_data,
        PATH_TO_BALLER2VEC_INFO,
        CLE_POTENTIAL_STARTER_NAMES_2_ENTITY_IDXS,
        PLAYER_NAMES_IN_DATASET_2_POSITIONS,
        event_idxs=None,
        sampling_rate_Hz=5,
        discard_nonstandard_hoop_sides=False,
        verbose=False,
    )


def test_that_the_largest_x_and_y_steps_on_court_for_all_players_in_one_basketball_game_lie_at_example_boundaries(game):
    xs_raw = game.player_coords_unnormalized
    diffs_raw = xs_raw[1:] - xs_raw[:-1]

    for j in range(10):
        for d in range(2):
            idx_of_step_with_large_magnitude_on_one_dim = np.argmax(abs(diffs_raw[:, j, d]))
            step_with_large_magnitude_on_one_dim = diffs_raw[idx_of_step_with_large_magnitude_on_one_dim, j]
            step_with_large_magnitude_on_one_dim_is_at_example_boundary = (
                idx_of_step_with_large_magnitude_on_one_dim in game.example_stop_idxs
            )
            print(
                f"Step {step_with_large_magnitude_on_one_dim} for player {j} is large on dim {d}.  Found at idx {idx_of_step_with_large_magnitude_on_one_dim}. At example boundary? {step_with_large_magnitude_on_one_dim_is_at_example_boundary}"
            )
            assert step_with_large_magnitude_on_one_dim_is_at_example_boundary
