import numpy as np

from dynagroup.model2a.basketball.court import flip_player_coords_unnormalized


def test_flip_player_coords_unnormalized():
    player_coords_unnormalized = np.array([[0, 0], [47, 25], [94, 50]], dtype=float)
    player_coords_unnormalized_flipped_computed = flip_player_coords_unnormalized(
        player_coords_unnormalized
    )
    player_coords_unnormalized_flipped_expected = np.array(
        [[94, 50], [47, 25], [0, 0]], dtype=float
    )
    assert (
        player_coords_unnormalized_flipped_computed == player_coords_unnormalized_flipped_expected
    ).all()

    player_coords_unnormalized = np.array([[57, 35]], dtype=float)
    player_coords_unnormalized_flipped_computed = flip_player_coords_unnormalized(
        player_coords_unnormalized
    )
    player_coords_unnormalized_flipped_expected = np.array([[37, 15]], dtype=float)
    assert np.allclose(
        player_coords_unnormalized_flipped_computed, player_coords_unnormalized_flipped_expected
    )
