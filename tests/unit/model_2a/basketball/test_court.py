import numpy as np

from dynagroup.model2a.basketball.court import flip_coords_unnormalized


def test_flip_coords_unnormalized():
    coords_unnormalized = np.array([[0, 0], [47, 25], [94, 50]], dtype=float)
    coords_unnormalized_flipped_computed = flip_coords_unnormalized(coords_unnormalized)
    coords_unnormalized_flipped_expected = np.array([[94, 50], [47, 25], [0, 0]], dtype=float)
    assert (coords_unnormalized_flipped_computed == coords_unnormalized_flipped_expected).all()

    coords_unnormalized = np.array([[57, 35]], dtype=float)
    coords_unnormalized_flipped_computed = flip_coords_unnormalized(coords_unnormalized)
    coords_unnormalized_flipped_expected = np.array([[37, 15]], dtype=float)
    assert np.allclose(coords_unnormalized_flipped_computed, coords_unnormalized_flipped_expected)
