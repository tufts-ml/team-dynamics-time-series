import numpy as np

from dynagroup.model2a.basketball.court import X_MAX_COURT
from dynagroup.model2a.basketball.data.baller2vec.TOR_vs_CHA import (
    get_basketball_data_for_TOR_vs_CHA,
)


np.set_printoptions(precision=3, suppress=True)


def test_discarding_of_nonstandard_hoop_sides():
    data_with_no_discarding = get_basketball_data_for_TOR_vs_CHA(
        event_idxs=None,
        sampling_rate_Hz=5,
        discard_nonstandard_hoop_sides=False,
        verbose=False,
    )
    n_timesteps_with_no_discarding = np.shape(data_with_no_discarding.coords_unnormalized)[0]

    data_with_discarding = get_basketball_data_for_TOR_vs_CHA(
        event_idxs=None,
        sampling_rate_Hz=5,
        discard_nonstandard_hoop_sides=True,
        verbose=False,
    )
    n_timesteps_with_discarding = np.shape(data_with_discarding.coords_unnormalized)[0]
    assert n_timesteps_with_no_discarding > n_timesteps_with_discarding


def test_symmetry_of_player_locations_on_court_width_when_we_rotate_the_court_to_align_hoop_sides():
    for discard_nonstandard_hoop_sides in [True, False]:
        # If we don't discard nonstandard hoop sides, we rotate the court whenever we encounter
        # the nonstandard hoop side in order to align dynamics for offense vs defense.
        print(
            f"Now performing test when discarding of nonstandard hoop sides is {discard_nonstandard_hoop_sides}"
        )
        data_with_discarding = get_basketball_data_for_TOR_vs_CHA(
            event_idxs=None,
            sampling_rate_Hz=5,
            discard_nonstandard_hoop_sides=discard_nonstandard_hoop_sides,
            verbose=False,
        )
        un = data_with_discarding.coords_unnormalized

        x_dists = np.zeros((10, 2))
        y_dists = np.zeros((10, 2))
        left_tail_percentiles = [0.5, 1.0]

        for left_tail_percentile in left_tail_percentiles:
            print(
                f"---Now checking symmetry of distance from court boundary at left tail percentile {left_tail_percentile}."
            )
            percentile_queries = [left_tail_percentile, 100 - left_tail_percentile]
            for j in range(10):
                x_dists[j, :] = np.percentile(un[:, j, 0], percentile_queries)
                y_dists[j, :] = np.percentile(un[:, j, 1], percentile_queries)

            avg_horizontal_distance_from_left_boundary_at_percentile = np.mean(x_dists[:, 0])
            avg_horizontal_distance_from_right_boundary_at_percentile = X_MAX_COURT - np.mean(
                x_dists[:, 1]
            )

            print(
                f"Avg x distance from left boundary: {avg_horizontal_distance_from_left_boundary_at_percentile:.02f}"
            )
            print(
                f"Avg x distance from right boundary: {avg_horizontal_distance_from_right_boundary_at_percentile:.02f}"
            )
            # check that the distances are within 1 foot of each other
            assert np.isclose(
                avg_horizontal_distance_from_left_boundary_at_percentile,
                avg_horizontal_distance_from_right_boundary_at_percentile,
                atol=1,
            )
