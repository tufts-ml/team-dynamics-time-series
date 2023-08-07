import numpy as np

from dynagroup.model2a.basketball.data.baller2vec.TOR_vs_CHA import (
    get_basketball_data_for_TOR_vs_CHA,
)


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
