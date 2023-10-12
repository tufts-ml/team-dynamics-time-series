import numpy as np
import pytest

from dynagroup.model2a.basketball.data.baller2vec.disk import (
    DataSamplingConfig,
    DataSplitConfig,
    ForecastConfig,
    load_processed_data_to_analyze,
)
from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    get_start_and_stop_timestep_idxs_from_event_idx,
)


@pytest.fixture
def T_FORECAST():
    return 30


@pytest.fixture
def DATA(T_FORECAST):
    # Note that the configs below are meant to match what's on disk; otherwise the
    # `load_processed_data_to_analyze` function will generate new data

    processed_data_dir = "tests/artifacts/basketball/processed_data/"
    data_sampling_config = DataSamplingConfig(sampling_rate_Hz=5)
    data_split_config = DataSplitConfig(n_train_games_list=[1, 5, 20], n_val_games=4, n_test_games=5)
    forecast_config = ForecastConfig(T_test_event_min=50, T_context_min=20, T_forecast=T_FORECAST)
    return load_processed_data_to_analyze(
        data_sampling_config,
        data_split_config,
        forecast_config,
        processed_data_dir,
    )


def test_that__generate_random_context_times__gives_enough_space_for_the_designated_forecasting_window(
    DATA, T_FORECAST
):
    # Rk: Warning - we test the function indirectly, by testing data that it generated and stored to disk.
    # It's actually more of a "data quality" test.

    random_context_times = DATA.random_context_times

    example_end_times_test = DATA.test.example_stop_idxs
    xs_test = np.asarray(DATA.test.player_coords)

    event_idxs_to_analyze = [
        i for (i, random_context_time) in enumerate(random_context_times) if not np.isnan(random_context_time)
    ]

    for e, event_idx_to_analyze in enumerate(event_idxs_to_analyze):
        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(
            example_end_times_test, event_idx_to_analyze
        )
        xs_test_example = xs_test[start_idx - 1 : stop_idx]

        T_context = int(random_context_times[event_idx_to_analyze])

        continuous_states_for_one_example = xs_test_example
        continuous_states_during_forecast_window = continuous_states_for_one_example[T_context : T_context + T_FORECAST]
        assert len(continuous_states_during_forecast_window) == T_FORECAST
