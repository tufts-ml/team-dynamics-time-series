from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.court import normalize_coords
from dynagroup.model2a.basketball.data.baller2vec.CLE_starters_dataset import (
    get_basketball_games_for_CLE_dataset,
)
from dynagroup.model2a.basketball.data.baller2vec.data import (
    make_basketball_data_from_games,
)
from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import Event
from dynagroup.model2a.basketball.forecasts import (
    generate_random_context_times_for_events,
)
from dynagroup.types import NumpyArray2D, NumpyArray3D


"""
For more information, see  EXPORTED_DATA_README.md
"""

###
# Structs
###


### Configs
@dataclass
class DataSplitConfig:
    n_train_games_list: List[int]
    n_val_games: int
    n_test_games: int


@dataclass
class ForecastConfig:
    T_test_event_min: int
    T_context_min: int
    T_forecast: int


@dataclass
class DataSamplingConfig:
    sampling_rate_Hz: float


### Data
@dataclass
class DataSplit:
    player_coords: NumpyArray3D
    ball_coords: NumpyArray2D
    example_stop_idxs: List[int]
    play_start_stop_idxs: List[Tuple[int]]
    events: Optional[List[Event]] = None
    player_data: Optional[Dict] = None


@dataclass
class Processed_Data_To_Analyze:
    train_dict: Dict[int, DataSplit]
    val: DataSplit
    test: DataSplit
    random_context_times: List[int]


###
# Helpers
###
def compare_elements(element1, element2):
    """Written by ChatGPT"""
    if isinstance(element1, (list, np.ndarray)):
        return np.all(element1 == element2)
    return element1 == element2


def data_classes_match(instance1, instance2):
    """Written by ChatGPT"""
    fields = instance1.__dataclass_fields__.values()
    return all(
        compare_elements(getattr(instance1, field.name), getattr(instance2, field.name))
        for field in fields
    )


###
# Main functions
###
def write_processed_data_to_disk(
    data_sampling_config: DataSamplingConfig,
    data_split_config: DataSplitConfig,
    forecast_config: ForecastConfig,
    processed_data_dir: str,
):
    # TODO: Save config and check that our specs match the config

    ###
    # I/O
    ###
    ensure_dir(processed_data_dir)

    ###
    # Config alias and unpacking
    ###
    DSC = data_split_config
    FC = forecast_config
    n_train_games_list, n_val_games, n_test_games = (
        DSC.n_train_games_list,
        DSC.n_val_games,
        DSC.n_test_games,
    )

    ###
    # Generate Games
    ###
    games = get_basketball_games_for_CLE_dataset(
        sampling_rate_Hz=data_sampling_config.sampling_rate_Hz
    )
    plays_per_game = [len(game.events) for game in games]
    print(f"The plays per game are {plays_per_game}.")

    ###
    # Data splitting and preprocessing
    ###

    data_train_dict = {}
    player_coords_train_dict = {}
    ball_coords_train_dict = {}

    for n_train_games in n_train_games_list:
        # TODO: Factor out train better so we're not redundantly doing val and test.
        games_train = games[
            -(n_train_games + n_test_games + n_val_games) : -(n_test_games + n_val_games)
        ]
        data_train_dict[n_train_games] = make_basketball_data_from_games(games_train)
        player_coords_train_dict[n_train_games] = normalize_coords(
            data_train_dict[n_train_games].player_coords_unnormalized
        )
        ball_coords_train_dict[n_train_games] = normalize_coords(
            data_train_dict[n_train_games].ball_coords_unnormalized
        )

    games_val = games[-(n_test_games + n_val_games) : -n_test_games]
    data_val = make_basketball_data_from_games(games_val)
    player_coords_val = normalize_coords(data_val.player_coords_unnormalized)
    ball_coords_val = normalize_coords(data_val.ball_coords_unnormalized)

    games_test = games[-n_test_games:]
    data_test = make_basketball_data_from_games(games_test)
    player_coords_test = normalize_coords(data_test.player_coords_unnormalized)
    ball_coords_test = normalize_coords(data_test.ball_coords_unnormalized)

    ###
    # Random context times
    ###
    random_context_times = generate_random_context_times_for_events(
        data_test.example_stop_idxs,
        FC.T_test_event_min,
        FC.T_context_min,
        FC.T_forecast,
    )

    ###
    # Writing to disk
    ###

    ### Write Datasplits
    for n_train_games in n_train_games_list:
        np.save(
            f"{processed_data_dir}/player_coords_train__with_{n_train_games}_games.npy",
            player_coords_train_dict[n_train_games],
        )
        np.save(
            f"{processed_data_dir}/ball_coords_train__with_{n_train_games}_games.npy",
            ball_coords_train_dict[n_train_games],
        )
        np.save(
            f"{processed_data_dir}/example_stop_idxs_train__with_{n_train_games}_games.npy",
            data_train_dict[n_train_games].example_stop_idxs,
        )
        np.save(
            f"{processed_data_dir}/play_start_stop_idxs_train__with_{n_train_games}_games.npy",
            data_train_dict[n_train_games].play_start_stop_idxs,
        )
        np.save(
            f"{processed_data_dir}/events_train__with_{n_train_games}_games.npy",
            data_train_dict[n_train_games].events,
        )

    np.save(
        f"{processed_data_dir}/player_coords_val__with_{n_val_games}_games.npy", player_coords_val
    )
    np.save(f"{processed_data_dir}/ball_coords_val__with_{n_val_games}_games.npy", ball_coords_val)
    np.save(
        f"{processed_data_dir}/example_stop_idxs_val__with_{n_val_games}_games.npy",
        data_val.example_stop_idxs,
    )
    np.save(
        f"{processed_data_dir}/play_start_stop_idxs_val__with_{n_val_games}_games.npy",
        data_val.play_start_stop_idxs,
    )
    np.save(f"{processed_data_dir}/events_val__with_{n_val_games}_games.npy", data_val.events)

    np.save(
        f"{processed_data_dir}/player_coords_test__with_{n_test_games}_games.npy",
        player_coords_test,
    )
    np.save(
        f"{processed_data_dir}/ball_coords_test__with_{n_test_games}_games.npy", ball_coords_test
    )
    np.save(
        f"{processed_data_dir}/example_stop_idxs_test__with_{n_test_games}_games.npy",
        data_test.example_stop_idxs,
    )
    np.save(
        f"{processed_data_dir}/play_start_stop_idxs_test__with_{n_test_games}_games.npy",
        data_test.play_start_stop_idxs,
    )
    np.save(f"{processed_data_dir}/events_test__with_{n_test_games}_games.npy", data_test.events)

    ### Write Player dataset data
    # We could grab this from train, test or val, it doesn't matter because it's the same for all.
    # The player data is given by `preprocessed/<*>/info/baller2vec_info.pydict`, and so is
    # not a function of individual games, but of the whole dataset.  For more info, see the
    # BasketballData class definition.
    np.savez(f"{processed_data_dir}/player_data_from_all_games.npz", data_test.player_data)

    ### Write Random Context Times
    np.save(f"{processed_data_dir}/random_context_times.npy", random_context_times)

    ### Write configs
    np.savez(f"{processed_data_dir}/data_sampling_config_dict.npz", **data_sampling_config.__dict__)
    np.savez(f"{processed_data_dir}/data_split_config_dict.npz", **data_split_config.__dict__)
    np.savez(f"{processed_data_dir}/forecast_config_dict.npz", **forecast_config.__dict__)


def load_processed_data_to_analyze(
    data_sampling_config: DataSamplingConfig,
    data_split_config: DataSplitConfig,
    forecast_config: ForecastConfig,
    processed_data_dir: str,
) -> Processed_Data_To_Analyze:
    ###
    # Check if requested configs match those on disk.
    ###
    data_exists = False
    try:
        data_sampling_config_dict = np.load(f"{processed_data_dir}/data_sampling_config_dict.npz")
        data_split_config_dict = np.load(f"{processed_data_dir}/data_split_config_dict.npz")
        forecast_config_dict = np.load(f"{processed_data_dir}/forecast_config_dict.npz")
        data_exists = True
    except FileNotFoundError:
        print("The processed dir data directory is empty; will populate.")

    if data_exists:
        data_sampling_config_loaded = DataSamplingConfig(**data_sampling_config_dict)
        data_split_config_loaded = DataSplitConfig(**data_split_config_dict)
        forecast_config_loaded = ForecastConfig(**forecast_config_dict)

        requested_config_match_those_on_disk = True
        if not data_classes_match(data_sampling_config_loaded, data_sampling_config):
            requested_config_match_those_on_disk = False
        elif not data_classes_match(data_split_config_loaded, data_split_config):
            requested_config_match_those_on_disk = False
        elif not data_classes_match(forecast_config_loaded, forecast_config):
            requested_config_match_those_on_disk = False

    ###
    # Write data if we don't already have data with those configs on disk.
    ###

    # TODO: Currently we overwrite old configs.  Handle this better.
    # If nothing else, raise a warning or error when we overwrite old data.

    if not data_exists or (not requested_config_match_those_on_disk):
        print("Configs on disk don't match request. Writing new data to disk")
        write_processed_data_to_disk(
            data_sampling_config,
            data_split_config,
            forecast_config,
            processed_data_dir,
        )
    else:
        print("Configs on disk match request. Loading from disk.")

    ###
    # Load Data
    ###

    player_data_from_all_games = np.load(f"{processed_data_dir}/player_data_from_all_games.npz")

    train_by_sample_dict = {}
    for n_train_games in data_split_config.n_train_games_list:
        ball_coords = np.load(
            f"{processed_data_dir}/ball_coords_train__with_{n_train_games}_games.npy"
        )
        player_coords = np.load(
            f"{processed_data_dir}/player_coords_train__with_{n_train_games}_games.npy"
        )
        example_stop_idxs = np.load(
            f"{processed_data_dir}/example_stop_idxs_train__with_{n_train_games}_games.npy"
        )
        play_start_stop_idxs = np.load(
            f"{processed_data_dir}/play_start_stop_idxs_train__with_{n_train_games}_games.npy",
        )
        events = np.load(
            f"{processed_data_dir}/events_train__with_{n_train_games}_games.npy", allow_pickle=True
        )
        train_by_sample_dict[n_train_games] = DataSplit(
            player_coords,
            ball_coords,
            example_stop_idxs,
            play_start_stop_idxs,
            events,
            player_data_from_all_games,
        )

    n_val_games = data_split_config.n_val_games
    val_data = DataSplit(
        player_coords=np.load(
            f"{processed_data_dir}/player_coords_val__with_{n_val_games}_games.npy"
        ),
        ball_coords=np.load(f"{processed_data_dir}/ball_coords_val__with_{n_val_games}_games.npy"),
        example_stop_idxs=np.load(
            f"{processed_data_dir}/example_stop_idxs_val__with_{n_val_games}_games.npy"
        ),
        play_start_stop_idxs=np.load(
            f"{processed_data_dir}/play_start_stop_idxs_val__with_{n_val_games}_games.npy"
        ),
        events=np.load(
            f"{processed_data_dir}/events_val__with_{n_val_games}_games.npy", allow_pickle=True
        ),
        player_data=player_data_from_all_games,
    )

    n_test_games = data_split_config.n_test_games
    test_data = DataSplit(
        player_coords=np.load(
            f"{processed_data_dir}/player_coords_test__with_{n_test_games}_games.npy"
        ),
        ball_coords=np.load(
            f"{processed_data_dir}/ball_coords_test__with_{n_test_games}_games.npy"
        ),
        example_stop_idxs=np.load(
            f"{processed_data_dir}/example_stop_idxs_test__with_{n_test_games}_games.npy"
        ),
        play_start_stop_idxs=np.load(
            f"{processed_data_dir}/play_start_stop_idxs_test__with_{n_test_games}_games.npy"
        ),
        events=np.load(
            f"{processed_data_dir}/events_val__with_{n_val_games}_games.npy", allow_pickle=True
        ),
        player_data=player_data_from_all_games,
    )

    ### Load Random Context Times
    random_context_times = np.load(f"{processed_data_dir}/random_context_times.npy")
    return Processed_Data_To_Analyze(
        train_by_sample_dict, val_data, test_data, random_context_times
    )
