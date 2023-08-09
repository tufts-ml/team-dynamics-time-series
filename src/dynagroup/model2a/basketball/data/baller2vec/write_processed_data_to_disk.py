import numpy as np

from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.court import normalize_coords
from dynagroup.model2a.basketball.data.baller2vec.CLE_starters_dataset import (
    get_basketball_games_for_CLE_dataset,
)
from dynagroup.model2a.basketball.data.baller2vec.data import (
    make_basketball_data_from_games,
)
from dynagroup.model2a.basketball.forecasts import (
    generate_random_context_times_for_events,
)


"""
For more information, see  EXPORTED_DATA_README.md
"""

###
# Configs
###

# Save dir
DATA_EXPORT_DIR = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/processed/"

# Data split
n_train_games_list = [1, 5, 20]
n_val_games = 4
n_test_games = 5

# Sampling rate
sampling_rate_Hz = 5

# Forecasts
T_test_event_min = 50
T_context_min = 10
T_forecast = 20
n_forecasts = 10


###
# I/O
###
ensure_dir(DATA_EXPORT_DIR)

games = get_basketball_games_for_CLE_dataset(sampling_rate_Hz=sampling_rate_Hz)
plays_per_game = [len(game.events) for game in games]
print(f"The plays per game are {plays_per_game}.")


###
# Data splitting and preprocessing
###

data_train_dict = {}
xs_train_dict = {}

for n_train_games in n_train_games_list:
    # TODO: Factor out train better so we're not redundantly doing val and test.
    games_train = games[
        -(n_train_games + n_test_games + n_val_games) : -(n_test_games + n_val_games)
    ]
    data_train_dict[n_train_games] = make_basketball_data_from_games(games_train)
    xs_train_dict[n_train_games] = normalize_coords(
        data_train_dict[n_train_games].coords_unnormalized
    )


games_val = games[-(n_test_games + n_val_games) : -n_test_games]
data_val = make_basketball_data_from_games(games_val)
xs_val = normalize_coords(data_val.coords_unnormalized)

games_test = games[-n_test_games:]
data_test = make_basketball_data_from_games(games_test)
xs_test = normalize_coords(data_test.coords_unnormalized)


###
# Random context times
###
random_context_times = generate_random_context_times_for_events(
    data_test.example_stop_idxs,
    T_test_event_min,
    T_context_min,
    T_forecast,
)


###
# Writing to disk
###

### Write Datasplits
for n_train_games in n_train_games_list:
    np.save(
        f"{DATA_EXPORT_DIR}/xs_train__with_{n_train_games}_games.npy", xs_train_dict[n_train_games]
    )
    np.save(
        f"{DATA_EXPORT_DIR}/event_stop_idxs_train__with_{n_train_games}_games.npy",
        data_train_dict[n_train_games].example_stop_idxs,
    )


np.save(f"{DATA_EXPORT_DIR}/xs_test.npy", xs_test)
np.save(f"{DATA_EXPORT_DIR}/event_stop_idxs_val.npy", data_val.example_stop_idxs)


np.save(f"{DATA_EXPORT_DIR}/xs_val.npy", xs_val)
np.save(f"{DATA_EXPORT_DIR}/event_stop_idxs_test.npy", data_test.example_stop_idxs)


### Write Random Context Times
np.save(f"{DATA_EXPORT_DIR}/random_context_times.npy", random_context_times)
