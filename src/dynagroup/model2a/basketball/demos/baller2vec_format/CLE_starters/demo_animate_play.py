import os

import numpy as np

from dynagroup.model2a.basketball.animate import (
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.data.baller2vec.disk import (
    load_processed_data_to_analyze,
)
from dynagroup.params import load_params


### SPECIFY PATHS TO SAVED MODEL
RESULTS_DIR = "results/basketball/CLE_starters/artifacts/"
BASENAME_HEAD = "rebuttal_L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_20_CAVI_its_10_timestamp__12-04-2023_00h50m51s"

BASENAME_TAIL__FOR__EXPECTED_ENTITY_REGIMES = "expected_regimes_qZ.npy"
BASENAME_TAIL__FOR__PARAMS = "params.pkl"

PATH_TO_EXPECTED_ENTITY_REGIMES = os.path.join(
    RESULTS_DIR, f"{BASENAME_HEAD}_{BASENAME_TAIL__FOR__EXPECTED_ENTITY_REGIMES}"
)
PATH_TO_EXPECTED_PARAMS = os.path.join(RESULTS_DIR, f"{BASENAME_HEAD}_{BASENAME_TAIL__FOR__PARAMS}")

### LOAD IN TRAINING DATA
DATA = load_processed_data_to_analyze()
n_train_games_to_use = 20
DATA_TRAIN = DATA.train_dict[n_train_games_to_use]


### LOAD IN TRAINED MODEL OBJECTS
expected_entity_regimes = np.load(PATH_TO_EXPECTED_ENTITY_REGIMES)
params = load_params(PATH_TO_EXPECTED_PARAMS)

### MAKE ANIMATION
# Rk: J_FOCAL, first_event_idx = 0,12 gives a play which is interesting b/c at the end,
# the vector field has Lebron running towards the three point line. Note in particular that,
# despite the k-means initializations which just had vector autoregressions with identity state
# matrix and non-zero bias terms (i.e. just traced out a constant movement direction), the
# model is learning entity states which are more interesting than that - i.e. have attractors
# at the three point lines.

J_FOCAL = 0
first_event_idx = 12  # Interesting ones: [12,7]
last_event_idx = first_event_idx + 1

# TODO: Should we by default have the animation match the forecasting entity?
animate_events_over_vector_field_for_one_player(
    DATA_TRAIN.events[first_event_idx:last_event_idx],
    DATA_TRAIN.play_start_stop_idxs[first_event_idx:last_event_idx],
    expected_entity_regimes,
    params.CSP,
    J_FOCAL,
)

### Attempt to save
# TODO: For some reason I can run the movie on my laptop, but cannot save it.
# animate_events_over_vector_field_for_one_player(
#     DATA_TRAIN.events[first_event_idx:last_event_idx],
#     DATA_TRAIN.play_start_stop_idxs[first_event_idx:last_event_idx],
#     expected_entity_regimes,
#     params.CSP,
#     J_FOCAL,
#     save_dir=""
# )
