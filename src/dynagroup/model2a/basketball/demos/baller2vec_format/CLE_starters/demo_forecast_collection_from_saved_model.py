import numpy as np

from dynagroup.model2a.basketball.data.baller2vec.disk import (
    load_processed_data_to_analyze,
)
from dynagroup.model2a.basketball.forecast_collection import (
    make_forecast_collections_for_all_basketball_examples,
)
from dynagroup.model2a.basketball.model import (
    load_model_from_model_type_string_filepath,
)
from dynagroup.params import load_params


###
# Configs
###

### Set run description

# 1) 20 train, our model
# run_description = "L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_20_CAVI_its_2_timestamp__10-12-2023_00h19m22s"

# 2) 20 train, no recurrence
# run_description = "L=5_K=10_model_type_No_Recurrence_train_20_CAVI_its_2_timestamp__10-12-2023_00h19m53s"

# 3) 5 train, our model
# run_description = "L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_5_CAVI_its_2_timestamp__10-12-2023_00h20m21s"

# 4) 5 train, no recurrence
# run_description = "L=5_K=10_model_type_No_Recurrence_train_5_CAVI_its_2_timestamp__10-12-2023_00h20m34s"

# 5) 1 train, our model
# run_description = "L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_CAVI_its_2_timestamp__10-12-2023_00h20m38s"

# 6) 1 train, no recurrence
run_description = "L=5_K=10_model_type_No_Recurrence_train_1_CAVI_its_2_timestamp__10-12-2023_00h20m43s"

### Model and params location
model_type_string_filepath = f"results/basketball/CLE_starters/artifacts/{run_description}_model_type_string.txt"
params_filepath = f"results/basketball/CLE_starters/artifacts/{run_description}_params.pkl"
artifacts_dir = f"results/basketball/CLE_starters/artifacts/"

# Unknown training settings
system_covariates = None  # Warning! We currently need to assume this matches what was used during training.

# Forecasting
random_forecast_starting_points = True
n_cavi_iterations_for_forecasting = 5
n_forecasts_per_example = 20
n_forecasting_examples_to_analyze = np.inf
n_forecasting_examples_to_plot = 0
T_forecast = 30

###
# Main
###

### Load in Params and Model
params_learned = load_params(params_filepath)
model_basketball = load_model_from_model_type_string_filepath(model_type_string_filepath)

### Load in Data
DATA = load_processed_data_to_analyze()
random_context_times = DATA.random_context_times

### Make quantiative forecasts

make_forecast_collections_for_all_basketball_examples(
    DATA,
    model_basketball,
    params_learned,
    system_covariates,
    n_cavi_iterations_for_forecasting,
    n_forecasts_per_example,
    random_forecast_starting_points,
    T_forecast,
    artifacts_dir,
    n_forecasting_examples_to_plot,
    n_forecasting_examples_to_analyze,
    subdir_prefix=run_description,
)
