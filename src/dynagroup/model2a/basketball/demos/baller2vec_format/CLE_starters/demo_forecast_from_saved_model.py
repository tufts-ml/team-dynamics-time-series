import numpy as np

from dynagroup.model2a.basketball.data.baller2vec.disk import (
    load_processed_data_to_analyze,
)
from dynagroup.model2a.basketball.forecasts import run_basketball_forecasts
from dynagroup.model2a.basketball.model import (
    load_model_from_model_type_string_filepath,
)
from dynagroup.params import load_params


###
# Configs
###

# Model and params location
model_type_string_filepath = "results/basketball/CLE_starters/models/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_val_4_test_5_CAVI_its_2_timestamp__10-11-2023_21h54m28s_model_type_string.txt"
params_filepath = "results/basketball/CLE_starters/models/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_val_4_test_5_CAVI_2_its__10-11-2023_21h54m28s_params.pkl"
analysis_dir = f"results/basketball/CLE_starters/analysis/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_val_4_test_5_CAVI_its_2_timestamp__10-11-2023_21h54m28s/"
# TODO: get analysis dir programmatically

# Unknown training settings
system_covariates = None  # Warning! We currently need to assume this matches what was used during training.

# Forecasting
random_forecast_starting_points = True
n_cavi_iterations_for_forecasting = 5
n_forecasts_per_example = 20
n_forecasting_examples_to_analyze = np.inf
n_forecasting_examples_to_plot = 0
T_forecast = 25  # note this is an "off-label" compared to what was generated on disk.

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

run_basketball_forecasts(
    DATA,
    model_basketball,
    params_learned,
    system_covariates,
    n_cavi_iterations_for_forecasting,
    n_forecasts_per_example,
    random_forecast_starting_points,
    T_forecast,
    analysis_dir,
    n_forecasting_examples_to_plot,
    n_forecasting_examples_to_analyze,
)
