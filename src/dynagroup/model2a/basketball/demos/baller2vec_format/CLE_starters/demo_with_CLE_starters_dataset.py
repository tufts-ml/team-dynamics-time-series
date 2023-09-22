import jax.numpy as jnp
import numpy as np

from dynagroup.eda.show_derivatives import plot_discrete_derivatives
from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.animate import (
    animate_event,
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.data.baller2vec.disk import (
    DataSamplingConfig,
    DataSplitConfig,
    ForecastConfig,
    load_processed_data_to_analyze,
)
from dynagroup.model2a.basketball.forecasts import (
    generate_random_context_times_for_events,
    get_forecast_MSEs_by_event,
)
from dynagroup.model2a.basketball.model import model_basketball
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import Dims
from dynagroup.util import get_current_datetime_as_string
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


"""
Module purpose: To "demo" (and, later, run) our training and forecasting pipeline
for the "CLE Starters Dataset".  Note that we reinitialize the model whenever 
the coordinates have changed a huge amount over timesteps, as can happen over halftime,
over excluded plays (because the lineup is not of interest), or across games.
"""

###
# Configs
###

# Processed data dir
processed_data_dir = "data/basketball/baller2vec_format/processed/"

# Data split
n_train_games_list = [1, 5, 20]
n_train_games_to_use = 1
n_val_games = 4
n_test_games = 5

# Sampling rate
sampling_rate_Hz = 5

# Directories
datetime_as_string = get_current_datetime_as_string()
save_dir = f"results/basketball/analyses/DEVEL_CLE_training_with_{n_train_games_to_use}_train_{n_val_games}_val_and_{n_test_games}_test_games__{datetime_as_string}/"

# Exploratory Data Analysis
animate_raw_data = False
save_plots_of_initialization_diagnostics = True

# Model specification
K = 10
L = 5

# Model adjustments
model_adjustment = None  # Options: None, "one_system_regime"

# Initialization
animate_initialization = False
make_verbose_initialization_plots = True
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE

# Inference
n_cavi_iterations = 1
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# Forecasts
T_test_event_min = 50
T_context_min = 20
T_forecast = 20
n_forecasts = 3

###
# I/O
###

ensure_dir(save_dir)

###
# Data splitting and preprocessing
###

data_sampling_config = DataSamplingConfig(sampling_rate_Hz)
data_split_config = DataSplitConfig(n_train_games_list, n_val_games, n_test_games)
forecast_config = ForecastConfig(T_test_event_min, T_context_min, T_forecast)

DATA = load_processed_data_to_analyze(
    data_sampling_config,
    data_split_config,
    forecast_config,
    processed_data_dir,
)
DATA_TRAIN, DATA_TEST, DATA_VAL = DATA.train_dict[n_train_games_to_use], DATA.test, DATA.val
random_context_times = DATA.random_context_times


###
# MASKING
###
use_continuous_states = None


###
# Raw data diagnostics
###

# animate
if animate_raw_data:
    n_events_to_animate = 5
    for event in DATA_TRAIN.events[-n_events_to_animate:]:
        animate_event(event)

plot_discrete_derivatives(
    DATA_TRAIN.player_coords, DATA_TRAIN.example_stop_idxs, use_continuous_states, save_dir
)


###
# Setup Model
###

#### Setup Dims

J = np.shape(DATA_TRAIN.player_coords)[1]
D, D_t = 2, 2
N = 0
M_s, M_e = 0, 0  # for now!
DIMS = Dims(J, K, L, D, D_t, N, M_s, M_e)

### Setup Prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

print("Running smart initialization.")

### TODO: Make smart initialization better. E.g.
# 1) Run init x times, pick the one with the best ELBO.
# 2) Find a way to do smarter init for the recurrence parameters
# 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

results_init = smart_initialize_model_2a(
    DIMS,
    DATA_TRAIN.player_coords,
    DATA_TRAIN.example_stop_idxs,
    model_basketball,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
    use_continuous_states,
    save_dir * save_plots_of_initialization_diagnostics,
    verbose = True, 
    plotbose = make_verbose_initialization_plots,
)
params_init = results_init.params
most_likely_entity_states_after_init = results_init.record_of_most_likely_entity_states[
    :, :, -1
]  # TxJ


elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    DATA_TRAIN.player_coords,
    model_basketball,
    DATA_TRAIN.example_stop_idxs,
    system_covariates,
)
print(f"ELBO after init: {elbo_init:.02f}")


###
# Animation Diagnostics
###

### Animate some plays along with vector fields
# RK: Focal team (blue) has scoring hoop on left.
if animate_initialization:
    J_FOCAL = 0
    first_event_idx, last_event_idx = 5, 10
    # TODO: Should we by default have the animation match the forecasting entity?
    animate_events_over_vector_field_for_one_player(
        DATA_TRAIN.events[first_event_idx:last_event_idx],
        DATA_TRAIN.play_start_stop_idxs[first_event_idx:last_event_idx],
        results_init.EZ_summaries.expected_regimes,
        params_init.CSP,
        J_FOCAL,
        DATA_TRAIN.player_data,
    )

####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(DATA_TRAIN.player_coords),
    n_cavi_iterations,
    results_init,
    model_basketball,
    DATA_TRAIN.example_stop_idxs,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    system_covariates,
    use_continuous_states,
)


###
# Forecasts
###


random_context_times = generate_random_context_times_for_events(
    DATA_TEST.example_stop_idxs,
    T_test_event_min,
    T_context_min,
    T_forecast,
)

forecast_MSEs_by_event = get_forecast_MSEs_by_event(
    DATA_TEST.player_coords,
    DATA_TEST.example_stop_idxs,
    params_learned,
    model_basketball,
    random_context_times,
    T_forecast,
    n_cavi_iterations,
    n_forecasts,
    system_covariates,
)
