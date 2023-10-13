import jax.numpy as jnp
import numpy as np

from dynagroup.diagnostics.steps_in_state import (
    plot_steps_within_examples_assigned_to_each_entity_state,
)
from dynagroup.eda.show_derivatives import plot_discrete_derivatives
from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.animate import (
    animate_event,
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.data.baller2vec.disk import (
    load_processed_data_to_analyze,
)
from dynagroup.model2a.basketball.forecast_collection import (
    make_forecast_collections_for_all_basketball_examples,
)
from dynagroup.model2a.basketball.model import (
    Model_Type,
    get_basketball_model,
    save_model_type,
)
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import (
    Dims,
    get_dim_of_entity_recurrence_output,
    get_dim_of_system_recurrence_output,
    save_params,
)
from dynagroup.util import get_current_datetime_as_string
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


"""
Module purpose: To explore the forecasts made on the "CLE Starters Dataset".  

Compared to the full analysis:
1) We run forecasts on the initialized model, not the model fully trained by CAVI.
2) We just look at one test set event at a time.
3) We make plots.
"""

###
# Configs
###

# Model specification
n_train_games_to_use = 20
model_type = Model_Type.Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence
# model_type = Model_Type.No_Recurrence
K = 10
L = 1

# Exploratory Data Analysis
animate_raw_data = False

# Initialization
animate_initialization = False
make_verbose_initialization_plots = False  # True
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE

# Inference
n_cavi_iterations = 2
make_verbose_CAVI_plots = False
M_step_toggle_for_STP = "gradient_descent"  # "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# Forecasting
random_forecast_starting_points = True
n_cavi_iterations_for_forecasting = 5
n_forecasts_per_example = 20
n_forecasting_examples_to_analyze = np.inf
n_forecasting_examples_to_plot = 0
T_forecast = 30

# Directories
datetime_as_string = get_current_datetime_as_string()
run_description = f"L={L}_K={K}_model_type_{model_type.name}_train_{n_train_games_to_use}_CAVI_its_{n_cavi_iterations}_timestamp__{datetime_as_string}"
plots_dir = f"results/basketball/CLE_starters/plots/{run_description}/"
artifacts_dir = f"results/basketball/CLE_starters/artifacts/"

###
# I/O
###
ensure_dir(plots_dir)
ensure_dir(artifacts_dir)

###
# Data splitting and preprocessing
###

DATA = load_processed_data_to_analyze()
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

plot_discrete_derivatives(DATA_TRAIN.player_coords, DATA_TRAIN.example_stop_idxs, use_continuous_states, plots_dir)


###
# Setup Model
###


### Setup Prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

### Setup Model Form
model_basketball = get_basketball_model(model_type)

#### Setup Dims

J = np.shape(DATA_TRAIN.player_coords)[1]
D = np.shape(DATA_TRAIN.player_coords)[2]
D_e = get_dim_of_entity_recurrence_output(D, model_basketball)
D_s = get_dim_of_system_recurrence_output(D, J, system_covariates, model_basketball)
M_e = 0  # for now!
N = 0
DIMS = Dims(J, K, L, D, D_e, N, D_s, M_e)

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
    plots_dir,
    verbose=True,
    plotbose=make_verbose_initialization_plots,
)
params_init = results_init.params
most_likely_entity_states_after_init = results_init.record_of_most_likely_entity_states[:, :, -1]  # TxJ

elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    DATA_TRAIN.player_coords,
    model_basketball,
    DATA_TRAIN.example_stop_idxs,
    system_covariates,
)
print(f"ELBO after init: {elbo_init:.02f}")

if make_verbose_initialization_plots:
    plot_steps_within_examples_assigned_to_each_entity_state(
        continuous_states=jnp.asarray(DATA_TRAIN.player_coords),
        continuous_state_labels=results_init.record_of_most_likely_entity_states[:, :, -1],
        example_end_times=DATA_TRAIN.example_stop_idxs,
        use_continuous_states=None,
        K=DIMS.K,
        plots_dir=plots_dir,
        show_plot=False,
        basename_prefix="post_init",
    )


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

if make_verbose_CAVI_plots:
    plot_steps_within_examples_assigned_to_each_entity_state(
        continuous_states=jnp.asarray(DATA_TRAIN.player_coords),
        continuous_state_labels=np.array(np.argmax(VEZ_summaries.expected_regimes, 2), dtype=int),
        example_end_times=DATA_TRAIN.example_stop_idxs,
        use_continuous_states=None,
        K=DIMS.K,
        plots_dir=plots_dir,
        show_plot=False,
        basename_prefix=f"post_CAVI_{n_cavi_iterations}_iterations",
    )

### Save model and learned params
save_model_type(model_type, artifacts_dir, basename_prefix=run_description)
save_params(params_learned, artifacts_dir, basename_prefix=run_description)


###
# Make quantiative forecasts
###

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
