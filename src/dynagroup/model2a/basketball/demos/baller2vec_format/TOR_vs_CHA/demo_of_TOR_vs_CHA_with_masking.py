import jax.numpy as jnp
import numpy as np

from dynagroup.diagnostics.occupancies import (
    print_multi_level_regime_occupancies_after_init,
)
from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.animate import (
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.court import normalize_coords
from dynagroup.model2a.basketball.data.baller2vec.TOR_vs_CHA import (
    get_basketball_data_for_TOR_vs_CHA,
)
from dynagroup.model2a.basketball.diagnostics.posterior_mean_and_forward_simulation import (
    write_model_evaluation_via_posterior_mean_and_forward_simulation_on_slice,
)
from dynagroup.model2a.basketball.mask import (
    make_mask_of_which_continuous_states_to_use,
)
from dynagroup.model2a.basketball.model import Model_Type, get_basketball_model
from dynagroup.model2a.basketball.plot_vector_fields import plot_vector_fields
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import Dims
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


"""
We model multiple back-to-back plays, treated as one long time series.
We then show animations along with the inferred system states.
Do the inferred system states track changes in plays? 
"""

###
# Configs
###

# Directories
save_dir = "results/basketball/analyses/EXPLORE_init_on_TOR_dataset_all_events/"

# Data properties
animate_raw_data = False
event_stop_idxs = None
event_idxs = None  # [i for i in range(25)]

# Model specification
model_type = Model_Type.Linear_Entity_Recurrence
K = 20
L = 5

# Model adjustments
model_adjustment = None  # Options: None, "one_system_regime"

# Initialization
animate_initialization = True
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE

# Inference
n_cavi_iterations = 3
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# Forecasting
entities_to_mask = [0, 1, 2, 3, 4]
forecast_horizon = 20

# CAVI diagnostics
animate_diagnostics = False
forward_simulation_seeds = [0, 1, 2]
forward_sim_and_posterior_mean_entity_idxs = [i for i in range(10)]

###
# MODEL ADJUSTMENTS
###
if model_adjustment == "one_system_regime":
    L = 1
    save_dir = save_dir.rstrip("/") + "_L=1/"


###
# I/O
###

ensure_dir(save_dir)

basketball_data = get_basketball_data_for_TOR_vs_CHA(
    event_idxs,
    sampling_rate_Hz=5,
)

###
# Preprocess Data
###

xs = normalize_coords(basketball_data.player_coords_unnormalized)

###
# MASKING
###
use_continuous_states = make_mask_of_which_continuous_states_to_use(
    xs,
    entities_to_mask,
    forecast_horizon,
)


###
# Setup Model
###

#### Setup Dims

J = np.shape(xs)[1]
D, D_t = 2, 2
N = 0
M_s, M_e = 0, 0  # for now!
DIMS = Dims(J, K, L, D, D_t, N, M_s, M_e)

### Setup Prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

### Setup Model Form
model_basketball = get_basketball_model(model_type)

###
# INITIALIZATION
###


print("Running smart initialization.")

### TODO: Make smart initialization better. E.g.
# 1) Run init x times, pick the one with the best ELBO.
# 2) Find a way to do smarter init for the recurrence parameters
# 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

results_init = smart_initialize_model_2a(
    DIMS,
    xs,
    event_stop_idxs,
    model_basketball,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
    use_continuous_states,
    save_dir,
)
params_init = results_init.params
CSP_init = params_init.CSP  # JxKxDxD


###
# Initialization Diagnostics
###

### Compute ELBO
elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    xs,
    model_basketball,
    basketball_data.example_stop_idxs,
    system_covariates,
)
print(f"ELBO after init: {elbo_init:.02f}")


### Animate some plays along with vector fields
if animate_initialization:
    J_FOCAL = 0
    first_event_idx, last_event_idx = 0, 5
    # TODO: Give jersey label of the focal player in the title of the animation.
    # TODO: Should we by default have the animation match the forecasting entity?
    animate_events_over_vector_field_for_one_player(
        basketball_data.events[first_event_idx:last_event_idx],
        basketball_data.play_start_stop_idxs[first_event_idx:last_event_idx],
        results_init.EZ_summaries.expected_regimes,
        CSP_init,
        J_FOCAL,
        basketball_data.player_data,
    )


### Print regime occupancies
print_multi_level_regime_occupancies_after_init(results_init)

### Plot vector fields
plot_vector_fields(results_init.params.CSP, J=5)

### Plot posterior means and forward simulations
find_forward_sim_t0_for_entity_sample = lambda x: np.shape(xs)[0] - forecast_horizon
(
    MSEs_via_posterior_mean_after_init,
    MSEs_via_forward_sims_after_init,
    MSEs_via_velocity_baseline_after_init,
) = write_model_evaluation_via_posterior_mean_and_forward_simulation_on_slice(
    xs,
    params_init,
    results_init.ES_summary,
    results_init.EZ_summaries,
    model_basketball,
    forward_simulation_seeds,
    save_dir,
    use_continuous_states,
    forward_sim_and_posterior_mean_entity_idxs,
    find_forward_sim_t0_for_entity_sample,
    system_covariates=system_covariates,
    max_forward_sim_window=forecast_horizon,
    find_posterior_mean_t0_for_entity_sample=find_forward_sim_t0_for_entity_sample,
    max_posterior_mean_window=forecast_horizon,
    filename_prefix="AFTER_INITIALIZATION_",
    figsize=(8, 4),
)


####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(xs),
    n_cavi_iterations,
    results_init,
    model_basketball,
    event_stop_idxs,
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
# CAVI  Diagnostics
###

### Plot vector fields
plot_vector_fields(params_learned.CSP, J=5)

### Plot posterior mean and forward simulation
(
    MSEs_via_posterior_mean_after_CAVI,
    MSEs_via_forward_sims_after_CAVI,
    MSEs_via_velocity_baseline_after_CAVI,
) = write_model_evaluation_via_posterior_mean_and_forward_simulation_on_slice(
    xs,
    params_learned,
    VES_summary,
    VEZ_summaries,
    model_basketball,
    forward_simulation_seeds,
    save_dir,
    use_continuous_states,
    forward_sim_and_posterior_mean_entity_idxs,
    find_forward_sim_t0_for_entity_sample,
    system_covariates=system_covariates,
    max_forward_sim_window=forecast_horizon,
    find_posterior_mean_t0_for_entity_sample=find_forward_sim_t0_for_entity_sample,
    max_posterior_mean_window=forecast_horizon,
    filename_prefix="AFTER_CAVI_",
    figsize=(8, 4),
)

### Plot animation with learned vector fields
if animate_diagnostics:
    J_FOCAL = 0
    s_maxes = np.argmax(VES_summary.expected_regimes, 1)
    CSP_after_CAVI = params_learned.CSP  # JxKxDxD

    # TODO: Give jersey label of the focal player in the title of the animation.
    animate_events_over_vector_field_for_one_player(
        basketball_data.events,
        basketball_data.play_start_stop_idxs,
        VEZ_summaries.expected_regimes,
        CSP_after_CAVI,
        J_FOCAL,
        basketball_data.player_data,
        s_maxes=s_maxes,
    )
