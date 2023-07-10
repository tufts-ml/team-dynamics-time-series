import copy

import jax.numpy as jnp
import numpy as np

from dynagroup.diagnostics.fit_and_forecasting import (
    evaluate_posterior_mean_and_forward_simulation_on_slice,
)
from dynagroup.diagnostics.occupancies import (
    print_multi_level_regime_occupancies_after_init,
)
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.animate import (
    animate_event,
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.court import X_MAX_COURT, Y_MAX_COURT
from dynagroup.model2a.basketball.data.baller2vec_format import (
    coords_from_moments,
    get_event_in_baller2vec_format,
)
from dynagroup.model2a.basketball.mask import (
    make_mask_of_which_continuous_states_to_use,
)
from dynagroup.model2a.basketball.model import model_basketball
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
data_load_dir = "/Users/mwojno01/Desktop/"
save_dir = "/Users/mwojno01/Desktop/DEVEL_Basketball_with_proper_forecasting/"

# Data properties
animate_raw_data = False
event_end_times = None


# Initialization
animate_initialization = False
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE


# Model specification
K = 4
L = 5


# Inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# Forecasting
entity_to_mask = 9
forecast_horizon = 20


# CAVI diagnostics
animate_diagnostics = False
forecast_seeds = [0, 1, 2]
forecasting_entity_idxs = [i for i in range(10)]


###
# I/O
###

event_idxs = [0, 1, 2, 3, 4]

# get moments
events = []
moments = []
event_start_stop_idxs = []


num_moments_so_far = 0
for event_idx in event_idxs:
    event = get_event_in_baller2vec_format(event_idx, sampling_rate_Hz=5)
    if animate_raw_data:
        print(f"Now animating event idx {event_idx}, which has type {event.label}")
        animate_event(event)
    moments.extend(event.moments)
    event_first_moment = num_moments_so_far
    num_moments = len(event.moments)
    num_moments_so_far += num_moments
    event_last_moment = num_moments_so_far
    event_start_stop_idxs.extend([(event_first_moment, event_last_moment)])
    events.extend([event])

xs_unnormalized = coords_from_moments(moments)

ensure_dir(save_dir)

###
# Preprocess Data
###

# TODO: Move this (and `unnorm` function from the `animate` module) to the `court` module, which should control
# all operations that have to do with the size of the basketball court (including normalizing and unnormalizing)
xs = copy.copy(xs_unnormalized)
xs[:, :, 0] /= X_MAX_COURT
xs[:, :, 1] /= Y_MAX_COURT

###
# MASKING
###
use_continuous_states = make_mask_of_which_continuous_states_to_use(
    xs,
    entity_to_mask,
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
    event_end_times,
    model_basketball,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
    use_continuous_states,
)
params_init = results_init.params
most_likely_entity_states_after_init = results_init.record_of_most_likely_entity_states[
    :, :, -1
]  # TxJ
CSP_init = params_init.CSP  # JxKxDxD

# elbo_init = compute_elbo_from_initialization_results(
#     initialization_results, system_transition_prior, sample.xs, model, event_end_times, system_covariates
# )
# print(f"ELBO after init: {elbo_init:.02f}")

###
# Initialization Diagnostics
###

### Print regime occupancies
print_multi_level_regime_occupancies_after_init(results_init)

### Plot vector fields
plot_vector_fields(results_init.params.CSP, J=5)


### Animate some plays along with vector fields
if animate_initialization:
    J_FOCAL = 0
    # TODO: Give jersey label of the focal player in the title of the animation.
    animate_events_over_vector_field_for_one_player(
        events,
        event_start_stop_idxs,
        most_likely_entity_states_after_init,
        CSP_init,
        J_FOCAL,
    )


####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(xs),
    n_cavi_iterations,
    results_init,
    model_basketball,
    event_end_times,
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

### Plot fit and forecasting
forecasting_max_T = np.shape(xs)[0]
forecasting_find_t0_for_entity_sample = lambda x: forecasting_max_T - forecast_horizon

MSEs_posterior_mean, MSEs_forward_sim = evaluate_posterior_mean_and_forward_simulation_on_slice(
    xs,
    params_learned,
    VES_summary,
    VEZ_summaries,
    forecasting_max_T,
    model_basketball,
    forecast_seeds,
    save_dir,
    forecasting_entity_idxs,
    forecasting_find_t0_for_entity_sample,
    use_continuous_states,
    x_lim=(0, 1),
    y_lim=(0, 1),
    filename_prefix="",
    figsize=(8, 4),
)
MMSE_fit, MMSE_forecast = np.mean(MSEs_posterior_mean), np.mean(MSEs_forward_sim)
print(
    f"The mean (across entities) MSEs for fit is {MMSE_fit:.03f} and forecasting is {MMSE_forecast:.03f}."
)

### Plot animation with learned vector fields
if animate_diagnostics:
    J_FOCAL = 0
    s_maxes = np.argmax(VES_summary.expected_regimes, 1)
    most_likely_entity_states_after_CAVI = np.argmax(VEZ_summaries.expected_regimes, -1)
    CSP_after_CAVI = params_learned.CSP  # JxKxDxD

    # TODO: Give jersey label of the focal player in the title of the animation.
    animate_events_over_vector_field_for_one_player(
        events,
        event_start_stop_idxs,
        most_likely_entity_states_after_CAVI,
        CSP_after_CAVI,
        J_FOCAL,
        s_maxes,
    )
