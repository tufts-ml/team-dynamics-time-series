import jax.numpy as jnp
import numpy as np

from dynagroup.diagnostics.fit_and_forecasting import plot_fit_and_forecast_on_slice
from dynagroup.diagnostics.occupancies import (
    print_multi_level_regime_occupancies_after_init,
)
from dynagroup.diagnostics.team_slice import plot_team_slice
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.data.orig_format import (
    get_data_in_original_formatting,
)
from dynagroup.model2a.basketball.model import model_basketball
from dynagroup.model2a.gaussian.diagnostics.mean_regime_trajectories import (
    get_deterministic_trajectories,
    plot_deterministic_trajectories,
)
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import Dims
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import run_CAVI_with_JAX
from dynagroup.vi.prior import SystemTransitionPrior_JAX


###
# Configs
###

# Directories
save_dir = "/Users/mwojno01/Desktop/just_init/"

# Initialization
do_init_plots = False
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE


# Model specification
K = 5
L = 5

# For inference
do_post_inference_plots = True
n_cavi_iterations = 10
M_step_toggle_for_STP = "gradient_descent"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 50.0


###
# Get data
###
DATA = get_data_in_original_formatting()
system_covariates = DATA.has_ball_players
M_s = np.shape(system_covariates)[1]

###
# Setup Model
###

#### Setup Dims

J = np.shape(DATA.positions)[1]
D, D_t = 2, 2
N = 0
M_e = 0  # for now!
DIMS = Dims(J, K, L, D, D_t, N, M_s, M_e)

### Setup Prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)


###
# I/O
###
ensure_dir(save_dir)

###
# INITIALIZATION
###


print("Running smart initialization.")

### TODO: Make smart initialization better. E.g.
# 1) Run init x times, pick the one with the best ELBO.
# 2) Find a way to do smarter init for the recurrence parameters
# 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

# TODO; Develop this and bring in system covariates.
results_init = smart_initialize_model_2a(
    DIMS,
    DATA.positions,
    DATA.event_end_times,
    model_basketball,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
)

params_init = results_init.params

# initialization_results

# elbo_init = compute_elbo_from_initialization_results(
#     initialization_results, system_transition_prior, sample.positions, model, event_end_times, system_covariates
# )
# print(f"ELBO after init: {elbo_init:.02f}")

###
# Initialization Diagnostics
###
if do_init_plots:
    ### plot learned dynamical modes
    deterministic_trajectories = get_deterministic_trajectories(
        params_init.CSP.As, params_init.CSP.bs, num_time_samples=100, x_init=DATA.positions[0]
    )
    plot_deterministic_trajectories(deterministic_trajectories, "Init", save_dir=save_dir)

# ### print regime occupancies
print_multi_level_regime_occupancies_after_init(results_init)

###
# TRY FORECASTING
###
if do_init_plots:
    event_idx = 4
    pct_event_to_skip = 0.0

    event_start = DATA.event_end_times[event_idx] + 1
    event_end = DATA.event_end_times[event_idx + 1]
    event_duration = event_end - event_start

    T_start = int(event_start + pct_event_to_skip * (event_duration))
    T_slice_max = event_end - T_start

    plot_fit_and_forecast_on_slice(
        DATA.positions,
        params_init,
        results_init.ES_summary,
        results_init.EZ_summaries,
        T_slice_max,
        model_basketball,
        forecast_seeds=[i + 1 for i in range(3)],
        save_dir=save_dir,
        entity_idxs=None,
        find_t0_for_entity_sample=lambda x: T_start,
        x_lim=(0, 2),
        y_lim=(0, 1),
        filename_prefix=f"AFTER_INITIALIZATION_",
        figsize=(8, 4),
    )


# # ####
# # # Inference
# # ####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(DATA.positions),
    n_cavi_iterations,
    results_init,
    model_basketball,
    DATA.event_end_times,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    system_covariates=jnp.asarray(system_covariates),
)

s_hats = np.argmax(VES_summary.expected_regimes, 1)

if do_post_inference_plots:
    ### Fit and Forecast
    event_idx = 4
    pct_event_to_skip = 0.50

    event_start = DATA.event_end_times[event_idx] + 1
    event_end = DATA.event_end_times[event_idx + 1]
    event_duration = event_end - event_start

    T_start = int(event_start + pct_event_to_skip * (event_duration))
    T_slice_max = event_end - T_start

    plot_team_slice(
        DATA.positions,
        T_start,
        T_slice_max,
        s_hats,
        x_lim=(0, 2),
        y_lim=(0, 1),
        save_dir=save_dir,
        show_plot=True,
        figsize=(8, 6),
    )

    plot_fit_and_forecast_on_slice(
        DATA.positions,
        params_init,
        results_init.ES_summary,
        results_init.EZ_summaries,
        T_slice_max,
        model_basketball,
        forecast_seeds=[i + 1 for i in range(3)],
        save_dir=save_dir,
        entity_idxs=None,
        find_t0_for_entity_sample=lambda x: T_start,
        x_lim=(0, 2),
        y_lim=(0, 1),
        filename_prefix=f"AFTER_CAVI_",
        figsize=(8, 4),
    )
