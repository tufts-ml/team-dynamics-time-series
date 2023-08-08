import jax.numpy as jnp
import numpy as np

from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.animate import (
    animate_events_over_vector_field_for_one_player,
)
from dynagroup.model2a.basketball.court import normalize_coords
from dynagroup.model2a.basketball.data.baller2vec.CLE_starters_dataset import (
    get_basketball_games_for_CLE_dataset,
)
from dynagroup.model2a.basketball.data.baller2vec.data import (
    make_basketball_data_from_games,
)
from dynagroup.model2a.basketball.forecasts import (
    chunkify_xs_into_events_which_have_sufficient_length,
    generate_random_context_times_for_x_chunks,
)
from dynagroup.model2a.basketball.model import model_basketball
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.params import Dims
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX
from dynagroup.vi.vi_forecast import get_forecasting_MSEs_on_test_set


"""
Module purpose: To "demo" (and, later, run) our training and forecasting pipeline
for the "CLE Starters Dataset".  Note that we reinitialize the model whenever 
the coordinates have changed a huge amount over timesteps, as can happen over halftime,
over excluded plays (because the lineup is not of interest), or across games.
"""

###
# Configs
###

# Data split
n_train_games = 25
n_val_games = 2
n_test_games = 2

# Sampling rate
sampling_rate_Hz = 5


# Directories
save_dir = f"/Users/mwojno01/Desktop/DEVEL_CLE_training_with_{n_train_games}_games/"

# Model specification
K = 4
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
T_context_min = 10
T_forecast = 20
n_forecasts = 10

###
# I/O
###

ensure_dir(save_dir)

###
# Data splitting and preprocessing
###

# TODO: This information should be grabbed from the processed data that is written
# by `write_processed_data_to_disk.py`.  The current blocker is that we don't write
# all the information we need for this demo.  In particular, the animation uses
# some additional information (e.g. the "events" themselves and the provided  rather than inferred
# event boundaries).  The events are structs; while they can be written to and loaded from disk
# successfully by setting `allow_pickle=True`, this is dangerous because code changes (e.g. adding
# an attribute to Events) will make it impossible to read data back in from disk. It would be
# better to work directly with primitives -- e.g. np.arrays.

games = get_basketball_games_for_CLE_dataset(sampling_rate_Hz=sampling_rate_Hz)
plays_per_game = [len(game.events) for game in games]
print(f"The plays per game are {plays_per_game}.")

games_train = games[-(n_train_games + n_test_games + n_val_games) : -(n_test_games + n_val_games)]
games_val = games[-(n_test_games + n_val_games) : -n_test_games]
games_test = games[-n_test_games:]

data_train = make_basketball_data_from_games(games_train)
data_val = make_basketball_data_from_games(games_val)
data_test = make_basketball_data_from_games(games_test)

xs_train = normalize_coords(data_train.coords_unnormalized)
xs_val = normalize_coords(data_val.coords_unnormalized)
xs_test = normalize_coords(data_test.coords_unnormalized)


###
# MASKING
###
use_continuous_states = None

###
# Setup Model
###

#### Setup Dims

J = np.shape(xs_train)[1]
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
    xs_train,
    data_train.inferred_event_stop_idxs,
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


###
# Animation Diagnostics
###

### Animate some plays along with vector fields
if animate_initialization:
    J_FOCAL = 0
    n_events_to_animate = 2
    # TODO: Give jersey label of the focal player in the title of the animation.
    # TODO: Should we by default have the animation match the forecasting entity?
    animate_events_over_vector_field_for_one_player(
        data_train.events[:n_events_to_animate],
        data_train.provided_event_start_stop_idxs[:n_events_to_animate],
        most_likely_entity_states_after_init,
        params_init.CSP,
        J_FOCAL,
        save_dir,
        "post_init",
    )

####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(xs_train),
    n_cavi_iterations,
    results_init,
    model_basketball,
    data_train.inferred_event_stop_idxs,
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

# TODO: Move this into a function in the basketball.forecasts.py module.
# The return value is a List whose elements have type `dynagroup.forecasts.Forecast_MSEs`

# TODO: The two chunking functions, `chunkify_xs_into_events_which_have_sufficient_length`
# and `generate_random_context_times_for_x_chunks` involve dynamically processed data.
#  ut these are unnecessary if we use the `generate_random_context_times_for_events` function,
# which generates static data on disk for exporting to Preetish for AgentFormer.
# Ideally we should rewrite our forecasting so that we only need
# one kind of function, presumably `generate_random_context_times_for_events`.  Then we only need
# to have 1 function here instead of 3, and we remove redundancies that can cause problems upon
# further development. I am holding off on this until after the NeurIPS rebuttal period.
# Note that this is related to the proposed change in the `Data splitting and preprocessing`
# section above.

x_chunks_test = chunkify_xs_into_events_which_have_sufficient_length(
    data_test.inferred_event_stop_idxs, xs_test, T_test_event_min
)
T_contexts = generate_random_context_times_for_x_chunks(
    x_chunks_test,
    T_context_min,
    T_forecast,
)

forecasting_MSEs_by_chunk = [None] * len(x_chunks_test)
for i, (x_chunk_test, T_context) in enumerate(zip(x_chunks_test, T_contexts)):
    print(f"--- --- Now making forecasts on chunk {i+1}/{len(x_chunks_test)}. --- ---")
    forecasting_MSEs_by_chunk[i] = get_forecasting_MSEs_on_test_set(
        x_chunk_test,
        params_learned,
        model_basketball,
        T_context,
        T_forecast,
        n_cavi_iterations,
        n_forecasts,
        system_covariates,
    )


for i, forecasting_MSEs in enumerate(forecasting_MSEs_by_chunk):
    mean_median_forward_sim = np.mean(
        np.median(forecasting_MSEs.forward_simulation, 0)[0]
    )  # median over S, mean over J
    mean_fixed_velocity = np.mean(forecasting_MSEs.fixed_velocity[0])  # mean over J
    print(
        f"For chunk {i}, forward sim: {mean_median_forward_sim:.02f}, mean_fixed_velocity: {mean_fixed_velocity:.02f}"
    )
