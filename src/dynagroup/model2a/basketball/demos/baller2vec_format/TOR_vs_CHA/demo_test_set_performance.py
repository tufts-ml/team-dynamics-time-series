import jax.numpy as jnp
import numpy as np

from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.court import normalize_coords
from dynagroup.model2a.basketball.data.baller2vec.TOR_vs_CHA import (
    get_basketball_data_for_TOR_vs_CHA,
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
Added test set API for CAVI.
This allows us to apply a trained model to multiple separate test set events/plays, 
where each play has an initial context window and then a forecasting window.
"""

###
# Configs
###

# Directories
data_load_dir = "/Users/mwojno01/Desktop/"
save_dir = "/Users/mwojno01/Desktop/DEVEL_25_plays_complete_forecasting_test_set_performance/"

# Data properties
event_idxs_train = [i for i in range(25)]
event_idxs_test = [i + 25 for i in range(5)]

event_end_times_train = None
event_end_times_test = None


# Model specification
K = 4
L = 5

# Model adjustments
model_adjustment = None  # Options: None, "one_system_regime"

# Initialization
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE

# Inference
n_cavi_iterations = 2
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# Forecasts
T_context = 20
T_forecast = 20
n_forecasts = 3


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

basketball_data_train = get_basketball_data_for_TOR_vs_CHA(
    event_idxs_train,
    sampling_rate_Hz=5,
)

basketball_data_test = get_basketball_data_for_TOR_vs_CHA(
    event_idxs_test,
    sampling_rate_Hz=5,
)


###
# Preprocess Data
###

xs_train = normalize_coords(basketball_data_train.coords_unnormalized)
xs_test = normalize_coords(basketball_data_test.coords_unnormalized)

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
    xs_train,
    event_end_times_train,
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

####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(xs_train),
    n_cavi_iterations,
    results_init,
    model_basketball,
    event_end_times_train,
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
# Forecasting
###

### Warning: TODO:  We want the test set API to perform one forecast per play;
# right now it's just making a forecast at the end of ALL the plays.

forecasting_MSEs = get_forecasting_MSEs_on_test_set(
    xs_test,
    params_learned,
    model_basketball,
    T_context,
    T_forecast,
    n_cavi_iterations,
    n_forecasts,
    system_covariates,
)