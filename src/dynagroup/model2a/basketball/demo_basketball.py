import copy

import jax.numpy as jnp
import numpy as np

from dynagroup.diagnostics.occupancies import (
    print_multi_level_regime_occupancies_after_init,
)
from dynagroup.io import ensure_dir
from dynagroup.model import Model
from dynagroup.model2a.figure8.initialize import smart_initialize_model_2a
from dynagroup.model2a.figure8.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_continuous_state_emissions_at_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)
from dynagroup.model2a.gaussian.diagnostics.mean_regime_trajectories import (
    get_deterministic_trajectories,
    plot_deterministic_trajectories,
)
from dynagroup.params import Dims
from dynagroup.types import JaxNumpyArray1D
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


###
# Configs
###

# Directories
data_load_dir = "/Users/mwojno01/Desktop/"
save_dir = "/Users/mwojno01/Desktop/basketball_devel/"

# Data properties
# t_every


# Initialization
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20

# Model specification
K = 3
L = 5

# For inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

###
# I/O
###
ensure_dir(save_dir)
xs_unnormalized = np.load(data_load_dir + "basketball_play.npy")

###
# Preprocess Data
###

# From: https://github.com/linouk23/NBA-Player-Movements/blob/master/Constant.py
X_MAX = 100
Y_MAX = 50

xs = copy.copy(xs_unnormalized)
xs[:, 0] /= X_MAX
xs[:, 1] /= Y_MAX


###
# Setup Model
###

#### Setup Dims

J = np.shape(xs)[1]
D, D_t = 2, 2
N = 0
M_s, M_e = 0, 0  # for now!
DIMS = Dims(J, K, L, D, D_t, N, M_s, M_e)

#### Setup Recurrence


# TODO: can I still the identity transformation  be a default?
def transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    KAPPA = 0.05
    return KAPPA * x_vec


# TODO: Can I set up the entity and system to be generic across fig8 and circles so that we
# call a single function each time?
model_basketball = Model(
    compute_log_continuous_state_emissions_at_initial_timestep_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)

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
    model_basketball,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
)
params_init = results_init.params

# initialization_results

# elbo_init = compute_elbo_from_initialization_results(
#     initialization_results, system_transition_prior, sample.xs, model, system_covariates
# )
# print(f"ELBO after init: {elbo_init:.02f}")

###
# Initialization Diagnostics
###

### plot learned dynamical modes
deterministic_trajectories = get_deterministic_trajectories(
    params_init.CSP.As, params_init.CSP.bs, num_time_samples=100, x_init=xs[0]
)
plot_deterministic_trajectories(deterministic_trajectories, "Init", save_dir=save_dir)

### plot regime occupancies
print_multi_level_regime_occupancies_after_init(results_init)

####
# Inference
####

VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(xs),
    n_cavi_iterations,
    results_init,
    model_basketball,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    system_covariates,
)
