import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.io import ensure_dir
from dynagroup.model2a.circle.initialize import smart_initialize_model_2a_for_circles
from dynagroup.model2a.circle.model_factors import circle_model_JAX
from dynagroup.params import Dims
from dynagroup.plotting.paneled_series import plot_time_series_with_regime_panels
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX
from dynagroup.von_mises.util import degrees_to_radians


###
# Configs
###

### Sample selection

# start of contact: 9:18:50.  Around timestep 203100
# end of contact 9:28:00.
#
# roughly (based on a single clip), 13 timesteps is about 1/10 of a second.
# 20 timestep downsampling is good to get regimes that correspond to turning
# directions and back in the clip below
# t_start, t_end, t_every = 130000, 134000, 20
# entity_idx = 2
# num_entity_regimes = 2
#
# t_start, t_end, t_every = 19000, 23000, 20
# t_start, t_end, t_every = 203100, 207100, 20
# t_start, t_end, t_every = 130000, 134000, 20
t_start, t_end, t_every = 203100, 211100, 20


### Model specification
num_entity_regimes = 4
num_system_regimes = 4
alpha_system_prior, kappa_system_prior = 1.0, 10.0

### Initialization
show_plots_after_init = True
bottom_half_self_transition_prob = 0.995
bottom_half_changepoint_penalty = 10.0
bottom_half_min_segment_size = 10
bottom_half_num_EM_iterations = 3
top_half_num_EM_iterations = 20
initialization_seed = 0

### Diagnostics
save_dir = "/Users/mwojno01/Desktop/supra_devel/"

### Inference
n_cavi_iterations = 15
M_step_toggle_for_STP = "gradient_descent"
# can i do M_step_toggle_for_ETP in closed form, even though there are L of them?
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_von_mises"
M_step_toggle_for_IP = "closed_form_von_mises"
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0
initialization_seed = 2

###
# Get sample
###

filepath = "/Users/mwojno01/Library/CloudStorage/Box-Box/IRB_Approval_Required/MASTR_E_Program_Data/data/18_003_SUPRA_Data/Results_Files/MASTRE_SUPRA_P2S1_ITDG.csv"

if not "df" in globals():
    df = pd.read_csv(filepath)


entity_names = ["SL", "ATL", "AGRN", "AAR", "BTL", "BRM", "BGRN", "BAR"]  # no ARM
feature_name = "HELMET_HEADING"
system_covariate_names = [
    "SQUAD_SECURITYN",
    "SQUAD_SECURITYW",
    "SQUAD_SECURITYE",
    "SQUAD_SECURITYS",
    "SQUAD_SECURITY",
]

T = len(df)
J = len(entity_names)
heading_angles_as_degrees = np.zeros((T, J))
for j, entity_name in enumerate(entity_names):
    heading_angles_as_degrees[:, j] = df[f"{entity_name}_{feature_name}"]
heading_angles = degrees_to_radians(heading_angles_as_degrees)

# clock_times_all= np.array([x.split(" ")[1] for x in df["DATETIME"]])
clock_times_all = np.array([x.split(" ")[1].split(".")[0] for x in df["DATETIME"]])

###
# Subset data
###

squad_angles = heading_angles[t_start:t_end:t_every]
clock_times = clock_times_all[t_start:t_end:t_every]

# Per Lee Clifford Hancock's email on 5/8/23,
# PLT1-3 need directional relabelings
# (N to E, W to S, E to N, and S to W)

security_E = np.asarray(df["SQUAD_SECURITYN"])[t_start:t_end:t_every]
security_S = np.asarray(df["SQUAD_SECURITYW"])[t_start:t_end:t_every]
security_N = np.asarray(df["SQUAD_SECURITYE"])[t_start:t_end:t_every]
security_W = np.asarray(df["SQUAD_SECURITYS"])[t_start:t_end:t_every]
security_four_directions = np.vstack((security_N, security_E, security_S, security_W)).T

# we use security from last timestep as a system-level covariate
# (actually skip-level recurrence, but formally it's the same as a covariate)
system_covariates_zero_to_hundred = np.vstack((np.zeros(4), security_four_directions[:-1]))

# TODO: Consider whether standardizing this is a good idea or not.
system_covariates = (
    system_covariates_zero_to_hundred - np.mean(system_covariates_zero_to_hundred)
) / np.std(system_covariates_zero_to_hundred)


####
# Smart Initialization for HSRDM (WIP)
###


# Setup DIMS
J = np.shape(squad_angles)[1]
D, D_t, N, M_s, M_e = 1, 0, 0, 4, 0
DIMS = Dims(J, num_entity_regimes, num_system_regimes, D, D_t, N, M_s, M_e)

# Initialization
results_init = smart_initialize_model_2a_for_circles(
    DIMS,
    squad_angles,
    system_covariates,
    circle_model_JAX,
    bottom_half_self_transition_prob,
    bottom_half_changepoint_penalty,
    bottom_half_min_segment_size,
    bottom_half_num_EM_iterations,
    top_half_num_EM_iterations,
    initialization_seed,
)
params_init = results_init.params

s_hat = np.array(results_init.record_of_most_likely_system_states[:, -1], dtype=int)

###
# Post-Initialization Diagnostics
###

ensure_dir(save_dir)

### Post-Initialization ELBO
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)
elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    squad_angles,
    circle_model_JAX,
    system_covariates,
)
print(f"\nELBO after init: {elbo_init:.02f}")


### Are the system-level states related to security scores? : Correlations

for compass_direction in range(4):
    # Compute the point-biserial correlation coefficient
    corr, pval = stats.pointbiserialr(
        s_hat, system_covariates_zero_to_hundred[:, compass_direction]
    )
    # Print the correlation coefficient and p-value
    print(
        f"Compass dir: {compass_direction}. Correlation coefficient: {corr:.02f}, P-value: {pval:.03f}"
    )


### Are the system-level states related to security scores? : Plots
fig, ax = plot_time_series_with_regime_panels(
    system_covariates_zero_to_hundred, s_hat, clock_times, dim_labels=["N", "E", "S", "W"]
)
plt.ylabel("Security scores")
plt.tight_layout
fig.savefig(save_dir + "system_segmentations_with_security_scores_after_init.pdf")
if show_plots_after_init:
    plt.show()

###
# Inference
###

# TODO: Check if results_init (or something ele?) needs to be jax numpyified
VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(squad_angles),
    n_cavi_iterations,
    results_init,
    circle_model_JAX,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    jnp.asarray(system_covariates),
)

###
# Post Inference Plots
###
### Are the system-level states related to security scores? : Plots
s_hat = np.array(np.argmax(VES_summary.expected_regimes, 1), dtype=int)
fig, ax = plot_time_series_with_regime_panels(
    system_covariates_zero_to_hundred, s_hat, clock_times, dim_labels=["N", "E", "S", "W"]
)
plt.ylabel("Security scores")
plt.tight_layout
fig.savefig(save_dir + "system_segmentations_with_security_scores_after_learning.pdf")
if show_plots_after_init:
    plt.show()
