import jax.numpy as jnp
import numpy as np
from scipy import stats


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.io import ensure_dir
from dynagroup.model2a.circle.initialize import smart_initialize_model_2a_for_circles
from dynagroup.model2a.circle.model_factors import circle_model_JAX
from dynagroup.model2a.supra.diagnostics.soldier_dynamics import (
    report_on_directional_attractors,
)
from dynagroup.model2a.supra.diagnostics.soldier_segmentations import (
    compute_likely_soldier_regimes,
    panel_plot_the_soldier_headings_with_learned_segmentations,
    polar_plot_the_soldier_headings_with_learned_segmentations,
)
from dynagroup.model2a.supra.eda.show_squad_headings import (
    polar_plot_the_squad_headings,
)
from dynagroup.model2a.supra.get_data import (
    get_df,
    make_data_snippet,
    make_time_snippet_based_on_desired_elapsed_secs,
)
from dynagroup.params import Dims
from dynagroup.plotting.paneled_series import plot_time_series_with_regime_panels
from dynagroup.util import normalize_matrix_by_mean_and_std_of_columns
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


###
# Configs
###

### Events
event_end_times = None

### Model specification
num_entity_regimes = 4
num_system_regimes = 3
alpha_system_prior, kappa_system_prior = 1.0, 50.0
only_use_north_security = True

### Initialization
show_plots_after_init = True
bottom_half_self_transition_prob = 0.995
bottom_half_changepoint_penalty = 10.0
bottom_half_min_segment_size = 10
bottom_half_num_EM_iterations = 3
top_half_num_EM_iterations = 20
initialization_seed = 0

### Diagnostics
save_dir = "/Users/mwojno01/Desktop/supra_devel_only_north_security_L=3_K=4_still_stickier/"

### Inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "gradient_descent"
# can i do M_step_toggle_for_ETP in closed form, even though there are L of them?
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_von_mises"
M_step_toggle_for_IP = "closed_form_von_mises"
num_M_step_iters = 50
initialization_seed = 2

###
# Procedural
###

ensure_dir(save_dir)

###
# Get sample
###

if not "df" in globals():
    df = get_df()

time_snippet = make_time_snippet_based_on_desired_elapsed_secs(
    df,
    elapsed_secs_after_contact_start_for_starting=0,
    elapsed_secs_after_start_for_snipping=60.0,
    timestep_every=20,
)

snip = make_data_snippet(df, time_snippet)

polar_plot_the_squad_headings(snip.squad_angles, snip.clock_times, save_dir)

# TODO: Consider whether standardizing this is a good idea or not.
system_covariates = (
    snip.system_covariates_zero_to_hundred - np.mean(snip.system_covariates_zero_to_hundred)
) / np.std(snip.system_covariates_zero_to_hundred)


if only_use_north_security:
    INDEX_OF_NORTH_DIRECTION = 0
    system_covariates_zero_to_hundred = snip.system_covariates_zero_to_hundred[
        :, INDEX_OF_NORTH_DIRECTION
    ][:, None]
    system_covariates_dim_labels = ["N"]
else:
    system_covariates_zero_to_hundred = snip.system_covariates_zero_to_hundred
    system_covariates_dim_labels = ["N", "E", "S", "W"]

system_covariates = normalize_matrix_by_mean_and_std_of_columns(system_covariates)

####
# Smart Initialization for HSRDM (WIP)
###


# Setup DIMS
J = np.shape(snip.squad_angles)[1]
D, D_t, N, M_s, M_e = 1, 0, 0, 4, 0
DIMS = Dims(J, num_entity_regimes, num_system_regimes, D, D_t, N, M_s, M_e)

# Initialization
results_init = smart_initialize_model_2a_for_circles(
    DIMS,
    snip.squad_angles,
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


###
# Post-Initialization Diagnostics
###

### Post-Initialization ELBO
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)
elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    snip.squad_angles,
    circle_model_JAX,
    event_end_times,
    system_covariates,
)
print(f"\nELBO after init: {elbo_init:.02f}")


### Are the system-level states related to security scores? : Plots
s_hat = np.array(results_init.record_of_most_likely_system_states[:, -1], dtype=int)
fig, ax = plot_time_series_with_regime_panels(
    system_covariates_zero_to_hundred,
    s_hat,
    snip.clock_times,
    dim_labels=system_covariates_dim_labels,
)
plt.ylabel("Security scores")
plt.tight_layout
fig.savefig(save_dir + "system_segmentations_with_security_scores_after_init.pdf")
if show_plots_after_init:
    plt.show()

### Are the system-level states related to security scores? : Correlations

for compass_direction in range(4):
    # Compute the point-biserial correlation coefficient
    corr, pval = stats.pointbiserialr(
        s_hat, snip.system_covariates_zero_to_hundred[:, compass_direction]
    )
    # Print the correlation coefficient and p-value
    print(
        f"Compass dir: {compass_direction}. Correlation coefficient: {corr:.02f}, P-value: {pval:.03f}"
    )


###
# Inference
###

# TODO: Check if results_init (or something ele?) needs to be jax numpyified
VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(snip.squad_angles),
    n_cavi_iterations,
    results_init,
    circle_model_JAX,
    event_end_times,
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
    system_covariates_zero_to_hundred,
    s_hat,
    snip.clock_times,
    dim_labels=system_covariates_dim_labels,
)
plt.ylabel("Security scores")
plt.tight_layout
fig.savefig(save_dir + "system_segmentations_with_security_scores_after_learning.pdf")
if show_plots_after_init:
    plt.show()

### Solder-level segmentations

likely_soldier_regimes = compute_likely_soldier_regimes(VEZ_summaries.expected_regimes)

polar_plot_the_soldier_headings_with_learned_segmentations(
    snip.squad_angles, snip.clock_times, likely_soldier_regimes, save_dir
)
panel_plot_the_soldier_headings_with_learned_segmentations(
    snip.squad_angles, snip.clock_times, likely_soldier_regimes, save_dir
)

###
# Post Inference Reports
###

report_on_directional_attractors(params_learned)
