import jax.numpy as jnp
import numpy as np
import os

from dynagroup.io import ensure_dir
from dynagroup.model2a.figure8.diagnostics.fit_and_forecasting import (
    evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8,
)
from dynagroup.model2a.figure8.generate import (
    ALL_PARAMS,
    sample,
    times_of_system_regime_changepoints,
)
from dynagroup.model2a.figure8.mask import make_mask_of_which_continuous_states_to_use
from dynagroup.model2a.figure8.model_factors import figure8_model_JAX
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.util import (
    get_current_datetime_as_string,
)
from dynagroup.params import dims_from_params
from dynagroup.plotting.entity_regime_changepoints import (
    plot_entity_regime_changepoints_for_figure_eight_dataset,
)
from dynagroup.plotting.sampling import plot_sample_with_system_regimes
from dynagroup.plotting.unfolded_time_series import plot_unfolded_time_series
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX


"""
Demo Model 2a.

Model 1 is the model with "top-level" recurrence from entity regimes to system regimes.
Model 2 is the "top-down meta-switching model" from the notes.
    This model has the advantage that there is no exponential complexity
    in the number of entities.
Model 2a is the special case of Model 2 where the continuous states are directly observed.
    In the notes, it's called a meta-switching recurrent AR-HMM
    This has the advantage that CAVI is exact (no weird sampling), except for the Laplace VI.

For inference, we use JAX.
"""

### TODO: Integrate this into the main fig 8 demo.  It should just be one variant.

###
# CONFIGS
###

# For sample generation
show_plots_of_samples = False

# Masking and model adjustments
mask_final_regime_transition_for_entity_2 = True
MODEL_ADJUSTMENT = "complete_pooling"  # Options: None, "one_system_regime", "remove_recurrence", "complete_pooling"

# Events
example_end_times = None

# For initialization
seed_for_initialization = 123
show_plots_after_init = False
num_em_iterations_for_bottom_half_init = 5
num_em_iterations_for_top_half_init = 20
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.LOCATION


# For inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "closed_form_tpm" #"gradient_descent"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0

# For diagnostics
show_plots_after_learning = False
T_snippet_for_fit_to_observations = 400
seeds_for_forecasting = [120, 121, 122, 123, 124]
entity_idxs_for_forecasting = [2]
T_slice_for_forecasting = 120

# Directories
datetime_as_string = get_current_datetime_as_string()
run_description = f"seed_{seed_for_initialization}_timestamp__{datetime_as_string}_pooling"
home_dir = os.path.expanduser("~")
plots_dir = f"{home_dir}/team-dynamics-time-series/src/dynagroup/model2a/figure8/results/plots/{run_description}/"


###
# PLOT SAMPLE
###

### Grab dimensions
params_true = ALL_PARAMS
DIMS = dims_from_params(params_true)

### Plot sample
if show_plots_of_samples:
    plot_entity_regime_changepoints_for_figure_eight_dataset(
        sample.z_probs,
        times_of_system_regime_changepoints,
        which_changepoint_to_show=2,
        which_entity_regime_to_show=1,
    )

    for j in range(DIMS.J):
        print(f"Now plotting results for entity {j}")
        plot_sample_with_system_regimes(sample.xs[:, j, :], sample.ys[:, j, :], sample.zs[:, j], sample.s)

    plot_unfolded_time_series(sample.xs, period_to_use=4)

###
# MASKING
###
if mask_final_regime_transition_for_entity_2:
    use_continuous_states = make_mask_of_which_continuous_states_to_use(sample.xs)
else:
    use_continuous_states = None

###
# SPECIFY MODEL
###
model = figure8_model_JAX

###
# MAKE PRIOR
###
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)


###
# MODEL ADJUSTMENTS
###
xs_for_inference = sample.xs
if MODEL_ADJUSTMENT == "one_system_regime":
    DIMS.L = 1
elif MODEL_ADJUSTMENT == "remove_recurrence":
    model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e)  # noqa
    )
elif MODEL_ADJUSTMENT == "complete_pooling":
    DIMS.J = 1
    # TODO: move this to model adjustment repo
    T, J, D = np.shape(sample.xs)
    T_pooled = T * J
    xs_stacked = np.zeros((T_pooled, 1, D))
    for j in range(J):
        xs_stacked[T * j : T * (j + 1), 0, :] = sample.xs[:, j, :]
    xs_for_inference = xs_stacked
    example_end_times = np.array([-1] + [T * (i + 1) for i in range(J)])
    if mask_final_regime_transition_for_entity_2:
        # TODO: Do this via a proper function, don't hardcode it
        use_continuous_states = np.full((T_pooled, J), True)
        use_continuous_states[1080:,] = False
###
# I/O
###

ensure_dir(plots_dir)

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
    xs_for_inference,
    example_end_times,
    figure8_model_JAX,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
    use_continuous_states,
    plots_dir,
)
params_init = results_init.params
VES_init, VEZ_init = results_init.ES_summary, results_init.EZ_summaries


###
# CAVI
###


VES_summary, VEZ_summaries, params_learned, elbo_decomposed, classification_list = run_CAVI_with_JAX(
    params_init,
    VES_init, VEZ_init,
    system_transition_prior,
    model,
    jnp.asarray(xs_for_inference),
    example_end_times,
    n_cavi_iterations,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_covariates,
    use_continuous_states,
)

most_likely_entity_regimes = np.argmax(VEZ_summaries.expected_regimes, axis=2)
print(most_likely_entity_regimes)

###
# Forecasting...adjusted...
###
entity_idxs_for_forecasting = [0]
evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8(
    xs_for_inference,
    params_learned,
    VES_summary,
    VEZ_summaries,
    T_slice_for_forecasting,
    model,
    seeds_for_forecasting,
    plots_dir,
    use_continuous_states,
    entity_idxs_for_forecasting,
    filename_prefix=f"adjustment_{MODEL_ADJUSTMENT}_",
)
