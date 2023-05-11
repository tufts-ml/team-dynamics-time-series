import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from dynagroup.hmm_posterior import convert_hmm_posterior_summaries_from_jax_to_numpy
from dynagroup.initialize import (
    compute_elbo_from_initialization_results,
    inspect_entity_level_segmentations_over_EM_iterations,
    inspect_system_level_segmentations_over_EM_iterations,
)
from dynagroup.io import ensure_dir
from dynagroup.model2a.figure8.diagnostics.entity_transitions import (
    investigate_entity_transition_probs_in_different_contexts,
)
from dynagroup.model2a.figure8.diagnostics.fit_and_forecasting import (
    plot_fit_and_forecast_on_slice_for_figure_8,
)
from dynagroup.model2a.figure8.diagnostics.next_step import (
    compute_next_step_predictive_means,
)
from dynagroup.model2a.figure8.diagnostics.old_forecasting import (
    plot_results_of_old_forecasting_test,
)
from dynagroup.model2a.figure8.diagnostics.trajectories import (
    get_deterministic_trajectories,
    plot_deterministic_trajectories,
)
from dynagroup.model2a.figure8.generate import (
    ALL_PARAMS,
    sample,
    times_of_system_regime_changepoints,
)
from dynagroup.model2a.figure8.initialize import smart_initialize_model_2a
from dynagroup.model2a.figure8.model_factors import figure8_model_JAX
from dynagroup.params import dims_from_params, numpy_params_from_params
from dynagroup.plotting.entity_regime_changepoints import (
    plot_entity_regime_changepoints_for_figure_eight_dataset,
)
from dynagroup.plotting.sampling import plot_sample_with_system_regimes
from dynagroup.plotting.unfolded_time_series import plot_unfolded_time_series
from dynagroup.util import normalize_log_potentials_by_axis
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


###
# CONFIGS
###

# For sample generation
show_plots_of_samples = False

# For initialization
show_plots_after_init = False
seed_for_initialization = 1

# For model adjustments
model_adjustment = None  # Options: None, "one_system_regime", "remove_recurrence"

# For inference
n_iterations = 10
init_to_true_params = True
M_step_toggle_for_STP = "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0
initialization_seed = 2

# For diagnostics
show_plots_after_learning = False
save_dir = "/Users/mwojno01/Desktop/tmp3/"
T_snippet_for_fit_to_observations = 400
seeds_for_forecasting = [i + 1 for i in range(5)]
entity_idxs_for_forecasting = [2]
T_slice_for_forecasting = 200
T_slice_for_old_forecasting = 200

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
        plot_sample_with_system_regimes(
            sample.xs[:, j, :], sample.ys[:, j, :], sample.zs[:, j], sample.s
        )

    plot_unfolded_time_series(sample.xs, period_to_use=4)

###
# SPECIFY MODEL
###
model = figure8_model_JAX

###
# MAKE PRIOR
###
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)


###
# ADJUSTMENTS TO HANDLE SPECIAL CASES
###
if model_adjustment == "one_system_regime":
    DIMS.L = 1
elif model_adjustment == "remove_recurrence":
    model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX = (
        lambda x_vec: np.zeros(DIMS.D_t)  # noqa
    )

###
# INITIALIZATION
###

print("Running smart initialization.")

### TODO: Make smart initialization better. E.g.
# 1) Run init x times, pick the one with the best ELBO.
# 2) Find a way to do smarter init for the recurrence parameters
# 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

initialization_results = smart_initialize_model_2a(
    DIMS,
    sample.xs,
    figure8_model_JAX,
    seed_for_initialization,
)
params_init = initialization_results.params

### inspect quality of initialization
inspect_entity_level_segmentations_over_EM_iterations(
    initialization_results.record_of_most_likely_entity_states, sample.zs
)
inspect_system_level_segmentations_over_EM_iterations(
    initialization_results.record_of_most_likely_system_states, sample.s
)

elbo_init = compute_elbo_from_initialization_results(
    initialization_results, system_transition_prior, sample.xs, model, system_covariates
)
print(f"ELBO after init: {elbo_init:.02f}")

### Show plots of init
if show_plots_after_init:
    xs = get_deterministic_trajectories(
        params_true.CSP.As, params_true.CSP.bs, num_time_samples=100
    )
    plot_deterministic_trajectories(xs, "True")

    xs = get_deterministic_trajectories(
        params_init.CSP.As, params_init.CSP.bs, num_time_samples=100
    )
    plot_deterministic_trajectories(xs, "Initialized")

if show_plots_after_learning:
    plot_results_of_old_forecasting_test(
        params_true,
        T_slice_for_old_forecasting,
        model,
        title_prefix="forecasted (via true params)",
    )
    plot_results_of_old_forecasting_test(
        params_init,
        T_slice_for_old_forecasting,
        model,
        title_prefix="forecasted (via init params)",
    )

###
# CAVI
###


VES_summary, VEZ_summaries, params_learned = run_CAVI_with_JAX(
    jnp.asarray(sample.xs),
    n_iterations,
    initialization_results,
    model,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    system_covariates,
    true_system_regimes=sample.s,
    true_entity_regimes=sample.zs,
)

####
# PLOTS AND DIAGNOSTICS
###

### Make save dir
ensure_dir(save_dir)


### Plot Deterministic Trajectories (by regime)

if show_plots_after_learning:
    xs = get_deterministic_trajectories(
        params_true.CSP.As, params_true.CSP.bs, num_time_samples=100
    )
    plot_deterministic_trajectories(xs, "True")

    xs = get_deterministic_trajectories(
        params_learned.CSP.As, params_learned.CSP.bs, num_time_samples=100
    )
    plot_deterministic_trajectories(xs, "Learned")

# Print the norms of the eigenvalues of the state matrix.  The
# eigenvalues need to lie within the unit circle (in the complex plane)
# to prevent the trajectories from going off to infinity.
for j in range(DIMS.J):
    for k in range(DIMS.K):
        eigenvalues, eigenvectors = np.linalg.eig(params_learned.CSP.As[j, k])
        eigenvalue_norms = [np.linalg.norm(lam) for lam in eigenvalues]
        print(f"For entity {j} and regime {k}, the eigenvalue norms are {eigenvalue_norms}")


### Plot forecasting test

plot_fit_and_forecast_on_slice_for_figure_8(
    sample.xs,
    params_learned,
    VES_summary,
    VEZ_summaries,
    T_slice_for_forecasting,
    model,
    seeds_for_forecasting,
    save_dir,
    entity_idxs_for_forecasting,
)

### Plot Old Forecasting test
if show_plots_after_learning:
    plot_results_of_old_forecasting_test(
        params_true,
        T_slice_for_old_forecasting,
        model,
        title_prefix="forecasted (via true params)",
    )
    plot_results_of_old_forecasting_test(
        params_learned,
        T_slice_for_old_forecasting,
        model,
        title_prefix="forecasted (via learned params)",
    )


###  Plot Fit to Observations
if show_plots_after_learning:
    d = 0
    for j in range(DIMS.J):
        for after_learning in [False, True]:
            # TODO: This function needs to be sped up now that it can use JAX values as well.
            predictive_means = compute_next_step_predictive_means(
                numpy_params_from_params(params_learned),
                sample,
                convert_hmm_posterior_summaries_from_jax_to_numpy(VEZ_summaries),
                after_learning,
            )
            print(
                f"Plotting next-step predictive means for entity {j} and dim {d}. After learning ? {after_learning}"
            )
            plt.plot(predictive_means[0:T_snippet_for_fit_to_observations, j, d], color="red")
            plt.plot(sample.xs[0:T_snippet_for_fit_to_observations, j, d], color="black")
            plt.title(f"Entity {j+1}. After learning ? {after_learning}")
            plt.show()


### Plot Estimated Regimes
if show_plots_after_learning:
    most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
    most_likely_entity_regimes = np.argmax(VEZ_summaries.expected_regimes, axis=2)
    for j in range(DIMS.J):
        print(f"Now plotting results for entity {j}")
        plot_sample_with_system_regimes(
            sample.xs[:, j, :],
            sample.ys[:, j, :],
            most_likely_entity_regimes[:, j],
            most_likely_system_regimes,
        )


### Diagnostics Parameter investigation

# Parameter investigation :  endogenous transition preferences
learned_system_tpm = np.exp(normalize_log_potentials_by_axis(params_learned.STP.Pi, axis=1))
print(f"Learned system tpm: \n{learned_system_tpm}")
for j in range(DIMS.J):
    for l in range(DIMS.L):
        learned_entity_tpm = np.exp(
            normalize_log_potentials_by_axis(params_learned.ETP.Ps[j, l], axis=1)
        )
        print(f"Learned tpm for entity {j} under system regime {l}: \n{learned_entity_tpm}")

# Parameter investigation: Entity transition probabilities under different system regimes and closeness-to-origin statuses.
investigate_entity_transition_probs_in_different_contexts(
    params_true.ETP,
    sample.xs,
    model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
input("Above is report with TRUE params.  Press any key to continue.")
investigate_entity_transition_probs_in_different_contexts(
    params_init.ETP,
    sample.xs,
    model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
input("Above is report with INITIALIZED params.  Press any key to continue.")
investigate_entity_transition_probs_in_different_contexts(
    params_learned.ETP,
    sample.xs,
    model.transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    save_dir,
)
input("Above is report with LEARNED params.  Press any key to continue.")
