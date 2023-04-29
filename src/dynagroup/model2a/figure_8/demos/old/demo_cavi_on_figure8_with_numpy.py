from matplotlib import pyplot as plt

from dynagroup.model2a.figure_8.diagnostics.next_step import (
    compute_next_step_predictive_means,
)
from dynagroup.model2a.figure_8.generate import (
    ALL_PARAMS,
    sample,
    times_of_system_regime_changepoints,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
)
from dynagroup.model2a.vi.initialize import smart_initialize_model_2a
from dynagroup.model2a.vi_numpy.core_numpy import run_CAVI_with_numpy
from dynagroup.params import dims_from_params
from dynagroup.plotting.entity_regime_changepoints import (
    plot_entity_regime_changepoints_for_figure_eight_dataset,
)
from dynagroup.plotting.sampling import plot_sample_with_system_regimes
from dynagroup.plotting.unfolded_time_series import plot_unfolded_time_series


"""
Demo Model 2a.

Model 1 is the model with "top-level" recurrence from entity regimes to system regimes.
Model 2 is the "top-down meta-switching model" from the notes.
    This model has the advantage that there is no exponential complexity
    in the number of entities.
Model 2a is the special case of Model 2 where the continuous states are directly observed.
    In the notes, it's called a meta-switching recurrent AR-HMM
    This has the advantage that CAVI is exact (no weird sampling), except for the Laplace VI.
"""


###
# CONFIGS
###

n_iterations = 3
run_M_step = False
init_to_true_params = False

###
# PLOT SAMPLE
###

### Grab dimensions
all_params = ALL_PARAMS
DIMS = dims_from_params(all_params)

### Plot sample
for j in range(DIMS.J):
    print(f"Now plotting results for entity {j}")
    plot_sample_with_system_regimes(
        sample.xs[:, j, :], sample.ys[:, j, :], sample.zs[:, j], sample.s
    )

plot_unfolded_time_series(sample.xs)

plot_entity_regime_changepoints_for_figure_eight_dataset(
    sample.z_probs,
    times_of_system_regime_changepoints,
    which_changepoint_to_show=2,
    which_entity_regime_to_show=1,
)


###
# CAVI
###

# RK: We are using a a --uniform-- initialization of q(Z) - should be bad!
# TODO: Need to smart initialize the expected joints; see Linderman

params_init = (
    all_params if init_to_true_params else smart_initialize_model_2a(DIMS, sample.xs).params
)


VES_summary, VEZ_summaries, all_params = run_CAVI_with_numpy(
    sample,
    n_iterations,
    params_init,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix,
    run_M_step,
)


for after_learning in [False, True]:
    predictive_means = compute_next_step_predictive_means(
        all_params,
        sample,
        VEZ_summaries,
        after_learning,
    )
    j, d = 0, 0
    print(
        f"Plotting next-step predictive means for entity {j} and dim {d}. After learning ? {after_learning}"
    )
    plt.plot(predictive_means[0:500, j, d], color="red")
    plt.plot(sample.xs[0:500, j, d], color="black")
    plt.show()
