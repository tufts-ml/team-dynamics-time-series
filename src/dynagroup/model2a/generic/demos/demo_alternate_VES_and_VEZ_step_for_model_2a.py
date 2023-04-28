from dynagroup.model2a.generic.generate import ALL_PARAMS, sample
from dynagroup.model2a.vi import run_CAVI_with_numpy
from dynagroup.params import dims_from_params
from dynagroup.plotting.sampling import plot_sample_with_system_regimes


"""
Demo Model 2a.

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


###
# PLOT SAMPLE
###

### Grab dimensions
DIMS = dims_from_params(ALL_PARAMS)

### Plot sample
for j in range(DIMS.J):
    print(f"Now plotting results for entity {j}")
    plot_sample_with_system_regimes(
        sample.xs[:, j, :], sample.ys[:, j, :], sample.zs[:, j], sample.s
    )


###
# CAVI
###

# RK: We are using a a --uniform-- initialization of q(Z) - should be bad!
# TODO: Need to smart initialize the expected joints; see Linderman

run_CAVI_with_numpy(sample, n_iterations, ALL_PARAMS)
