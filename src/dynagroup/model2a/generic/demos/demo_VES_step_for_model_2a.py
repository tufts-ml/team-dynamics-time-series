import numpy as np

from dynagroup.model2a.figure8.diagnostics import compute_regime_labeling_accuracy
from dynagroup.model2a.generic.generate import ALL_PARAMS, ETP, STP, sample
from dynagroup.model2a.vi import generate_expected_joints_uniformly, run_VES_step
from dynagroup.params import dims_from_params


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
# MAIN
###


### Grab dimensions
DIMS = dims_from_params(ALL_PARAMS)
J, K, L, D, N = DIMS.J, DIMS.K, DIMS.L, DIMS.D, DIMS.N
T = len(sample.s)

### Construct VES Step inputs #1: Uniform initialization

# TODO: Need to smart initialize the expected joints; see Linderman
variationally_expected_joints_for_entity_regimes = generate_expected_joints_uniformly(T, J, K)
variationally_expected_initial_entity_regimes = np.ones((J, K)) / K
continuous_states = sample.xs
init_dist_over_system_regimes = np.ones(L) / L
init_dists_over_entity_regimes = np.ones((J, K)) / K

### Construct VES Step inputs #2: Ground truth initialization
# Motivation: we should get a higher log normalizer (over entity z's)
# from the VES step when we make the incoming variational expectations from the VEZ
# step match the truth
ground_truth_joint_probs_for_entity_regimes = np.zeros((T - 1, J, K, K))
for j in range(J):
    for t in range(1, T):
        prev_regime, next_regime = sample.zs[t - 1, j], sample.zs[t, j]
        ground_truth_joint_probs_for_entity_regimes[t - 1, j, prev_regime, next_regime] = 1.0

ground_truth_probs_for_initial_entity_regimes = np.zeros((J, K))
for j in range(J):
    true_regime = sample.zs[0, j]
    ground_truth_probs_for_initial_entity_regimes[j, true_regime] = 1.0


###
# TEST
###

print(
    f"\n\nHere we give a demo of the VES step for model 2a (the meta-switching recurrent AR-HMM). "
    f"In particular, we compare the log_normalizers and the percentage of correct classifications "
    f"when we feed in the ground truth vs. a uniform initialization for q(z) from the VEZ step. "
)
print(
    f"\nOur motivation: we should get a higher log normalizer (over entity z's) and more "
    f"correct classifications from the VES step when we make the incoming variational expectations "
    f"from the VEZ step match the truth. "
)
input("\nPress any key to continue")

VES_summary = run_VES_step(
    STP,
    ETP,
    sample.xs,
    ground_truth_joint_probs_for_entity_regimes,
    ground_truth_probs_for_initial_entity_regimes,
    init_dist_over_system_regimes,
    init_dists_over_entity_regimes,
)
most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
pct_correct = compute_regime_labeling_accuracy(most_likely_system_regimes, sample.s)
print(
    f"\nVES step's log normalizer for entity regimes when we use ground truth inits for q(Z): {VES_summary.log_normalizer:.02f}"
)
print(f"Percent correct classifications {pct_correct:.02f}")

VES_summary = run_VES_step(
    STP,
    ETP,
    sample.xs,
    variationally_expected_joints_for_entity_regimes,
    variationally_expected_initial_entity_regimes,
    init_dist_over_system_regimes,
    init_dists_over_entity_regimes,
)
most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
pct_correct = compute_regime_labeling_accuracy(most_likely_system_regimes, sample.s)
print(
    f"\nVES step's log normalizer for entity regimes when we use uniform inits for q(Z): {VES_summary.log_normalizer:.02f}"
)
print(f"Percent correct classifications {pct_correct:.02f}")


# On one run, when we feed in truth for q(z)
#   log_normalizer = -22
# The other way
#   log_normalizer = -78402239.84339495.
# Nice!
