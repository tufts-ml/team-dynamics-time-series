import numpy as np

from dynagroup.model2a.figure8.diagnostics import compute_regime_labeling_accuracy
from dynagroup.model2a.generic.generate import ALL_PARAMS, CSP, ETP, IP, sample
from dynagroup.model2a.vi import generate_expected_state_regimes_uniformly, run_VEZ_step
from dynagroup.params import dims_from_params


np.set_printoptions(precision=2)

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


### Construct VEZ Step inputs, Option #1: Uniform initialization of q(s)
init_dists_over_entity_regimes = np.ones((J, K)) / K
variationally_expected_system_regimes = generate_expected_state_regimes_uniformly(T, L)

### Construct VEZ Step inputs, Option #2:  Initialize q(s) to ground truth
ground_truth_probs_for_system_regimes = np.zeros((T, L))
for t in range(T):
    true_regime = sample.s[t]
    ground_truth_probs_for_system_regimes[t, true_regime] = 1.0

###
# TEST
###

print(
    f"Here we give a demo of the VEZ step for model 2a (the meta-switching recurrent AR-HMM). "
    f"In particular, we compare the log_normalizers and the percentage of correct classifications "
    f"when we feed in the ground truth vs. a uniform initialization for q(s) from the VES step. "
)
print(
    f"\nOur motivation: we should get a higher log normalizer (over entity z's) and more "
    f"correct classifications from the VEZ step when we make the incoming variational expectations "
    f"from the VES step match the truth. "
)
input("\nPress any key to continue")

VEZ_summaries = run_VEZ_step(
    CSP,
    ETP,
    IP,
    sample.xs,
    ground_truth_probs_for_system_regimes,
    init_dists_over_entity_regimes,
)

pct_corrects = np.empty(J)
for j in range(J):
    most_likely_system_regimes = np.argmax(VEZ_summaries[j].expected_regimes, axis=1)
    pct_corrects[j] = compute_regime_labeling_accuracy(most_likely_system_regimes, sample.zs[:, j])

log_normalizers = np.array([VEZ_summaries[j].log_normalizer for j in range(J)])

print(
    f"\nVEZ step's log normalizer by entities for continuous state emissions when we use ground truth inits for q(S): {log_normalizers}"
)
print(f"Percent correct classifications by entity {pct_corrects}")

VEZ_summaries = run_VEZ_step(
    CSP,
    ETP,
    IP,
    sample.xs,
    variationally_expected_system_regimes,
    init_dists_over_entity_regimes,
)

pct_corrects = np.empty(J)
for j in range(J):
    most_likely_system_regimes = np.argmax(VEZ_summaries[j].expected_regimes, axis=1)
    pct_corrects[j] = compute_regime_labeling_accuracy(most_likely_system_regimes, sample.zs[:, j])

log_normalizers = np.array([VEZ_summaries[j].log_normalizer for j in range(J)])

print(
    f"\nVEZ step's log normalizer by entities for continuous state emissions when we use uniform inits for q(S): {log_normalizers}"
)
print(f"Percent correct classifications by entity {pct_corrects}")
