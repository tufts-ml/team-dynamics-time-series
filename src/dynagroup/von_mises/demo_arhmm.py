import numpy as np


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from dynagroup.plotting.paneled_series import plot_time_series_with_regime_panels
from dynagroup.von_mises.generate import sample_from_switching_von_mises_AR_with_drift
from dynagroup.von_mises.inference.ar import VonMisesParams
from dynagroup.von_mises.inference.arhmm import run_EM_for_von_mises_arhmm


"""
Goal: try to learn von mises AR-HMM when we don't know the true
emissions parameters or tpm.

TODO:
- Make the param specifications look more like what's in the VI module.
"""

###
# Configs
###

### Generation
emissions_params_by_regime_true = [
    VonMisesParams(drift=np.pi / 8, kappa=10, ar_coef=0.2),
    VonMisesParams(drift=-np.pi / 8, kappa=100, ar_coef=-0.5),
]
list_of_regime_id_and_num_timesteps = [(0, 100), (1, 100), (0, 100), (1, 100)]
init_angle = 0.0


### Inference
num_regimes = 2
self_transition_prob_init = 0.995
num_EM_iterations = 3

###
# Generate
###
angles = sample_from_switching_von_mises_AR_with_drift(
    emissions_params_by_regime_true,
    list_of_regime_id_and_num_timesteps,
    init_angle,
)

###
# Plot
###

# plt.scatter(angles[:-1], angles[1:])
# plt.show()

plt.plot([i for i in range(len(angles))], angles)
plt.ylabel("Angle (in [-pi,pi])")
plt.xlabel("Timestep")
plt.show()


###
# Inference
###

posterior_summary, emissions_params_by_regime_learned, transitions = run_EM_for_von_mises_arhmm(
    angles, num_regimes, self_transition_prob_init, num_EM_iterations
)

###
# Diagnostics
###

# TODO: This comparison could be plagued by label misalignment!
for k in range(num_regimes):
    print(
        f"\n---For regime {k}, the params after learning were {emissions_params_by_regime_learned[k]}."
    )
    print(f"The true params are {emissions_params_by_regime_true[k]}.")


###
# PLot results
###

# TODO: get actual viterbi, not marginal MAP estimats.
fitted_regime_sequence = np.argmax(posterior_summary.expected_regimes, 1)

plot_time_series_with_regime_panels(
    angles,
    fitted_regime_sequence,
)
plt.show()
