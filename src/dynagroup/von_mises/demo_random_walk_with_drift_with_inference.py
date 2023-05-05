import numpy as np


np.set_printoptions(precision=3, suppress=True)

from dynagroup.von_mises.core import (
    estimate_drift_angle_for_von_mises_random_walk_with_drift,
    estimate_kappa_for_random_walk_with_drift,
    sample_from_von_mises_random_walk_with_drift,
)


"""
We show that we can do inference (by gradient descent, at least for now) on the drift angle "theta"
from a Von Mises random walk with drift.

TODO:
- Integrate with the other inference code for IID and random walk without drift.  See "model_type" 
- Extend to where we have a weight on the random walk part (so we can recover IID as a special case.)
"""


###
# Configs
###

# sampling
T = 1000
kappa_true = 100
plot_angle_time_series = False
plot_time_series_on_circle = True
init_angle = 0.0
true_drift_angles = np.array([1 / 24, 1 / 12, 1 / 4, 1 / 2, 1]) * np.pi

# optimization
optimizer_state = None
optimizer_init_strategy = "smart"  # possible values ["smart" or "zero"]
num_M_step_iters = 100


###
# Sample from a Von Mises Random Walk with Drift
###
for true_drift_angle in true_drift_angles:
    angles = sample_from_von_mises_random_walk_with_drift(
        kappa_true, T, init_angle, true_drift_angle
    )
    estimated_drift_angle = estimate_drift_angle_for_von_mises_random_walk_with_drift(
        angles,
        num_M_step_iters,
        optimizer_init_strategy,
    )

    estimated_kappa = estimate_kappa_for_random_walk_with_drift(angles, estimated_drift_angle)

    print(
        f"\nTrue kappa {kappa_true:.02f}. Estimated kappa : {estimated_kappa : .02f}."
        f"\nTrue drift {true_drift_angle:.02f}. Estimated drift: {estimated_drift_angle:.02f}."
    )

    # # TMP: from a devel script
    # theta_eric=closed_form_ML_from_eric(points)
    # print(f"Eric's closed form solution is {theta_eric:.03f}")
