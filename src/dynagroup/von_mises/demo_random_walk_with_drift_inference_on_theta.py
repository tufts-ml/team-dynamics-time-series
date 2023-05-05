import numpy as np


np.set_printoptions(precision=3, suppress=True)

from dynagroup.von_mises.core import (
    estimate_drift_angle_for_von_mises_random_walk_with_drift,
    sample_from_von_mises_random_walk_with_drift,
)


"""
We show that we can do inference (by gradient descent, at least for now) on the drift angle "theta"
from a Von Mises random walk with drift.

TODO:
- Add inference for the concentration parameter, kappa
- Integrate with the other inference code for IID and random walk without drift.  See "model_type" 
"""


###
# Configs
###

# sampling
T = 1000
kappa_true = 100.0
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

    print(
        f"Kappa true {kappa_true:.03}. True drift {true_drift_angle:.03}. Estimated drift: {estimated_drift_angle:.03}."
    )

    # # TMP: from a devel script
    # theta_eric=closed_form_ML_from_eric(points)
    # print(f"Eric's closed form solution is {theta_eric:.03f}")
