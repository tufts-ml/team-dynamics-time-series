import functools

import numpy as np
from dynamax.utils.optimize import run_gradient_descent

from dynagroup.von_mises.core import (
    compute_mean_angle_between_neighbors,
    negative_log_likelihood_up_to_a_constant_in_drift_angle_theta_JAX,
    points_from_angles,
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
drift_angles = np.array([1 / 24, 1 / 12, 1 / 4, 1 / 2, 1]) * np.pi

# optimization
optimizer_state = None
optimizer_init_strategy = "smart"  # possible values ["smart" or "zero"]
num_M_step_iters = 100


###
# Sample from a Von Mises Random Walk with Drift
###
for drift_angle in drift_angles:
    angles = sample_from_von_mises_random_walk_with_drift(kappa_true, T, init_angle, drift_angle)
    # mean of angles: np.mean(angles[1:]-angles[:1])

    ### (smart) initialization.
    # note that we get very bad results for large drift angles if we initialize at 0.
    if optimizer_init_strategy == "smart":
        mean_angle_between_neighbors = compute_mean_angle_between_neighbors(angles)
        optimizer_init_angle = mean_angle_between_neighbors
    elif optimizer_init_strategy == "zero":
        optimizer_init_angle = 0.0
    else:
        raise ValueError("What is your preferred optimizer initialization strategy?!")

    ###
    # Do inference on drift (rotation angle) for Von Mises Random Walk with Drift
    ###

    # TODO: can I get the solution without gradient descent on theta?!
    points = points_from_angles(angles)
    cost_function = functools.partial(
        negative_log_likelihood_up_to_a_constant_in_drift_angle_theta_JAX, points=points
    )

    (
        theta_hat,
        optimizer_state_new,
        losses,
    ) = run_gradient_descent(
        cost_function,
        optimizer_init_angle,
        optimizer_state=optimizer_state,
        num_mstep_iters=num_M_step_iters,
    )

    print(
        f"Kappa true {kappa_true:.03}. True drift {drift_angle:.03}. Estimated drift: {theta_hat:.03}. Optimizer init: {optimizer_init_angle :.03f}."
    )
    print(f"The mean angle between neighbors is {mean_angle_between_neighbors:.03f}")
