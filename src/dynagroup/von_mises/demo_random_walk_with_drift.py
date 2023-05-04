import numpy as np
from matplotlib import pyplot as plt

from dynagroup.von_mises.core import (
    points_from_angles,
    sample_from_von_mises_random_walk_with_drift,
)


"""
We sample from, and (eventually) do inference on, a von Mises random walk with drift.
"""

###
# Configs
###

T = 25
kappa_trues = [100, 10, 1]
plot_angle_time_series = False
plot_time_series_on_circle = True
init_angle = 0.0
drift_angle = np.pi / 25

for kappa_true in kappa_trues:
    ###
    # Sample from a Von Mises Random Walk with Drift
    ###
    angles = sample_from_von_mises_random_walk_with_drift(kappa_true, T, init_angle, drift_angle)

    ###
    # Plot samples
    ###

    if plot_angle_time_series:
        plt.scatter(np.arange(T), angles, c=np.arange(T), cmap="cool")
        plt.xlabel("time")
        plt.ylabel(r"angle $\in [-\pi, \pi]$")
        plt.tight_layout()
        plt.show()

    if plot_time_series_on_circle:
        points = points_from_angles(angles)
        plt.scatter(points[:, 0], points[:, 1], c=np.arange(len(points)), cmap="cool")
        plt.title(f"Samples of a von Mises random walk with drift (kappa={kappa_true:.02f})")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.show()

    # ####
    # # Estimate parameters
    # ###

    # params_learned = estimate_von_mises_params(angles, VonMisesModelType.IID)
    # print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
