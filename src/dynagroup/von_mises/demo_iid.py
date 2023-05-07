import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import vonmises

from dynagroup.von_mises.inference.ar import (
    VonMisesModelType,
    estimate_von_mises_params,
)
from dynagroup.von_mises.util import points_from_angles


"""
We sample from, and do inference on, a von Mises random walk.
"""
###
# Configs
###

T = 1000
locs_true = [np.pi / 4, np.pi / 2]
kappas_true = [1.0, 20.0, 100.0]
plot_angle_time_series = False
plot_time_series_on_circle = True

###
# Sample from a Von Mises
###

for loc_true in locs_true:
    for kappa_true in kappas_true:
        angles = vonmises.rvs(kappa_true, loc=loc_true, size=T)

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
            plt.title(f"Samples on the circle (loc={loc_true:.02f}, kappa={kappa_true:.02f})")
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.show()

        ####
        # Estimate parameters
        ###

        params_learned = estimate_von_mises_params(angles, VonMisesModelType.IID)
        print(f"True loc: {loc_true:.02f}, Estimated: {params_learned.drift:.02f}")
        print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
