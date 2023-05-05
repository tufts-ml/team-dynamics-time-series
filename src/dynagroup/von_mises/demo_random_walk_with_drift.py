import numpy as np


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from dynagroup.von_mises.generate import sample_from_von_mises_random_walk_with_drift
from dynagroup.von_mises.inference import VonMisesModelType, estimate_von_mises_params
from dynagroup.von_mises.util import points_from_angles


"""
Demo of a von Mises random walk with drift.

1. We show samples.
2. We show inference works. 
    - For the concentration parameter, kappa, we do "closed form" inference (up to numerically solving the root of an equation involving 
        modified bessel functions). 
    - For the drift parameter, we do gradient descent (at least for now; Eric seems to have worked out a closed-form solution, although
        I will probably generalize this anyways.)

TODO:
- Extend to where we have a weight on the random walk part (so we can recover IID as a special case.)

"""

###
# Configs
###

T = 25
kappa_trues = [100, 50, 1]
plot_angle_time_series = False
plot_time_series_on_circle = True
init_angle = 0.0
true_drift_angles = np.array([1 / 24, 1 / 12, 1 / 4, 1 / 2, 1]) * np.pi

for true_drift_angle in true_drift_angles:
    for kappa_true in kappa_trues:
        ###
        # Sample from a Von Mises Random Walk with Drift
        ###
        angles = sample_from_von_mises_random_walk_with_drift(
            kappa_true, T, init_angle, true_drift_angle
        )

        ###
        # Plot samples
        ###
        if plot_angle_time_series and true_drift_angle == true_drift_angles[0]:
            plt.scatter(np.arange(T), angles, c=np.arange(T), cmap="cool")
            plt.xlabel("time")
            plt.ylabel(r"angle $\in [-\pi, \pi]$")
            plt.tight_layout()
            plt.show()

        if plot_time_series_on_circle and true_drift_angle == true_drift_angles[0]:
            points = points_from_angles(angles)
            plt.scatter(points[:, 0], points[:, 1], c=np.arange(len(points)), cmap="cool")
            plt.title(
                f"Samples of a von Mises random walk with drift \n (drift angle={true_drift_angle:.02f}, kappa={kappa_true:.02f})"
            )
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.show()

        ####
        # Estimate parameters
        ###
        params_learned = estimate_von_mises_params(angles, VonMisesModelType.RANDOM_WALK_WITH_DRIFT)

        print(
            f"\nTrue kappa {kappa_true:.02f}. Estimated kappa : {params_learned.kappa : .02f}."
            f"\nTrue drift {true_drift_angle:.02f}. Estimated drift: {params_learned.drift :.02f}."
        )
