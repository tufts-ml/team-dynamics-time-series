import matplotlib.pyplot as plt
import numpy as np

from dynagroup.von_mises.generate import sample_from_von_mises_AR_with_drift
from dynagroup.von_mises.inference import VonMisesModelType, estimate_von_mises_params
from dynagroup.von_mises.util import points_from_angles, two_angles_are_close


###
# Configs
###

T = 1000
kappa_true = 100
plot_points_on_circle = True
plot_angle_time_series = True

###
# Inference
###


### Suspected Limitations:
# Can't reliably do inference if ...
# 1) true ar coef is exactly 1 or -1. This is possibly
# due to the hyperbolic tangent tranformation.  If we want to have a random walk or random
# walk with drift for sure, maybe use dedicated inference for those models.
# 2) drift angle are too close to pi or -pi (which are equal).
# 3) kappa is too low (i.e. variance too high), especially if ar coef is high.

for drift_true in np.array([-0.8, 0.0, 0.8]) * np.pi:
    for ar_coef_true in [-0.99, 0.0, 0.99]:  # [-0.9, -0.5, 0.0, 0.5, 0.9]:
        print("\n --- New test ---")
        print(f"ar coef true:{ar_coef_true:.02f}, drift true:{drift_true:.02f}")

        init_angle = drift_true
        angles = sample_from_von_mises_AR_with_drift(
            kappa_true, T, ar_coef_true, init_angle, drift_true
        )
        points = points_from_angles(angles)

        ###
        # PLOTTING
        ###
        if plot_angle_time_series:
            T_to_plot = 100
            plt.scatter(
                np.arange(T_to_plot), angles[-T_to_plot:], c=np.arange(T_to_plot), cmap="cool"
            )
            plt.xlabel("time")
            plt.ylabel(r"angle $\in [-\pi, \pi]$")
            plt.tight_layout()
            plt.show()

        if plot_points_on_circle:
            T_to_plot = 25
            plt.scatter(
                points[-T_to_plot:, 0], points[-T_to_plot:, 1], c=np.arange(T_to_plot), cmap="cool"
            )
            plt.title(
                f"Last 25 samples of a von Mises AR with drift \n (drift angle={drift_true:.02f}, ar_coef = {ar_coef_true:02f}, kappa={kappa_true:.02f})"
            )
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.show()

        params = estimate_von_mises_params(angles, VonMisesModelType.AUTOREGRESSION)

        print(f"ar coef true:{ar_coef_true:.02f}, learned:{params.ar_coef:.02f}")
        print(f"drift true:{drift_true:.02f}, learned:{params.drift:.02f}")
        print(f"kappa true:{kappa_true:.02f}, learned:{params.kappa:.02f}")
        assert np.isclose(params.ar_coef, ar_coef_true, atol=0.20)
        assert two_angles_are_close(params.drift, drift_true, atol=np.pi / 16)
        assert np.isclose(params.kappa, kappa_true, rtol=0.20)
