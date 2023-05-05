from typing import Optional

import numpy as np
from scipy.stats import vonmises

from dynagroup.types import NumpyArray1D
from dynagroup.util import make_2d_rotation_matrix
from dynagroup.von_mises.util import angles_from_points, points_from_angles


"""
We try to learn von Mises parameters for points distributed on the circle.
The points can be either IID on the circle, or a Von Mises random walk on the circle.
See `VonMisesModelType`.
"""


def sample_from_von_mises_random_walk(
    kappa: float,
    T: int,
    init_angle: float,
) -> NumpyArray1D:
    drift_angle = 0.0
    ar_coef = 1.0
    return sample_from_von_mises_AR_with_drift(kappa, T, ar_coef, init_angle, drift_angle)


def sample_from_von_mises_random_walk_ANOTHER_WAY(
    kappa: float,
    T: int,
    init_angle: float,
) -> NumpyArray1D:
    """
    Returns T samples from a von Mises random walk,
        theta_t ~ VonMises(theta_{t-1}, kappa)

    Arguments:
        kappa: concentration parameter for von mises distribution
        init_angle: In [-pi, pi]
    """

    angles = np.zeros(T)
    angles[0] = init_angle
    for t in range(1, T):
        angles[t] = vonmises.rvs(kappa, loc=angles[t - 1])
    return angles


def sample_from_von_mises_random_walk_with_drift(
    kappa: float,
    T: int,
    init_angle: float,
    drift_angle: Optional[float],
) -> NumpyArray1D:
    ar_coef = 1
    return sample_from_von_mises_AR_with_drift(kappa, T, ar_coef, init_angle, drift_angle)


def sample_from_von_mises_random_walk_with_drift_ANOTHER_WAY(
    kappa: float,
    T: int,
    init_angle: float,
    drift_angle: Optional[float],
) -> NumpyArray1D:
    """
    Returns T samples from a von Mises random walk with drift,
        theta_t ~ VonMises(theta_{t-1} + angular_drift, kappa)
    We create the angular drift by rotating the previous value by the drift angle

    Arguments:
        kappa: concentration parameter for von mises distribution
        init_angle: In [-pi, pi]
        drift_angle: In [-pi, pi]
            If drift_angle = 0, this produces an identity rotation matrix
            to operate on the previous observation
    """

    # The rotation matrix `R` is the identity if the drift angle is 0.0
    R = make_2d_rotation_matrix(drift_angle)

    angles = np.zeros(T)
    angles[0] = init_angle
    for t in range(1, T):
        mu_t_as_point = (R @ points_from_angles(angles[t - 1]).T).T
        mu_t = angles_from_points(mu_t_as_point)[0]
        angles[t] = vonmises.rvs(kappa, loc=mu_t)
    return angles


def sample_from_von_mises_AR_with_drift(
    kappa: float,
    T: int,
    ar_coef: float,
    init_angle: float,
    drift_angle: Optional[float],
) -> NumpyArray1D:
    """
    Returns T samples from a von Mises autoregression with drift,
        theta_t ~ VonMises(alpha*theta_{t-1} + angular_drift, kappa)
    We create the angular drift by rotating the previous value by the drift angle

    Arguments:
        kappa: concentration parameter for von mises distribution
        init_angle: In [-pi, pi]
        drift_angle: In [-pi, pi]
            If drift_angle = 0, this produces an identity rotaton matrix
            to operate on the previous observation
        ar_coef in [0,1].
            if 1 this is a random walk with drift.
            if 0 this is an IID model.
    """

    angles = np.zeros(T)
    angles[0] = init_angle
    for t in range(1, T):
        mu_t_as_angle = drift_angle + ar_coef * angles[t - 1]
        angles[t] = vonmises.rvs(kappa, loc=mu_t_as_angle)
    return angles
