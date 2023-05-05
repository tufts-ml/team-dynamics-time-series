import functools
import math
from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
import scipy
from scipy import stats
from scipy.special import (  # modified bessel of first kind
    iv as modified_bessel_first_kind,
)
from scipy.stats import vonmises

from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.util import make_2d_rotation_JAX, make_2d_rotation_matrix


"""
We try to learn von Mises parameters for points distributed on the circle.
The points can be either IID on the circle, or a Von Mises random walk on the circle.
See `VonMisesModelType`.
"""

###
# Structs
###


class VonMisesModelType:
    """
    Values:
        IID
            x_t ~ VonMises(loc, kappa)
        Random Walk
            x_t ~ VonMises(x_{t-1}, kappa)
    """

    IID = 1
    RANDOM_WALK = 2


@dataclass
class VonMisesParams:
    """
    Attributes:
        loc: an angle in [-pi, pi].
            This parameter is optional, because it not needed to model a Von Mises random walk (see `VonMisesModelType`)
        kappa: concentration parameter, a non-negative real.
            When kappa=0, the Von Mises distribution is uniform on the circle

    References:
        https://stats.stackexchange.com/questions/18692/estimating-kappa-of-von-mises-distribution
        https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    """

    loc: Optional[float]
    kappa: float


###
# Sampling
###


def sample_from_von_mises_random_walk(
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
    """
    Returns T samples from a von Mises random walk with drift,
        theta_t ~ VonMises(theta_{t-1} + angular_drift, kappa)
    We create the angular drift by rotating the previous value by the drift angle

    Arguments:
        kappa: concentration parameter for von mises distribution
        init_angle: In [-pi, pi]
        drift_angle: In [-pi, pi]
            If drift_angle = 0, this produces an identity rotaton matrix
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


###
# Helpers for Inference
###


def _compute_norm_of_mean_Cartesian_coordinate(angles: NumpyArray1D) -> float:
    """
    If {theta_i} are angles, then we convert to Cartesian coordinates {x_i}, and then compute
    the norm of the mean,  i.e. || mean x_i ||

    This is an intermediate quantity for estimating kappa, the concentration parameter
    for the von Mises distribution

    For reference, see https://stats.stackexchange.com/questions/18692/estimating-kappa-of-von-mises-distribution,
    where we might call this r_bar,  the Euclidean distance from the barycenter to the origin (i.e. the norm of the barycenter)
    r_bar = || mean x_i ||

    """
    points = points_from_angles(angles)
    return np.linalg.norm(np.mean(points, 0))


def points_from_angles(angles: NumpyArray1D) -> NumpyArray2D:
    """Compute the Cartesian coordinates of a point on the unit circle, given the angle."""
    xs = np.cos(angles)
    ys = np.sin(angles)
    return np.vstack((xs, ys)).T


def angles_from_points(points: NumpyArray2D) -> NumpyArray1D:
    """Compute the angle of a point on the unit circle, given its Cartesian coordinates."""
    return np.arctan2(points[:, 1], points[:, 0])


def equation_whose_root_is_the_kappa_MLE(kappa: float, RHS: float) -> float:
    """
    The root of this equation gives the maximum likelihood estimate for kappa.

    I_1(kappa)/I_0(kappa) = RHS

    where I_r() is the modified Bessel function of the first kind and order r.

    For iid samples, the RHS is given by the norm of the mean point, i.e. || mean x_i ||
    For a Von Mises random walk, the RHS is given by the mean of x_{t-1}^T x_t

    References
        https://stats.stackexchange.com/questions/18692/estimating-kappa-of-von-mises-distribution.
        https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    """
    return modified_bessel_first_kind(1, kappa) / modified_bessel_first_kind(0, kappa) - RHS


###
# Inference for Von Mises: IID or random walk (without drift)
###


def estimate_kappa_for_iid_samples(angles: np.array):
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    given iid samples from a Von Mises distribution (on the circle).  The computation of this equation can be obtained
    by the argument in Appendix A.1, Banerjee et al 2005 JMLR,  Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
    """
    norm_of_cartesian_mean = _compute_norm_of_mean_Cartesian_coordinate(angles)
    this_equation = functools.partial(
        equation_whose_root_is_the_kappa_MLE, RHS=norm_of_cartesian_mean
    )
    # TODO: Don't hardcode kappa init.
    kappa_init = 1.0
    return scipy.optimize.fsolve(this_equation, kappa_init)[0]


def estimate_kappa_for_random_walk(angles: np.array):
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    of a Von Mises random walk.  The computation of this equation can be obtained from MTW's notes,
    and/or by a slight tweak to the argument in Appendix A.1, Banerjee et al 2005 JMLR,
    Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
    """
    points = points_from_angles(angles)

    # TODO: We are ignoring the first measurement and estimating this from T-1 measurements.
    # We could also posit an initial measurement x_0 and then estimate this from T measurements.

    RHS = np.mean(np.einsum("td,td->t", points[1:], points[:-1]))
    this_equation = functools.partial(equation_whose_root_is_the_kappa_MLE, RHS=RHS)

    # TODO: Don't hardcode kappa init.
    kappa_init = 1.0
    return scipy.optimize.fsolve(this_equation, kappa_init)[0]


def estimate_von_mises_params(
    angles: np.array, model_type: VonMisesModelType = VonMisesModelType.IID
) -> VonMisesParams:
    """
    Arguments:
        angles: an np.array of shape (N,) where N is the number of examples.
            The n-th entry is in [-pi, pi]
    """

    # Compute the ML estimate for the concentration parameter
    if model_type == VonMisesModelType.IID:
        # Compute the MoM estimate for the mean direction
        loc = stats.circmean(angles, high=np.pi, low=-np.pi)
        kappa = estimate_kappa_for_iid_samples(angles)
        # TODO: Is it weird to use MOM for one estimate and ML for another?
    elif model_type == VonMisesModelType.RANDOM_WALK:
        loc = None
        kappa = estimate_kappa_for_random_walk(angles)
    else:
        raise ValueError("What model type do you want?")

    return VonMisesParams(loc, kappa)


###
# [WIP] Inference for Von Mises: random walk WITH drift
###


def negative_log_likelihood_up_to_a_constant_in_drift_angle_theta_JAX(theta, points):
    """
    Estimates rotation matrix R=R(theta) for the Von Mises random walk with drift:
        theta_t ~ VonMises(theta_{t-1} + angular_drift, kappa)

    The von mises log likelihood, up to a contant in the drift angle theta, is given by
        log_like(theta)= sum_{t=1}^T x_{t-1}^T R(theta) x_t  (*)
    for some choice of x_0 and where x_t \in R^2 is the "point" representation of the angle theta_t, i.e.
        x_t := [cos(theta_t), sin(theta_t)]

    The equation (*) is clear from the von Mises likelihood (presented in terms of points x.)
    For instance, see  Banerjee et al 2005 JMLR.

    Arguments:
        theta : drift angle
        points: has shape (T,2)
    """
    # Note: i'm discarding the parts of von mises log likelihood that are irrelevant to finding the drift
    # TODO: Can't I just optimize the above equation in closed form?
    R = make_2d_rotation_JAX(theta)
    next_points = points[1:]
    rotated_previous_points = (R @ points[:-1].T).T
    dot_products = jnp.einsum("td,td->t", next_points, rotated_previous_points)
    return -jnp.sum(dot_products)


###
# Initialization for inference for Von Mises (random walk WITH drift ) via an intuitive estimator
###


def _angle_between_zero_and_2pi(v1, v2):
    "returns value between 0 and 2pi"
    return math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compute_mean_angle_between_neighbors(angles):
    points = points_from_angles(angles)
    neighbor_angles = [_angle_between_zero_and_2pi(x, y) for (x, y) in zip(points[1:], points[:-1])]
    return np.mean(neighbor_angles)
