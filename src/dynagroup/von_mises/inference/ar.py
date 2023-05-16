import functools
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import scipy
from dynamax.utils.optimize import run_gradient_descent
from numpy import arctanh, tanh
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from scipy import stats
from scipy.special import (  # modified bessel of first kind
    iv as modified_bessel_first_kind,
)
from sklearn.linear_model import LinearRegression

from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.util import make_2d_rotation_JAX
from dynagroup.von_mises.patches import try_pomegranate_model_up_to_n_times
from dynagroup.von_mises.util import (
    force_angles_to_be_from_neg_pi_to_pi,
    points_from_angles,
)


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
            theta_t ~ VonMises(loc, kappa)
                for kappa >=0, loc in [-pi, pi]
        Random Walk
            theta_t ~ VonMises(theta_{t-1}, kappa)
                for kappa >=0
        Random Walk with Drift
            theta_t ~ VonMises(theta_{t-1} + + angular_drift, kappa)
                for angular_drift in [-pi, pi], kappa >=0,
        AUTOREGRESSION
            theta_t ~ VonMises( alpha *theta_{t-1} + angular_drift, kappa)
                for alpha in [-1,1], angular_drift in [-pi, pi], kappa >=0

    Note that the AUTOREGRESSION model generalizes all of the model types
        - If ar_coef==1 and drift!=0, then this is a random walk with drift.
        - If ar_coef==1 and drift_angle==0, then this is a random walk without drift.
        - If ar_coef==0, then this gives IID samples with location=drift.

    However we allow specification of special cases to aid in inference.
    For example, parameter inference on the full AR model, at least in the implementation as of 5/5/23, can become
    dicey when the angular_drift is large (greater than |pi/2| ) or when the AUTOREGRESSION coefficient
    is exactly 1.0 or -1.0
    """

    IID = 1
    RANDOM_WALK = 2
    RANDOM_WALK_WITH_DRIFT = 3
    AUTOREGRESSION = 4


@jdc.pytree_dataclass
class VonMisesParams:
    """
    Parameters for governing a Von Mises Autoregression with Drift
        theta_t ~ VonMises(ar_coef*theta_{t-1} + drift, kappa)

    Note the special cases:
        - If ar_coef==1 and drift!=0, then this is a random walk with drift.
        - If ar_coef==1 and drift_angle==0, then this is a random walk without drift.
        - If ar_coef==0, then this gives IID samples with location=drift.

    Attributes:
        drift: an angle in [-pi, pi].
        kappa: concentration parameter, a non-negative real.
            When kappa=0, the Von Mises distribution is uniform on the circle
        ar_coef : currently in {0,1}
            This parameter is optional. When set to 0, we get IID samples.
            it might make sense t have the ar coef be in [0,1], but i'm not sure.

    References:
        https://stats.stackexchange.com/questions/18692/estimating-kappa-of-von-mises-distribution
        https://www.jmlr.org/papers/volume6/banerjee05a/banerjee05a.pdf
    """

    drift: float
    kappa: float
    ar_coef: Optional[float] = 0.0


###
# Enforcers
###
def enforce_von_mises_params_to_have_kappa_at_least_unity(
    von_mises_params: VonMisesParams,
) -> VonMisesParams:
    kappa = von_mises_params.kappa
    if kappa <= 1.0:
        new_kappa = 1.0
        von_mises_params = VonMisesParams(
            von_mises_params.drift, new_kappa, von_mises_params.ar_coef
        )
    return von_mises_params


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


def _estimate_kappa_for_general_von_mises_model(RHS: float) -> float:
    """
    Estimate the concentration parameter, kappa, for a von mises model (IID, Random Walk, Random Walk with drift)
    Each model has a different RHS for the estimating equation

        c'(kappa)/c(kappa) - RHS =0

    where c(kappa) is the normalizing constant.  See Appendix A.1, Banerjee et al 2005 JMLR for the IID case,
    and my notes for the other cases.
    """
    this_equation = functools.partial(
        equation_whose_root_is_the_kappa_MLE,
        RHS=RHS,
    )
    # TODO: Don't hardcode kappa init.
    kappa_init = 1.0
    return scipy.optimize.fsolve(this_equation, kappa_init)[0]


def estimate_kappa_for_iid_samples_USING_THE_MINOR_BANERJEE_SIMPLIFICATION(
    angles: np.array,
) -> float:
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    given iid samples from a Von Mises distribution (on the circle).  The computation of this equation can be obtained
    by the argument in Appendix A.1, Banerjee et al 2005 JMLR,  Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.

    Remarks:
        The minor banerjee simplification just involves a cancelation between the numerator and the denominator.
        We leave `estimate_kappa_for_iid_samples` as the standard (unmarked) function because it utilizes a more
        obviously generalizable presentation of the logic in the function body
    """
    norm_of_cartesian_mean = _compute_norm_of_mean_Cartesian_coordinate(angles)
    return _estimate_kappa_for_general_von_mises_model(norm_of_cartesian_mean)


def estimate_kappa_for_iid_samples(
    angles: np.array, loc_estimate: float, sample_weights: Optional[np.array] = None
) -> float:
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    given iid samples from a Von Mises distribution (on the circle).  The computation of this equation can be obtained
    by the argument in Appendix A.1, Banerjee et al 2005 JMLR,  Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
    """
    return estimate_kappa_for_autoregression(
        angles, drift_angle=loc_estimate, ar_coef=0.0, sample_weights=sample_weights
    )


def estimate_kappa_for_random_walk(
    angles: np.array, sample_weights: Optional[np.array] = None
) -> float:
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    of a Von Mises random walk.  The computation of this equation can be obtained from MTW's notes,
    and/or by a slight tweak to the argument in Appendix A.1, Banerjee et al 2005 JMLR,
    Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
    """
    return estimate_kappa_for_autoregression(
        angles, drift_angle=0.0, ar_coef=1.0, sample_weights=sample_weights
    )


def estimate_kappa_for_random_walk_with_drift(
    angles: np.array, drift_angle: float, sample_weights: Optional[np.array] = None
) -> float:
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    of a Von Mises random walk with drift.  The computation of this equation can be obtained from MTW's notes,
    and/or by a slight tweak to the argument in Appendix A.1, Banerjee et al 2005 JMLR,
    Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.
    """
    return estimate_kappa_for_autoregression(
        angles, drift_angle, ar_coef=1.0, sample_weights=sample_weights
    )


def estimate_kappa_for_autoregression(
    angles: np.array,
    drift_angle: float,
    ar_coef: float,
    sample_weights: Optional[np.array] = None,
    min_kappa_allowed: float = 1.0,
) -> float:
    """
    We find the root of an equation whose root gives the MLE for the concentration parameter, kappa,
    of a Von Mises autoregression.  The computation of this equation can be obtained from MTW's notes,
    and/or by a slight tweak to the argument in Appendix A.1, Banerjee et al 2005 JMLR,
    Clustering on the Unit Hypersphere using von Mises-Fisher Distributions.

    Arguments:
        angles: array of shape (T,)
            Observations, which live in [-pi,pi]
        drift_angle: drift in terms of an angle in [-pi, pi]
        ar_coef: autoregressive coefficient on the previous angle, in [-1,1]
        sample_weights: optional array of shape (T,)
            By default, all observations are equally weights.
    """
    if sample_weights is None:
        sample_weights = np.ones_like(angles)

    observed_points = points_from_angles(angles)
    expected_angles = ar_coef * angles[:-1] + drift_angle
    expected_points = points_from_angles(expected_angles)

    # TODO: We are ignoring the first measurement and estimating this from T-1 measurements.
    # We could also posit an initial measurement x_0 and then estimate this from T measurements.
    #
    # RK: What we are reallyd doing here is a mean inner product between observations and expectations.
    #   RHS = np.mean(np.einsum("td,td->t", observed_points[1:], expected_points))
    # but we are allowing for differential sample weightings.

    inner_product_over_time = np.einsum("td,td->t", observed_points[1:], expected_points)
    RHS = np.dot(inner_product_over_time, sample_weights[1:]) / np.sum(sample_weights[1:])

    kappa_estimated = _estimate_kappa_for_general_von_mises_model(RHS)

    if kappa_estimated > min_kappa_allowed:
        return kappa_estimated
    else:
        return min_kappa_allowed


###
# General function!
###


def estimate_von_mises_params(
    angles: np.array,
    model_type: VonMisesModelType = VonMisesModelType.IID,
    ar_coef_init: Optional[float] = None,
    drift_angle_init: Optional[float] = None,
    sample_weights: Optional[np.array] = None,
    suppress_warnings: bool = False,
    fix_ar_kappa_to_unity_rather_than_estimate: bool = False,
) -> VonMisesParams:
    """
    Arguments:
        angles: an np.array of shape (N,) where N is the number of examples.
            The n-th entry is in [-pi, pi]
        ar_coef_init : If this and `drift_angle_init` are none, we use a smart initialization.
            But when we are calling this as the M-step in an EM algo,
            we have an init from the previous iteration.
        drift_init : If this and `ar_coef_init` are none, we use a smart initialization.
            But when we are calling this as the M-step in an EM algo,
            we have an init from the previous iteration.
    """

    # Compute the ML estimate for the concentration parameter
    if model_type == VonMisesModelType.IID:
        if sample_weights is not None:
            raise NotImplementedError(
                "Need to implement circular mean with sample weighting; implement or try autoregression model."
            )

        # Compute the MoM estimate for the mean direction
        loc = stats.circmean(angles, high=np.pi, low=-np.pi)
        kappa = estimate_kappa_for_iid_samples(angles, loc, sample_weights=sample_weights)
        # TODO: Is it weird to use MOM for one estimate and ML for another?
        return VonMisesParams(loc, kappa=kappa)
    elif model_type == VonMisesModelType.RANDOM_WALK:
        drift = 0.0
        kappa = estimate_kappa_for_random_walk(angles)
        return VonMisesParams(drift, kappa=kappa)
    elif model_type == VonMisesModelType.RANDOM_WALK_WITH_DRIFT:
        if sample_weights is not None:
            raise NotImplementedError(
                "Need to implement sample weighting for random walk with drift; implement or try autoregression model."
            )

        drift = estimate_drift_angle_for_von_mises_random_walk_with_drift(angles)
        kappa = estimate_kappa_for_random_walk_with_drift(
            angles, drift, sample_weights=sample_weights
        )
        return VonMisesParams(drift, kappa=kappa)
    elif model_type == VonMisesModelType.AUTOREGRESSION:
        drift, ar_coef = estimate_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
            angles,
            num_coord_ascent_iterations=10,
            ar_coef_init=ar_coef_init,
            drift_angle_init=drift_angle_init,
            sample_weights=sample_weights,
            suppress_warnings=suppress_warnings,
        )
        if fix_ar_kappa_to_unity_rather_than_estimate:
            kappa = 1.0
        else:
            # RK: kappa can be poorly estimated - even negative! - if drift and ar_coef are poorly estimated.
            kappa = estimate_kappa_for_autoregression(
                angles, drift, ar_coef, sample_weights=sample_weights
            )
        return VonMisesParams(drift, kappa, ar_coef)
    else:
        raise ValueError("What model type do you want?")


###
# Inference for drift angle for Von Mises random walk WITH drift
###


def negative_log_likelihood_up_to_a_constant_in_drift_angle_theta_JAX(drift_angle, points):
    """
    Estimates rotation matrix R=R(theta) for the Von Mises random walk with drift:
        theta_t ~ VonMises(theta_{t-1} + drift_angle, kappa)

    The von mises log likelihood, up to a contant in the drift angle theta, is given by
        log_like(theta)= sum_{t=1}^T x_t^T R(theta) x_{t-1}  (*)
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
    R = make_2d_rotation_JAX(drift_angle)
    next_points = points[1:]
    rotated_previous_points = (R @ points[:-1].T).T
    dot_products = jnp.einsum("td,td->t", next_points, rotated_previous_points)
    return -jnp.sum(dot_products)


def estimate_drift_angle_for_von_mises_random_walk_with_drift(
    angles: NumpyArray1D,
    num_M_step_iters: int = 100,
    optimizer_init_strategy: Optional[str] = "smart",
    verbose: Optional[bool] = False,
) -> float:
    """
    Do inference on drift (rotation angle) for Von Mises Random Walk with Drift.
    Uses gradient descent.

    Arguments:
        optimizer_init_strategy: str, in ["smart", "zero"].
            "zero" does not work well when the true angle is large; it's just kept here for the purposes of demonstrating
            that gradient descent is sensitive to inits (or possibly just need to run it longer.)
    """

    ###
    # Initialization (by default, a smart one)
    ###

    # note that we get very bad results for large drift angles if we initialize at 0.
    if optimizer_init_strategy == "smart":
        mean_angle_between_neighbors = compute_mean_angle_between_neighbors(angles)
        optimizer_init_angle = mean_angle_between_neighbors
    elif optimizer_init_strategy == "zero":
        # This performs poorly; it's just here to illustrate that.
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
        estimated_drift_angle,
        optimizer_state_new,
        losses,
    ) = run_gradient_descent(
        cost_function,
        optimizer_init_angle,
        optimizer_state=None,
        num_mstep_iters=num_M_step_iters,
    )
    if verbose:
        print(f"The first 5 losses were {losses[:5]}. The last 5 losses were {losses[-5:]}.")

    return estimated_drift_angle


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


###
# Inference for drift angle and AUTOREGRESSION coefficient for Von Mises autoregression
###


def equation_whose_root_is_the_drift_MLE_given_ar_coef(
    drift: float, ar_coef: float, angles: NumpyArray1D, sample_weights: NumpyArray1D
) -> float:
    points = points_from_angles(angles)
    weighted_sum = 0.0
    T = len(points)
    for t in range(1, T):
        sample_contribution = -points[t, 0] * np.sin(drift + ar_coef * angles[t - 1]) + points[
            t, 1
        ] * np.cos(drift + ar_coef * angles[t - 1])
        weighted_sample_contribution = sample_contribution[0] * sample_weights[t]
        weighted_sum += weighted_sample_contribution
    return weighted_sum


def equation_whose_root_is_the_arctanh_ar_coef_MLE_given_drift(
    arctanh_ar_coef: float, drift: float, angles: NumpyArray1D, sample_weights: NumpyArray1D
) -> float:
    ar_coef = tanh(arctanh_ar_coef)
    points = points_from_angles(angles)
    weighted_sum = 0.0
    T = len(points)
    for t in range(1, T):
        sample_contribution = -points[t, 0] * angles[t - 1] * np.sin(
            drift + ar_coef * angles[t - 1]
        ) + points[t, 1] * angles[t - 1] * np.cos(drift + ar_coef * angles[t - 1])
        weighted_sample_contribution = sample_contribution[0] * sample_weights[t]
        weighted_sum += weighted_sample_contribution
    return weighted_sum


@dataclass
class GMM_Results:
    num_components: int
    log_like: float
    responsibilities: Optional[NumpyArray2D] = None


def fit_Gaussian(X, sample_weights):
    """
    Arguments:
        X: array of size (T, D)
        sample_weights: array of size (T,)
    """
    gmm1 = Normal().fit(X, sample_weight=sample_weights)
    log_like = gmm1.log_probability(X).sum()
    return GMM_Results(1, log_like)


def fit_two_component_Gaussian_mixture(X, sample_weights) -> Optional[GMM_Results]:
    """
    Arguments:
        X: array of size (T, D)
        sample_weights: array of size (T,)
    """
    gmm2 = GeneralMixtureModel([Normal(), Normal()], init="random").fit(
        X, sample_weight=sample_weights
    )
    if gmm2 is not None:
        log_like = gmm2.log_probability(X).sum()
        responsibilities = np.asarray(gmm2.predict_proba(X))
        return GMM_Results(2, log_like, responsibilities)
    else:
        return None


def smart_initialize_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
    angles: NumpyArray1D, sample_weights: NumpyArray1D
) -> Tuple[float, float]:
    f"""
    The smart initialization iniitalizes the parameters for the expected value of the autoregression
        theta_t = drift + ar_coef * theta_t-1
    but it handles the "phase-wrapping" problem of theta in [-pi, pi] as follows:

    We try to detect multiple clusters (currently only seek up to 2, and currently use a heuristic to answer the question).
    If multiple clusters are detected, we fit separate linear regressions to each cluster,
    extract the parameters, and take a weighted sum of the parameters to get our initialized parameters.
    """
    # TODO: I think this method will fail if there are more than two clusters.  This can happen
    # when we have extreme (i.e. near-boundary) values for ar-coef or drift.

    # TODO: We are using this detection/transformation procedure on clusters to handle the problem of
    # wrapping.  But there is probably a more natural way to do this.  Is there a circular regression approach
    # that we could use to help us estimate y=a+bx, for y,x on unit circle? Or should we move to a different
    # representation.

    ###
    # Check if two clusters
    ###
    predictors, responses = angles[:-1], angles[1:]
    T_minus_1 = len(predictors)
    X = np.vstack([predictors, responses]).T

    # pomegranate works with torch, and so needs float32's
    X = X.astype(np.float32)
    sample_weights = sample_weights.astype(np.float32)

    ###
    # Fit Gaussian mixtures via pomegranate library.
    ###

    # TODO: This is so ugly and hacky.  I just need an api which lets me fit a Gaussian mixture model
    # with sample weights, so that I can grab the evidence and responsibilities.
    # Is there an alternative to pomegranate, which errors out easily?
    # sklearn doesn't let me provide sample weights - although perhaps there's a workaround?
    #
    # We add some noise b/c sometimes the inverse covariance matrix is singular, which causes an error to be raised
    # when fitting the 2-component Gaussian mixture.

    # noisy_sample_weights = sample_weights + np.ones(len(sample_weights), dtype=np.float32) * 0.05
    # noisy_X = X + np.random.normal(loc=0.0, scale=0.1, size=X.shape)

    # Rk: I wanted to just use sklearn, as per below, but sklearn's GaussianMixture().fit() method doesn't support sample weights

    gmm1_results = try_pomegranate_model_up_to_n_times(fit_Gaussian, n=10)(X, sample_weights[1:])
    gmm2_results = try_pomegranate_model_up_to_n_times(fit_two_component_Gaussian_mixture, n=10)(
        X, sample_weights[1:]
    )
    if gmm2_results is None:
        gmm2_results = GMM_Results(2, -np.inf, None)

    # TODO: This is an arbitrary way to decide if there are 2 clusters. Come up with something more principled.
    THRESHOLD_IN_MEAN_LOG_LIKE_INCREASE_FROM_TWO_COMPONENTS = 0.25
    mean_log_like_increase_from_two_components = (
        gmm2_results.log_like - gmm1_results.log_like
    ) / T_minus_1

    there_are_two_clusters = (
        mean_log_like_increase_from_two_components
        > THRESHOLD_IN_MEAN_LOG_LIKE_INCREASE_FROM_TWO_COMPONENTS
    )

    if not there_are_two_clusters:
        reg = LinearRegression().fit(
            predictors[:, None], responses, sample_weight=sample_weights[1:]
        )
        ar_coef = reg.coef_[0]
        drift = reg.intercept_
    else:
        K = 2
        ar_coefs = np.zeros(K)
        drifts = np.zeros(K)
        n_effectives = np.zeros(K)
        for k in range(K):
            reg = LinearRegression().fit(
                predictors[:, None],
                responses,
                sample_weight=sample_weights[1:] * gmm2_results.responsibilities[:, k],
            )
            ar_coefs[k] = reg.coef_[0]
            drifts[k] = force_angles_to_be_from_neg_pi_to_pi(
                reg.intercept_
            )  # may need to move into [-pi, pi] if outside of it.
            n_effectives[k] = np.sum(gmm2_results.responsibilities[:, k])

        cluster_weights = n_effectives / np.sum(n_effectives)
        ar_coef = np.sum(ar_coefs * cluster_weights)
        drift = np.sum(drifts * cluster_weights)

    return ar_coef, force_angles_to_be_from_neg_pi_to_pi(drift)


def estimate_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
    angles: NumpyArray1D,
    num_coord_ascent_iterations: int,
    ar_coef_init: Optional[float] = None,
    drift_angle_init: Optional[float] = None,
    sample_weights: Optional[NumpyArray1D] = None,
    verbose=False,
    suppress_warnings=False,
) -> Tuple[float, float]:
    """
    Do inference on drift (rotation angle) for Von Mises Random Walk with Drift.
    Uses gradient descent

    Arguments:
        angles: np.array of shape (T,)
        sample_weights: np.array of shape (T,)
    """
    if suppress_warnings:
        warnings.filterwarnings("ignore")

    warnings.warn(
        f" The strategy used here is numerical optimization.  It can sometimes struggle when (1) the drift angle is very "
        f" close to -pi or pi (these are equal) and/or (2) the ar coefficient is EXACTLY 1.0 or -1.0.  "
        f" and/or (3) kappa is very low (i.e. there is too much noise), especially if these are combined."
        f""
        f" Problem (2) may be related to fact that we use the hyperbolic tangent to map from [-1,1] to unconstrained space. "
        f" If the AR coefficient = 1.0 exactly, we have a random walk (with or without drift); there are separate "
        f" inference functions for those models that are more reliable.  Just specify the appropriate Model in the "
        f" model enum. Note that in these cases the initialization can be WAY off. But note that problem (2) is not "
        f" a problem in isolation. "
        f""
        f" More to the point, look at the smart initialization (via linear regression) for these boundary cases."
        f" We have worked to address the wrapping problem (where -pi==pi), but weird things can happen near the boundary "
        f" (Like a linear regression with four clusters).  Perhaps we can initialize outside theta space?!"
    )

    if sample_weights is None:
        sample_weights = np.ones_like(angles)

    ###
    # Smart Init via Linear Regression
    ###
    # TODO: Do I just need the intercept from linear regression? If so that seems easier
    if ar_coef_init is None or drift_angle_init is None:
        ar_coef, drift = smart_initialize_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
            angles, sample_weights
        )
        print(f"After smart initialization: ar_coef:{ar_coef:.02f}, drift:{drift:.02f}")
    else:
        ar_coef, drift = ar_coef_init, drift_angle_init

    ###
    # Do inference on drift (rotation angle) for Von Mises Random Walk with Drift
    ###
    for it in range(num_coord_ascent_iterations):
        this_drift_equation = functools.partial(
            equation_whose_root_is_the_drift_MLE_given_ar_coef,
            ar_coef=ar_coef,
            angles=angles,
            sample_weights=sample_weights,
        )
        drift = scipy.optimize.fsolve(this_drift_equation, drift)[0]
        this_ar_coef_equation = functools.partial(
            equation_whose_root_is_the_arctanh_ar_coef_MLE_given_drift,
            drift=drift,
            angles=angles,
            sample_weights=sample_weights,
        )
        arctanh_ar_coef = scipy.optimize.fsolve(this_ar_coef_equation, arctanh(ar_coef))[0]
        ar_coef = tanh(arctanh_ar_coef)
        if verbose:
            print(
                f"it: {it+1}/{num_coord_ascent_iterations}, drift:{drift:.02f}, ar_coef:{ar_coef:.02f}"
            )

    return drift, ar_coef
