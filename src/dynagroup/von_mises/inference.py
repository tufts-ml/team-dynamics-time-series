import contextlib
import functools
import io
import math
import warnings
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

from dynagroup.types import NumpyArray1D
from dynagroup.util import make_2d_rotation_JAX
from dynagroup.von_mises.util import (
    force_angles_to_be_from_neg_pi_to_pi,
    points_from_angles,
)


# In order to Create a file-like object that discards output from external libraries
# with io.StringIO() as f, contextlib.redirect_stdout(f):


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

    return _estimate_kappa_for_general_von_mises_model(RHS)


###
# General function!
###


def estimate_von_mises_params(
    angles: np.array,
    model_type: VonMisesModelType = VonMisesModelType.IID,
    sample_weights: Optional[np.array] = None,
) -> VonMisesParams:
    """
    Arguments:
        angles: an np.array of shape (N,) where N is the number of examples.
            The n-th entry is in [-pi, pi]
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
            angles, num_coord_ascent_iterations=10, sample_weights=sample_weights
        )
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


def are_there_two_clusters_when_plotting_consecutive_angles(
    angles: NumpyArray1D, sample_weights: NumpyArray1D
):
    predictors, responses = angles[:-1], angles[1:]
    T_minus_1 = len(predictors)
    X = np.vstack([predictors, responses]).T

    # pomegranate works with torch, and so needs float32's
    X = X.astype(np.float32)
    sample_weights = sample_weights.astype(np.float32)

    with io.StringIO() as f, contextlib.redirect_stdout(f):
        # Suppress the output from this external library
        gmm1 = Normal().fit(X, sample_weight=sample_weights[1:])
        ll_one_component = gmm1.log_probability(X).sum()

        gmm2 = GeneralMixtureModel([Normal(), Normal()], verbose=True).fit(
            X, sample_weight=sample_weights[1:]
        )
        ll_two_components = gmm2.log_probability(X).sum()

        # RK: I wanted to just use sklearn, as per below, but sklearn's GaussianMixture().fit() method doesn't support sample weights
        # gmm1 = GaussianMixture(n_components=1)
        # gmm1.fit(X)
        # ll_one_component = np.sum(gmm1.score_samples(X))

        # gmm2 = GaussianMixture(n_components=2)
        # gmm2.fit(X)
        # ll_two_components = np.sum(gmm2.score_samples(X))

    THRESHOLD_IN_MEAN_LOG_LIKE_INCREASE_FROM_TWO_COMPONENTS = 0.25
    mean_log_like_increase_from_two_components = (ll_two_components - ll_one_component) / T_minus_1

    return (
        mean_log_like_increase_from_two_components
        > THRESHOLD_IN_MEAN_LOG_LIKE_INCREASE_FROM_TWO_COMPONENTS
    )


def make_transformed_predictors_and_responses_for_two_clusters(
    angles: NumpyArray1D,
) -> Tuple[NumpyArray1D, NumpyArray1D]:
    """
    The gist of this function is to do initialization via linear regression:
        theta_t = drift + ar_coef theta_{t-1}
    However, scatterplots of theta_t against theta_{t-1} reveal a wrapping problem due to angle domain being [-pi,pi].
    As soon as the response gets above pi, it wraps back down to -pi, destroying the linear relationship.  (Indeed,
    plots may reveal 2 lines -- or possibly even more)

    As a fix, we find the minimum response from the first 10% of the predictors, use that to establish the relationship,
    and add 2pi to any responses that are below that minimum.
    """
    predictors, responses = angles[:-1], angles[1:]
    idxs = np.argsort(predictors)
    predictors_sorted_by_predictor = predictors[idxs]
    responses_sorted_by_predictor = responses[idxs]

    T = len(predictors)
    T_init = int(T / 10)
    response_cutoff = np.min(responses_sorted_by_predictor[:T_init])

    transform_response = lambda x: x + (2 * np.pi) if x < response_cutoff else x
    responses_sorted_by_predictor_transformed = np.array(
        [transform_response(y) for y in responses_sorted_by_predictor]
    )

    if np.min(responses_sorted_by_predictor_transformed) != response_cutoff:
        raise ValueError(
            f"The smart initialization failed.  Check if the scatterplot of previous and subsequent angles "
            "has more than two lines."
        )
    return predictors_sorted_by_predictor, responses_sorted_by_predictor_transformed


def smart_initialize_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
    angles: NumpyArray1D, sample_weights: NumpyArray1D
) -> Tuple[float, float]:
    # TODO: Think about whether the determination of two clusters should use the sample weights.
    # My current feeling is no.

    # TODO: We are using this detection/transformation procedure on clusters to handle the problem of
    # wrapping.  But there is probably a more natural way to do this.  Is there a circular regression approach
    # that we could use to help us estimate y=a+bx, for y,x on unit circle? Or should we move to a different
    # representation.

    two_clusters = are_there_two_clusters_when_plotting_consecutive_angles(angles, sample_weights)

    if two_clusters:
        predictors, responses = make_transformed_predictors_and_responses_for_two_clusters(angles)
    else:
        predictors, responses = angles[:-1], angles[1:]

    reg = LinearRegression().fit(predictors[:, None], responses, sample_weight=sample_weights[1:])
    ar_coef = reg.coef_[0]
    drift = force_angles_to_be_from_neg_pi_to_pi(
        reg.intercept_
    )  # may need to move into [-pi, pi] if outside of it.
    return ar_coef, drift


def estimate_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
    angles: NumpyArray1D,
    num_coord_ascent_iterations: int,
    sample_weights: Optional[NumpyArray1D] = None,
    verbose=False,
) -> Tuple[float, float]:
    """
    Do inference on drift (rotation angle) for Von Mises Random Walk with Drift.
    Uses gradient descent

    Arguments:
        angles: np.array of shape (T,)
        sample_weights: np.array of shape (T,)
    """
    warnings.warn(
        f" The strategy used here is numerical optimization.  It appears to struggle when (1) the drift angle is too "
        f" close to -pi or pi (these are equal) and/or (2) when the ar coefficient is EXACTLY 1.0 or -1.0.  "
        f" and/or (3) when kappa is too low (i.e. there is too much noise), especially if (1) or (2) are challenging."
        f""
        f" Problem (2) may be related to fact that we use the hyperbolic tangent to map from [-1,1] to unconstrained space. "
        f" If the AR coefficient = 1.0 exactly, we have a random walk (with or without drift); there are separate "
        f" inference functions for those models that are more reliable.  Just specify the appropriate Model in the "
        f" model enum. Note that in these cases the initialization can be WAY off."
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

    ar_coef, drift = smart_initialize_drift_angle_and_ar_coef_for_von_mises_ar_with_drift(
        angles, sample_weights
    )

    print(f"After smart initialization: ar_coef:{ar_coef:.02f}, drift:{drift:.02f}")

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
