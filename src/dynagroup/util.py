import warnings

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.scipy.special import logsumexp as logsumexp_JAX
from scipy.special import logsumexp

from dynagroup.types import NumpyArray2D


###
# Normalization (to simplex) and Log Normalization
###

# We import `logsumexp` from jax instead of doing
# so that we can use the functions here when doing
# numerical optimization procedures on parameters.


def normalize_log_potentials(log_potentials: NumpyArray2D) -> NumpyArray2D:
    """
    Arguments:
        log_potentials:  A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED log probability
            of transitioning from state k to state k'

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
            of transitioning from state k to state k'
    """
    log_normalizer = logsumexp(log_potentials, axis=1)
    return log_potentials - log_normalizer[:, None]


def normalize_log_potentials_by_axis(log_potentials: np.array, axis: int) -> np.array:
    log_normalizer = logsumexp(log_potentials, axis)
    return log_potentials - np.expand_dims(log_normalizer, axis)


def normalize_log_potentials_by_axis_JAX(log_potentials: jnp.array, axis: int) -> jnp.array:
    log_normalizer = logsumexp_JAX(log_potentials, axis)
    return log_potentials - jnp.expand_dims(log_normalizer, axis)


def normalize_potentials_by_axis(potentials: np.array, axis: int) -> np.array:
    log_probs = normalize_log_potentials_by_axis(np.log(potentials), axis)
    return np.exp(log_probs)


def normalize_potentials_by_axis_JAX(potentials: jnp.array, axis: int) -> jnp.array:
    log_probs = normalize_log_potentials_by_axis_JAX(jnp.log(potentials), axis)
    return jnp.exp(log_probs)


###
# Generate random objects
###


def generate_random_covariance_matrix(dim, var=1.0):
    A = np.random.randn(dim, dim) * np.sqrt(var)
    return np.dot(A, A.transpose())


def make_2d_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def make_2d_rotation_JAX(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])


def random_rotation(n, theta=None):
    """
    From Linderman's state space modeling repo

    Reference:
        https://github.com/lindermanlab/ssm/blob/master/ssm/util.py#L73-L86

    Remark:
        It's not random in 2dim.
    """
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = make_2d_rotation_matrix(theta)
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


###
# TPMs
###


def make_fixed_sticky_tpm(self_transition_prob: float, num_states: int) -> np.array:
    if num_states == 1:
        warnings.warn(
            "Sticky tpm has only 1 state; ignoring self transition prob and creating a `1` matrix."
        )
        return np.array([[1]])
    external_transition_prob = (1.0 - self_transition_prob) / (num_states - 1)
    return (
        np.eye(num_states) * self_transition_prob
        + (1.0 - np.eye(num_states)) * external_transition_prob
    )


def make_fixed_sticky_tpm_JAX(self_transition_prob: float, num_states: int) -> jnp.array:
    if num_states == 1:
        warnings.warn(
            "Sticky tpm has only 1 state; ignoring self transition prob and creating a `1` matrix."
        )
        return jnp.array([[1]])
    external_transition_prob = (1.0 - self_transition_prob) / (num_states - 1)
    return (
        jnp.eye(num_states) * self_transition_prob
        + (1.0 - jnp.eye(num_states)) * external_transition_prob
    )


###
# Represent transition probability matrices (tpm) in unconstrained manner
###


def unconstrained_tpm_from_tpm(tpm_or_tpms: jnp.array) -> jnp.array:
    """
    Represents a tpm over K destinations in unconstrained space R^{K-1} through a bijection
    If X is the unconstrained representation of a pmf and Y is the pmf, we can write
        Y = g(X) = exp([X 0]) / sum(exp([X 0])).
    i.e., we identify the softmax by forcing the last potential to be 0.
    See https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/bijectors/softmax_centered.py#LL34C30-L34C72.

    Arguments:
        tpm_or_tpms: A jnp.array whose last two axes give a  tpm (a transition probability matrix), of size (K,K) (say),
            whose (k,k')-th entry gives the probability of transitioning FROM the k-th state
            to the k'-th state, and where each k-th row is constrained to live on the simplex.

            So for instance, `tpm_or_tpms` could have shape (J,L,K,K),
            where tpm_or_tpms[j,l] gives a KxK tpm.


    Returns:
        A (...,K,K-1) matrix whose values in the last axis are unconstrained reals; there is a bijection betweeen
        the set of (K-1)-vectors and the simplex with K entries.


    Remark:
        The entries of the tpm must live in (0,1).  It cannot contain exact 0's or exact 1's.
    """
    if (tpm_or_tpms == 0).any() or (tpm_or_tpms == 1).any():
        raise ValueError(
            "Transition probability matrices cannot be converted to an unconstrained representation if any entry is exactly 0 or 1."
        )

    softmax_bijector = tfb.SoftmaxCentered()
    unconstrained_tpm_list = []
    for row in tpm_or_tpms:
        row_identified = softmax_bijector.inverse(row)
        unconstrained_tpm_list.append(row_identified)
    return jnp.asarray(unconstrained_tpm_list)


def tpm_from_unconstrained_tpm(
    unconstrained_tpm_or_unconstrained_tpms: jnp.array,
) -> jnp.array:
    """
    The inverse of `unconstrained_tpm_from_tpm`
    """
    softmax_bijector = tfb.SoftmaxCentered()
    tpm_list = []
    for row in unconstrained_tpm_or_unconstrained_tpms:
        row_on_simplex = softmax_bijector.forward(row)
        tpm_list.append(row_on_simplex)
    return jnp.asarray(tpm_list)
