from warnings import warn

import jax.numpy as jnp
import numpy as np

from dynagroup.types import JaxNumpyArray1D, NumpyArray1D


# TODO: Relate one-step ahead transitions to the total transitions


def _transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_OLD(
    x_vec: NumpyArray1D,
) -> NumpyArray1D:
    """
    Arguments:
        x_vec : the continuous state for some entity j at some time t-1.

    Returns:
        f(x_vec), a transformation of x_vec; if Psi is a recurrence matrix, then
        the contribution of recurrence to the entity-level regime destinatons is Psi @ f(x_vec).

    Remarks:
        1. This recurrence function evaluates to very large number (23) at (x1,x2)=(0,2).
        However, the intention of the function was to produce large numbers ONLY
        when x is close to the origin (0,0). Hence it has been replaced by a Gaussian density.
        2. Back when we wrote this function, we required the transformed continuous states, x-tilde  = f(x),
        to have the same dimension at the continuous states, x. This required needlessly duplicating entries of x-tilde.
        However, the new version of this function allows x_tilde to have a different dimension, D_t, because we allow
        the matrix Psi to have D_t columns instead of D columns. This makes more sense conceptually, reduces the
        chance of a mismatch between generation and inference variants, and might (?) make inference easier.

    """
    warn("This fucnction is now deprecated", DeprecationWarning)
    EPSILON = 1e-10
    return np.array([-np.log(np.prod(np.abs(x_vec)) + EPSILON) for x_dim in x_vec])


def _transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_OLD_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Arguments:
        x_vec : the continuous state for some entity j at some time t-1.

    Returns:
        f(x_vec), a transformation of x_vec; if Psi is a recurrence matrix, then
        the contribution of recurrence to the entity-level regime destinatons is Psi @ f(x_vec).

    Remarks:
        1. This recurrence function evaluates to very large number (23) at (x1,x2)=(0,2).
        However, the intention of the function was to produce large numbers ONLY
        when x is close to the origin (0,0). Hence it has been replaced by a Gaussian density.
        2. Back when we wrote this function, we required the transformed continuous states, x-tilde  = f(x),
        to have the same dimension at the continuous states, x. This required needlessly duplicating entries of x-tilde.
        However, the new version of this function allows x_tilde to have a different dimension, D_t, because we allow
        the matrix Psi to have D_t columns instead of D columns. This makes more sense conceptually, reduces the
        chance of a mismatch between generation and inference variants, and might (?) make inference easier.
    """
    warn("This fucnction is now deprecated", DeprecationWarning)
    EPSILON = 1e-10
    return jnp.array([-jnp.log(jnp.prod(jnp.abs(x_vec)) + EPSILON) for x_dim in x_vec])


SIGMA = 0.2
KAPPA = 20.0


def transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix(
    x_vec: NumpyArray1D,
) -> NumpyArray1D:
    """
    A Gaussian [N(0, sigma^2)] density function, scaled by a constant kappa.
    The intent is to return high values iff x_vec is close to the origin.

    Arguments:
        x_vec : the continuous state for some entity j at some time t-1.

    Returns:
        f(x_vec), a transformation of x_vec; if Psi is a recurrence matrix, then
        the contribution of recurrence to the entity-level regime destinatons is Psi @ f(x_vec).

    Remarks:
        This particular version of the function returns a 1D array with a single entry
    """
    return np.asarray([KAPPA * np.exp(-np.sum(x_vec**2) / (2 * SIGMA**2))])


def transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    A Gaussian [N(0, sigma^2)] density function, scaled by a constant kappa.
    The intent is to return high values iff x_vec is close to the origin.

    Arguments:
        x_vec : the continuous state for some entity j at some time t-1.

    Returns:
        f(x_vec), a transformation of x_vec; if Psi is a recurrence matrix, then
        the contribution of recurrence to the entity-level regime destinatons is Psi @ f(x_vec).
    """
    return jnp.asarray([KAPPA * jnp.exp(-jnp.sum(x_vec**2) / (2 * SIGMA**2))])
