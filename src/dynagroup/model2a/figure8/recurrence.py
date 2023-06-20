import jax.numpy as jnp

from dynagroup.types import JaxNumpyArray1D


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

    SIGMA = 0.2
    KAPPA = 20.0

    return jnp.asarray([KAPPA * jnp.exp(-jnp.sum(x_vec**2) / (2 * SIGMA**2))])
