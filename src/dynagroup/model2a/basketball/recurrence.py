import jax.numpy as jnp

from dynagroup.types import JaxNumpyArray1D


def LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    # TODO: can I still the identity transformation  be a default?
    KAPPA = 1.0
    return KAPPA * x_vec


def ZERO_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    return jnp.zeros_like(x_vec)
