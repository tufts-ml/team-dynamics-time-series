import jax.numpy as jnp

from dynagroup.types import JaxNumpyArray1D


def ZERO_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    return jnp.zeros_like(x_vec)


def LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    # TODO: can I still the identity transformation  be a default?
    KAPPA = 1.0
    return KAPPA * x_vec


def LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    out_of_bounds_to_the_left, out_of_bounds_to_the_right = x_vec[0] < 0.0, x_vec[0] > 1.0
    out_of_bounds_to_the_south, out_of_bounds_to_the_north = x_vec[1] < 0.0, x_vec[1] > 1.0
    return jnp.array(
        [
            x_vec[0],
            x_vec[1],
            out_of_bounds_to_the_left,
            out_of_bounds_to_the_right,
            out_of_bounds_to_the_south,
            out_of_bounds_to_the_north,
        ]
    )


def OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    out_of_bounds_to_the_left, out_of_bounds_to_the_right = x_vec[0] < 0.0, x_vec[0] > 1.0
    out_of_bounds_to_the_north, out_of_bounds_to_the_south = x_vec[1] < 0.0, x_vec[1] > 1.0
    return jnp.array(
        [
            out_of_bounds_to_the_left,
            out_of_bounds_to_the_right,
            out_of_bounds_to_the_north,
            out_of_bounds_to_the_south,
        ]
    )


LIST_OF_RECURRENCES = [
    ZERO_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
]
