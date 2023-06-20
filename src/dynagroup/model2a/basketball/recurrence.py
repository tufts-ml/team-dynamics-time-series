from dynagroup.types import JaxNumpyArray2D


def transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray2D,
) -> JaxNumpyArray2D:
    # TODO: can I still the identity transformation  be a default?
    KAPPA = 0.05
    return KAPPA * x_vec
