import jax
import jax.numpy as jnp
import pytest

from dynagroup.model2a.basketball.recurrence import LIST_OF_RECURRENCES


@pytest.fixture
def T():
    return 100


@pytest.fixture
def J():
    return 10


@pytest.fixture
def D():
    return 2


@pytest.fixture
def x_prevs(T, J, D):
    """
    x_prevs : jnp.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D
        for t=1,...,T-1.   If `sample` is an instance of the `Sample` class, this
        object can be obtained by doing sample.xs[:-1], which gives all the x's except
        the one at the final timestep.
    """
    return jax.random.normal(jax.random.PRNGKey(1), shape=(T - 1, J, D))


@pytest.mark.parametrize(
    "transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX", LIST_OF_RECURRENCES
)
def test_that_entity_recurrence_transformations_can_be_applied_without_error(
    x_prevs, transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX
):
    """ """

    # Rk: This application of recurrence is given by
    # the `compute_log_entity_transition_probability_matrices_JAX` function
    # in the `model_factors` module of the `gaussian` subpackage.
    x_prev_tildes = jnp.apply_along_axis(
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
        2,
        x_prevs,
    )

    assert x_prev_tildes
