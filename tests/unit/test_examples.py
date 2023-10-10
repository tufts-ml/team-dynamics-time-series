import jax.numpy as jnp
import numpy as np
import pytest

from dynagroup.examples import (
    fix__log_emissions_from_entities__at_example_boundaries,
    fix_log_entity_transitions_at_example_boundaries,
    fix_log_system_transitions_at_example_boundaries,
    get_non_initialization_times,
)
from dynagroup.model2a.figure8.model_factors import figure8_model_JAX
from dynagroup.params import Dims, InitializationParameters_Gaussian_JAX
from dynagroup.util import make_fixed_sticky_tpm


@pytest.fixture
def DIMS():
    return Dims(J=3, K=2, L=5, D=2, D_e=1, N=0, D_s=0, M_e=0)


@pytest.fixture
def IP(DIMS):
    return InitializationParameters_Gaussian_JAX(
        pi_system=jnp.ones(DIMS.L) / DIMS.L,
        pi_entities=jnp.ones((DIMS.J, DIMS.K)) / DIMS.K,
        mu_0s=jnp.ones((DIMS.J, DIMS.K, DIMS.D)),
        Sigma_0s=jnp.ones((DIMS.J, DIMS.K, DIMS.D, DIMS.D)),
    )


@pytest.fixture
def T():
    return 10


@pytest.fixture
def log_entity_transitions(DIMS, T):
    self_transition_prob = 0.95
    tpm = make_fixed_sticky_tpm(self_transition_prob, DIMS.K)
    return jnp.tile(tpm, (T - 1, DIMS.J, 1, 1))


@pytest.fixture
def log_system_transitions(DIMS, T):
    self_transition_prob = 0.95
    tpm = make_fixed_sticky_tpm(self_transition_prob, DIMS.L)
    return jnp.tile(tpm, (T - 1, 1, 1))


@pytest.fixture
def example_end_times(T):
    """e.g. if T=10 this returns [-1,4,10]"""
    return [-1, int(T / 2) - 1, T]


@pytest.fixture
def continuous_states(DIMS, T):
    return jnp.array(np.random.rand(T, DIMS.J, DIMS.D))


@pytest.fixture
def log_state_emissions(DIMS, T):
    DUMMY_VALUE = -5.0
    return jnp.full((T, DIMS.J, DIMS.K), DUMMY_VALUE)


@pytest.fixture
def model():
    return figure8_model_JAX


def test_fix_log_system_transitions_at_example_boundaries(IP, log_system_transitions, example_end_times, T, DIMS):
    log_system_transitions_fixed = fix_log_system_transitions_at_example_boundaries(
        log_system_transitions,
        IP,
        example_end_times,
    )
    for t in range(T - 1):
        if t in example_end_times:
            assert not jnp.allclose(log_system_transitions_fixed[t], log_system_transitions[t])
            for k in range(DIMS.K):
                assert jnp.allclose(log_system_transitions_fixed[t, k, :], jnp.log(IP.pi_system))
        else:
            assert jnp.allclose(log_system_transitions_fixed[t], log_system_transitions[t])


def test_fix_log_entity_transitions_at_example_boundaries(IP, log_entity_transitions, example_end_times, T, DIMS):
    log_entity_transitions_fixed = fix_log_entity_transitions_at_example_boundaries(
        log_entity_transitions,
        IP,
        example_end_times,
    )
    for t in range(T - 1):
        if t in example_end_times:
            assert not jnp.allclose(log_entity_transitions_fixed[t], log_entity_transitions[t])
            for j in range(DIMS.J):
                for k in range(DIMS.K):
                    assert jnp.allclose(log_entity_transitions_fixed[t, j, k, :], jnp.log(IP.pi_entities[j]))
        else:
            assert jnp.allclose(log_entity_transitions_fixed[t], log_entity_transitions[t])


def test_fix__log_emissions_from_entities__at_example_boundaries(
    log_state_emissions, continuous_states, IP, model, example_end_times, T
):
    log_state_emissions_fixed = fix__log_emissions_from_entities__at_example_boundaries(
        log_state_emissions, continuous_states, IP, model, example_end_times
    )

    # if there is an event boundary, then the NEXT continuous state is drawn from the initial distribution.
    for t in range(T - 1):
        if t in example_end_times:
            assert not jnp.allclose(log_state_emissions_fixed[t + 1], log_state_emissions[t + 1])
            # TODO: test that the new value is exactly the continuous state evaluated at the normal model with IP parameters.
        else:
            assert jnp.allclose(log_state_emissions_fixed[t + 1], log_state_emissions[t + 1])


def test_get_non_initialization_times(example_end_times):
    expected_result = np.array([1, 2, 3, 4, 6, 7, 8, 9])
    assert np.allclose(get_non_initialization_times(example_end_times), expected_result)
