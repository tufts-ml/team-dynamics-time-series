import jax.numpy as jnp
import jax.random as jr
import pytest
from scipy.stats import vonmises

from dynagroup.model2a.circle.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
)
from dynagroup.params import ContinuousStateParameters_VonMises_JAX


@pytest.fixture
def K():
    return 2


@pytest.fixture
def J():
    return 3


@pytest.fixture
def T():
    return 5


@pytest.fixture
def CSP(J, K):
    key = jr.PRNGKey(0)

    ar_coefs = jr.uniform(key, shape=(J, K))
    drifts = jr.uniform(key, shape=(J, K))
    kappas = jr.uniform(key, shape=(J, K), maxval=200.0)

    return ContinuousStateParameters_VonMises_JAX(ar_coefs, drifts, kappas)


@pytest.fixture
def group_angles(T, J):
    """
    Returns:
        array of shape (T,J)
    """
    if T != 5:
        raise NotImplementedError
    if J != 3:
        raise NotImplementedError

    group_angles_for_entity_1 = jnp.array([-1, -0.5, 0, 0.5, 1]) * jnp.pi
    group_angles_for_entity_2 = group_angles_for_entity_1 * 0.5
    group_angles_for_entity_3 = group_angles_for_entity_1 * 0.25
    return jnp.array(
        [group_angles_for_entity_1, group_angles_for_entity_2, group_angles_for_entity_3]
    ).T


def test__compute_log_continuous_state_emissions_after_initial_timestep_JAX(CSP, group_angles):
    log_emissions = compute_log_continuous_state_emissions_after_initial_timestep_JAX(
        CSP, group_angles
    )  # (T-1, J,K); t=1,...,T

    t, j, k = 1, 0, 0
    mean = CSP.ar_coefs[j, k] * group_angles[t - 1, j] + CSP.drifts[j, k]
    concentration = CSP.kappas[j, k]
    assert jnp.isclose(
        log_emissions[t - 1, j, k], vonmises.logpdf(group_angles[t, j], concentration, loc=mean)
    )
