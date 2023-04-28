import jax.numpy as jnp
import pytest

from dynagroup.util import tpm_from_unconstrained_tpm, unconstrained_tpm_from_tpm


def test__unconstrained_tpm_from_tpm__then__tpm_from_unconstrained_tpm__gives_identity():
    tpm_1 = jnp.eye(5)
    tpm_2 = jnp.ones((3, 3)) / 3
    tpm_3 = jnp.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])

    with pytest.raises(ValueError):
        # The tpm can't contain exact 0's or 1's.
        unconstrained_tpm_from_tpm(tpm_1)

    for tpm in [tpm_2, tpm_3]:
        unconstrained_tpm = unconstrained_tpm_from_tpm(tpm)
        tpm_recovered = tpm_from_unconstrained_tpm(unconstrained_tpm)
        assert jnp.allclose(tpm, tpm_recovered)


def test__unconstrained_tpm_from_tpm__then__tpm_from_unconstrained_tpm__gives_identity_even_with_batched_tpms():
    tpms = jnp.ones((10, 3, 3)) / 3
    unconstrained_tpms = unconstrained_tpm_from_tpm(tpms)
    tpms_recovered = tpm_from_unconstrained_tpm(unconstrained_tpms)
    assert jnp.allclose(tpms, tpms_recovered)
