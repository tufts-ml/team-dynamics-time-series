import jax.numpy as jnp
import numpy as np

from dynagroup.covariances import (
    cholesky_nzvals_from_covariance_JAX,
    covariance_from_cholesky_nzvals_JAX,
)
from dynagroup.util import generate_random_covariance_matrix


def test__cholesky_nzvals_from_covariance_JAX__then__covariance_from_cholesky_nzvals_JAX():
    Sigma = generate_random_covariance_matrix(dim=3, var=1.0)
    cholesky_nzvals = cholesky_nzvals_from_covariance_JAX(Sigma)
    Sigma_reconstructed = covariance_from_cholesky_nzvals_JAX(cholesky_nzvals)
    assert np.allclose(Sigma, Sigma_reconstructed)


def test__covariance_from_cholesky_nzvals_JAX__then__cholesky_nzvals_from_covariance_JAX():
    D = 5  # pick a dimension of a covariance matrix
    N = int(D * (D + 1) / 2)  # this determines a possible num of nzvals
    cholesky_nzvals = jnp.asarray(np.random.rand(N), dtype=np.float32)
    covariance = covariance_from_cholesky_nzvals_JAX(cholesky_nzvals)
    cholesky_nzvals_reconstructed = cholesky_nzvals_from_covariance_JAX(covariance)
    assert np.allclose(cholesky_nzvals, cholesky_nzvals_reconstructed, atol=1e-5)
