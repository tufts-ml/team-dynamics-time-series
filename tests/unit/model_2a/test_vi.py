import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy

from dynagroup.params import (
    CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian,
    ContinuousStateParameters_Gaussian_JAX,
    cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX,
    covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX,
    ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances,
)


def test_invertibility_of__cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX():
    J, K, D = 5, 3, 2
    covariances = scipy.stats.wishart(scale=np.eye(D)).rvs(
        (J, K)
    )  # There are (JxK) instances of DxD covariances.
    cholesky_nzvals_for_JK = cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX(covariances)
    covariances_reconstructed = covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX(
        cholesky_nzvals_for_JK
    )
    assert np.allclose(covariances, covariances_reconstructed)


def test_invertibility_of__CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian():
    J, K, D = 5, 4, 3
    key = jr.PRNGKey(1)
    As = jr.normal(key, (J, K, D, D))
    bs = jr.normal(key, (J, K, D))
    Qs = jnp.asarray(scipy.stats.wishart(scale=np.eye(D)).rvs((J, K)))
    CSP = ContinuousStateParameters_Gaussian_JAX(As, bs, Qs)
    CSP_WUC = CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian(CSP)
    CSP_recovered = ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(CSP_WUC)
    for attribute, attribute_recovered in zip(
        CSP.__dict__.values(), CSP_recovered.__dict__.values()
    ):
        assert jnp.allclose(attribute, attribute_recovered)
