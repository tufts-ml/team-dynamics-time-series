import jax.numpy as jnp
import numpy as np

from dynagroup.types import JaxNumpyArray1D, JaxNumpyArray2D


###
# Covariances <->  Non-zero entries of cholesky
###

# We want to take a covariance matrix, (1) compute the cholesky factorization, and then
# (2) represent solely the non-zero indices of the lower-triangular matrix,
# The purpose of (1) is because doing gradient descent on a cov matrix doesn't respect psd-ness.
# The purpose of (2) is to keep gradient descent from changing the zero values of the cholesky factor.


def cholesky_nzvals_from_covariance_JAX(Sigma: JaxNumpyArray2D) -> JaxNumpyArray1D:
    """
    Returns the non-zero values of the Cholesky factor (which is lower triangular)
    of a covariance matrix.

    I.e. if Sigma = LL^T, we return the nzvals of L.

    Arguments:
        Sigma: A covariance matrix.

    Returns:
        A 1d array giving the non-zero values of the Cholesky factor (which is lower triangular)
        of a covariance matrix.

    """
    L = jnp.linalg.cholesky(Sigma)
    return L[jnp.tril_indices_from(L)]


def covariance_from_cholesky_nzvals_JAX(
    cholesky_nzvals: JaxNumpyArray1D,
) -> JaxNumpyArray2D:
    """
    Arguments:
        cholesky_nzvals: A 1d array giving the non-zero values of L, the Cholesky factor (which is lower triangular)
        of a covariance matrix.

    Returns:
        A covariance matrix, LL^T
    """

    D = _compute_dim_of_lower_triangular_matrix_from_number_of_nzvals(len(cholesky_nzvals))
    idxs = np.tril_indices(D)
    L_reconstructed = jnp.zeros((D, D), dtype=cholesky_nzvals.dtype).at[idxs].set(cholesky_nzvals)
    return L_reconstructed @ L_reconstructed.T


def _compute_dim_of_lower_triangular_matrix_from_number_of_nzvals(n: int) -> int:
    """
    Overview
        Compute the dimension of a square lower triangular matrix
        if there are `n` nonzero values.

    Details:
        For a lower triangular (square) matrix with D rows and columns,
        the number of nonzero values is N=D(D+1)/2.

        Thus, given N, we can solve for D via the quadratic formula:
            D^2 + D - 2N = 0
        gives, taking the positive square root
            D = -1 + sqrt(1+8N)
                ---------------
                    2
    """
    return int((-1 + np.sqrt(1 + 8 * n)) / 2)
