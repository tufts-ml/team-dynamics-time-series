import jax.numpy as jnp
import jax.random as jr
import numpy as np


def sample_sticky_transition_matrix(K: int, alpha: float, kappa: float, seed: int) -> np.array:
    """
    Construct a transition matrix (source x destination) by independently sampling each
    row (which gives probabilities of transitioning into each destination) from a Dirichlet
    disitribution.

    Each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted, i.e.
        pi_k ~ Dir(alpha * 1_K + kappa * e_k)

    In other words, the Dirichlet on the k-th row is given by (alpha, ..., alpha, alpha+kappa, alpha...,alpha)
    with kappa added to the k-th location, in order to encourage self-transitions.

    Remark:
        We use jax to seed this function so that we can have more "local" control over the random number generation.
        See https://jax.readthedocs.io/en/latest/jep/263-prng.html for a further justification for why to prefer
        jax.numpy over numpy for handling of random numbers.
    """
    key = jr.PRNGKey(seed)
    Ps = np.zeros((K, K))
    for k in range(K):
        dirichlet_param = np.ones(K) * alpha
        dirichlet_param[k] += kappa
        Ps[k] = jr.dirichlet(key, dirichlet_param)
        key, _ = jr.split(key)
    return Ps


def evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
    log_P: jnp.array,
    alpha: float,
    kappa: float,
) -> float:
    """
    Arguments:
        log_P : the logarithm of the transition matrix, P, which we want to
            evaluate against a sticky tpm distribution, in particular one
            with independent Dirichlet priors on each of the k=1,...,K rows
            where each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted, i.e.
            pi_k ~ Dir(alpha * 1_K + kappa * e_k)

    Note that the normalizing constant, which is a beta function on the parameter alpha * 1_K +kappa * e_k
    is ignored here, presumably because the ELBO does not depend on it (it's a hyperparmeter.)
    """
    K = len(log_P)

    lp = 0
    for k in range(K):
        dirichlet_param = alpha * jnp.ones(K) + kappa * (jnp.arange(K) == k)
        lp += jnp.dot((dirichlet_param - 1), log_P[k])
    return lp
