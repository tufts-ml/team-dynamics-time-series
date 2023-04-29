import jax_dataclasses as jdc


@jdc.pytree_dataclass
class SystemTransitionPrior_JAX:
    """
    Gives the parameters of a "sticky" Dirichlet prior on tpm's.
    In particular, the prior has independent Dirichlet priors on each of the k=1,...,K rows
    where each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted, i.e.
    pi_k ~ Dir(alpha * 1_K + kappa * e_k)
            pi_k ~ Dir(alpha * 1_K + kappa * e_k)
    """

    alpha: float
    kappa: float
