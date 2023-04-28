from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    HMM_Posterior_Summary_NUMPY,
)


@dataclass
class Variational_Dims:
    """
    Attributes:
        T: number of timesteps
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
    """

    T: int
    J: int
    K: int
    L: int


@jdc.pytree_dataclass
class Variational_Dims_JAX:
    """
    Attributes:
        T: number of timesteps
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
    """

    T: int
    J: int
    K: int
    L: int


def variational_dims_from_summaries(
    VES_summary: HMM_Posterior_Summary_NUMPY,
    VEZ_summaries: List[HMM_Posterior_Summary_NUMPY],
) -> Variational_Dims:
    return Variational_Dims(
        T=np.shape(VES_summary.expected_regimes)[0],
        J=len(VEZ_summaries),
        K=np.shape(VEZ_summaries[0].expected_joints)[1],
        L=np.shape(VES_summary.expected_regimes)[1],
    )


def variational_dims_from_summaries_JAX(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> Variational_Dims_JAX:
    return Variational_Dims_JAX(
        T=jnp.shape(VES_summary.expected_regimes)[0],
        J=jnp.shape(VEZ_summaries.expected_regimes)[1],
        K=jnp.shape(VEZ_summaries.expected_regimes)[2],
        L=jnp.shape(VES_summary.expected_regimes)[1],
    )
