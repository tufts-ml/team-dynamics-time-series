from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import numpy.random as npr
from numpy.random import multivariate_normal as mvn

from dynagroup.params import AllParameters, dims_from_params
from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)


@dataclass
class Sample:
    """
    Attributes:
        s : has shape (T,)
            Each entry is in {1,...,L}
        z_probs : has shape (T,J,K)
            The (t,j)-th vector lies on the (K-1)-dim simplex
        zs : has shape (T, J)
            Each entry is in {1,...,K}
        xs: has shape (T, J, D)
        ys: has shape (T, J, N)

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    s: NumpyArray1D
    z_probs: NumpyArray3D
    zs: NumpyArray2D
    xs: NumpyArray3D
    ys: NumpyArray3D


@jdc.pytree_dataclass
class Sample_JAX:
    """
    Attributes:
        s : has shape (T,)
            Each entry is in {1,...,L}
        z_probs : has shape (T,J,K)
            The (t,j)-th vector lies on the (K-1)-dim simplex
        zs : has shape (T, J)
            Each entry is in {1,...,K}
        xs: has shape (T, J, D)
        ys: has shape (T, J, N)

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    s: JaxNumpyArray1D
    z_probs: JaxNumpyArray3D
    zs: JaxNumpyArray2D
    xs: JaxNumpyArray3D
    ys: JaxNumpyArray3D


def jax_sample_from_sample(sample: Sample) -> Sample_JAX:
    return Sample_JAX(
        jnp.asarray(sample.s),
        jnp.asarray(sample.z_probs),
        jnp.asarray(sample.zs),
        jnp.asarray(sample.xs),
        jnp.asarray(sample.ys),
    )


def sample_team_dynamics(
    AP: AllParameters,
    T: int,
    log_probs_for_one_step_ahead_system_transitions: Callable,
    log_probs_for_one_step_ahead_entity_transitions: Callable,
    seed: int = 0,
    fixed_system_regimes: Optional[NumpyArray1D] = None,
    fixed_init_entity_regimes: Optional[NumpyArray1D] = None,
    fixed_init_continuous_states: Optional[NumpyArray2D] = None,
) -> Sample:
    """
    Assumes we have a state space model on the bottom of the switches.

    Arguments:
        fixed_system_regimes:  has shape (T,)
            Each entry is in {1,...,L}.
        fixed_init_entity_regimes: has shape (J,).
            Each entry is in {1,...,K}.
        fixed_init_continuous_states: has shape (J,D)
            Each entry is in R^D


    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    np.random.seed(seed)

    # pre-allocate
    dims = dims_from_params(AP)

    s = np.zeros(T, dtype=int) if fixed_system_regimes is None else fixed_system_regimes
    z_probs = np.zeros((T, dims.J, dims.K))
    zs = np.zeros((T, dims.J), dtype=int)
    xs = np.zeros((T, dims.J, dims.D))
    ys = np.zeros((T, dims.J, dims.N))

    ### initialize
    if fixed_system_regimes is None:
        s[0] = npr.choice(range(dims.L), p=AP.IP.pi_system)

    if fixed_init_entity_regimes is None:
        for j in range(dims.J):
            z_probs[0, j, :] = AP.IP.pi_entities[j]
            zs[0, j] = npr.choice(range(dims.K), p=AP.IP.pi_entities[j])
    else:
        z_probs[0] = np.full_like(z_probs[0], np.nan)
        zs[0] = fixed_init_entity_regimes

    if fixed_init_continuous_states is None:
        for j in range(dims.J):
            k = zs[0, j]
            mu_0, Sigma_0 = AP.IP.mu_0s[j, k], AP.IP.Sigma_0s[j, k]
            xs[0, j] = mvn(mu_0, Sigma_0)
    else:
        xs[0] = fixed_init_continuous_states

    for j in range(dims.J):
        C = AP.EP.Cs[j]
        d = AP.EP.ds[j]
        R = AP.EP.Rs[j]
        ys[0, j] = mvn(C @ xs[0, j] + d, R)

    ### generate
    for t in range(1, T):
        ### sample next system regime
        if fixed_system_regimes is None:
            log_probs_next_sys = log_probs_for_one_step_ahead_system_transitions(
                AP.STP, zs[t - 1], s[t - 1]
            )
            s[t] = npr.choice(range(dims.L), p=np.exp(log_probs_next_sys))

        ### sample next entity regimes
        log_probs_next_entities = log_probs_for_one_step_ahead_entity_transitions(
            AP.ETP, zs[t - 1], xs[t - 1], s[t]
        )

        for j in range(dims.J):
            z_probs[t, j, :] = np.exp(log_probs_next_entities[j])
            zs[t, j] = npr.choice(range(dims.K), p=np.exp(log_probs_next_entities[j]))

        ### sample next entity continuous states
        for j in range(dims.J):
            A = AP.CSP.As[j, zs[t, j]]
            b = AP.CSP.bs[j, zs[t, j]]
            Q = AP.CSP.Qs[j, zs[t, j]]
            xs[t, j] = mvn(A @ xs[t - 1, j] + b, Q)

        ### sample next entity observations
        for j in range(dims.J):
            C = AP.EP.Cs[j]
            d = AP.EP.ds[j]
            R = AP.EP.Rs[j]
            ys[t, j] = mvn(C @ xs[t, j] + d, R)

    return Sample(s, z_probs, zs, xs, ys)
