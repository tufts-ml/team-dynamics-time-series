from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np

from dynagroup.model import Model
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
        example_end_times: optional, has shape (E+1,)
            An `event` takes an ordinary sampled group time series of shape (T,J,:) and interprets it as (T_grand,J,:),
            where T_grand is the sum of the number of timesteps across i.i.d "events".  An event might induce a large
            time gap between timesteps, and a discontinuity in the continuous states x.

            If there are E events, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th eveent ended.
            So to get the timesteps for the e-th event, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
        E: number of events
    """

    s: NumpyArray1D
    z_probs: NumpyArray3D
    zs: NumpyArray2D
    xs: NumpyArray3D
    ys: NumpyArray3D
    example_end_times: Optional[NumpyArray1D] = None


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


def get_multiple_samples_of_team_dynamics(
    S: int,
    AP: AllParameters,
    T: int,
    model: Model,
    seed: int = 0,
    fixed_system_regimes: Optional[NumpyArray1D] = None,
    fixed_init_system_regime: Optional[int] = None,
    fixed_init_entity_regimes: Optional[NumpyArray1D] = None,
    fixed_init_continuous_states: Optional[NumpyArray2D] = None,
    system_covariates: Optional[np.array] = None,
) -> List[Sample]:
    """
    Assumes we have a state space model on the bottom of the switches.

    Arguments:
        S: number of samples
        T: length of time series
        fixed_system_regimes: Optional, has shape (T,)
            Each entry is in {1,...,L}.
            If not None, `fixed_init_system_regime` must be None
        fixed_init_system_regime: Optional, has type int
            If not None, `fixed_system_regimes` must be None,
        fixed_init_entity_regimes: Optional,  has shape (J,).
            Each entry is in {1,...,K}.
        fixed_init_continuous_states: Optional, has shape (J,D)
            Each entry is in R^D


    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """
    ### Up-front material
    if (fixed_init_system_regime is not None) and (fixed_system_regimes is not None):
        raise ValueError(
            f"At most ONE of `fixed_init_system_regime` and `fixed_system_regimes` can be non-None. "
            f"You have set them BOTH to non-None, and so I don't know what to use for the system "
            f"regime at the initial timestep, s[0]."
        )

    key = jax.random.PRNGKey(seed)

    ### Pre-allocation
    dims = dims_from_params(AP)

    # Rk: Below is a 'safe mode' I tried to enforce, but it proved to be too restrictive in practice,
    # especially since we may run forecasting on very many examples in the test set, so we might want to
    # handle many, wildly varying T's.
    #
    # if S in (dims.K, dims.L, T, dims.J):
    #     raise ValueError(
    #         f"Pick a value for S that does not equal K, L, J or T, so that we don't get unexpected results "
    #         f"from numpy broadcasting when making S samples in parallel"
    #     )

    s = np.zeros((S, T), dtype=int)
    z_probs = np.zeros((S, T, dims.J, dims.K))
    zs = np.zeros((S, T, dims.J), dtype=int)
    xs = np.zeros((S, T, dims.J, dims.D))
    ys = np.full((S, T, dims.J, dims.N), np.nan)

    ### Initialize
    if fixed_init_system_regime is not None:
        s[:, 0] = fixed_init_system_regime
    elif fixed_system_regimes is not None:
        s[:, :] = fixed_system_regimes
    else:
        s[:, 0] = jax.random.choice(key, dims.L, shape=(S,), p=AP.IP.pi_system)
        key, _ = jax.random.split(key)

    if fixed_init_entity_regimes is None:
        for j in range(dims.J):
            z_probs[:, 0, j, :] = AP.IP.pi_entities[j]
            zs[:, 0, j] = jax.random.choice(key, dims.K, shape=(S,), p=AP.IP.pi_entities[j])
            key, _ = jax.random.split(key)

    else:
        z_probs[:, 0] = np.full_like(z_probs[:, 0], np.nan)
        zs[:, 0] = fixed_init_entity_regimes

    if fixed_init_continuous_states is None:
        for j in range(dims.J):
            # we get a separate entity state, k, for each sample s.
            ks = zs[:, 0, j]
            mu_0s, Sigma_0s = AP.IP.mu_0s[j, ks], AP.IP.Sigma_0s[j, ks]
            xs[:, 0, j] = jax.random.multivariate_normal(key, mu_0s, Sigma_0s)
            key, _ = jax.random.split(key)

    else:
        xs[:, 0] = fixed_init_continuous_states

    for j in range(dims.J):
        C = AP.EP.Cs[j]
        d = AP.EP.ds[j]
        R = AP.EP.Rs[j]
        emissions_exist = C.any() or d.any() or R.any()
        if emissions_exist:
            raise NotImplementedError(
                "Need to update multi-sample generation to handle emissions of y. See commit on 10/6/23."
            )
            # Rk: it should be something like
            #   ys[:,0, j] = mvn(C @ xs[:,0, j] + d, R, size=S)
            # but I probably need einsum rather than @.

    ### Generate
    for t in range(1, T):
        # `dummy_time_index` allows us to use the model factors but apply them to the case
        # where we're looking at a single time observation at a time
        dummy_time_index = 0

        ###
        # Sample next system regime
        ###
        if fixed_system_regimes is None:
            for sample in range(S):
                system_covariates_at_this_timestep = system_covariates[t] if system_covariates is not None else None
                log_probs_next_sys = model.compute_log_system_transition_probability_matrices_JAX(
                    AP.STP,
                    T_minus_1=1,
                    system_covariates=system_covariates_at_this_timestep,
                    x_prevs=xs[sample, t - 1][None, :, :],
                    system_recurrence_transformation=model.transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX,
                )

                # select the probabilities that are relevant to the current system regime s_t and previous entity regimes zs[t-1]
                # TODO: To handle Model 1, the system transitions probabilities should depend
                # on the previous entity regimes, zs[t-1].
                s_probs = np.exp(log_probs_next_sys[dummy_time_index, s[:, t - 1]])

                # TODO: I want to do vectorized version of npr choice.
                # I'm getting "ValueError: 'p' must be 1-dimensional
                # On first glance, jax didn't help here
                # s[sample,t] = npr.choice(range(dims.L), p=s_probs[sample])
                s[sample, t] = jax.random.choice(key, dims.L, p=s_probs[sample])
                key, _ = jax.random.split(key)

        ###
        # Sample next entity regimes
        ###

        #  Old way of doing it
        #
        # log_probs_next_entities = log_probs_for_one_step_ahead_entity_transitions(
        #     AP.ETP, zs[t - 1], xs[t - 1], s[t]
        # )

        for sample in range(S):
            log_probs_next_entities = model.compute_log_entity_transition_probability_matrices_JAX(
                AP.ETP,
                xs[sample, t - 1][None, :, :],
                model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
            )
            # select the probabilities that are relevant to the current system regime s_t and previous entity regimes zs[t-1]
            z_probs[sample, t] = np.exp(
                log_probs_next_entities[dummy_time_index, np.arange(dims.J), s[sample, t], zs[sample, t - 1], :]
            )

            for j in range(dims.J):
                z_probs[sample, t, j] /= np.sum(z_probs[sample, t, j])
                zs[sample, t, j] = jax.random.choice(key, dims.K, p=z_probs[sample, t, j])
                key, _ = jax.random.split(key)

        ###
        # Sample next entity continuous states
        ###
        for j in range(dims.J):
            As = AP.CSP.As[j, zs[:, t, j]]
            bs = AP.CSP.bs[j, zs[:, t, j]]
            Qs = AP.CSP.Qs[j, zs[:, t, j]]
            xs[:, t, j] = jax.random.multivariate_normal(
                key, mean=np.einsum("sde, se -> sd", As, xs[:, t - 1, j]) + bs, cov=Qs
            )
            key, _ = jax.random.split(key)

        ###
        # Sample next entity observations
        ###
        for j in range(dims.J):
            C = AP.EP.Cs[j]
            d = AP.EP.ds[j]
            R = AP.EP.Rs[j]
            emissions_exist = C.any() or d.any() or R.any()
            if emissions_exist:
                raise NotImplementedError(
                    "Need to update multi-sample generation to handle emissions of y. See commit on 10/6/23."
                )
                # It should be something like
                #   ys[t, j] = mvn(C @ xs[:,t, j] + d, R, size=S)
                # but I probably need einsum rather than @.

    return list(Sample(s[sample], z_probs[sample], zs[sample], xs[sample], ys[sample]) for sample in range(S))


def sample_team_dynamics(
    AP: AllParameters,
    T: int,
    model: Model,
    seed: int = 0,
    fixed_system_regimes: Optional[NumpyArray1D] = None,
    fixed_init_system_regime: Optional[int] = None,
    fixed_init_entity_regimes: Optional[NumpyArray1D] = None,
    fixed_init_continuous_states: Optional[NumpyArray2D] = None,
    system_covariates: Optional[np.array] = None,
) -> Sample:
    """
    Assumes we have a state space model on the bottom of the switches.

    Arguments:
        fixed_system_regimes: Optional, has shape (T,)
            Each entry is in {1,...,L}.
            If not None, `fixed_init_system_regime` must be None
        fixed_init_system_regime: Optional, has type int
            If not None, `fixed_system_regimes` must be None,
        fixed_init_entity_regimes: Optional,  has shape (J,).
            Each entry is in {1,...,K}.
        fixed_init_continuous_states: Optional, has shape (J,D)
            Each entry is in R^D


    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """
    num_samples = 1
    return get_multiple_samples_of_team_dynamics(
        num_samples,
        AP,
        T,
        model,
        seed,
        fixed_system_regimes,
        fixed_init_system_regime,
        fixed_init_entity_regimes,
        fixed_init_continuous_states,
        system_covariates,
    )[0]
