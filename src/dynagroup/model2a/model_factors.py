from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX
from scipy.stats import multivariate_normal as mvn

from dynagroup.params import (
    ContinuousStateParameters,
    ContinuousStateParameters_JAX,
    EntityTransitionParameters,
    EntityTransitionParameters_JAX,
    InitializationParameters,
    InitializationParameters_JAX,
    SystemTransitionParameters,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import JaxNumpyArray3D, JaxNumpyArray5D, NumpyArray3D, NumpyArray5D
from dynagroup.util import (
    normalize_log_potentials,
    normalize_log_potentials_by_axis_JAX,
)


# We import from autograd instead of doing `import numpy as np`
# so that we can use use the functions here when doing
# numerical optimization procedures on parameters.


"""
Gives the system transition, entity transitions, 
dynamics, emission, and init functions for the model.
"""


def compute_log_system_transition_probability_matrices(
    STP: SystemTransitionParameters,
    T: int,
):
    """
    Compute log system transition probability matrices.
    These are time varying

    Returns:
        np.array of shape (T-1,L,L).  The (t,l,l')-th element gives the probability of transitioning
            from regime l to regime l' when transitioning into time t+1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
    """
    # TODO: Incorporate covariates

    J, L, _ = np.shape(STP.Gammas)

    # We initialize with a list instead of with np.zeros((T-1,L,L)) so this will
    # work with autograd.numpy and therefore autodiff.
    log_probs_for_each_t = []
    for t in range(1, T):
        # we add the destination bias to the columns of Pi, the logartihm of the baseline tpm
        log_potentials_matrix = STP.Pi
        log_probs_at_t = normalize_log_potentials(log_potentials_matrix)
        log_probs_for_each_t.append(log_probs_at_t)
    log_probs = np.array(log_probs_for_each_t)  # (T-1) x L x L

    return log_probs


def compute_log_system_transition_probability_matrices_JAX(
    STP: SystemTransitionParameters_JAX,
    T: int,
):
    """
    Compute log system transition probability matrices.
    These are time varying

    Returns:
        np.array of shape (T-1,L,L).  The (t,l,l')-th element gives the probability of transitioning
            from regime l to regime l' when transitioning into time t+1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
    """
    # TODO: Incorporate covariates
    return jnp.tile(STP.Pi, (T - 1, 1, 1))  # (T-1, J, K, K)


def compute_log_entity_transition_probability_matrices(
    ETP: EntityTransitionParameters,
    continuous_states: NumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: Callable = None,
) -> NumpyArray5D:
    """
    Compute log entity transition probability matrices.

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Returns:
        np.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the probability of
            the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives P(z_{t+1}^j = k' | z_t^j =k, s_{t+1}=l).

    Notation:
        T: number of timesteps
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    if transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix is None:
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix = (
            lambda x: x
        )
    func = transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix

    # TODO: Add covariates

    J, L, K, D = np.shape(ETP.Psis)
    T = np.shape(continuous_states)[0]

    log_probs = np.zeros((T - 1, J, L, K, K))
    for t in range(1, T):
        for j in range(J):
            for l in range(L):
                # we add the destination bias to the columns of Ps, the logarithm of the baseline tpm
                bias = ETP.Psis[j, l] @ func(continuous_states[t - 1, j])
                log_potentials_matrix = ETP.Ps[j, l] + bias
                log_probs_at_t = normalize_log_potentials(log_potentials_matrix)
                log_probs[t - 1, j, l] = log_probs_at_t
    return log_probs


def compute_log_entity_transition_probability_matrices_JAX(
    ETP_JAX: EntityTransitionParameters_JAX,
    xs: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> JaxNumpyArray5D:
    """
    Compute log entity transition probability matrices.

    Arguments:
        Psis: Transition parameter representing recurrence effect,
            jnp.array of shape (T,L,K,D)
        Ps: Transition parameter representing baseline transition preferences.
            jnp.array of shape (T,J,K,K)
            See `EntityTransitionParameters` class definiton for more details.
        xs : jnp.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Returns:
        jnp.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the probability of
            the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives P(z_{t+1}^j = k' | z_t^j =k, s_{t+1}=l).

    Notation:
        T: number of timesteps
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    if transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX is None:
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX = (
            lambda x: x
        )
    # TODO: Add covariates
    x_tildes = jnp.apply_along_axis(
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
        2,
        xs,
    )
    bias_from_recurrence = jnp.einsum(
        "jlkd,tjd->tjlk", ETP_JAX.Psis, x_tildes[:-1]
    )  # (T-1, J, L, K)
    log_potentials = (
        bias_from_recurrence[:, :, :, None, :] + ETP_JAX.Ps[None, :, :, :, :]
    )  # (T-1, J, L, K, K)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=4)


def compute_log_continuous_state_emissions(
    CSP: ContinuousStateParameters,
    IP: InitializationParameters,
    continuous_states: NumpyArray3D,
):
    """
    Compute the log (autoregressive, switching) emissions for the continuous states, where we have
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
        x_t^j ~ N( A[j,k] @ x_{t-1}^j + b[j,k] , Q[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D

    Returns:
        np.array of shape (T,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the t-th continuous state (given the (t-1)-st continuous state)
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    T, J, _ = np.shape(continuous_states)
    K = np.shape(CSP.As)[1]
    log_emissions = np.zeros((T, J, K))

    for j in range(J):
        for k in range(K):
            # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
            mu_0, Sigma_0 = IP.mu_0s[j, k], IP.Sigma_0s[j, k]
            log_emissions[0, j, k] = mvn(mu_0, Sigma_0).logpdf(continuous_states[0, j])

    # TODO: This triple-looping is kind of slow.  Vectorize?  I miss Julia. Sigh.
    for t in range(1, T):
        for j in range(J):
            for k in range(K):
                # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
                mu_t = CSP.As[j, k] @ continuous_states[t - 1, j] + CSP.bs[j, k]
                Sigma_t = CSP.Qs[j, k]
                log_emissions[t, j, k] = mvn(mu_t, Sigma_t).logpdf(continuous_states[t, j])

    return log_emissions


def compute_log_continuous_state_emissions_JAX(
    CSP: ContinuousStateParameters_JAX,
    IP: InitializationParameters_JAX,
    continuous_states: JaxNumpyArray3D,
):
    """
    Compute the log (autoregressive, switching) emissions for the continuous states, where we have
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
        x_t^j ~ N( A[j,k] @ x_{t-1}^j + b[j,k] , Q[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D

    Returns:
        np.array of shape (T,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the t-th continuous state (given the (t-1)-st continuous state)
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    # TODO: Have this be computed from `compute_log_continuous_state_emissions_after_initial_timestep_JAX`
    # and `compute_log_continuous_state_emissions_at_initial_timestep_JAX` to avoid DRY.

    T = len(continuous_states)
    K = np.shape(CSP.As)[1]

    ### Initial times
    # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    means_init_time, covs_init_time = IP.mu_0s, IP.Sigma_0s
    log_pdfs_init_time = mvn_JAX.logpdf(
        continuous_states[0, :, None, :], means_init_time, covs_init_time
    )

    # Pre-vectorized version for clarity
    # for j in range(J):
    #     for k in range(K):
    #         # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    #         mu_0, Sigma_0 = IP.mu_0s[j, k], IP.Sigma_0s[j, k]
    #         log_emissions.at[0, j, k].set(mvn_JAX.logpdf(continuous_states[0, j], mu_0, Sigma_0))

    #### Remaining times
    # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
    # TODO: DO I need to tile the covs and the continuous states?
    means_remaining_times = jnp.einsum(
        "jkde,tje->tjkd", CSP.As, continuous_states[:-1]
    )  # (T-1,J,K,D)
    means_remaining_times += CSP.bs[None, :, :, :]
    covs_remaining_times = jnp.tile(CSP.Qs, (T - 1, 1, 1, 1, 1))  # (T-1,J,K,D, D)
    continuous_states_remaining_times_axes_poorly_ordered = jnp.tile(
        continuous_states[1:], (K, 1, 1, 1)
    )  # (K,T-1,J,D)
    continuous_states_remaining_times = jnp.moveaxis(
        continuous_states_remaining_times_axes_poorly_ordered, [0, 1, 2], [2, 0, 1]
    )  # WANT: (T_1,J,K,D)
    log_pdfs_remaining_times = mvn_JAX.logpdf(
        continuous_states_remaining_times, means_remaining_times, covs_remaining_times
    )

    # Pre-vectorized version for clarity
    # for t in range(1, T):
    #     for j in range(J):import numpy as np
    #         for k in range(K):
    #             # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
    #             mu_t = CSP.As[j, k] @ continuous_states[t - 1, j] + CSP.bs[j, k]
    #             Sigma_t = CSP.Qs[j, k]
    #             log_emissions.at[t, j, k].set(mvn_JAX.logpdf(continuous_states[t, j], mu_t, Sigma_t))

    log_emissions = jnp.vstack((log_pdfs_init_time[None, :, :], log_pdfs_remaining_times))

    return log_emissions


def compute_log_continuous_state_emissions_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
) -> JaxNumpyArray3D:
    """
    Compute the log (autoregressive, switching) emissions for the continuous states, where we have
        x_t^j ~ N( A[j,k] @ x_{t-1}^j + b[j,k] , Q[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Note that we do NOT include the initial state
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
    which is computed elsewhere.

    Arguments:
        continuous_states : array of shape (T,J,D) where the (t,j)-th entry is
            in R^D

    Returns:
        array of shape (T-1,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the (t+1)-st continuous state (given the (t)-th continuous state)
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    T = len(continuous_states)
    K = np.shape(CSP.As)[1]

    #### Remaining times
    # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
    # TODO: DO I need to tile the covs and the continuous states?
    means_after_initial_timestep = jnp.einsum(
        "jkde,tje->tjkd", CSP.As, continuous_states[:-1]
    )  # (T-1,J,K,D)
    means_after_initial_timestep += CSP.bs[None, :, :, :]
    covs_after_initial_timestep = jnp.tile(CSP.Qs, (T - 1, 1, 1, 1, 1))  # (T-1,J,K,D, D)
    continuous_states_after_initial_timestep_axes_poorly_ordered = jnp.tile(
        continuous_states[1:], (K, 1, 1, 1)
    )  # (K,T-1,J,D)
    continuous_states_after_initial_timestep = jnp.moveaxis(
        continuous_states_after_initial_timestep_axes_poorly_ordered,
        [0, 1, 2],
        [2, 0, 1],
    )  # WANT: (T_1,J,K,D)
    log_pdfs_after_initial_timestep = mvn_JAX.logpdf(
        continuous_states_after_initial_timestep,
        means_after_initial_timestep,
        covs_after_initial_timestep,
    )

    # Pre-vectorized version for clarity
    # for t in range(1, T):
    #     for j in range(J):
    #         for k in range(K):
    #             # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
    #             mu_t = CSP.As[j, k] @ continuous_states[t - 1, j] + CSP.bs[j, k]
    #             Sigma_t = CSP.Qs[j, k]
    #             log_emissions.at[t, j, k].set(mvn_JAX.logpdf(continuous_states[t, j], mu_t, Sigma_t))

    # log_emissions = jnp.vstack(
    #     (log_pdfs_init_time[None, :, :], log_pdfs_remaining_times)
    # )

    return log_pdfs_after_initial_timestep


def compute_log_continuous_state_emissions_at_initial_timestep_JAX(
    IP: InitializationParameters_JAX,
    continuous_states: JaxNumpyArray3D,
):
    """
    Compute the log (autoregressive, switching) emissions for the continuous states at the INITIAL timestep
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Arguments:
        continuous_states : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D

    Returns:
        np.array of shape (J,K), where the (j,k)-th element gives the log emissions
        probability of the initial continuous state
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    ### Initial times <- Not computed
    # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    means_init_time, covs_init_time = IP.mu_0s, IP.Sigma_0s
    log_pdfs_init_time = mvn_JAX.logpdf(
        continuous_states[0, :, None, :], means_init_time, covs_init_time
    )
    # Pre-vectorized version of initial times...for clarity
    # for j in range(J):
    #     for k in range(K):
    #         # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    #         mu_0, Sigma_0 = IP.mu_0s[j, k], IP.Sigma_0s[j, k]
    #         log_emissions.at[0, j, k].set(mvn_JAX.logpdf(continuous_states[0, j], mu_0, Sigma_0))

    return log_pdfs_init_time