from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX

from dynagroup.model import Model
from dynagroup.params import (
    ContinuousStateParameters_JAX,
    EntityTransitionParameters_JAX,
    InitializationParameters_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import JaxNumpyArray1D, JaxNumpyArray3D, JaxNumpyArray5D
from dynagroup.util import normalize_log_potentials_by_axis_JAX


def zero_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX(
    x_vec: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Arguments:
        x_vec : the continuous state for some entity j at some time t-1.

    Returns:
        f(x_vec), a transformation of x_vec; if Psi is a recurrence matrix, then
        the contribution of recurrence to the entity-level regime destinatons is Psi @ f(x_vec).
    """
    return jnp.zeros(1)


def compute_log_system_transition_probability_matrices_JAX(
    STP: SystemTransitionParameters_JAX,
    T_minus_1: int,
    system_covariates: Optional[jnp.array] = None,
):
    """
    Compute log system transition probability matrices.
    These are time varying, but only if at least one of the following conditions are true:
        * system-level covariates exist (in which case the function signature needs to be updated).
            The covariate effect is governed by the parameters in STP.Upsilon
        * there is recurrent feedback from the previous entities zs[t-1], as in Model 1.
            The recurrence effect is governed by the parameters in STP.Gammas

    Arguments:
        T_minus_1: The number of timesteps minus 1.  This is used instead of T because the initial
            system regime probabilities are governed by the initial parameters.
        system_covariates: An optional array of shape (T, M_s).

    Returns:
        np.array of shape (T-1,L,L).  The (t,l,l')-th element gives the probability of transitioning
            from regime l to regime l' when transitioning into time t+1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
    """
    # TODO: Check t vs t-1
    bias_from_system_covariates = jnp.einsum(
        "ld,td->tl", STP.Upsilon, system_covariates[:-1]
    )  # (T-1, L)

    # Pi: has shape (L, L)
    log_potentials = (
        bias_from_system_covariates[:, None, :] + STP.Pi[None, :, :]
    )  # (T-1, None, L) + (None,L,L) = (T-1, L,L)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=2)


def compute_log_entity_transition_probability_matrices_JAX(
    ETP_JAX: EntityTransitionParameters_JAX,
    x_prevs: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable = None,
) -> JaxNumpyArray5D:
    """
    Compute log entity transition probability matrices.

    Arguments:
        ETP_JAX:
            See `EntityTransitionParameters` class definition for more details.
        x_prevs : jnp.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D
            for t=1,...,T-1.   If `sample` is an instance of the `Sample` class, this
            object can be obtained by doing sample.xs[:-1], which gives all the x's except
            the one at the final timestep.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Returns:
        jnp.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the probability of
            the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives P(z_{t+1}^j = k' | z_t^j =k, s_{t+1}=l).
            for t=1,...,T-1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    T_minus_1 = jnp.shape(x_prevs)[0]
    # TODO: Consider adding recurrence
    return jnp.tile(ETP_JAX.Ps, (T_minus_1, 1, 1, 1, 1))  # (T-1, J, L, K, K)


def compute_log_continuous_state_emissions_JAX(
    CSP: ContinuousStateParameters_JAX,
    IP: InitializationParameters_JAX,
    continuous_states: JaxNumpyArray3D,
    compute_log_continuous_state_emissions_at_initial_timestep_JAX: Callable,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX: Callable,
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

    ### Initial times
    # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    log_pdfs_init_time = compute_log_continuous_state_emissions_at_initial_timestep_JAX(
        IP,
        continuous_states,
    )

    # Pre-vectorized version for clarity
    # for j in range(J):
    #     for k in range(K):
    #         # We have x_0^j ~ N(mu_0[j,k], Sigma_0[j,k])
    #         mu_0, Sigma_0 = IP.mu_0s[j, k], IP.Sigma_0s[j, k]
    #         log_emissions.at[0, j, k].set(mvn_JAX.logpdf(continuous_states[0, j], mu_0, Sigma_0))

    #### Remaining times
    log_pdfs_remaining_times = compute_log_continuous_state_emissions_after_initial_timestep_JAX(
        CSP, continuous_states
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


# TODO: We haven't yet actually defined the below properly.  TODO!
compute_log_continuous_state_emissions_JAX_NONE = None
compute_log_continuous_state_emissions_at_initial_timestep_JAX_NONE = None
compute_log_continuous_state_emissions_after_initial_timestep_JAX_NONE = None

circle_model_JAX = Model(
    compute_log_continuous_state_emissions_JAX_NONE,
    compute_log_continuous_state_emissions_at_initial_timestep_JAX_NONE,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX_NONE,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    zero_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
