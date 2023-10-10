from typing import Callable, Optional, Union

import jax.numpy as jnp
from scipy.stats import vonmises as vonmises

from dynagroup.model import Model
from dynagroup.params import (
    ContinuousStateParameters_VonMises_JAX,
    EntityTransitionParameters_JAX,
    InitializationParameters_VonMises_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
)
from dynagroup.util import normalize_log_potentials_by_axis_JAX


def zero_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX(
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
    # RK: In the gaussian model, Upsilon also handles recurrence, but we haven't updated the circle
    # model to handle this yet.
    bias_from_system_covariates = jnp.einsum("ld,td->tl", STP.Upsilon, system_covariates[:-1])  # (T-1, L)

    # Pi: has shape (L, L)
    log_potentials = (
        bias_from_system_covariates[:, None, :] + STP.Pi[None, :, :]
    )  # (T-1, None, L) + (None,L,L) = (T-1, L,L)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=2)


def compute_log_entity_transition_probability_matrices_JAX(
    ETP_JAX: EntityTransitionParameters_JAX,
    x_prevs: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX: Callable = None,
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


def compute_log_continuous_state_emissions_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_VonMises_JAX,
    group_angles: Union[JaxNumpyArray2D, JaxNumpyArray3D],
) -> JaxNumpyArray3D:
    """
    Compute the log (autoregressive, switching) emissions for the continuous states, where we have
        x_t^j ~ VM( ar_coefs[j,k] @ x_{t-1}^j + drift[j,k] , kappa[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Note that we do NOT include the initial state
        x_0^j ~ ????
    which is computed elsewhere.

    Arguments:
        group_angles : array of shape (T,J) or (T,J,1) where the (t,j)-th entry is an angle in [-pi,pi]

    Returns:
        array of shape (T-1,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the (t+1)-st angle (given the (t)-th angle)
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        K: number of entity-level regimes
    """
    # force there to be a third array dimensions for D
    T, J = jnp.shape(group_angles)[:2]
    K = jnp.shape(CSP.ar_coefs)[1]
    group_angles = group_angles.reshape((T, J))  # throw away D=1 axis dimension if it exists.

    means_after_initial_timestep = jnp.einsum("jk,tj->tjk", CSP.ar_coefs, group_angles[:-1])  # (T-1,J,K)
    means_after_initial_timestep += CSP.drifts[None, :, :]
    concentrations_after_initial_timestep = jnp.tile(CSP.kappas, (T - 1, 1, 1))  # (T-1,J,K)

    group_angles_after_initial_timestep_axes_poorly_ordered = jnp.tile(group_angles[1:], (K, 1, 1))  # (K,T-1,J)
    group_angles_after_initial_timestep = jnp.moveaxis(
        group_angles_after_initial_timestep_axes_poorly_ordered,
        [0, 1, 2],
        [2, 0, 1],
    )  # WANT: (T_1,J,K)

    # The jax version of the `vonmises.logpdf` function gives
    #   TypeError: logpdf() takes 2 positional arguments but 3 were given
    # Using the numpy version doesn't seem to be slow, so I guess we'll go with that?
    # Hopefully we won't run into autodiff problems, but I'm running my own code
    # for the M-step so hopefully it will be ok
    log_pdfs_after_initial_timestep = vonmises.logpdf(
        group_angles_after_initial_timestep,
        concentrations_after_initial_timestep,
        loc=means_after_initial_timestep,
    )

    return log_pdfs_after_initial_timestep


def compute_log_initial_continuous_state_emissions_JAX(
    IP: InitializationParameters_VonMises_JAX,
    initial_group_angles: Union[JaxNumpyArray1D, JaxNumpyArray2D],
) -> JaxNumpyArray2D:
    """
    Compute the log (autoregressive, switching) emissions for the continuous states at the INITIAL timestep
        x_0^j ~ VM( loc[j,k], kappa[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    For now, we are just emitting equal numbers for all J,K.  See remark.

    Arguments:
        group_angles : array of shape (J,) or (J,1) where the (j)-th entry is an angle in [-pi,pi]

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

    Remarks:
        How should we handle the initial emission, since there is no previous emission from which we can compute the mean?
            1. Ignore the initial observation.
                I’ve noticed that just assigning equal likelihoods to the initial observation across all regimes
                works pretty well; any sticky transition probability matrix will then assign the initial observation
                to the the same cluster as the 2nd observation, which seems nice.
            2. Specify a distribution to model the initial observation.  The problem is how to learn this.
                The mean could be set to the observation itself.  But then what would the variance be?
                A Bayesian approach helps here, since a the single observation can be used to update a prior,
                whereas a frequentist approach can’t compute a variance with one observation.
            2a. Since we’re modeling grouped time series, we could perhaps learn about the initial distribution
                by taking the mean and variance of the initial observations across entities.
                But I think it doesn’t always make sense to assume that these initial observations
                all come from the same distribution.   (Indeed, we assume all the other observations
                come from different distributions.
            3.  Set the mean equal to the observed value, and leave the variance at the initialized value.

        For now, we're taking approach #3... but it's effectively the same as approach #1.
    """

    ### Initial times
    # We have  x_0^j ~ VM( loc[j,k], kappa[j,k] )

    # TODO: Group angles could be (T,J,1) or (T,J). That's causing problems.  Pick a representation
    # and use it throughout!
    initial_group_angles = jnp.squeeze(initial_group_angles)
    log_pdfs_init_time = vonmises.logpdf(initial_group_angles[:, None], IP.kappas, loc=IP.locs)
    return log_pdfs_init_time


circle_model_JAX = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    zero_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)
