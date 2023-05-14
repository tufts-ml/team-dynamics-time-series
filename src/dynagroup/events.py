from typing import Optional

import jax.numpy as jnp
import numpy as np

from dynagroup.model import Model
from dynagroup.params import InitializationParameters
from dynagroup.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray4D,
    NumpyArray1D,
)


"""
An `event` takes an ordinary sampled group time series of shape (T,J,:) and interprets it as (T_grand,J,:),
where T_grand is the sum of the number of timesteps across i.i.d "events".  An event might induce a large
time gap between timesteps, and a discontinuity in the continuous states x.

If there are E events, then along with the observations, we store 
    end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th event ended.  

So to get the timesteps for the e-th event, you can index from 1,…,T_grand by doing
        [end_times[e-1]+1 : end_times[e]].

Example:
    data has shape (T,D)=(10,2)
    event_times=[-1, 5, 10]
        
"""

"""

TODO: 
[DONE] 1. patch up entity transitions 
[DONE] 2. patch up emissions for continuous states
3. patch up system transitions 
4. patch up - overall function 

"""


###
# Helpers
###


def event_end_times_are_proper(event_end_times: NumpyArray1D, T: int) -> Optional[bool]:
    """event_end_times should look like [-1, <bunch of times giving ends of all segments besides the last one>, T]"""
    return event_end_times[0] == -1 and event_end_times[-1] == T


def get_initialization_times(event_end_times: NumpyArray1D) -> NumpyArray1D:
    """
    These are times just after event boundaries.
    """
    # want to use np and not lists here, so that we can add 1 to the event end times.
    return np.array(event_end_times[:-1]) + 1


def get_non_initialization_times(event_end_times: NumpyArray1D) -> NumpyArray1D:
    """
    These are times NOT just after event boundaries.
    """

    # TODO: The returned numpy array (of all NORMAL times) could get large if there are lots of timesteps. Is there an easier way to view all
    # indices EXCEPT?  I wanted to return a generator, but it seems that I can't index a numpy array
    # with a generator

    T = event_end_times[-1]
    initialization_times = get_initialization_times(event_end_times)
    return np.array([i for i in range(T) if i not in initialization_times])


def eligible_transitions_to_next(event_end_times: NumpyArray1D) -> NumpyArray1D:
    """
    The t-th timestep is not an eligible transition source if there is en event boundary between the t-th and the
    (t+1)-st timestep.

    Returns:
        A numpy array of booleans telling whether each timestep in (0,...,T-1) should be selected when performing
        an inference operation on transitions.

    Example:
        If event_end_times=[-1,4,10], this function returns
            array([ True,  True,  True,  True, False,  True,  True,  True,  True]).
        The value at index 4 is False because the transition from 4 to 5 is not eligible for doing inference.
        (It crosses an event boundary.)
    """
    T = event_end_times[-1]
    return np.isin(np.arange(T - 1), event_end_times[1:-1], invert=True)


###
# Fixing up emissions and transitions (for forward-backward when there are events)
###


def fix_log_system_transitions_at_event_boundaries(
    log_system_transitions: JaxNumpyArray4D,
    IP: InitializationParameters,
    event_end_times: NumpyArray1D,
) -> JaxNumpyArray4D:
    """
    Arguments:
        log_system_transitions: has shape (T-1, L, L)
    """
    L = np.shape(log_system_transitions)[2]

    log_system_transitions_fixed = np.array(log_system_transitions)

    # TODO: Vectorize this!
    # pi_system: has shape (L,).
    # We reshape this so that there the transitions to entity K are uniform across the rows
    log_transitions_to_destinations_per_init_dist = np.tile(np.log(IP.pi_system), (L, 1))
    for end_time in event_end_times[1:-1]:
        log_system_transitions_fixed[end_time] = log_transitions_to_destinations_per_init_dist
    return jnp.array(log_system_transitions_fixed)


def fix_log_entity_transitions_at_event_boundaries(
    log_entity_transitions: JaxNumpyArray4D,
    IP: InitializationParameters,
    event_end_times: NumpyArray1D,
) -> JaxNumpyArray4D:
    """
    Arguments:
        log_entity_transitions: has shape (T-1, J, K, K)
    """
    _, J, K, _ = np.shape(log_entity_transitions)

    log_entity_transitions_fixed = np.array(log_entity_transitions)

    # TODO: Vectorize all this
    for j in range(J):
        # pi_entities : has shape (J, K).
        # We reshape this so that there the transitions to entity K are uniform across the rows
        log_transitions_to_destinations_per_init_dist = np.tile(np.log(IP.pi_entities[j]), (K, 1))
        for end_time in event_end_times[1:-1]:
            log_entity_transitions_fixed[
                end_time, j
            ] = log_transitions_to_destinations_per_init_dist
    return jnp.array(log_entity_transitions_fixed)


def fix__log_emissions_from_system__at_event_boundaries(
    log_emissions_from_system: JaxNumpyArray2D,
    VEZ_expected_regimes: JaxNumpyArray3D,
    IP: InitializationParameters,
    event_end_times: NumpyArray1D,
) -> JaxNumpyArray3D:
    """
    Arguments:
        log_emissions_from_system: has shape (T, L)
        VEZ_expected_regimes: has shape (T,J,K)
        continuous_states: has shape (T,J,D)
    """
    L = np.shape(log_emissions_from_system)[1]

    log_emissions_from_system_fixed = np.array(log_emissions_from_system)

    ###
    # Reconstruct the initial log emissions from system
    ###

    # `iinitial_log_emissions_from_system` has shape (L,) and is obtained by summing over (J,K) objects
    initial_log_emission_for_each_system_regime = jnp.sum(
        VEZ_expected_regimes * np.log(IP.pi_entities)
    )
    initial_log_emissions_from_system = jnp.repeat(initial_log_emission_for_each_system_regime, L)

    # TODO: Maybe vectorize this
    for end_time in event_end_times[1:-1]:
        log_emissions_from_system_fixed[end_time + 1] = initial_log_emissions_from_system

    return jnp.array(log_emissions_from_system_fixed)


def fix__log_emissions_from_entities__at_event_boundaries(
    log_emissions_from_entities: JaxNumpyArray3D,
    continuous_states: JaxNumpyArray3D,
    IP: InitializationParameters,
    model: Model,
    event_end_times: NumpyArray1D,
) -> JaxNumpyArray3D:
    """
    Arguments:
        log_emissions_from_entities: has shape (T, J, K).  These are the log emissions associated to an entity
            transition function.  They are the continuous states, x.
        continuous_states: has shape (T,J,D)
    """
    log_emissions_from_entities_fixed = np.array(log_emissions_from_entities)

    # TODO: Vectorize all this
    for end_time in event_end_times[1:-1]:
        log_emissions_from_entities_fixed[
            end_time + 1
        ] = model.compute_log_initial_continuous_state_emissions_JAX(
            IP, continuous_states[end_time + 1]
        )

    return jnp.array(log_emissions_from_entities_fixed)
