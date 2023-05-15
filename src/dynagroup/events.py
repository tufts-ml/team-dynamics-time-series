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

What is an event?

    An event can be considered as an (iid) training example from a group dynamics model.
    A separate event might induce a large time gap between time-steps, 
    and a discontinuity in the continuous states x. Thus is should be handled correctly. 

What is the data representation strategy in the presence of multiple events?

    The sequences for each event are stacked on top of one another. 
    So the dimensionality of observations will be (T,J,N), 
    where T is the total number of timesteps, J is the number of entities, 
    and N is the observation dimension. Here, T= T_1 + T_2 + ... +T_E,
    where E  is the number of events (or training examples). We track the ending indices of each event 
    for efficient indexing in `event_end_times`, which looks like 
    [-1, <bunch of times giving ends of all segments besides the last one>, T]
    In particular, if there are E events, then along with the observations, we store 
    event_end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th event ended.  
    So to get the timesteps for the e-th event, you can index from 1,…,T_grand by doing
        [event_end_times[e-1]+1 : event_end_times[e]].

    Examples:
        * If the data has shape (T,J,D)=(10,3,2), we might have event_end_times=[-1, 5, 10].
        * In the default case, where there are not multiple events, the `event_end_times` are 
    taken to be [-1,T].

What is the inference strategy in the presence of multiple events?

    Inspecting the proof for filtering and smoothing in HMM's with time-dependent transitions 
    (e.g., arHMMs) reveals that we can handle this situation as follows:

        * E-steps (VEZ or VES): Whenever we get to a cross-event boundary (so for any pair of timesteps 
            (t_1, t_2) where t_1 is in event_end_times), we replace the usual transition function with the 
            appropriate initial regime distribution. Similarly, whenever we have started a new event 
            (so for any timestep t where t+1 is in event_end_times), we replace the usual emissions with the 
            initial emissions.
        * M-steps: We update the initialization parameters IP using any data from any timesteps that ARE at 
            the beginning of an event (so any timesteps t where t+1 is in event_end_times). We update the continuous 
            state parameters CSP using data from any timesteps t that AREN'T at the beginning of an event 
            (so any timesteps t where t+1 is in event_end_times). We update the entity transition parameters ETP 
            and system transition parameters STP using any pair timesteps that don't straddle a transition boundary 
            (so we ignore any pair of timesteps (t_1, t_2) where t_1 is in event_end_times).
        
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
