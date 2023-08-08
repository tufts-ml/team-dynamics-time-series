import random
from typing import List

import numpy as np

from dynagroup.types import NumpyArray1D, NumpyArray3D


# TODO (MODULE-LEVEL):  The two chunking functions, `chunkify_xs_into_events_which_have_sufficient_length`
# and `generate_random_context_times_for_x_chunks` are used by the demo where we currently create
# processed data dynamically.  But these are unnecessary if we use the
# `generate_random_context_times_for_events` function, which generates static data on disk for exporting
# to Preetish for AgentFormer.  Ideally we should rewrite our forecasting so that we only need
# one kind of function, presumably `generate_random_context_times_for_events`.  Then we only need
# to have 1 function here instead of 3, and we remove redundancies that can cause problems upon
# further development. I am holding off on this currently because it's lower priority than
# some other things, and also because I'm waiting to see if we will even use random context
# times anyhow (it seems to be hard to pull off with AgentFormer).


def chunkify_xs_into_events_which_have_sufficient_length(
    event_end_times: List[int],
    xs: NumpyArray3D,
    min_event_length: int,
) -> List[NumpyArray3D]:
    """
    We want to divide the continguous x's into chunks/event/plays, but only
    if they are sufficiently long that we can define a sufficiently long
    context period and forecasting period.
    """
    event_end_times = np.array(event_end_times)
    x_chunks = []
    event_lengths_in_timesteps = event_end_times[1:] - event_end_times[:-1]
    for i, event_length_in_timesteps in enumerate(event_lengths_in_timesteps):
        if event_length_in_timesteps >= min_event_length:
            start_idx = max(event_end_times[i], 0)
            end_idx = event_end_times[i + 1]
            x_chunks.append(xs[start_idx:end_idx])
    return x_chunks


def generate_random_context_times_for_x_chunks(
    x_chunks: List[NumpyArray3D],
    T_context_min: int,
    T_forecast: int,
) -> List[int]:
    """
    Pick a random number in [T_context_min, T_chunk-T_forecast] to be the context size
    for the chunk.

    Arguments:
        x_chunks_test: return value of `chunkify_xs_into_events_which_have_sufficient_length`.
            The list has length n_chunks, and each entry is array with shape (T_chunk, J, D)

    Returns:
        List of length n_chunks
    """
    n_chunks = len(x_chunks)
    T_contexts_random = np.zeros(n_chunks, dtype=int)
    for i, x_chunk in enumerate(x_chunks):
        T_chunk = len(x_chunk)
        T_contexts_random[i] = random.randint(T_context_min, T_chunk - T_forecast)
    return T_contexts_random


def generate_random_context_times_for_events(
    event_end_times: List[int],
    min_event_length: int,
    T_context_min: int,
    T_forecast: int,
) -> NumpyArray1D:
    """
    Pick a random number in [T_context_min, T_chunk-T_forecast] to be the context size
    for the chunk.
    Arguments:
        event_end_times:  array of shape (E+1,), where E is the number of events.
            Note that the 0th element is always -1

    Returns:
        random_context_times: array of shape (E,) and dtype float, whose value is np.nan if the event
            doesn't have enough timesteps to be used, and otherwise is a floatified integer
            specifying how many timesteps to use as context when forecasting on this event.
    """
    event_end_times = np.array(event_end_times)
    event_lengths_in_timesteps = event_end_times[1:] - event_end_times[:-1]

    n_events = len(event_lengths_in_timesteps)
    T_contexts_random = np.full((n_events,), fill_value=np.nan)
    for i, T_this_event in enumerate(event_lengths_in_timesteps):
        use_event = T_this_event >= min_event_length
        if use_event:
            T_contexts_random[i] = random.randint(T_context_min, T_this_event - T_forecast)
    return T_contexts_random
