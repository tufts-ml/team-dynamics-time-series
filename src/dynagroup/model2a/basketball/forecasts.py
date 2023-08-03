import random
from typing import List

import numpy as np

from dynagroup.types import NumpyArray3D


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
