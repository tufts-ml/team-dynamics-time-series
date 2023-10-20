import random
from typing import List

import numpy as np

from dynagroup.types import NumpyArray1D


def generate_random_context_times_for_events(
    example_end_times: List[int],
    min_event_length: int,
    T_context_min: int,
    T_forecast: int,
) -> NumpyArray1D:
    """
    Pick a random number in [T_context_min, T_chunk-T_forecast] to be the context size
    for the chunk.

    Arguments:
        example_end_times:  array of shape (E+1,), where E is the number of events.
            Note that the 0th element is always -1

    Returns:
        random_context_times: array of shape (E,) and dtype float, whose value is np.nan if the event
            doesn't have enough timesteps to be used, and otherwise is a floatified integer
            specifying how many timesteps to use as context when forecasting on this event.
    """
    example_end_times = np.array(example_end_times)
    event_lengths_in_timesteps = example_end_times[1:] - example_end_times[:-1]

    n_events = len(event_lengths_in_timesteps)
    T_contexts_random = np.full((n_events,), fill_value=np.nan)
    for i, T_this_event in enumerate(event_lengths_in_timesteps):
        use_event = T_this_event >= min_event_length
        if use_event:
            T_contexts_random[i] = random.randint(T_context_min, T_this_event - T_forecast)
    return T_contexts_random
