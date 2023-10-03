import random
from dataclasses import dataclass
from typing import List

import numpy as np

from dynagroup.forecasts import Forecast_MSEs, MSEs_from_forecasts
from dynagroup.model import Model
from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    get_start_and_stop_timestep_idxs_from_event_idx,
)
from dynagroup.params import AllParameters_JAX
from dynagroup.types import NumpyArray1D, NumpyArray3D
from dynagroup.vi.vi_forecast import get_forecasts_on_test_set


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


@dataclass
class Forecast_MSEs_By_Event:
    """
    Includes raw and summary info.
        `forecasting_MSEs_by_examples` is raw.
        `mean_over_<*>` is summary.
    """

    forecasting_MSEs_by_examples: List[Forecast_MSEs]
    mean_over_J_median_forward_sims_by_example: List[float]
    mean_over_J_fixed_velocities_by_example: List[float]


def get_forecast_MSEs_by_event(
    xs_test: NumpyArray3D,
    example_stop_idxs_test: List[int],
    params_learned: AllParameters_JAX,
    model_basketball: Model,
    random_context_times: NumpyArray1D,
    T_forecast: int,
    n_cavi_iterations: int,
    n_forecasts: int,
    system_covariates,
) -> Forecast_MSEs_By_Event:
    """
    Arguments:
        random_context_times: array of shape (E,) and dtype float, whose value is np.nan if the event
            doesn't have enough timesteps to be used, and otherwise is a floatified integer
            specifying how many timesteps to use as context when forecasting on this event.
    """
    ### Get forecasting MSEs by inferred events
    forecasting_MSEs_by_examples = []
    num_events = len(example_stop_idxs_test) - 1
    for event_idx in range(num_events):
        print(f"--- --- Now making forecasts for event {event_idx+1}/{num_events}. --- ---")
        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(example_stop_idxs_test, event_idx)
        random_context_time_float = random_context_times[event_idx]

        if np.isnan(random_context_time_float):
            continue

        forecasts = get_forecasts_on_test_set(
            xs_test[start_idx:stop_idx],
            params_learned,
            model_basketball,
            int(random_context_time_float),
            T_forecast,
            n_cavi_iterations,
            n_forecasts,
            system_covariates,
        )
        forecasting_MSEs = MSEs_from_forecasts(forecasts)
        forecasting_MSEs_by_examples.append(forecasting_MSEs)

    ### Summarize
    mean_over_J_median_forward_sims_by_example = []
    mean_over_J_fixed_velocities_by_example = []
    for i, forecasting_MSEs in enumerate(forecasting_MSEs_by_examples):
        mean_median_forward_sim = np.mean(
            np.median(forecasting_MSEs.forward_simulation, 0)[0]
        )  # median over S, mean over J
        mean_fixed_velocity = np.mean(forecasting_MSEs.fixed_velocity[0])  # mean over J
        print(
            f"For event {i}, forward sim: {mean_median_forward_sim:.02f}, mean_fixed_velocity: {mean_fixed_velocity:.02f}"
        )
        mean_over_J_median_forward_sims_by_example.append(mean_median_forward_sim)
        mean_over_J_fixed_velocities_by_example.append(mean_fixed_velocity)

    return Forecast_MSEs_By_Event(
        forecasting_MSEs_by_examples,
        mean_over_J_median_forward_sims_by_example,
        mean_over_J_fixed_velocities_by_example,
    )
