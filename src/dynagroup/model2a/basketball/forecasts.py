from typing import Optional

import numpy as np

from dynagroup.forecasts import (
    MSEs_from_forecasts,
    make_forecast_MSEs_summary,
    plot_forecasts,
    save_forecasts,
)
from dynagroup.model import Model
from dynagroup.model2a.basketball.data.baller2vec.disk import Processed_Data_To_Analyze
from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    get_start_and_stop_timestep_idxs_from_event_idx,
)
from dynagroup.model2a.gaussian.forecasts import get_forecasts_on_test_set_example
from dynagroup.params import AllParameters_JAX


###
# Make forecasts on test set examples
###


def run_basketball_forecasts(
    data: Processed_Data_To_Analyze,
    model: Model,
    params: AllParameters_JAX,
    system_covariates: Optional[np.array],
    n_cavi_iterations_for_forecasting: int,
    n_forecasts_per_example: int,
    random_forecast_starting_points: bool,
    T_forecast: int,
    save_dir: str,
    n_forecasting_examples_to_plot: int = 0,
    n_forecasting_examples_to_analyze: float = np.inf,
):
    """
    Makes forecasts, saves to disk, saves desired number of plots, gives evaluation metrics along the way.
    """

    xs_test = np.asarray(data.test.player_coords)
    example_end_times_test = data.test.example_stop_idxs
    random_context_times = data.random_context_times

    forecast_MSEs_summary_by_example_idx = {}
    event_idxs_to_analyze = [
        i for (i, random_context_time) in enumerate(random_context_times) if not np.isnan(random_context_time)
    ]

    for e, event_idx_to_analyze in enumerate(event_idxs_to_analyze):
        if (e + 1) > n_forecasting_examples_to_analyze:
            break

        start_idx, stop_idx = get_start_and_stop_timestep_idxs_from_event_idx(
            example_end_times_test, event_idx_to_analyze
        )

        if random_forecast_starting_points:
            # Use random starting times for forecast
            T_context = int(random_context_times[event_idx_to_analyze])
        else:
            # Use fixed starting times for forecast (last N steps)
            T_example = stop_idx - start_idx
            T_context = T_example - T_forecast

        forecasts = get_forecasts_on_test_set_example(
            xs_test[start_idx:stop_idx],
            params,
            model,
            T_context,
            T_forecast,
            n_cavi_iterations_for_forecasting,
            n_forecasts_per_example,
            system_covariates,
        )

        save_forecasts(
            forecasts,
            save_dir,
            forecast_description=f"random_forecast_starting_points_{random_forecast_starting_points}_T_forecast_{T_forecast}",
            event_description=f"example_idx_{e}_orig_example_idx_{event_idx_to_analyze}_start_idx_{start_idx}_stop_idx_{stop_idx}_T_context_{T_context}",
        )

        try:
            forecast_MSEs = MSEs_from_forecasts(forecasts)
        except TypeError:
            print(f"Couldn't process event {e}")
            continue

        forecast_MSEs_summary = make_forecast_MSEs_summary(forecast_MSEs)
        forecast_MSEs_summary_by_example_idx[event_idx_to_analyze] = forecast_MSEs_summary

        print(
            f"---Results for example-to-analyze {event_idx_to_analyze}, with start time  {start_idx} and stop time  {stop_idx} ---"
        )

        print(
            f"Mean MSEs:  "
            f"Fixed velocity: {forecast_MSEs_summary.mean_fixed_velocity:.03f}. "
            f"Forward_simulation: {forecast_MSEs_summary.mean_forward_simulation:.03f}. "
        )

        print(
            f"Mean MSEs (CLE only):  "
            f"Fixed velocity: {forecast_MSEs_summary.mean_fixed_velocity_CLE_only:.03f}. "
            f"Forward_simulation: {forecast_MSEs_summary.mean_forward_simulation_CLE_only:.03f}. "
        )

        print(
            f"Median MSEs:  "
            f"Fixed velocity: {forecast_MSEs_summary.median_fixed_velocity:.03f}. "
            f"Forward_simulation: {forecast_MSEs_summary.median_forward_simulation:.03f}. "
        )

        if (e + 1) <= n_forecasting_examples_to_plot:
            plot_forecasts(
                forecasts,
                forecast_MSEs,
                save_dir,
                filename_prefix=f"forecast_plot_example_idx_{e}_orig_example_idx_{event_idx_to_analyze}_start_idx_{start_idx}_stop_idx_{stop_idx}",
            )

        # ### running summary metrics
        # median_ours = np.median([v.median_forward_simulation for v in forecast_MSEs_summary_by_example_idx.values()])
        # median_fixed_velocity = np.median([v.median_fixed_velocity for v in forecast_MSEs_summary_by_example_idx.values()])
        # print(f"Median after {e+1} examples.  Fixed velocity: {median_fixed_velocity:.03f}. Ours: {median_ours:.03f}.")

        mean_ours = np.mean([v.mean_forward_simulation for v in forecast_MSEs_summary_by_example_idx.values()])
        mean_fixed_velocity = np.mean([v.mean_fixed_velocity for v in forecast_MSEs_summary_by_example_idx.values()])
        SEM_ours = np.std([v.mean_forward_simulation for v in forecast_MSEs_summary_by_example_idx.values()]) / np.sqrt(
            e + 1
        )
        SEM_fixed_velocity = np.std(
            [v.mean_fixed_velocity for v in forecast_MSEs_summary_by_example_idx.values()]
        ) / np.sqrt(e + 1)
        print(
            f"Mean (SEM) after {e+1} examples.  Fixed velocity: {mean_fixed_velocity:.03f} ({SEM_fixed_velocity:.03f}). Ours: {mean_ours:.03f} ({SEM_ours:.03f})."
        )
