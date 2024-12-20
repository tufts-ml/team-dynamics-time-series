import functools
from typing import List, Optional, Tuple

import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.model2a.figure8.diagnostics.posterior_mean_and_forward_simulation import (
    evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice,
)
from dynagroup.params import AllParameters_JAX
from dynagroup.types import JaxNumpyArray3D, NumpyArray1D, NumpyArray2D, NumpyArray4D


###
# HELPERS
###


def find_last_index_in_interval_where_array_value_is_close_to_desired_point(
    array: np.array,
    desired_point: np.array,
    starting_index: int,
    ending_index: int,
) -> float:
    closeness_threshold = 0.2

    for t in reversed(range(starting_index, ending_index)):
        if np.linalg.norm(array[t] - desired_point) < closeness_threshold:
            return t
    return np.nan


###
# MAIN
###


def evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    T_slice_max: int,
    model: Model,
    forward_simulation_seeds: List[int],
    save_dir: str,
    use_continuous_states: NumpyArray2D,
    entity_idxs: Optional[List[int]] = None,
    filename_prefix: Optional[str] = None,
) -> Tuple[NumpyArray1D, NumpyArray2D, NumpyArray1D, NumpyArray4D]:
    """
    Returns:
        MSEs_posterior_mean: An array of size (J,) that describes the model performance for each of the
            J entities over a time period requested for the posterior mean.
        MSEs_forward_sims: An array of size (J,S)  that describes the model performance for each of the
            J entities for each of S simulations over a time period requested for the forward sims.
            The value is NaN if the entity was not masked.
        MSEs_velocity_baseline: An array of size (J,) that describes the model performance for each of the
            J entities over the same time period as requested for the forward sims.
            The value is NaN if the entity was not masked.
        forecasts: An array of shape (S,T_forecast,J_forecast,D), where S is the number of simulations, J is the number of forecasted entities,
            D is the continuous state dimensionality, and T is the number of forecasted timesteps.  We put np.nan for any
            entities or timesteps where there was no forecasting.
    """
    # I want to work with when we transition from up to down (timesteps 100-200)

    find_forward_sim_t0_for_entity_sample = functools.partial(
        find_last_index_in_interval_where_array_value_is_close_to_desired_point,
        desired_point=np.array([0.6,0.1]),
        starting_index=0,
        ending_index=len(continuous_states),
    )

    return evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice(
        continuous_states,
        params,
        VES_summary,
        VEZ_summaries,
        model,
        forward_simulation_seeds,
        save_dir,
        use_continuous_states,
        entity_idxs,
        find_forward_sim_t0_for_entity_sample,
        max_forward_sim_window=T_slice_max,
        y_lim=(-2.5, 2.5),
        filename_prefix=filename_prefix,
    )
