import functools
from typing import List, Optional

import numpy as np

from dynagroup.diagnostics.fit_and_forecasting import (
    evaluate_fit_and_partial_forecast_on_slice,
)
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.params import AllParameters_JAX
from dynagroup.types import JaxNumpyArray3D


###
# HELPERS
###


def find_last_index_in_interval_where_array_value_is_close_to_desired_point(
    array: np.array,
    desired_point: np.array,
    starting_index: int,
    ending_index: int,
) -> float:
    closeness_threshold = 0.15

    for t in reversed(range(starting_index, ending_index)):
        if np.linalg.norm(array[t] - desired_point) < closeness_threshold:
            return t
    return np.nan


###
# MAIN
###


def evaluate_fit_and_partial_forecast_on_slice_for_figure_8(
    continuous_states: JaxNumpyArray3D,
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    T_slice_max: int,
    model: Model,
    forecast_seeds: List[int],
    save_dir: str,
    entity_idxs: Optional[List[int]] = None,
    filename_prefix: Optional[str] = None,
) -> None:
    # I want to work with when we transition from up to down (timesteps 100-200)

    find_t0_for_entity_sample = functools.partial(
        find_last_index_in_interval_where_array_value_is_close_to_desired_point,
        desired_point=np.array([1, 1]),
        starting_index=0,
        ending_index=len(continuous_states),
    )

    return evaluate_fit_and_partial_forecast_on_slice(
        continuous_states,
        params,
        VES_summary,
        VEZ_summaries,
        T_slice_max,
        model,
        forecast_seeds,
        save_dir,
        entity_idxs,
        find_t0_for_entity_sample,
        y_lim=(-2.5, 2.5),
        filename_prefix=filename_prefix,
    )
