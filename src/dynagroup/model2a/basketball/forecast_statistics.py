from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
from scipy.stats import circvar, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

from dynagroup.model2a.basketball.court import (
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
)
from dynagroup.types import NumpyArray1D, NumpyArray5D


###
# Structs
###


@dataclass
class Summary_Of_Forecast_Statistic:
    name: str
    num_valid_examples: int
    MEAN_STAT_E: NumpyArray1D
    MEAN_STAT: float
    SE_MEAN_STAT: float


###
# Computation of specific statistics
###


def compute_in_bounds_pcts_by_example(forecasts: NumpyArray5D) -> NumpyArray1D:
    # TODO: Should we restrict the computation to valid examples ? (Forecasts are made on ALL examples,
    # but ground truth is available only for VALID examples)
    E = np.shape(forecasts)[0]

    in_bounds_pcts = np.zeros(E)
    for e in range(E):
        in_bounds_pcts[e] = np.mean(
            (forecasts[e, ..., 0] >= X_MIN_COURT)
            * (forecasts[e, ..., 0] <= X_MAX_COURT)
            * (forecasts[e, ..., 1] >= Y_MIN_COURT)
            * (forecasts[e, ..., 1] <= Y_MAX_COURT)
        )
    return in_bounds_pcts


def compute_dispersions_by_example(forecasts: NumpyArray5D) -> NumpyArray1D:
    # TODO: Should we restrict the computation to valid examples ? (Forecasts are made on ALL examples,
    # but ground truth is available only for VALID examples)
    E = np.shape(forecasts)[0]

    dispersions_by_example = np.zeros(E)
    for e in range(E):
        typical_centroid_distance_by_sample = np.sqrt(np.sum(np.var(forecasts[e, :, -1, :5], axis=1), axis=1))
        dispersions_by_example[e] = np.mean(typical_centroid_distance_by_sample)
    return dispersions_by_example


def compute_directional_variability_by_example(forecasts: NumpyArray5D, CLE_only: bool = False) -> NumpyArray1D:
    """
    We interpret directional variability during the forecasting window as
        Circular variance of (X_forecasted_T2 - X_forecasted_T1).

    It can be taken as a measure of coordination.

    Arguments:
        forecasts: (E,S,T_forecast,J,D)
        CLE_only : If true, we only look at variability across the CAVS.
    """
    # TODO: Restrict to valid examples ?
    E, S = np.shape(forecasts)[:2]

    def get_circular_variance_for_one_example_and_sample(forecasts_by_example_and_sample):
        """
        Arguments:
            forecasts_by_example_and_sample: (T_forecast, J, D)
        """
        player_forecast_secants = forecasts_by_example_and_sample[-1] - forecasts_by_example_and_sample[0]  # J,D
        player_angles = np.arctan2(player_forecast_secants[:, 1], player_forecast_secants[:, 0])
        return circvar(player_angles)

    if CLE_only:
        num_players_to_use = 5
    else:
        num_players_to_use = 10

    directional_variability_by_example = np.zeros(E)
    for e in range(E):
        circular_variances_for_samples_on_this_example = np.zeros(S)
        for s in range(S):
            circular_variances_for_samples_on_this_example[s] = get_circular_variance_for_one_example_and_sample(
                forecasts[e, s, :, :num_players_to_use]
            )
        directional_variability_by_example[e] = np.nanmean(circular_variances_for_samples_on_this_example)
    return directional_variability_by_example


###
#  Summarize Statistics and Test whether models differ on them
###
class Forecast_Statistic(Enum):
    Pct_In_Bounds = 1
    Directional_Variabilities = 2


FORECAST_STATISTIC_TO_FUNCTION_FOR_COMPUTING_IT_OVER_EXAMPLES = {
    Forecast_Statistic.Pct_In_Bounds: compute_in_bounds_pcts_by_example,
    Forecast_Statistic.Directional_Variabilities: compute_directional_variability_by_example,
}
FORECAST_STATISTICS = list(FORECAST_STATISTIC_TO_FUNCTION_FOR_COMPUTING_IT_OVER_EXAMPLES.keys())

# Remark: We do not officially include `compute_dispersions_by_example` as a valid statistic.
# It was my first attempt at capturing coherence, but dispersion is not sufficiently dissociated
# from pct_in_bounds.  It seems to behave as a mix of the two.


def compute_summaries_of_forecast_statistic(
    forecast_statistic: Forecast_Statistic, forecasts_dict: Dict[str, NumpyArray5D]
) -> Dict[str, Summary_Of_Forecast_Statistic]:
    """
    Arguments
        name_of_statistic: Must be a key in STATISTIC_FUNCTION_BY_NAME dictionary
    """

    if forecast_statistic not in Forecast_Statistic:
        raise ValueError("I do not recognize the desired forecast statistic")

    print("\n")
    statistics_summary_dict = {}
    for model_name, forecasts in forecasts_dict.items():
        func_to_compute_statistics = FORECAST_STATISTIC_TO_FUNCTION_FOR_COMPUTING_IT_OVER_EXAMPLES[forecast_statistic]
        statistics_by_example = func_to_compute_statistics(forecasts)
        num_valid_examples = len(statistics_by_example)
        statistics_summary_dict[model_name] = Summary_Of_Forecast_Statistic(
            name=forecast_statistic.name,
            num_valid_examples=num_valid_examples,
            MEAN_STAT_E=statistics_by_example,
            MEAN_STAT=np.nanmean(statistics_by_example),
            SE_MEAN_STAT=np.nanstd(statistics_by_example) / np.sqrt(num_valid_examples),
        )
        print(
            f" {statistics_summary_dict[model_name].name} for {model_name} is {statistics_summary_dict[model_name].MEAN_STAT : .02f} ({statistics_summary_dict[model_name].SE_MEAN_STAT : .03f})"
        )
    return statistics_summary_dict


def compute_model_comparison_results_for_forecast_statistic(
    statistics_summary_dict: Dict[str, Summary_Of_Forecast_Statistic],
    focal_models_to_competitor_models: Dict[str, List[str]],
    alpha: float,
    alternative: str = "two-sided",
) -> Dict:
    """
    Arguments:
        statistics_summary_dict: return value of compute_summaries_of_forecast_statistics_dict()
        focal_models_to_competitor_models:  Maps a focal model (string which is a key of statistics_summary_dict and forecasts_dict)
            to a list of competitor models (strings which are keys of statistics_summary_dict and forecasts_dict)
                Example: = {"ours": ["no_system_switches", "no_recurrence"]}
        alpha: significance level of test
        alternative: refers to the nature of the alternative hypothesis. in ["less", "greater", "two sided"].
    """
    uncorrected_p_vals_dict = OrderedDict()
    SE_diffs = []

    ### Make uncorrected p-values
    for size in ["small", "medium", "large"]:
        for focal_model, competitor_models in focal_models_to_competitor_models.items():
            for competitor in competitor_models:
                diffs = (
                    statistics_summary_dict[f"{competitor}_{size}"].MEAN_STAT_E
                    - statistics_summary_dict[f"{focal_model}_{size}"].MEAN_STAT_E
                )
                SE_diff = np.nanstd(diffs) / np.sqrt(statistics_summary_dict[f"ours_{size}"].num_valid_examples)
                # t_stat_by_hand = np.nanmean(diffs)/np.nanstd(diffs)/np.sqrt(statistics_summary_dict[f"ours_{size}"].num_valid_examples)
                t_stat, p_val = ttest_1samp(diffs, popmean=0, nan_policy="omit", alternative=alternative)
                uncorrected_p_vals_dict[(size, focal_model, competitor)] = p_val
                SE_diffs.append(SE_diff)

    ### Make corrected p-values
    # TODO: confirm it makes sense to assume that tests are positively correlated for all comparisons

    reject_hypoth, pvals_corrected = fdrcorrection(
        list(uncorrected_p_vals_dict.values()), alpha=alpha, method="poscorr", is_sorted=False
    )
    test_results_dict = OrderedDict()
    for i, comparison in enumerate(uncorrected_p_vals_dict.keys()):
        test_results_dict.update([(comparison, (reject_hypoth[i], f"{pvals_corrected[i]:.3e}", f"{SE_diffs[i]:.3f}"))])
    return test_results_dict
