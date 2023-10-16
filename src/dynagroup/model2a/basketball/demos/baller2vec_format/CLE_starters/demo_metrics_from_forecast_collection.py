import numpy as np 

from dynagroup.model2a.basketball.forecast_analysis import (
    compute_metrics,
    load_agentformer_forecasts,
    load_dynagroup_forecasts,
)


### Get forecasts (agentformer)
forecasts_dict = {}
forecasts_dict["agentformer_small"] = load_agentformer_forecasts("small")
forecasts_dict["agentformer_medium"] = load_agentformer_forecasts("medium")
forecasts_dict["agentformer_large"] = load_agentformer_forecasts("large")

forecasts_ours_small_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_CAVI_its_2_timestamp__10-12-2023_00h20m38s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_ours_medium_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_5_CAVI_its_2_timestamp__10-12-2023_00h20m21s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_ours_large_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_20_CAVI_its_2_timestamp__10-12-2023_00h19m22s_forecasts_random_forecast_starting_points_True_T_forecast_30/"

forecasts_no_recurrence_small_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_No_Recurrence_train_1_CAVI_its_2_timestamp__10-12-2023_00h20m43s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_no_recurrence_medium_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_No_Recurrence_train_5_CAVI_its_2_timestamp__10-12-2023_00h20m34s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_no_recurrence_large_dir = "results/basketball/CLE_starters/artifacts/L=5_K=10_model_type_No_Recurrence_train_20_CAVI_its_2_timestamp__10-12-2023_00h19m53s_forecasts_random_forecast_starting_points_True_T_forecast_30/"

forecasts_no_system_switches_small_dir = "results/basketball/CLE_starters/artifacts/L=1_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_1_CAVI_its_2_timestamp__10-13-2023_10h07m00s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_no_system_switches_medium_dir = "results/basketball/CLE_starters/artifacts/L=1_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_5_CAVI_its_2_timestamp__10-13-2023_10h06m44s_forecasts_random_forecast_starting_points_True_T_forecast_30/"
forecasts_no_system_switches_large_dir = "results/basketball/CLE_starters/artifacts/L=1_K=10_model_type_Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence_train_20_CAVI_its_2_timestamp__10-13-2023_10h04m37s_forecasts_random_forecast_starting_points_True_T_forecast_30/"


forecasts_ours_small, fixed_velocity, ground_truth = load_dynagroup_forecasts(forecasts_ours_small_dir)
forecasts_ours_medium, _, _ = load_dynagroup_forecasts(forecasts_ours_medium_dir)
forecasts_ours_large, _, _ = load_dynagroup_forecasts(forecasts_ours_large_dir)

forecasts_no_recurrence_small, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_small_dir)
forecasts_no_recurrence_medium, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_medium_dir)
forecasts_no_recurrence_large, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_large_dir)

forecasts_no_system_switches_small, _, _ = load_dynagroup_forecasts(forecasts_no_system_switches_small_dir)
forecasts_no_system_switches_medium, _, _ = load_dynagroup_forecasts(forecasts_no_system_switches_medium_dir)
forecasts_no_system_switches_large, _, _ = load_dynagroup_forecasts(forecasts_no_system_switches_large_dir)


forecasts_dict["ours_small"] = forecasts_ours_small
forecasts_dict["ours_medium"] = forecasts_ours_medium
forecasts_dict["ours_large"] = forecasts_ours_large

forecasts_dict["no_system_switches_small"] = forecasts_no_system_switches_small
forecasts_dict["no_system_switches_medium"] = forecasts_no_system_switches_medium
forecasts_dict["no_system_switches_large"] = forecasts_no_system_switches_large

forecasts_dict["no_recurrence_small"] = forecasts_no_recurrence_small
forecasts_dict["no_recurrence_medium"] = forecasts_no_recurrence_medium
forecasts_dict["no_recurrence_large"] = forecasts_no_recurrence_large


forecasts_dict["fixed_velocity_small"] = fixed_velocity
forecasts_dict["fixed_velocity_medium"] = fixed_velocity
forecasts_dict["fixed_velocity_large"] = fixed_velocity

###  Run metrics
print("\n")
metrics_dict = {}
for model_name, forecasts in forecasts_dict.items():
    metrics_dict[model_name] = compute_metrics(forecasts, ground_truth)
    print(
        f"MEAN DIST (SE) - Both teams for {model_name} is {metrics_dict[model_name].BOTH_TEAMS__MEAN_DIST : .01f} ({metrics_dict[model_name].BOTH_TEAMS__SE_MEAN_DIST : .01f})"
    )

print("\n")
for model_name in metrics_dict.keys():
    print(
        f"MEAN DIST (SE) - CLE for {model_name} is {metrics_dict[model_name].CLE__MEAN_DIST : .02f} ({metrics_dict[model_name].CLE__SE_MEAN_DIST : .02f} )"
    )


####
# Paired difference test
###

import pprint
from collections import OrderedDict

from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection


focal_models_to_competitor_models = {
    "ours": ["agentformer", "no_system_switches", "no_recurrence", "fixed_velocity"],
}

uncorrected_p_vals_dict = OrderedDict()
SE_diffs = []

### Make uncorrected p-values
for size in ["small", "medium", "large"]:
    for focal_model, competitor_models in focal_models_to_competitor_models.items():
        for competitor in competitor_models:
            diffs = (
                metrics_dict[f"{competitor}_{size}"].BOTH_TEAMS__MEAN_DIST_E
                - metrics_dict[f"{focal_model}_{size}"].BOTH_TEAMS__MEAN_DIST_E
            )
            SE_diff = np.nanstd(diffs) / np.sqrt(metrics_dict[f"ours_{size}"].num_valid_examples)
            # t_stat_by_hand = np.nanmean(diffs)/np.nanstd(diffs)/np.sqrt(metrics_dict[f"ours_{size}"].num_valid_examples)
            t_stat, p_val = ttest_1samp(diffs, popmean=0, nan_policy="omit")
            uncorrected_p_vals_dict[(size, focal_model, competitor)] = p_val
            SE_diffs.append(SE_diff)


### Make corrected p-values
# TODO: confirm it makes sense to assume that tests are positively correlated for all comparisons

reject_hypoth, pvals_corrected = fdrcorrection(
    list(uncorrected_p_vals_dict.values()), alpha=0.05, method="poscorr", is_sorted=False
)
test_results_dict = OrderedDict()
for i, comparison in enumerate(uncorrected_p_vals_dict.keys()):
    test_results_dict.update([(comparison, (reject_hypoth[i], f"{pvals_corrected[i]:.3e}", f"{SE_diffs[i]:.3f}"))])

### Print results

print(f"\nFormal comparisons")
pprint.pprint(test_results_dict)


###
# Sanity checks
###

# # forecasts should start at about the same place.
# e=-2
# j=2
# s=5

# print(f"Ground truth: {ground_truth[e,:,j,:]}")
# print(f"Agent former: {forecasts_dict['agentformer_small'][e,s,:,j,:]}")
# print(f"Ours: {forecasts_dict['ours_small'][e,s,:,j,:]}")
# print(f"Fixed velocity: {forecasts_dict['fixed_velocity'][e,:,j,:]}")


###
# PLots
###

import numpy as np

from dynagroup.model2a.basketball.forecast_plots import plot_team_forecasts


### arguments
model_1 = "ours_large"
model_2 = "no_system_switches_large"


### setup
forecasts_1 = forecasts_dict[model_1]
forecasts_2 = forecasts_dict[model_2]

metrics_1 = metrics_dict[model_1]
metrics_2 = metrics_dict[model_2]

# find example where model_1 is most better than model_2.
# argsort goes lowest to highest
# by default, argsort treats NaN as larger than any other value
# so we want to set things up so that we're always looking for a low value.
rank_of_e_to_use = 37
e = np.argsort(metrics_1.CLE__MEAN_DIST_E - metrics_2.CLE__MEAN_DIST_E)[rank_of_e_to_use - 1]

for r in [1, 5, 10, 15, 20]:
    print(
        f"plotting with the {r}-th best forecasting samples for each model from the {e}-th best example for showing a difference"
    )
    rank_of_s_to_use = r
    s_1 = np.argsort(metrics_1.CLE__MEAN_DIST_ES[e])[rank_of_s_to_use - 1]
    s_2 = np.argsort(metrics_2.CLE__MEAN_DIST_ES[e])[rank_of_s_to_use - 1]
    # s_2=0 #fixed velocity

    plot_team_forecasts(
        forecasts_1,
        forecasts_2,
        ground_truth,
        metrics_1,
        metrics_2,
        e,
        s_1,
        s_2,
        show_plot=False,
        save_dir="/Users/miw267/Desktop/",
        basename_before_extension=f"{model_1}_vs_{model_2}_example_rank_{rank_of_e_to_use}_sample_ranks_{rank_of_s_to_use}",
    )


####
# Statistics
###

from typing import Tuple

from dynagroup.model2a.basketball.court import (
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
)
from dynagroup.types import NumpyArray1D, NumpyArray5D


### IN Bounds
def compute_in_bounds_pcts_by_example(forecasts: NumpyArray5D) -> NumpyArray1D:
    # TODO: Restrict to valid examples ?
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


def compute_in_bounds_pct_mean_and_SE(forecasts: NumpyArray5D) -> Tuple[float, float]:
    in_bounds_pcts = compute_in_bounds_pcts_by_example(forecasts)
    return np.mean(in_bounds_pcts), np.std(in_bounds_pcts) / np.sqrt(len(in_bounds_pcts))


print("Pct in bounds analysis.")
for method, forecasts in forecasts_dict.items():
    mean_pct, SE_mean_pct = compute_in_bounds_pct_mean_and_SE(forecasts)
    print(f"{method} : mean: {mean_pct:.02f}, SE: {SE_mean_pct:.02f}")

### Dispersions (at the last timestep)


def compute_dispersions_by_example(forecasts: NumpyArray5D) -> NumpyArray1D:
    # TODO: Restrict to valid examples ?
    E = np.shape(forecasts)[0]

    dispersions_by_example = np.zeros(E)
    for e in range(E):
        typical_centroid_distance_by_sample = np.sqrt(np.sum(np.var(forecasts[e, :, -1, :5], axis=1), axis=1))
        dispersions_by_example[e] = np.mean(typical_centroid_distance_by_sample)
    return dispersions_by_example


# TODO: combine with similar function above for pct out of bounds
def compute_dispersions_by_example_mean_and_SE(forecasts: NumpyArray5D) -> Tuple[float, float]:
    dispersions = compute_dispersions_by_example(forecasts)
    return np.mean(dispersions), np.std(dispersions) / np.sqrt(len(dispersions))


print("Dispersions analysis.")
for method, forecasts in forecasts_dict.items():
    mean_pct, SE_mean_pct = compute_dispersions_by_example_mean_and_SE(forecasts)
    print(f"{method} : mean: {mean_pct:.02f}, SE: {SE_mean_pct:.02f}")

### Directional variability

from scipy.stats import circvar


# Compute the circular variance
# circular_variance = circvar(theta)


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


# TODO: combine with similar function above for pct out of bounds
def compute_directional_variability_by_example_mean_and_SE(
    forecasts: NumpyArray5D, CLE_only: bool
) -> Tuple[float, float]:
    directional_variabilities = compute_directional_variability_by_example(forecasts, CLE_only)
    return np.mean(directional_variabilities), np.std(directional_variabilities) / np.sqrt(
        len(directional_variabilities)
    )


print("Directional variability analysis.")
for method, forecasts in forecasts_dict.items():
    mean_pct, SE_mean_pct = compute_directional_variability_by_example_mean_and_SE(forecasts, CLE_only=False)
    print(f"{method} : mean: {mean_pct:.02f}, SE: {SE_mean_pct:.02f}")
