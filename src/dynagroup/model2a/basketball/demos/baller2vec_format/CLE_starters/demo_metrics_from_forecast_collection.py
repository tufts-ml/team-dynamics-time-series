import numpy as np

from dynagroup.model2a.basketball.forecast.main_analysis import (
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

# TODO: Much of this material (metrics computation, muliple comparisons testing)
# is redundant with what's done in the forecasting statistics section.  Combine these.

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
# Forecasting statistics
###

from dynagroup.model2a.basketball.forecast.statistics import (
    Forecast_Statistic,
    compute_model_comparison_results_for_forecast_statistic,
    compute_summaries_of_forecast_statistic,
)


focal_models_to_competitor_models = {
    "ours": ["no_system_switches", "no_recurrence"],
}

parameters_for_model_comparison_on_forecast_statistics = {
    Forecast_Statistic.Pct_In_Bounds: {"alpha": 0.01, "alternative": "less"},
    Forecast_Statistic.Directional_Variabilities: {"alpha": 0.01, "alternative": "greater"},
}


for forecast_statistic, test_params in parameters_for_model_comparison_on_forecast_statistics.items():
    statistics_summary_dict = compute_summaries_of_forecast_statistic(forecast_statistic, forecasts_dict)
    test_results_dict = compute_model_comparison_results_for_forecast_statistic(
        statistics_summary_dict,
        focal_models_to_competitor_models,
        alpha=test_params["alpha"],
        alternative_hypothesis_for_competitor_minus_focal=test_params["alternative"],
    )

    print(f"\nFormal comparisons")
    pprint.pprint(test_results_dict)


###
# PLots
###

from dynagroup.model2a.basketball.forecast.plots import (
    get_plot_lims_for_column_and_row,
    plot_team_forecasts_giving_multiple_samples_for_one_model_and_one_player,
)


### arguments
models_list = ["ours_large", "no_system_switches_large", "no_recurrence_large", "agentformer_large"]
forecasts_list = [forecasts_dict[model] for model in models_list]
metrics_list = [metrics_dict[model] for model in models_list]


# find median example for our model
# argsort goes lowest to highest
# by default, argsort treats NaN as larger than any other value
# so we want to set things up so that we're always looking for a low value.
rank_of_e_to_use = 37
e = np.argsort(metrics_dict["ours_large"].CLE__MEAN_DIST_E)[rank_of_e_to_use - 1]


# sample ranks to use (for each model)
r_list = [1, 5, 10]

# convert to sample idxs
M, S = len(models_list), len(r_list)
s_matrix = np.zeros((M, S), dtype=int)
for m, model in enumerate(models_list):
    for i, r in enumerate(r_list):
        rank_of_s_to_use = r
        s_matrix[m, i] = np.argsort(metrics_dict[model].CLE__MEAN_DIST_ES[e])[rank_of_s_to_use - 1]


### get x lims by column and row
x_mins, x_maxes, y_mins, y_maxes = get_plot_lims_for_column_and_row(
    models_list, forecasts_dict, ground_truth, s_matrix, e
)


for m, model in enumerate(models_list):
    for j in range(5):
        plot_team_forecasts_giving_multiple_samples_for_one_model_and_one_player(
            forecasts_dict[model],
            ground_truth,
            e,
            j,
            s_matrix[m],
            np.min(x_mins),
            np.max(x_maxes),
            np.min(y_mins),
            np.max(y_maxes),
            show_plot=False,
            save_dir="/Users/miw267/Repos/aistats-2024/images/basketball/",
            basename_before_extension=f"forecasts_by_{model}_for_player_{j}",
        )


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