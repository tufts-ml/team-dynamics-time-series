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


forecasts_ours_small, fixed_velocity, ground_truth = load_dynagroup_forecasts(forecasts_ours_small_dir)
forecasts_ours_medium, _, _ = load_dynagroup_forecasts(forecasts_ours_medium_dir)
forecasts_ours_large, _, _ = load_dynagroup_forecasts(forecasts_ours_large_dir)

forecasts_no_recurrence_small, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_small_dir)
forecasts_no_recurrence_medium, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_medium_dir)
forecasts_no_recurrence_large, _, _ = load_dynagroup_forecasts(forecasts_no_recurrence_large_dir)

forecasts_dict["ours_small"] = forecasts_ours_small
forecasts_dict["ours_medium"] = forecasts_ours_medium
forecasts_dict["ours_large"] = forecasts_ours_large

forecasts_dict["no_recurrence_small"] = forecasts_no_recurrence_small
forecasts_dict["no_recurrence_medium"] = forecasts_no_recurrence_medium
forecasts_dict["no_recurrence_large"] = forecasts_no_recurrence_large

forecasts_dict["fixed_velocity"] = fixed_velocity


###  Run metrics
print("\n")
metrics_dict = {}
for model_name, forecasts in forecasts_dict.items():
    metrics_dict[model_name] = compute_metrics(forecasts, ground_truth)
    print(
        f"MSE (SE) - Both teams for {model_name} is {metrics_dict[model_name].BOTH_TEAMS__MSE : .02f} ({metrics_dict[model_name].BOTH_TEAMS__SE_MSE : .02f} )"
    )

print("\n")
for model_name in metrics_dict.keys():
    print(
        f"MSE (SE) - CLE for {model_name} is {metrics_dict[model_name].CLE__MSE : .02f} ({metrics_dict[model_name].CLE__SE_MSE : .02f} )"
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


####
# Paired difference test
###

# diffs = metrics_dict["agentformer_small"].BOTH_TEAMS__MSE_E - metrics_dict["ours_small"].BOTH_TEAMS__MSE_E
# t_stat = np.nanmean(diffs)/(np.nanstd(diffs)/np.sqrt(75))

###
# PLots
###

import numpy as np

from dynagroup.model2a.basketball.forecast_plots import plot_team_forecasts


### arguments
model_1 = "ours_small"
model_2 = "agentformer_small"


### setup
forecasts_1 = forecasts_dict[model_1]
forecasts_2 = forecasts_dict[model_2]

metrics_1 = metrics_dict[model_1]
metrics_2 = metrics_dict[model_2]

# find example where model_1 is most better than model_2.
# argsort goes lowest to highest
# by default, argsort treats NaN as larger than any other value
# so we want to set things up so that we're always looking for a low value.
e = np.argsort(metrics_1.CLE__MSE_E - metrics_2.CLE__MSE_E)[37]

for r in [1, 5, 10, 15]:
    rank_of_s_to_use = r
    s_1 = np.argsort(metrics_1.CLE__MSE_ES[e])[rank_of_s_to_use - 1]
    s_2 = np.argsort(metrics_2.CLE__MSE_ES[e])[rank_of_s_to_use - 1]
    # s_2=0 #fixed velocity

    plot_team_forecasts(forecasts_1, forecasts_2, ground_truth, metrics_1, metrics_2, e, s_1, s_2, show_plot=True)
