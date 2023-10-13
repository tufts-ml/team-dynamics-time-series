import numpy as np
import pandas as pd


np.set_printoptions(precision=3, suppress=True)

from matplotlib import pyplot as plt

from dynagroup.model2a.circle.directions import (
    LABELS_OF_DIRECTIONS,
    RADIANS_OF_DIRECTIONS,
)
from dynagroup.model2a.supra import MY_IRB_APPROVED_USERNAME
from dynagroup.plotting.paneled_series import plot_time_series_with_regime_panels
from dynagroup.von_mises.inference.arhmm import run_EM_for_von_mises_arhmm
from dynagroup.von_mises.util import degrees_to_radians


###
# Configs
###

### Sample selection

# start of contact: 9:18:50.  Around timestep 203100
# end of contact 9:28:00.
#
# roughly (based on a single clip), 13 timesteps is about 1/10 of a second.
# 20 timestep downsampling is good to get regimes that correspond to turning
# directions and back in the clip below
# t_start, t_end, t_every = 130000, 134000, 20
# entity_idx = 2
# num_regimes = 2
#
# t_start, t_end, t_every = 19000, 23000, 20
# t_start, t_end, t_every = 130000, 134000, 20
# t_start, t_end, t_every = 203100, 207100, 20
t_start, t_end, t_every = 203100, 211100, 20
entity_idx = 2

### Inference
num_regimes = 4
init_self_transition_prob = 0.995
init_changepoint_penalty = 10.0
init_min_segment_size = 10
num_EM_iterations = 3


###
# Get sample
###

filepath = f"/Users/{MY_IRB_APPROVED_USERNAME}/Library/CloudStorage/Box-Box/IRB_Approval_Required/MASTR_E_Program_Data/data/18_003_SUPRA_Data/Results_Files/MASTRE_SUPRA_P2S1_ITDG.csv"

if not "df" in globals():
    df = pd.read_csv(filepath)


entity_names = ["SL", "ATL", "AGRN", "AAR", "BTL", "BRM", "BGRN", "BAR"]  # no ARM
feature_name = "HELMET_HEADING"
system_covariate_names = [
    "SQUAD_SECURITYN",
    "SQUAD_SECURITYW",
    "SQUAD_SECURITYE",
    "SQUAD_SECURITYS",
    "SQUAD_SECURITY",
]

T = len(df)
J = len(entity_names)
heading_angles_as_degrees = np.zeros((T, J))
for j, entity_name in enumerate(entity_names):
    heading_angles_as_degrees[:, j] = df[f"{entity_name}_{feature_name}"]
heading_angles = degrees_to_radians(heading_angles_as_degrees)

# clock_times_all= np.array([x.split(" ")[1] for x in df["DATETIME"]])
clock_times_all = np.array([x.split(" ")[1].split(".")[0] for x in df["DATETIME"]])

###
# Subset data
###

angles = heading_angles[t_start:t_end:t_every, entity_idx]
clock_times = clock_times_all[t_start:t_end:t_every]

###
# Inference
###

posterior_summary, emissions_params_by_regime_learned, transitions = run_EM_for_von_mises_arhmm(
    angles,
    num_regimes,
    num_EM_iterations,
    init_self_transition_prob,
    init_changepoint_penalty,
    init_min_segment_size,
)

###
# PLot results
###

# TODO: get actual viterbi, not marginal MAP estimats.
fitted_regime_sequence = np.argmax(posterior_summary.expected_regimes, 1)

fig, ax = plot_time_series_with_regime_panels(angles, fitted_regime_sequence, clock_times)
ax.set_yticks(RADIANS_OF_DIRECTIONS)
ax.set_yticklabels(LABELS_OF_DIRECTIONS)
plt.show()
