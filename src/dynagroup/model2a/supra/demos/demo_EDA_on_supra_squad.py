import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from dynagroup.model2a.circle.changepoints import make_changepoints_dict
from dynagroup.model2a.circle.plot_changepoints import plot_changepoint_dict
from dynagroup.model2a.supra import MY_IRB_APPROVED_USERNAME
from dynagroup.von_mises.util import degrees_to_radians


###
# Configs
###
# t_start, t_end, t_every = 19000, 23000, 20
t_start, t_end, t_every = 130000, 134000, 20

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
entity_idxs_to_plot = range(J)

heading_angles_as_degrees = np.zeros((T, J))
for j, entity_name in enumerate(entity_names):
    heading_angles_as_degrees[:, j] = df[f"{entity_name}_{feature_name}"]
heading_angles = degrees_to_radians(heading_angles_as_degrees)


###
# Subset data
###

angles = heading_angles[t_start:t_end:t_every]
security = np.asarray(df["SQUAD_SECURITY"])[t_start:t_end:t_every]

# Per Lee Clifford Hancock's email on 5/8/23,
# PLT1-3 need directional relabelings
# (N to E, W to S, E to N, and S to W)

security_E = np.asarray(df["SQUAD_SECURITYN"])[t_start:t_end:t_every]
security_S = np.asarray(df["SQUAD_SECURITYW"])[t_start:t_end:t_every]
security_N = np.asarray(df["SQUAD_SECURITYE"])[t_start:t_end:t_every]
security_W = np.asarray(df["SQUAD_SECURITYS"])[t_start:t_end:t_every]
data_dict = {
    "security_N": security_N,
    "security_W": security_W,
    "security_E": security_E,
    "security_S": security_S,
}

for j in entity_idxs_to_plot:
    data_dict.update({f"heading_{j}": angles[:, j]})

###
# Make and plot changepoint dict
###

changepoints_dict = make_changepoints_dict(data_dict, changepoint_penalty=100)

plot_changepoint_dict(changepoints_dict)
plt.show()


###
#
###
