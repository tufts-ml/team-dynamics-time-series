from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.von_mises.util import degrees_to_radians


def get_df() -> DataFrame:
    """
    Takes a bit to load.  Recommended usage:

    if not "df" in globals():
        df=get_df()
    """
    FILEPATH = "/Users/mwojno01/Library/CloudStorage/Box-Box/IRB_Approval_Required/MASTR_E_Program_Data/data/18_003_SUPRA_Data/Results_Files/MASTRE_SUPRA_P2S1_ITDG.csv"
    return pd.read_csv(FILEPATH)


@dataclass
class DataSnippet:
    squad_angles: NumpyArray2D  # (T,J)
    clock_times: NumpyArray1D  # (T,)
    system_covariates_zero_to_hundred: NumpyArray2D  # (T,M=4) security score from (N,W,E,S)


@dataclass
class TimeSnippet:
    t_start: int
    t_end: int
    t_every: int


###
# Get sample
###

CONTACT_START = TimeSnippet(t_start=203100, t_end=211100, t_every=20)
CONTACT_ALL = TimeSnippet(t_start=203100, t_end=273427, t_every=20)
# this squad doesn't have actions on objective phase, so the final timestep for contact
# equals the final timestep for the whole time series


def make_data_snippet(df: DataFrame, time_snippet: TimeSnippet) -> DataSnippet:
    TS = time_snippet
    ENTITY_NAMES = ["SL", "ATL", "AGRN", "AAR", "BTL", "BRM", "BGRN", "BAR"]  # no ARM
    FEATURE_NAME = "HELMET_HEADING"

    T = len(df)
    J = len(ENTITY_NAMES)
    heading_angles_as_degrees = np.zeros((T, J))
    for j, entity_name in enumerate(ENTITY_NAMES):
        heading_angles_as_degrees[:, j] = df[f"{entity_name}_{FEATURE_NAME}"]
    heading_angles = degrees_to_radians(heading_angles_as_degrees)

    # clock_times_all= np.array([x.split(" ")[1] for x in df["DATETIME"]])
    clock_times_all = np.array([x.split(" ")[1].split(".")[0] for x in df["DATETIME"]])

    ###
    # Subset data
    ###

    squad_angles = heading_angles[TS.t_start : TS.t_end : TS.t_every]
    clock_times = clock_times_all[TS.t_start : TS.t_end : TS.t_every]

    # Per Lee Clifford Hancock's email on 5/8/23,
    # PLT1-3 need directional relabelings
    # (N to E, W to S, E to N, and S to W)

    security_E = np.asarray(df["SQUAD_SECURITYN"])[TS.t_start : TS.t_end : TS.t_every]
    security_S = np.asarray(df["SQUAD_SECURITYW"])[TS.t_start : TS.t_end : TS.t_every]
    security_N = np.asarray(df["SQUAD_SECURITYE"])[TS.t_start : TS.t_end : TS.t_every]
    security_W = np.asarray(df["SQUAD_SECURITYS"])[TS.t_start : TS.t_end : TS.t_every]
    security_four_directions = np.vstack((security_N, security_E, security_S, security_W)).T

    # we use security from last timestep as a system-level covariate
    # (actually skip-level recurrence, but formally it's the same as a covariate)
    system_covariates_zero_to_hundred = np.vstack((np.zeros(4), security_four_directions[:-1]))
    return DataSnippet(squad_angles, clock_times, system_covariates_zero_to_hundred)
