from dataclasses import dataclass
from datetime import datetime, time

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.von_mises.util import degrees_to_radians


"""

Obtaining Dataset (and Snippets) from the SUPRA data for Platoon 2 Squad 1.

    Info about "Contact" Phase of the simulated battle
        start of contact: 9:18:50.  Around timestep 203100
        end of contact 9:28:00. 

    Info about downsampling:
        roughly (based on a single clip), 13 timesteps is about 1/10 of a second.
        20 timestep downsampling is good to get regimes that correspond to turning
        directions and back in the clip below
            t_start, t_end, t_every = 130000, 134000, 20
            entity_idx = 2
            num_entity_regimes = 2
"""

###
# Structs
###


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
# Constants
###

DATA_FILEPATH_FOR_PLATOON_2_SQUAD_1 = "/Users/mwojno01/Library/CloudStorage/Box-Box/IRB_Approval_Required/MASTR_E_Program_Data/data/18_003_SUPRA_Data/Results_Files/MASTRE_SUPRA_P2S1_ITDG.csv"
TIMESTEP_OF_CONTACT_START_FOR_PLATOON_2_SQUAD_1 = 203100
ENTITY_NAMES = [
    "SL",
    "ATL",
    "AGRN",
    "AAR",
    "BTL",
    "BRM",
    "BGRN",
    "BAR",
]  # no ARM in this DATA_FILEPATH
FEATURE_NAME = "HELMET_HEADING"

###
# Helpers
###


def clock_time_as_datetime(clock_time: str) -> datetime:
    """
    Remarks:
        We use datetime instance instead of time instance because the
        former can be combined.  Note that to do this we create a fake day
        (today) at which these events happened.

    """
    hour, minute, second = [int(x) for x in clock_time.split(":")]
    time_instance = time(hour=hour, minute=minute, second=second)
    return datetime.combine(datetime.today().date(), time_instance)


def get_df() -> DataFrame:
    """
    Takes a bit to load.  Recommended usage:

    if not "df" in globals():
        df=get_df()
    """
    return pd.read_csv(DATA_FILEPATH_FOR_PLATOON_2_SQUAD_1)


###
# Key Time snippets of interest
###

CONTACT_FIRST_MINUTE = TimeSnippet(
    t_start=TIMESTEP_OF_CONTACT_START_FOR_PLATOON_2_SQUAD_1, t_end=210840, t_every=20
)
CONTACT_ALL = TimeSnippet(
    t_start=TIMESTEP_OF_CONTACT_START_FOR_PLATOON_2_SQUAD_1, t_end=273427, t_every=20
)
# this squad doesn't have actions on objective phase, so the final timestep for contact
# equals the final timestep for the whole time series

###
# Functions
###


def find_timestep_end_to_match_desired_elapsed_time(
    df: DataFrame, timestep_start: int, timestep_every: int, elapsed_secs_desired: float
) -> int:
    clock_times_all = np.array([x.split(" ")[1].split(".")[0] for x in df["DATETIME"]])

    timestep_end = len(clock_times_all)
    clock_time_start = clock_time_as_datetime(clock_times_all[timestep_start])

    for timestep_curr in range(timestep_start, timestep_end, timestep_every):
        clock_time_curr = clock_time_as_datetime(clock_times_all[timestep_curr])
        clock_time_diff = clock_time_curr - clock_time_start
        if clock_time_diff.total_seconds() > elapsed_secs_desired:
            return timestep_curr

    return timestep_end


def make_time_snippet_based_on_desired_elapsed_secs(
    df: DataFrame,
    elapsed_secs_after_contact_start_for_starting: float,
    elapsed_secs_after_start_for_snipping: float,
    timestep_every: int,
) -> TimeSnippet:
    timestep_start = find_timestep_end_to_match_desired_elapsed_time(
        df,
        TIMESTEP_OF_CONTACT_START_FOR_PLATOON_2_SQUAD_1,
        timestep_every,
        elapsed_secs_after_contact_start_for_starting,
    )
    timestep_end = find_timestep_end_to_match_desired_elapsed_time(
        df, timestep_start, timestep_every, elapsed_secs_after_start_for_snipping
    )
    return TimeSnippet(timestep_start, timestep_end, timestep_every)


def make_data_snippet(df: DataFrame, time_snippet: TimeSnippet) -> DataSnippet:
    ###
    # UP FRONT INFO RELATED TO FILE
    ###
    TS = time_snippet

    ###
    # organize all data
    ###

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
