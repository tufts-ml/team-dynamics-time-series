import copy
from dataclasses import dataclass

import numpy as np

from dynagroup.types import NumpyArray1D, NumpyArray3D


###
# Structs
###
@dataclass
class Data:
    xs: NumpyArray3D  # (T_grand,J,D)
    event_end_times: NumpyArray1D  # (E,)
    has_ball_team: NumpyArray1D  # (T_grand, )


def get_data() -> Data:
    ###
    # Load Raw Data
    ###

    # xs_raw are (T_grand,J,D), and involve vertically stacking disjoint basketball plays for the
    # J=5 basketball players on TOR.  D=2 gives the (x,y) coordinates on the court

    # covariates are (T_grand,M). The first two columns give the (x,y) coords of the ball. The third column
    # gives its radius (when in the air, the radius gets larger.)
    DATA_LOAD_DIR = "/Users/mwojno01/Repos/dynagroup/data/basketball/"
    POST_HALFTIME_TIMESTEP = 13841  # Visually extracted from filenames, below.
    xs_raw = np.load(
        DATA_LOAD_DIR
        + "basketball_game_TOR_vs_CHA_post_halftime_timestep_13841_stacked_starter_plays.npy"
    )
    event_end_times = np.load(
        DATA_LOAD_DIR + "basketball_game_TOR_vs_CHA_post_halftime_timestep_13841_end_times.npy"
    )
    covariates_raw = np.load(
        DATA_LOAD_DIR + "basketball_game_TOR_vs_CHA_post_halftime_timestep_13841_covariates_raw.npy"
    )

    ###
    # Preprocess Data
    ###

    ### Handle NANs
    # Patch up the few nans  (41/21467) that for some reason only exist in player 0's data
    T, J, D = np.shape(xs_raw)
    j = 0
    for t in range(T):
        if np.isnan(xs_raw[t, j, 0]):
            xs_raw[t, j, :] = xs_raw[t - 1, j, :]

    ### Normalize Basketball coordinates
    # We want x to be in [0,2] and y to be in [0,1]
    # The constants are obtained from the repo below, which holds the raw data.
    #   https://github.com/linouk23/NBA-Player-Movements/blob/master/Constant.py
    X_MAX = 100
    Y_MAX = 50

    xs_normalized_but_not_aligned_for_halftime = copy.copy(xs_raw)
    xs_normalized_but_not_aligned_for_halftime[:, :, 0] /= X_MAX
    xs_normalized_but_not_aligned_for_halftime[:, :, 1] /= Y_MAX

    xs_normalized_but_not_aligned_for_halftime[:, :, 0] *= 2  # let first dim be twice as big..

    ### "Flip" the x coordinates (length of court) at half time,
    # since the team is traveling in the opposite direction as before when they are on offense
    xs = xs_normalized_but_not_aligned_for_halftime
    xs[POST_HALFTIME_TIMESTEP:] = (
        1.0 - xs_normalized_but_not_aligned_for_halftime[POST_HALFTIME_TIMESTEP]
    )

    ###
    # Process the covariates
    ###

    T_grand, J = np.shape(xs)[:2]

    # if distances<1 and radius <6, player has ball
    has_ball_players = np.zeros((T_grand, J), dtype=int)
    for t in range(T_grand):
        DISTANCE_CUTOFF_FOR_HAVING_BALL = (
            3.0  # Determined by EDA.  Works for xs_raw, not normalized xs!!!
        )
        RADIUS_CUTOFF_FOR_HAVING_BALL = 8.0  # Determined by EDA
        ball_location = covariates_raw[t, :2]
        ball_radius = covariates_raw[t, 2]
        player_locations = xs_raw[t]
        distances = np.linalg.norm(player_locations - ball_location, axis=1)
        has_ball_players_bool = (distances < DISTANCE_CUTOFF_FOR_HAVING_BALL) * (
            ball_radius < RADIUS_CUTOFF_FOR_HAVING_BALL
        )
        has_ball_players[t] = has_ball_players_bool

    # Some sanity checks
    pct_time_with_ball_by_player = np.mean(has_ball_players, 0)
    print(f"The percentage of time each player on TOR had the ball {pct_time_with_ball_by_player}")
    print(
        f"The percentage of time each the team TOR had the ball {np.sum(pct_time_with_ball_by_player)}"
    )
    # About 93% of the time somebody has the ball. The other 8% it was in the air.

    # Team has ball
    has_ball_team = np.max(has_ball_players, axis=1)

    """
    TODO: Fill out the rest of this grid 
    Entity level covariates:
        * player has ball
        * ball location 
        
    System level covariates
        * team has ball
    """
    return Data(xs, event_end_times, has_ball_team)
