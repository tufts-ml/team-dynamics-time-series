import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import prettyprinter as pp


pp.install_extras()

from dynagroup.types import NumpyArray1D, NumpyArray2D


###
# Structs
###


@dataclass
class Moment:
    """
    Stores the information from one timestep within an "X" file as created by the
    `generate_game_numpy_arrays_simplified` module from the baller2vec repo.

    The X file is an np.array of shape (T,D), where D is the dimensionality of
    "information" about each of the T samples in a game.
    In particular, we have D=54, and
        0 = elapsed game_time (secs),
        1 = elapsed period_time (secs),
        2 = shot_clock,
        3 = period,
        4 = left_score,  The score of the team whose hoop is on the left.
        5 = right_score, The score of the team whose hoop is on the right.
        6 = left_score - right_score,
        7 = ball_x,
        8 = ball_y,
        9 = ball_z,
        10-19 = the (sorted) player ids of the players involved in the event's play
        20-29 = the player x values, players ordered as above
        30-39 = the player y values, players ordered as above
        40-49 = the player hoop sides, players ordered as above (0=left, 1=right)
        50 = event_id
        51 = wall_clock
    """

    game_time_elapsed_secs: float
    period_time_elapsed_secs: float
    shot_clock: float
    period: int
    left_score: int
    right_score: int
    ball_x: float
    ball_y: float
    ball_z: float
    player_ids: List[int]
    player_xs: NumpyArray1D  # float
    player_ys: NumpyArray1D  # float
    player_hoop_sides: List[int]
    event_id: int


@dataclass
class Event:
    moments: List[Moment]
    idx: int
    label: str
    player_names: List[str]
    start_game_secs_elapsed: float
    end_game_secs_elapsed: float


###
# Helpers
###


def moment_from_game_slice(slice: NumpyArray1D) -> Moment:
    """
    game_slice: has 54 elements. See Event class definition
    """
    return Moment(
        game_time_elapsed_secs=slice[0],
        period_time_elapsed_secs=slice[1],
        shot_clock=slice[2],
        period=int(slice[3]),
        left_score=int(slice[4]),
        right_score=int(slice[5]),
        ball_x=slice[7],
        ball_y=slice[8],
        ball_z=slice[9],
        player_ids=[int(x) for x in slice[10 : 19 + 1]],
        player_xs=slice[20 : 29 + 1],
        player_ys=slice[30 : 39 + 1],
        player_hoop_sides=[int(x) for x in slice[40 : 49 + 1]],
        event_id=int(slice[50]),
    )


def player_names_from_player_ids(PLAYER_DATA: Dict[int, Dict], player_ids: List[int]) -> List[str]:
    """
    Arguments:
        PLAYER_DATA: Constructed directly via the  load_player_data_from_pydict_config_path function here,
            traces back to the pydict config constructed by the `generate_game_numpy_arrays_simplified` module
            from the baller2vec repo.
    """
    names = []
    for player_id in player_ids:
        name = PLAYER_DATA[player_id]["name"]
        names.append(name)
    return names


def grab_event(
    GAME_DATA: NumpyArray2D,
    EVENT_LABEL_DATA: NumpyArray1D,  # ints
    EVENT_LABEL_DICT: Dict[str, int],
    PLAYER_DATA: Dict[int, Dict],
    event_idx: int,
    sampling_rate_Hz: int = 25,
    verbose: bool = True,
) -> Event:
    """
    Grabs an event (or play), represented as a list of Moments.   As per
    the baller2vec paper, the original data is sampled at 25Hz

    Arguments:
        GAME_DATA: an np.array of shape (T,D), where T is the number of timesteps and D
            is the dimensionality of  "information" about each of the T samples
            in a game.  This is what is loaded by an "X" file as created by the
            `generate_game_numpy_arrays_simplified` module from the baller2vec repo.
        EVENT_LABEL_DATA: an np.array of shape (T,) with dtype="int" classifying the event,
            with interpretation given by EVENT_LABEL_DICT
        PLAYER_DATA: Constructed directly via the  load_player_data_from_pydict_config_path function here,
            traces back to the pydict config constructed by the `generate_game_numpy_arrays_simplified` module
            from the baller2vec repo.
        event_idx: Which event (or play) we'd like to extract.
            An integer from 0,1,...,E, where E is the number of events (or plays)
            in the game.

    Returns:
        An event, which is a list of moments.
    """
    GAME_DATA_event_location = 50

    timestep_first = np.where(GAME_DATA[:, GAME_DATA_event_location] == event_idx)[0][0]
    timestep_last = np.where(GAME_DATA[:, GAME_DATA_event_location] == event_idx)[0][-1]

    ORIGINAL_SAMPLING_RATE_HZ = 25
    sample_every = int(ORIGINAL_SAMPLING_RATE_HZ / sampling_rate_Hz)

    moments = []
    for t in range(timestep_first, timestep_last + 1, sample_every):
        moment = moment_from_game_slice(GAME_DATA[t])
        moments.append(moment)

    ### Get player names
    # Probably safe to assume the names and order asre constant throughout a play.
    # TODO: Confirm this
    player_names = player_names_from_player_ids(PLAYER_DATA, moments[0].player_ids)

    ### Get start and end times
    start_game_secs_elapsed = moments[0].game_time_elapsed_secs
    end_game_secs_elapsed = moments[-1].game_time_elapsed_secs

    if verbose:
        time_of_play = end_game_secs_elapsed - start_game_secs_elapsed
        print(
            f"Event (play) {event_idx} lasts {time_of_play:.02f} seconds. Start: {start_game_secs_elapsed:.02f}. End: {end_game_secs_elapsed:.02f} "
        )

    ### Get event classification
    event_label_idx = EVENT_LABEL_DATA[
        timestep_first
    ]  # Rk: assume event label doesn't change over course of event.
    event_label = [k for k in EVENT_LABEL_DICT if EVENT_LABEL_DICT[k] == event_label_idx][0]

    return Event(
        moments,
        event_idx,
        event_label,
        player_names,
        start_game_secs_elapsed,
        end_game_secs_elapsed,
    )


def load_player_data_from_pydict_config_path(path_to_config):
    baller2vec_config = pickle.load(open(path_to_config, "rb"))
    return baller2vec_config["player_idx2props"]


def load_event_label_decoder_from_pydict_config_path(path_to_config):
    baller2vec_config = pickle.load(open(path_to_config, "rb"))
    return baller2vec_config["event2event_idx"]


def get_event_in_baller2vec_format(event_idx: int, sampling_rate_Hz=5) -> Event:
    """
    Arguments:
        event_idx: Which event (or play) to pull from the game. This can be hard to set since
            we don't know the maximum number of events in the file, but we at least print it out
            while running this function.
    """

    ### Configs
    path_to_game_data = "/Users/mwojno01/Repos/baller2vec_forked/data/games/0021500492_X.npy"
    path_to_event_label_data = "/Users/mwojno01/Repos/baller2vec_forked/data/games/0021500492_y.npy"
    path_to_baller2vec_config = (
        "/Users/mwojno01/Repos/baller2vec_forked/data/baller2vec_config.pydict"
    )

    ### Load the data
    GAME_DATA = np.load(path_to_game_data)
    EVENT_LABEL_DATA = np.load(path_to_event_label_data)
    EVENT_LABEL_DICT = load_event_label_decoder_from_pydict_config_path(path_to_baller2vec_config)
    PLAYER_DATA = load_player_data_from_pydict_config_path(path_to_baller2vec_config)

    num_events_in_this_game = int(GAME_DATA[-1][50])
    print(f"The number of events in this game is {num_events_in_this_game}.")
    if event_idx + 1 > num_events_in_this_game:
        raise ValueError(
            f"The desired event index {event_idx} is too large, since the number of "
            f"events in this game is {num_events_in_this_game}."
        )

    ### Explore a moment
    # moment = moment_from_game_slice(GAME_DATA[0])
    # pp.pprint(moment)

    ### Make event
    event = grab_event(
        GAME_DATA, EVENT_LABEL_DATA, EVENT_LABEL_DICT, PLAYER_DATA, event_idx, sampling_rate_Hz
    )

    ### TODO: Normalize the data?
    return event
