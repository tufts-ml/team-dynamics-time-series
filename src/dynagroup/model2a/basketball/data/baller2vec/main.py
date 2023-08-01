import copy
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import prettyprinter as pp


pp.install_extras()

import numpy as np

from dynagroup.model2a.basketball.court import flip_coords_unnormalized
from dynagroup.model2a.basketball.data.baller2vec.positions import (
    get_player_name_2_position,
    make_opponent_names_2_entity_idxs,
)
from dynagroup.types import NumpyArray1D, NumpyArray2D, NumpyArray3D


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


Coords = NumpyArray3D
# `Coords` has shape (T,J,D=2), where T is the number of timesteps, J is the number of players,
# and D=2 is the dimensionality.


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
    game_data: NumpyArray2D,
    event_label_data: NumpyArray1D,  # ints
    event_label_dict: Dict[str, int],
    player_data: Dict[int, Dict],
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

    timestep_first = np.where(game_data[:, GAME_DATA_event_location] == event_idx)[0][0]
    timestep_last = np.where(game_data[:, GAME_DATA_event_location] == event_idx)[0][-1]

    ORIGINAL_SAMPLING_RATE_HZ = 25
    sample_every = int(ORIGINAL_SAMPLING_RATE_HZ / sampling_rate_Hz)

    moments = []
    for t in range(timestep_first, timestep_last + 1, sample_every):
        moment = moment_from_game_slice(game_data[t])
        moments.append(moment)

    ### Get player names
    # Probably safe to assume the names and order asre constant throughout a play.
    # TODO: Confirm this
    player_names = player_names_from_player_ids(player_data, moments[0].player_ids)

    ### Get start and end times
    start_game_secs_elapsed = moments[0].game_time_elapsed_secs
    end_game_secs_elapsed = moments[-1].game_time_elapsed_secs

    if verbose:
        time_of_play = end_game_secs_elapsed - start_game_secs_elapsed
        print(
            f"Event (play) {event_idx} lasts {time_of_play:.02f} seconds. Start: {start_game_secs_elapsed:.02f}. End: {end_game_secs_elapsed:.02f} "
        )

    ### Get event classification
    event_label_idx = event_label_data[
        timestep_first
    ]  # Rk: assume event label doesn't change over course of event.
    event_label = [k for k in event_label_dict if event_label_dict[k] == event_label_idx][0]

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


def get_num_events_in_game(
    game_data: NumpyArray2D,
) -> int:
    """
    Arguments:
        game_data: A numpy array obtained from loading a "game data" file, e.g.
             "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/0021500492_X.npy",
            that was produced by the baller2vec_forked preprocessing.  These files have size
            (T,D=54)
    """
    return int(game_data[-1][50])


def get_event_in_baller2vec_format(
    event_idx: int,
    game_data: NumpyArray2D,
    event_label_data: NumpyArray1D,
    event_label_dict: Dict[str, int],
    player_data: Dict[int, Dict],
    sampling_rate_Hz=5,
) -> Event:
    """
    Arguments:
        event_idx: Which event (or play) to pull from the game. This can be hard to set since
            we don't know the maximum number of events in the file, but we at least print it out
            while running this function.
        game_data: Array of shape (T,D=54) obtained from loading a "game data" file, e.g.
             "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/0021500492_X.npy",
            that was produced by the baller2vec_forked preprocessing.
        event_label_data: Array of shape (T,) obtained from loading an "event label data" file, e.g.
             "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/0021500492_y.npy",
            that was produced by the baller2vec_forked preprocessing.
        event_label_dict: Dict obtained by loading a [MISNAMED] "baller2vec_config" file, e.g.
             e.g. "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/baller2vec_config.pydict",
            that was produces as the OUTPUT of baller2vec_forked preprocessing
        player_data: Dict obtained by loading a [MISNAMED] "baller2vec_config" file, e.g.
             e.g. "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/baller2vec_config.pydict",
            that was produces as the OUTPUT of baller2vec_forked preprocessing
    """

    ### Load the data
    num_events_in_this_game = get_num_events_in_game(game_data)

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
        game_data, event_label_data, event_label_dict, player_data, event_idx, sampling_rate_Hz
    )

    return event


###
# Get Basketball Data
###


@dataclass
class BasketballData:
    """
    Attributes:
        unnormalized_coords:  unnormalized coordinates for basketball players,
            array of shape (T_slice, J=10, D=2)
    """

    events: List[Event]
    event_start_stop_idxs: List[int]
    coords_unnormalized: NumpyArray3D


def coords_from_moments(moments: List[Moment]) -> Coords:
    T = len(moments)
    J = 10
    coords = np.zeros((T, J, 2))
    for t in range(T):
        coords[t, :, 0] = moments[t].player_xs
        coords[t, :, 1] = moments[t].player_ys
    return coords


def get_basketball_data_for_TOR_vs_CHA(
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
    filter_out_plays_where_TOR_hoop_side_is_1: bool = True,
) -> BasketballData:
    """
    Currently data is hardcoded to be from a single basketball game, TOR vs CHA.
    We only keep events with all 5 TOR starters.

    An incomplete list of steps along the way:
        1. Filter out all plays where we don't have the TOR starters.
        2. Run a preprocessing step to assign all available players in the game
        to position idxs.  The TOR starters are given indices [0,1,2,3,4], and the
        opponents are given indices [5,6,7,8,9] by mapping their positions to an index.
        3. The baller2vec `Event`s are extracted and reindexed according to the design above.
        4. We normalize the court so that the focal team (TOR) always has its basket on the left.
            (By "its basket", we mean that hoop_sides=0 in the baller2vec Event representation.
            To ascertain confidently whether this refers to the focal team's offensive or defensive
            hoop would require digging through the docs of both baller2vec and the original dataset.)

            NOTE: I assume that the center of the [0,100]x[0,50] court is [50,25].  But some code in
            the baller2vec repo suggests that the center on the x-axis might be 47 rather than 50
            (e.g. see https://github.com/airalcorn2/baller2vec/blob/master/settings.py#L17).
            Check into this.

    Returns:
        unnormalized coordinates for basketball players,
            array of shape (T_slice, J=10, D=2)
    """

    if not filter_out_plays_where_TOR_hoop_side_is_1:
        raise NotImplementedError(
            f"I need to implement some rotation strategy to align the switches "
            f"of hoop sides at half-time."
        )

    ### Specify hard-coded constants
    TOR_STARTER_NAMES_2_ENTITY_IDXS = {
        "DeMar DeRozan": 0,  # small forward/power forward
        "Luis Scola": 1,  # power forward
        "Jonas Valanciunas": 2,  # center
        "DeMarre Carroll": 3,  # shooting guard/small forward
        "Kyle Lowry": 4,  # point guard
    }
    TOR_STARTERS = set(TOR_STARTER_NAMES_2_ENTITY_IDXS.keys())

    PATH_TO_GAME_DATA = (
        "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/0021500492_X.npy"
    )
    PATH_TO_EVENT_LABEL_DATA = (
        "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/0021500492_y.npy"
    )
    PATH_TO_BALLER2VEC_CONFIG = (
        "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/baller2vec_config.pydict"
    )

    ### Load the data
    game_data = np.load(PATH_TO_GAME_DATA)
    event_label_data = np.load(PATH_TO_EVENT_LABEL_DATA)
    event_label_dict = load_event_label_decoder_from_pydict_config_path(PATH_TO_BALLER2VEC_CONFIG)
    player_data = load_player_data_from_pydict_config_path(PATH_TO_BALLER2VEC_CONFIG)

    ### Identify positions of all players in game
    PLAYER_NAME_2_POSITION = {
        "Spencer Hawes": "Center / power forward",
        "Marvin Williams": "Power forward / small forward",
        "Jeremy Lamb": "Shooting guard / small forward",
        "Jeremy Lin": "Point guard",
        "Aaron Harrison": "Shooting guard / small forward",
        "Kemba Walker": "Point guard",
        "PJ Hairston": "Small forward / shooting guard",
        "Brian Roberts": "Point guard",
        "Troy Daniels": "Shooting guard",
        "Cody Zeller": "Center / power forward",
        "Frank Kaminsky": "Center / power forward",
        "Tyler Hansbrough": "Power forward / center",
        "Al Jefferson": "Power forward / center",
        "James Johnson": "Power forward / small forward",
        "Luis Scola": "Power forward",
        "Cory Joseph": "Point guard",
        "Kyle Lowry": "Point guard",
        "Bismack Biyombo": "Center",
        "DeMar DeRozan": "Shooting guard / small forward",
        "Anthony Bennett": "Power forward",
        "Norman Powell": "Shooting guard",
        "Terrence Ross": "Shooting guard / small forward",
        "Patrick Patterson": "Power forward",
        "Lucas Nogueira": "Center",
        "DeMarre Carroll": "Small forward",
        "Jonas Valanciunas": "Center",
        "Bruno Caboclo": "Power forward",
        "Delon Wright": "Point guard / shooting guard",
        "Nicolas Batum": "Power forward / small forward",
    }
    player_name_2_position = PLAYER_NAME_2_POSITION

    if not player_name_2_position:
        print("Now scraping Wikipedia to identify the positions for all the players in the game...")
        player_name_2_position = get_player_name_2_position(player_data)
        print("...Done.")

    ### Use all events if we don't specify anything else
    if event_idxs is None:
        num_events_in_this_game = get_num_events_in_game(game_data)
        event_idxs = range(0, num_events_in_this_game)

    ### Organize the events into a BasketballData object.
    events = []
    moments = []
    event_start_stop_idxs = []
    n_events_without_TOR_starters = 0

    num_moments_so_far = 0
    for event_idx in event_idxs:
        try:
            event = get_event_in_baller2vec_format(
                event_idx,
                game_data,
                event_label_data,
                event_label_dict,
                player_data,
                sampling_rate_Hz=sampling_rate_Hz,
            )

            ### Filter out events that don't have all 5 TOR starters
            current_player_names = set(event.player_names)
            has_TOR_starters = TOR_STARTERS.issubset(current_player_names)
            if not has_TOR_starters:
                n_events_without_TOR_starters += 1
                continue

            ### Now we assign opponents to position indices.
            # TODO: if we dont have 5 unique names, then skip the play.
            # Q: can our model handle DIFFERENT identies from play to play,
            # as long as we've learned something about that role.
            current_opponent_names = current_player_names - TOR_STARTERS
            opponent_names_2_positions = {}
            for opponent_name in current_opponent_names:
                opponent_position = player_name_2_position[opponent_name]
                opponent_names_2_positions[opponent_name] = opponent_position

            opponent_names_2_entity_idxs = make_opponent_names_2_entity_idxs(
                opponent_names_2_positions
            )

            player_names_2_entity_idxs = {
                **TOR_STARTER_NAMES_2_ENTITY_IDXS,
                **opponent_names_2_entity_idxs,
            }

            ### Now we reorder any list-valued attributes within the `event` and `moment` objects
            # to match our new ordering for players.
            player_names_orig_order = event.player_names
            player_names_new_order = sorted(
                event.player_names, key=lambda name: player_names_2_entity_idxs[name]
            )
            permutation_indices = [
                player_names_orig_order.index(elem) for elem in player_names_new_order
            ]

            event_with_player_reindexing = copy.copy(event)
            event_with_player_reindexing.player_names = player_names_new_order

            for idx in range(len(event.moments)):
                event_with_player_reindexing.moments[idx].player_ids = [
                    event.moments[idx].player_ids[i] for i in permutation_indices
                ]
                event_with_player_reindexing.moments[idx].player_xs = np.array(
                    [event.moments[idx].player_xs[i] for i in permutation_indices]
                )
                event_with_player_reindexing.moments[idx].player_ys = np.array(
                    [event.moments[idx].player_ys[i] for i in permutation_indices]
                )
                event_with_player_reindexing.moments[idx].player_hoop_sides = [
                    event.moments[idx].player_hoop_sides[i] for i in permutation_indices
                ]

            ### Normalize hoop sides.   Assume focal team has hoop on left.  If not, we
            # flip the court 180 degrees around the center of the court (i.e. negate
            # both x and y coords w.r.t center of court). This controls for the effect of hoop
            # switches at half time on the court dynamics, in terms of both offense vs defense
            # direction, as well as in terms of player handedness.

            # NOTE: The center might be shifted slighly to the left of how I'm doing this.
            # See the note in the function docstring.

            NORMALIZED_HOOP_SIDES = [0] * 5 + [1] * 5  # focal team has hoop on left.
            if event_with_player_reindexing.moments[idx].player_hoop_sides != NORMALIZED_HOOP_SIDES:
                coords_unnormalized = np.vstack(
                    (
                        event_with_player_reindexing.moments[idx].player_xs,
                        event_with_player_reindexing.moments[idx].player_ys,
                    )
                ).T
                coords_unnormalized_flipped = flip_coords_unnormalized(coords_unnormalized)
                event_with_player_reindexing.moments[idx].player_xs = coords_unnormalized_flipped[
                    :, 0
                ]
                event_with_player_reindexing.moments[idx].player_ys = coords_unnormalized_flipped[
                    :, 1
                ]

            ### Now we extend our accumulating lists of moments, events, event_start_stop_idxs.
            moments.extend(event.moments)
            event_first_moment = num_moments_so_far
            num_moments = len(event.moments)
            num_moments_so_far += num_moments
            event_last_moment = num_moments_so_far
            event_start_stop_idxs.extend([(event_first_moment, event_last_moment)])
            events.extend([event])
        except:
            warnings.warn(f"Could not process event idx {event_idx}")
            continue

    print(
        f"\n\n --- There were {len(events)} events with TOR starters and {n_events_without_TOR_starters} without."
    )
    unnormalized_coords = coords_from_moments(moments)

    return BasketballData(events, event_start_stop_idxs, unnormalized_coords)
