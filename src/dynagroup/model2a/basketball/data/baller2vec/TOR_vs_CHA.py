import copy
import warnings
from typing import List, Optional

import prettyprinter as pp


pp.install_extras()

import numpy as np

from dynagroup.model2a.basketball.court import flip_coords_unnormalized
from dynagroup.model2a.basketball.data.baller2vec.core import (
    BasketballGame,
    coords_from_moments,
    get_event_in_baller2vec_format,
    get_num_events_in_game,
    load_event_label_decoder_from_pydict_config_path,
    load_player_data_from_pydict_config_path,
)
from dynagroup.model2a.basketball.data.baller2vec.positions import (
    get_player_name_2_position,
    make_opponent_names_2_entity_idxs,
)


def get_basketball_data_for_TOR_vs_CHA(
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
    filter_out_plays_where_TOR_hoop_side_is_1: bool = False,
) -> BasketballGame:
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

    ### Specify hard-coded constants
    TOR_STARTER_NAMES_2_ENTITY_IDXS = {
        "DeMar DeRozan": 0,  # small forward/power forward
        "Luis Scola": 1,  # power forward
        "Jonas Valanciunas": 2,  # center
        "DeMarre Carroll": 3,  # shooting guard/small forward
        "Kyle Lowry": 4,  # point guard
    }
    TOR_STARTERS = set(TOR_STARTER_NAMES_2_ENTITY_IDXS.keys())

    PATH_TO_GAME_DATA = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_X.npy"
    PATH_TO_EVENT_LABEL_DATA = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_y.npy"
    PATH_TO_BALLER2VEC_CONFIG = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/baller2vec_config.pydict"

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

    ### Organize the events into a BasketballGame object.
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
                if filter_out_plays_where_TOR_hoop_side_is_1:
                    continue
                else:
                    coords_unnormalized = np.vstack(
                        (
                            event_with_player_reindexing.moments[idx].player_xs,
                            event_with_player_reindexing.moments[idx].player_ys,
                        )
                    ).T
                    coords_unnormalized_flipped = flip_coords_unnormalized(coords_unnormalized)
                    event_with_player_reindexing.moments[
                        idx
                    ].player_xs = coords_unnormalized_flipped[:, 0]
                    event_with_player_reindexing.moments[
                        idx
                    ].player_ys = coords_unnormalized_flipped[:, 1]

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

    return BasketballGame(events, event_start_stop_idxs, unnormalized_coords)
