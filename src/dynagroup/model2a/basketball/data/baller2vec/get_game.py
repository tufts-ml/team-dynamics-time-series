import copy
import warnings
from time import time
from typing import Dict, List, Optional

import prettyprinter as pp


pp.install_extras()

import numpy as np

from dynagroup.model2a.basketball.court import flip_coords_unnormalized
from dynagroup.model2a.basketball.data.baller2vec.core import (
    BasketballGame,
    coords_from_moments,
    get_event_in_baller2vec_format,
    get_num_events_in_game,
    load_event_label_decoder_from_pydict_info_path,
    load_player_data_from_pydict_info_path,
)
from dynagroup.model2a.basketball.data.baller2vec.positions import (
    get_player_name_2_position,
    make_opponent_names_2_entity_idxs,
)


def get_basketball_game(
    path_to_game_data: str,
    path_to_event_label_data: str,
    path_to_baller2vec_info: str,
    focal_team_potential_starter_names_2_entity_idxs: Dict[str, int],
    player_names_in_dataset_2_positions: Optional[Dict[str, str]] = None,
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
) -> BasketballGame:
    """
    We only keep events with the starters for a given game.

    An incomplete list of steps along the way:
        1. Filter out all plays where we don't have the focal team starters.
        2. Run a preprocessing step to assign all available players in the game
        to position idxs.  The focal team starters are given indices [0,1,2,3,4], and the
        opponents are given indices [5,6,7,8,9] by mapping their positions to an index.
        3. The baller2vec `Event`s are extracted and reindexed according to the design above.
        4. We normalize the court so that the focal team  always has its basket on the left.
            (By "its basket", we mean that hoop_sides=0 in the baller2vec Event representation.
            To ascertain confidently whether this refers to the focal team's offensive or defensive
            hoop would require digging through the docs of both baller2vec and the original dataset.)

            NOTE: I assume that the center of the [0,100]x[0,50] court is [50,25].  But some code in
            the baller2vec repo suggests that the center on the x-axis might be 47 rather than 50
            (e.g. see https://github.com/airalcorn2/baller2vec/blob/master/settings.py#L17).
            Check into this.

    Arguments:
        focal_team_potential_starter_names_2_entity_idxs: A dict mapping potential starter names to entity indices.
            This allows us to handle substitutions from game to game.

            Example:
                focal_team_potential_starter_names_2_entity_idxs = {
                    "LeBron James": 0,  # everything
                    "Kevin Love": 1,  # power forward, center
                    "Timofey Mozgov": 2,  # center
                    "Tristan Thompson": 2,  # center
                    "J.R. Smith": 3,  # shooting guard/small forward
                    "Kyrie Irving": 4,  # guard
                    "Mo Williams": 4,  # guard
                    "Matthew Dellavedova": 4,  # guard
                }
        player_names_in_dataset_2_positions:
            Should handle the names and positions of (at least) all players in the game.
            A superset is fine and useful if we are processing multiple games simultaneously.
            If not provided, the code will try to infer it from Wikipedia, although there are
            usually some manual patch-ups required.

            Example:
                PLAYER_NAME_2_POSITION = {
                    "Pau Gasol": "Power forward / center",
                    "Kirk Hinrich": "Point guard / shooting guard",
                    "Joakim Noah": "Center",
                    "Aaron Brooks": "Point guard",
                    "Derrick Rose": "Point guard",
                    "Taj Gibson": "Power forward / center",
                    "Nikola Mirotic": "Power forward",
                    ...
                }

    Returns:
        unnormalized coordinates for basketball players,
            array of shape (T_slice, J=10, D=2)
    """

    potential_starters = set(focal_team_potential_starter_names_2_entity_idxs.keys())

    # E.g. for CLE, starters were:
    # \texttt{0: Lebron James, 1: Kevin Love, 2: J.R. Smith, 3: Starting Center, 4: Starting Guard}.
    # Depending on the game, the starting center was either T. Mazgov or T. Thompson.
    # Similarly, the starting guard was either K. Irving, M. Williams, or M. Dellavedova.

    ### Load the data
    game_data = np.load(path_to_game_data)
    event_label_data = np.load(path_to_event_label_data)
    event_label_dict = load_event_label_decoder_from_pydict_info_path(path_to_baller2vec_info)
    player_data = load_player_data_from_pydict_info_path(path_to_baller2vec_info)

    ### Identify positions of all players in game

    # Rk: I have the output hardcoded in here so we don't have to deal with Wikipedia scraping
    # each time we load the dataset. The scraping is (1) slow and (2) can sometimes miss positions
    # e.g., consider that some players may have the same name as different person with a Wiki entry,
    # in which case our guess at the Wiki url won't work.   In particular (2) happens for
    # 'Chris Johnson', "James Jones", 'Reggie Jackson',  'Ryan Anderson'
    # 'Jason Smith', 'Kevin Martin',  'Marcus Thornton'.
    # In addition, (3) I manually replaced "Centre" with "Center" for two of the players.
    # and (4) I manually ensured that posiitons after a "/" were lowercase, which affected 2 players.

    # TODO: Deal with "centre/power forward" and other anomalies

    if not player_names_in_dataset_2_positions:
        print("Now scraping Wikipedia to identify the positions for all the players in the game...")
        t0 = time()
        player_names_in_dataset_2_positions = get_player_name_2_position(player_data)
        t1 = time()
        print(f"...Done in {t1-t0:.02f} seconds")

    ### Use all events if we don't specify anything else
    if event_idxs is None:
        num_events_in_this_game = get_num_events_in_game(game_data)
        event_idxs = range(0, num_events_in_this_game)

    ### Find focal team starters for this game
    focal_team_starters_for_this_game = None

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

            current_player_names = set(event.player_names)
            focal_team_starters_for_this_game = set(
                [x for x in current_player_names if x in potential_starters]
            )
        except:
            warnings.warn(f"Could not process event {event_idx} to identify focal team starters")
            continue
        if focal_team_starters_for_this_game is not None:
            break

    focal_team_starter_names_2_entity_idxs = {
        starter: focal_team_potential_starter_names_2_entity_idxs[starter]
        for starter in focal_team_starters_for_this_game
    }

    ### Organize the events into a BasketballGame object.
    events = []
    moments = []
    event_start_stop_idxs = []
    n_events_without_focal_team_starters = 0

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
            has_focal_team_starters = focal_team_starters_for_this_game.issubset(
                current_player_names
            )
            if not has_focal_team_starters:
                n_events_without_focal_team_starters += 1
                continue

            ### Now we assign opponents to position indices.
            # TODO: if we dont have 5 unique names, then skip the play.
            # Q: can our model handle DIFFERENT identies from play to play,
            # as long as we've learned something about that role.
            current_opponent_names = current_player_names - focal_team_starters_for_this_game
            opponent_names_2_positions = {}
            for opponent_name in current_opponent_names:
                opponent_position = player_names_in_dataset_2_positions[opponent_name]
                opponent_names_2_positions[opponent_name] = opponent_position

            try:
                opponent_names_2_entity_idxs = make_opponent_names_2_entity_idxs(
                    opponent_names_2_positions,
                    override_lineups_with_insufficient_position_group_assignments=True,
                )
            except ValueError:
                raise ValueError("Could not convert opponent names to entity indices")

            player_names_2_entity_idxs = {
                **focal_team_starter_names_2_entity_idxs,
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
        f"\n\n --- There were {len(events)} events with focal team starters and {n_events_without_focal_team_starters} without."
    )
    if len(events) == 0:
        raise ValueError("This game had ZERO events retained. Check into this.")
    unnormalized_coords = coords_from_moments(moments)

    return BasketballGame(events, event_start_stop_idxs, unnormalized_coords)
