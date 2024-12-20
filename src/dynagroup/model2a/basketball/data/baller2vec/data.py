import copy
import warnings
from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Optional, Tuple

import prettyprinter as pp


pp.install_extras()


import numpy as np

from dynagroup.model2a.basketball.court import flip_player_coords_unnormalized
from dynagroup.model2a.basketball.data.baller2vec.event_boundaries import (
    clean_events_of_moments_with_too_small_intervals,
    get_example_stop_idxs,
    get_play_start_stop_idxs,
)
from dynagroup.model2a.basketball.data.baller2vec.moments_and_events import (
    Event,
    Moment,
    ball_coords_from_moments,
    get_event_in_baller2vec_format,
    get_num_events_in_game,
    load_event_label_decoder_from_pydict_info_path,
    load_player_data_from_pydict_info_path,
    player_coords_from_moments,
    player_names_from_player_ids,
)
from dynagroup.model2a.basketball.data.baller2vec.positions import (
    get_player_name_2_position,
    make_opponent_names_2_entity_idxs,
)
from dynagroup.types import NumpyArray2D, NumpyArray3D
from dynagroup.util import (
    are_lists_identical,
    construct_a_new_list_after_removing_multiple_items,
    flatten_list_of_lists,
)


###
# STRUCTS
###


@dataclass
class BasketballData:
    """
    Attributes:
        events: These have been mutated (to handle entity
            index alignment and court rotation) and filtered (we have removed events whose wall clock diffs are too small
            relative to the specified sampling rate).
        player_coords_unnormalized:  unnormalized coordinates for basketball players,
            array of shape (T_slice, J=10, D=2)
        ball_coords_unnormalized: unnormalized coordinates for balls,
            array of shape (T_slice, D=3)
        play_start_stop_idxs: List of tuples, each tuple has form (start_idx, stop_idx)
            giving the location where a play starts and stops.
        example_stop_idxs:  List of ints, giving the location where an example starts and stops.
        player_names_by_example: List of length E, where E is the number of examples.  Each list gives
            the 10 names representing the name of the basketball player who is playing during that example.
            We check that the players are constant throughout an example.
        player_data: Maps a player idx (in 0,..., N_Players) to a Dict containing the player's
            name, playing time (in the WHOLE dataset -- that is across ALL games in `preprocessed/<*>/games/`;
            this is determined by the object in `preprocessed/<*>/info/baller2vec_info.pydict` written by
            baller2vec), and playerid (in the sense of the `NBA-Player-Movements`; could be something like
            201567).   So (1) the players contained may be a superset of those who appear in "events" and
            (2) the playing times may be larger than what appears in "events".
    """

    # Rk: This is kind of a weird and redundant class since `play_start_stop_idxs` and `player_coords_unnormalized`
    # and `ball_coords_unnormalized` can all be derived from `events`.  Can I figure out what downstream tasks
    # need from `events` and justrepresent that?

    events: List[Event]
    player_coords_unnormalized: NumpyArray3D
    ball_coords_unnormalized: NumpyArray2D
    play_start_stop_idxs: List[Tuple[int]]
    example_stop_idxs: List[int]
    player_names_by_example: List[List[str]]
    player_data: Dict[int, Dict[str, Any]]
    sampling_rate_Hz: float


###
# Helper functions
###


def rotate_court_180_degrees_for_one_moment_of_an_event(
    event: Event, idx: int, normalized_hoop_sides: List[int]
) -> Event:
    """
    Rotates the (xy) coordinates for all 10 players, as well as the (xy, but not z) coordinates
    for the ball, by 180 degrees. Note that this is the same as just subtracting the current
    x coordinates from the COURT_LENGTH and the y coordinates from the COURT_WIDTH; see the
    rotation implementation in the `baller2vec` repo.

    Arguments:
        idx: idx of moment within an event where we want to flip coordinates.
    """

    ### Flip player coords
    player_coords_unnormalized = np.vstack((event.moments[idx].player_xs, event.moments[idx].player_ys)).T
    player_coords_unnormalized_flipped = flip_player_coords_unnormalized(player_coords_unnormalized)
    event.moments[idx].player_xs = player_coords_unnormalized_flipped[:, 0]
    event.moments[idx].player_ys = player_coords_unnormalized_flipped[:, 1]

    ### Flip ball coords
    ball_xy_player_coords_unnormalized = np.vstack((event.moments[idx].ball_x, event.moments[idx].ball_y)).T
    ball_xy_player_coords_unnormalized_flipped = flip_player_coords_unnormalized(ball_xy_player_coords_unnormalized)
    event.moments[idx].ball_x = ball_xy_player_coords_unnormalized_flipped[:, 0]
    event.moments[idx].ball_y = ball_xy_player_coords_unnormalized_flipped[:, 1]

    ### Flip player hoop sides
    event.moments[idx].player_hoop_sides = normalized_hoop_sides

    return event


def remove_events_with_player_substitutions(
    events: List[Event],
    verbose: bool = True,
) -> List[Event]:
    event_idxs_to_remove = []
    for event_idx, event in enumerate(events):
        if not are_lists_identical([m.player_ids for m in event.moments]):
            event_idxs_to_remove.append(event_idx)

    if verbose:
        print(f"The number of events with player substitutions are {len(event_idxs_to_remove)}/{len(events)}.")

    # Remove moments whose wall clock diffs that are too big.
    events_cleaned = construct_a_new_list_after_removing_multiple_items(events, event_idxs_to_remove)

    return events_cleaned


def player_identities_are_constant_throughout_examples(moments: List[Moment], example_stop_idxs: List[int]):
    for idx in range(len(example_stop_idxs) - 1):
        start_idx, stop_idx = example_stop_idxs[idx] + 1, example_stop_idxs[idx + 1]
        moments_in_example = moments[start_idx:stop_idx]
        if not are_lists_identical([m.player_ids for m in moments_in_example]):
            return False
    return True


def get_player_names_by_example(
    moments: List[Moment], example_stop_idxs: List[int], player_data: Dict[int, Dict[str, Any]]
) -> List[List[str]]:
    if not player_identities_are_constant_throughout_examples(moments, example_stop_idxs):
        raise ValueError("The player identities are not constant throughout examples. Check implementation.")

    player_names_by_example = []
    for idx in range(len(example_stop_idxs) - 1):
        example_start_idx = example_stop_idxs[idx] + 1
        player_names_for_this_example = player_names_from_player_ids(player_data, moments[example_start_idx].player_ids)
        player_names_by_example.append(player_names_for_this_example)
    return player_names_by_example


###
# Single game (from disk)
###


def construct_basketball_data_from_single_game_file(
    path_to_game_data: str,
    path_to_event_label_data: str,
    path_to_baller2vec_info: str,
    focal_team_potential_starter_names_2_entity_idxs: Dict[str, int],
    player_names_in_dataset_2_positions: Optional[Dict[str, str]] = None,
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
    discard_nonstandard_hoop_sides: bool = False,
    verbose: bool = True,
) -> BasketballData:
    """
    We take the `preprocessed` NBA player data (which is baller2vec's output after transforming NBA-Player-Movements
    data), and create `processed` data by adding some additional steps.

    An incomplete list of our processing steps are:
        1. Filter out all events (plays) where we don't have the focal team starters for the ENTIRE event (play).
        2. Run a preprocessing step to assign all available players in the game
        to position idxs.  The focal team starters are given indices [0,1,2,3,4], and the
        opponents are given indices [5,6,7,8,9] by mapping their positions to an index.
        3. The baller2vec `Event`s are extracted and reindexed according to the design above.
        4. We normalize the court so that the focal team  always has its basket on the left.
            (By "its basket", we mean that hoop_sides=0 in the baller2vec Event representation.
            To ascertain confidently whether this refers to the focal team's offensive or defensive
            hoop would require digging through the docs of both baller2vec and the original dataset.)

            NOTE: The center of the [0,94]x[0,50] court is [47,25].  See basetball.court module.

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
    ### Up front material
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
                verbose=verbose,
            )

            current_player_names = set(event.player_names)
            focal_team_starters_for_this_game = set([x for x in current_player_names if x in potential_starters])
        except:
            if verbose:
                warnings.warn(f"Could not process event {event_idx} to identify focal team starters")
            continue
        if focal_team_starters_for_this_game is not None:
            break

    focal_team_starter_names_2_entity_idxs = {
        starter: focal_team_potential_starter_names_2_entity_idxs[starter]
        for starter in focal_team_starters_for_this_game
    }

    ### Organize the events into a BasketballData object.
    events_mutated = []
    n_events_without_focal_team_starters = 0

    for event_idx in event_idxs:
        try:
            event = get_event_in_baller2vec_format(
                event_idx,
                game_data,
                event_label_data,
                event_label_dict,
                player_data,
                sampling_rate_Hz=sampling_rate_Hz,
                verbose=verbose,
            )

            ### Filter out events that don't have all 5 focal team starters
            current_player_names = set(event.player_names)
            has_focal_team_starters = focal_team_starters_for_this_game.issubset(current_player_names)

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
            player_names_new_order = sorted(event.player_names, key=lambda name: player_names_2_entity_idxs[name])
            permutation_indices = [player_names_orig_order.index(elem) for elem in player_names_new_order]

            event_with_player_reindexing = copy.deepcopy(event)
            event_with_player_reindexing.player_names = player_names_new_order

            for idx in range(len(event_with_player_reindexing.moments)):
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

            ### Optionally filter out events that don't have normalized hoop sides.
            NORMALIZED_HOOP_SIDES = [0] * 5 + [1] * 5  # focal team (blue) has scoring hoop on left.
            if (
                event_with_player_reindexing.moments[0].player_hoop_sides != NORMALIZED_HOOP_SIDES
            ) and discard_nonstandard_hoop_sides:
                continue

            ### Normalize hoop sides.   Assume focal team (blue) has scoring hoop on left.  If not, we
            # flip the court 180 degrees around the center of the court (i.e. negate
            # both x and y coords w.r.t center of court). This controls for the effect of hoop
            # switches at half time on the court dynamics, in terms of both offense vs defense
            # direction, as well as in terms of player handedness.
            event_with_player_reindexing_and_court_rotations_where_needed = copy.deepcopy(event_with_player_reindexing)
            for idx_moment in range(len(event_with_player_reindexing_and_court_rotations_where_needed.moments)):
                if (
                    event_with_player_reindexing_and_court_rotations_where_needed.moments[idx_moment].player_hoop_sides
                    != NORMALIZED_HOOP_SIDES
                ):
                    event_with_player_reindexing_and_court_rotations_where_needed = (
                        rotate_court_180_degrees_for_one_moment_of_an_event(
                            event_with_player_reindexing_and_court_rotations_where_needed,
                            idx_moment,
                            NORMALIZED_HOOP_SIDES,
                        )
                    )

            ### Now we extend our accumulating lists of events.
            events_mutated.extend([event_with_player_reindexing_and_court_rotations_where_needed])

        except:
            if verbose:
                warnings.warn(f"Could not process event idx {event_idx}")
            continue

    print(
        f"\n\n --- Of {len(event_idxs)} total events, I successfully constructed {len(events_mutated)} events with focal team starters."
        f"\nThe number of processed events without focal team starters was {n_events_without_focal_team_starters}."
    )
    if len(events_mutated) == 0:
        breakpoint()
        raise ValueError("This game had ZERO events retained. Check into this.")

    events_filtered_and_mutated = clean_events_of_moments_with_too_small_intervals(
        events_mutated, sampling_rate_Hz, verbose
    )

    # TODO: The `remove_events_with_player_substitution` function performs a second filtering.
    # Combine this with the first filtering, above, which extracts
    # events that have the focal team as starters in the first moment.
    # The aligning would be advantageous to reduce the complexity of the operations
    # (filtering -> mutating -> another, related filtering).
    # I think ideally we'd reduce the size of the huge try/catch above, and have separate functions
    # that each iterate through events and perform a fixed operation.

    events_processed = remove_events_with_player_substitutions(events_filtered_and_mutated)
    moments_processed = [moment for event in events_processed for moment in event.moments]
    player_coords_unnormalized = player_coords_from_moments(moments_processed)
    ball_coords_unnormalized = ball_coords_from_moments(moments_processed)
    example_stop_idxs = get_example_stop_idxs(events_processed, sampling_rate_Hz, verbose)
    play_start_stop_idxs = get_play_start_stop_idxs(events_processed)
    player_names_by_example = get_player_names_by_example(moments_processed, example_stop_idxs, player_data)

    ###
    # Some checks on the processed data
    ###

    if not len(moments_processed) == len(player_coords_unnormalized) == len(ball_coords_unnormalized):
        raise ValueError(
            "Somehow the number of player coords, ball coords, and moments don't match.  Check implementation."
        )

    return BasketballData(
        events_processed,
        player_coords_unnormalized,
        ball_coords_unnormalized,
        play_start_stop_idxs,
        example_stop_idxs,
        player_names_by_example,
        player_data,
        sampling_rate_Hz,
    )


###
# Multiple games (in memory)
###


def get_flattened_events_from_games(games: List[BasketballData]) -> List[Event]:
    """
    Concatentate all the events from a set of games
    """
    events_all = []
    for game in games:
        events_all.extend(game.events)
    return events_all


def get_flattened_player_coords_unnormalized_from_games(
    games: List[BasketballData],
) -> NumpyArray3D:
    """
    Concatentate all the unnormalized coords from a set of games
    """

    # concatentate the player_coords_unnormalized
    player_coords_unnormalized_as_tuple = ()
    for game in games:
        player_coords_unnormalized_as_tuple = player_coords_unnormalized_as_tuple + (game.player_coords_unnormalized,)
    player_coords_unnormalized = np.vstack(
        player_coords_unnormalized_as_tuple
    )  # T=total number of training samples about 4.5 hours..
    return player_coords_unnormalized


def make_basketball_data_from_games(games: List[BasketballData], verbose: bool = True):
    # RK: The games have already been constructed as Basketball Data; so we just need to concatentate
    # everything.
    sampling_rate_Hz = games[0].sampling_rate_Hz
    events_flattened = get_flattened_events_from_games(games)
    moments_flattened = [moment for event in events_flattened for moment in event.moments]

    player_coords_unnormalized = player_coords_from_moments(moments_flattened)
    ball_coords_unnormalized = ball_coords_from_moments(moments_flattened)
    play_start_stop_idxs = get_play_start_stop_idxs(events_flattened)
    example_stop_idxs = get_example_stop_idxs(events_flattened, sampling_rate_Hz, verbose)
    player_names_by_example = flatten_list_of_lists([game.player_names_by_example for game in games])

    # TODO: I think that the player_data will be the same for all games, by upstream construction.
    # But we should handle this more carefully. E.g. we could just check it here and raise an error
    # if false.
    player_data_from_all_games = games[0].player_data
    print(
        f"From {len(player_coords_unnormalized)} timesteps, there are {len(play_start_stop_idxs)} provided events (plays) and {len(example_stop_idxs)} inferred events (examples)."
    )
    return BasketballData(
        events_flattened,
        player_coords_unnormalized,
        ball_coords_unnormalized,
        play_start_stop_idxs,
        example_stop_idxs,
        player_names_by_example,
        player_data_from_all_games,
        sampling_rate_Hz,
    )
