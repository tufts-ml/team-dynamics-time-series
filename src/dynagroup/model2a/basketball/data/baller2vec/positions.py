from typing import Dict

import requests
from bs4 import BeautifulSoup


"""
Assign position-based entity indexing to opponent team before modeling, 
so that we can have index consistency even after player substitutions.
"""

###
# Constants
###


FORWARD_POSITIONS_ORDERED = [
    "Small forward / shooting guard",
    "Small forward",
    "Small forward / power forward",
    "Power forward / small forward",
    "Power forward",
    "Power forward / center",
    "Center / power forward",
    "Shooting guard / small forward",
]


GUARD_POSITIONS_ORDERED = [
    "Small forward / shooting guard",
    "Shooting guard / small forward",
    "Shooting guard",
    "Point guard / shooting guard",
    "Point guard",
]

CENTER_POSITIONS_ORDERED = [
    "Power forward / center",
    "Center / power forward",
    "Center",
]

POSITION_GROUP_ORDERINGS = [
    FORWARD_POSITIONS_ORDERED,
    CENTER_POSITIONS_ORDERED,
    GUARD_POSITIONS_ORDERED,
]
POSITIONS_ORDERED = [item for sublist in POSITION_GROUP_ORDERINGS for item in sublist]

POSITION_GROUP_TO_N_NEEDED = {"forward": 2, "center": 1, "guard": 2}
POSITION_GROUPS = ["forward", "center", "guard"]


###
# Get player position from name via Wikipedia
###


def _player_position_from_wikipedia_url(url: str) -> str:
    try:
        # Send an HTTP GET request to fetch the Wikipedia page
        response = requests.get(url)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the player infobox table, where the position is usually mentioned
        infobox = soup.find("table", {"class": "infobox vcard"})

        # Find the row in the infobox that contains the position information
        rows = infobox.find_all("tr")
        position = None
        for row in rows:
            if "Position" in row.get_text():
                position = row.find("td").get_text()
                if "forward" in position or "center" in position or "guard" in position:
                    break

        if position:
            return position.strip()
        else:
            return "Position not found for the given player name."
    except:
        return "Position not found for the given player name."


def get_player_position(player_name):
    """
    Given a basketball player name, get their position, by scraping Wikipedia.

    Thanks, Chat GPT!

    Example usage:
        player_name = input("Enter the NBA player's name: ")
        position = get_player_position(player_name)
        print(f"{player_name} plays as {position}")

    """
    # These are here because the wiki pages didn't follow the standard url
    PLAYER_POSITIONS_MANUAL = {"James Johnson": "Power forward / small forward"}

    # Format the player name to be used in the Wikipedia URL
    player_name_wiki = player_name.replace(" ", "_")
    urls_to_try = [
        f"https://en.wikipedia.org/wiki/{player_name_wiki}",
        f"https://en.wikipedia.org/wiki/{player_name_wiki}_(basketball)",
    ]
    for url in urls_to_try:
        try:
            position = _player_position_from_wikipedia_url(url)
            if position == "Position not found for the given player name.":
                continue
            else:
                return position
        except requests.exceptions.RequestException as e:
            print(f"Error accessing URL: {url} - {e}")
            continue  # Continue to the next URL if there's an error

    # If we've made it here, we have exceptions
    if player_name in PLAYER_POSITIONS_MANUAL:
        return PLAYER_POSITIONS_MANUAL[player_name]

    # If we've made it here, we messed up.
    return "Error: Unable to fetch data from Wikipedia or manual list."


###
# Positions from name
###


def get_player_name_2_position(player_data: Dict[int, Dict]) -> Dict[str, str]:
    """
    Arguments:
        player_data: Dict[int, Dict]. The return value of
            load_player_data_from_pydict_info_path

    Returns:
        dict mapping player name to position

        Sample clip:

            In [1]: player_name_2_position
            Out[1]:
            {
                'Spencer Hawes': 'Center / power forward',
                'Marvin Williams': 'Power forward / small forward',
                'Jeremy Lamb': 'Shooting guard / small forward',
                ....
            }

        All the different positions extracted from one game involving TOR:

            {
                'Small forward',
                'Small forward / power forward',
                'Shooting guard',
                'Power forward / small forward',
                'Shooting guard / small forward',
                'Center / power forward',
                'Power forward / center',
                'Small forward / shooting guard',
                'Center',
                'Power forward',
                'Point guard / shooting guard',
                'Point guard'
            }
    """
    player_name_2_position = {}
    for player_info in player_data.values():
        player_name = player_info["name"]
        try:
            position = get_player_position(player_name)
        except:
            print(f"Cannot find position for {player_name}")
        player_name_2_position[player_name] = position
    return player_name_2_position


###
# Orderings from position
###

### 1. Assign players to position groups (forward, guard, center)

from typing import Dict, List

import numpy as np


def make_position_group_to_player_names(
    opponent_names_2_positions: Dict[str, str]
) -> Dict[str, List[str]]:
    """
    Assign players to position groups (forward, guard, center)
    Note that positions are such that each player can be multiply classified.
    So we start with the rarest position group, and assign players to that rare group,
    starting with players who have the least classifications, until we have enough
    players to fill out the specs (1 center, 2 forwards, 2 guards)

    Arguments:
        Example:
            {
                'Marvin Williams': 'Power forward / small forward',
                'Kemba Walker': 'Point guard',
                'PJ Hairston': 'Small forward / shooting guard',
                'Cody Zeller': 'Center / power forward',
                'Nicolas Batum': 'Power forward / small forward'
            }

    Returns:
        Example:
            {
                'forward': ['PJ Hairston', 'Cody Zeller'],
                'guard': ['Kemba Walker', 'Nicolas Batum'],
                'center': ['Marvin Williams']
            }
    """
    ### Find whether each player coiuld be a forward, center, or guard.
    ### Often there are multiple categorizations
    names = list(opponent_names_2_positions.keys())
    matrix_bool__players_by_position_group_memberships = np.zeros(
        (5, 3), dtype=bool
    )  # (forward, center, guard)
    for p, name in enumerate(names):
        position = opponent_names_2_positions[name]
        for position_group in POSITION_GROUPS:
            if position_group in position.lower():
                position_group_idx = POSITION_GROUPS.index(position_group)
                matrix_bool__players_by_position_group_memberships[p, position_group_idx] = 1

    # Example:
    # matrix_bool__players_by_position_group_memberships= numpy.ndarray([
    #     [True, False, False],
    #     [False, False, True],
    #     [True, False, True],
    #     [True, True, False],
    #     [True, False, False]
    # ])

    position_group_totals = np.sum(matrix_bool__players_by_position_group_memberships, 0)
    position_groups_low_to_high = np.argsort(
        position_group_totals
    )  # order position group from least common to most common

    player_assignment_totals = np.sum(matrix_bool__players_by_position_group_memberships, 1)
    player_assignments_low_to_high = np.argsort(player_assignment_totals)

    # players for that group. Then we iterate.
    position_idxs_used = set()
    position_group_to_player_names = {"forward": [], "center": [], "guard": []}
    for position_group_idx in position_groups_low_to_high:
        position_group = POSITION_GROUPS[position_group_idx]
        n_used_for_position_group = 0
        for player_idx in player_assignments_low_to_high:
            if player_idx not in position_idxs_used:
                if (
                    matrix_bool__players_by_position_group_memberships[
                        player_idx, position_group_idx
                    ]
                    == True
                ):
                    player_name = names[player_idx]
                    position_group_to_player_names[position_group].append(player_name)
                    position_idxs_used.add(player_idx)
                    n_used_for_position_group += 1
            if n_used_for_position_group == POSITION_GROUP_TO_N_NEEDED[position_group]:
                break

    return position_group_to_player_names


### 2. Within each position group order players by position group


def get_opponent_player_names_ordered_by_position(
    opponent_names_2_positions: Dict[str, str],
) -> List[str]:
    position_group_to_player_names = make_position_group_to_player_names(opponent_names_2_positions)

    # Arguments: position_group_to_player_names, opponent_names_2_positions,
    player_names_in_order = []

    for g, position_group in enumerate(POSITION_GROUPS):
        player_names_for_this_group = position_group_to_player_names[position_group]
        positions_for_this_group = [
            opponent_names_2_positions[name] for name in player_names_for_this_group
        ]
        rankings_for_players_in_this_group = [
            POSITIONS_ORDERED.index(position) for position in positions_for_this_group
        ]
        internal_idxs_for_this_group = sorted(
            range(len(rankings_for_players_in_this_group)),
            key=lambda i: rankings_for_players_in_this_group[i],
        )
        names_in_order_for_this_group = [
            player_names_for_this_group[i] for i in internal_idxs_for_this_group
        ]
        player_names_in_order.extend(names_in_order_for_this_group)

    return player_names_in_order


def make_opponent_names_2_entity_idxs(
    opponent_names_2_positions: Dict[str, str], index_from_5_to_9: bool = True
) -> Dict[str, int]:
    """
    OVERVIEW:

        We need to elide over player substitutions from the opponent team.
        I’m able to scrape Wikipedia for positions, given player names.
        To be concrete, `POSITIONS_ORDERED` gives the set of positions listed from all the players
        in one particular game.   I’ve put this set into an order, for reasons described later.
        See `POSITIONS_ORDERED`.

        I think it would be easiest (and sufficient for our purposes), if we always assign indices 5-9 to each
        player on the opponent team.   (There is probably a way to be more general than this, and have
        a different entity model for all 12 positions listed above, or more, but then we could run into
        some issues like poor parameter learning for some of these positions due to sparsity - in the
        extreme a new position entirely could appear in the test set.  I don’t think this is worth dealing
        with for our purposes.)

        How to do this? That is,
        how can I automatically assign subsets of POSITIONS_ORDERED to have unique index labels in [5,6,7,8,9]?
        I’d like the player playing the role of center to always be assigned to some particular index (7, say).
        If there are multiple players playing the role of center, it’s fine if only one of them gets
        assigned to 7.

        My current solution to this is as follows:
            1.    Assign players to coarse "position groups" (forward, guard, center)
                Note that positions are such that each player can be multiply classified.
                So we start with the rarest position group, and assign players to that rare group,
                starting with players who have the least classifications, until we have enough
                players to fill out the specs (1 center, 2 forwards, 2 guards).
            2. Within each position group order players by position group, using the `POSITIONS_ORDERED` list.
            That is, for each position group, I scroll through `POSITIONS_ORDERED` in order,
            check for a match in the current positions, and if one is found I assign the lowest unassigned index.

    REMARK:
        What if there are multiple players with the same exact label on a given play?
        Then they will just get ordered randomly

    Argumments:
        index_from_5_to_9 : If true, the indices constructed range from 5-9 instead of 0-4.
            Set to true by default because we usually want the focal team to have indices 0-4.
    """

    opponent_names_ordered = get_opponent_player_names_ordered_by_position(
        opponent_names_2_positions,
    )
    opponent_names_2_entity_idxs = {name: idx for (idx, name) in enumerate(opponent_names_ordered)}

    if index_from_5_to_9:
        opponent_names_2_entity_idxs = {
            key: value + 5 for key, value in opponent_names_2_entity_idxs.items()
        }

    return opponent_names_2_entity_idxs
