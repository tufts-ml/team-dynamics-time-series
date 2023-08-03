from typing import List, Optional

import prettyprinter as pp


pp.install_extras()

from dynagroup.model2a.basketball.data.baller2vec.data import (
    BasketballData,
    load_basketball_data_from_single_game_file,
)


def get_basketball_data_for_TOR_vs_CHA(
    event_idxs: Optional[List[int]] = None,
    sampling_rate_Hz: int = 5,
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

    ### Specify hard-coded constants
    TOR_STARTER_NAMES_2_ENTITY_IDXS = {
        "DeMar DeRozan": 0,  # small forward/power forward
        "Luis Scola": 1,  # power forward
        "Jonas Valanciunas": 2,  # center
        "DeMarre Carroll": 3,  # shooting guard/small forward
        "Kyle Lowry": 4,  # point guard
    }

    PATH_TO_GAME_DATA = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_X.npy"
    PATH_TO_EVENT_LABEL_DATA = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/0021500492_y.npy"
    PATH_TO_BALLER2VEC_INFO = "/Users/mwojno01/Repos/dynagroup/data/basketball/baller2vec_format/TOR_vs_CHA/baller2vec_config.pydict"

    ### Identify positions of all players in game
    PLAYER_NAMES_IN_DATASET_2_POSITIONS = {
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
    return load_basketball_data_from_single_game_file(
        PATH_TO_GAME_DATA,
        PATH_TO_EVENT_LABEL_DATA,
        PATH_TO_BALLER2VEC_INFO,
        TOR_STARTER_NAMES_2_ENTITY_IDXS,
        PLAYER_NAMES_IN_DATASET_2_POSITIONS,
        event_idxs,
        sampling_rate_Hz,
    )
