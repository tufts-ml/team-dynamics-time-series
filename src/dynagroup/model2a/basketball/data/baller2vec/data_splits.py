from typing import List

import numpy as np

from dynagroup.model2a.basketball.data.baller2vec.core import Event
from dynagroup.model2a.basketball.data.baller2vec.game import BasketballGame
from dynagroup.types import NumpyArray3D


"""
Module Purpose: 
    Creating (flattened) data structures from many games
    This can be used to make training/test/validation sets from many games.
"""


def get_flattened_events_from_games(games: List[BasketballGame]) -> List[Event]:
    """
    Concatentate all the events from a set of games
    """
    events_all = []
    for game in games:
        events_all.extend(game.events)
    return events_all


def get_flattened_unnormalized_coords_from_games(games: List[BasketballGame]) -> NumpyArray3D:
    """
    Concatentate all the unnormalized coords from a set of games
    """

    # concatentate the coords_unnormalized
    coords_unnormalized_as_tuple = ()
    for game in games:
        coords_unnormalized_as_tuple = coords_unnormalized_as_tuple + (game.coords_unnormalized,)
    coords_unnormalized = np.vstack(
        coords_unnormalized_as_tuple
    )  # T=total number of training samples about 4.5 hours..
    return coords_unnormalized
