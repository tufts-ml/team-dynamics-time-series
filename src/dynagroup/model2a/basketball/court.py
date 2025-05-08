import copy
from typing import Union

import matplotlib.image as mpimg
import numpy as np
import os

from dynagroup.types import NumpyArray2D, NumpyArray3D


###
# CONSTS
###

### Court dimensions
"""
The dimensions of an NBA court are 95 feet by 50 feet.  

References:
    https://github.com/airalcorn2/baller2vec/issues/5
"""

X_MIN_COURT = 0
X_MAX_COURT = 94
Y_MIN_COURT = 0
Y_MAX_COURT = 50


# TODO: Can I make a scheme where I plot the court in the NORMALIZED coords,
# so that I don't have to unnormalize all the time?!
COURT_AXIS_UNNORM = [X_MIN_COURT, X_MAX_COURT, Y_MIN_COURT, Y_MAX_COURT]
COURT_IMAGE = mpimg.imread("image/nba_court_T.png")

###
# Normalize/Unnormalize
###


def normalize_coords(
    player_coords_unnormalized: Union[NumpyArray2D, NumpyArray3D]
) -> Union[NumpyArray2D, NumpyArray3D]:
    """
    Arguments:
        player_coords_unnormalized: NumpyArray whose last axis has shape D=2, representing x and y
        coordinates on the court

    """
    coords_normalized = copy.copy(player_coords_unnormalized)
    coords_normalized[..., 0] /= X_MAX_COURT
    coords_normalized[..., 1] /= Y_MAX_COURT
    return coords_normalized


def unnormalize_coords(coords_normalized: Union[NumpyArray2D, NumpyArray3D]) -> Union[NumpyArray2D, NumpyArray3D]:
    """
    Arguments:
        coords_normalized: NumpyArray whose last axis has shape D=2, representing x and y
        coordinates on the court
    """
    player_coords_unnormalized = copy.copy(coords_normalized)
    player_coords_unnormalized[..., 0] *= X_MAX_COURT
    player_coords_unnormalized[..., 1] *= Y_MAX_COURT
    return player_coords_unnormalized


###
# Flip
###


def flip_player_coords_unnormalized(player_coords_unnormalized: NumpyArray2D) -> NumpyArray2D:
    """
    We want to rotate the coordinates 180, where the center of the coordinate
    system is the center of the basketball court.

    In other words, we want to negate both the x and y coordinates, once we've represented
    the vectors with respect to the center of the court.

    The motivation here is as follows: at halftime, a team's own basket switches to the other side.
    We want to control for the direction of defense and offense, as well as for player handedness.
    """
    CENTER_OF_NORMALIZED_COURT = np.array([0.5, 0.5])

    coords_normalized = normalize_coords(player_coords_unnormalized)
    coords_normalized_and_centered = coords_normalized - CENTER_OF_NORMALIZED_COURT
    coords_normalized_and_centered_and_flipped = coords_normalized_and_centered * -1
    coords_normalized_and_flipped = coords_normalized_and_centered_and_flipped + CENTER_OF_NORMALIZED_COURT
    coords_flipped = unnormalize_coords(coords_normalized_and_flipped)
    return coords_flipped
