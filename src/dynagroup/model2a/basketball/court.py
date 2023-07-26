import copy
from typing import Union

from dynagroup.types import NumpyArray2D, NumpyArray3D


###
# CONSTS
###

### Court dimensions
# From: https://github.com/linouk23/NBA-Player-Movements/blob/master/Constant.py

X_MIN_COURT = 0
X_MAX_COURT = 100
Y_MIN_COURT = 0
Y_MAX_COURT = 50


###
# Normalize/Unnormalize
###


def normalize_coords(
    coords_unnormalized: Union[NumpyArray2D, NumpyArray3D]
) -> Union[NumpyArray2D, NumpyArray3D]:
    """
    Arguments:
        coords_unnormalized: NumpyArray whose last axis has shape D=2, representing x and y
        coordinates on the court

    """
    coords_normalized = copy.copy(coords_unnormalized)
    coords_normalized[..., 0] /= X_MAX_COURT
    coords_normalized[..., 1] /= Y_MAX_COURT
    return coords_normalized


def unnormalize_coords(
    coords_normalized: Union[NumpyArray2D, NumpyArray3D]
) -> Union[NumpyArray2D, NumpyArray3D]:
    """
    Arguments:
        coords_normalized: NumpyArray whose last axis has shape D=2, representing x and y
        coordinates on the court
    """
    coords_unnormalized = copy.copy(coords_normalized)
    coords_unnormalized[..., 0] *= X_MAX_COURT
    coords_unnormalized[..., 1] *= Y_MAX_COURT
    return coords_unnormalized
