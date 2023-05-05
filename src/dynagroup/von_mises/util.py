import numpy as np

from dynagroup.types import NumpyArray1D, NumpyArray2D


def points_from_angles(angles: NumpyArray1D) -> NumpyArray2D:
    """Compute the Cartesian coordinates of a point on the unit circle, given the angle."""
    xs = np.cos(angles)
    ys = np.sin(angles)
    return np.vstack((xs, ys)).T


def angles_from_points(points: NumpyArray2D) -> NumpyArray1D:
    """Compute the angle of a point on the unit circle, given its Cartesian coordinates."""
    return np.arctan2(points[:, 1], points[:, 0])
