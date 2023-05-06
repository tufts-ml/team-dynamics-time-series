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


def force_angles_to_be_from_neg_pi_to_pi(angles):
    # Use mod to adjust angles to be in the range [-2π, 2π]
    adjusted_angles = np.mod(angles, 2 * np.pi) - 2 * np.pi

    # Use a conditional statement to adjust angles to be in the range [-π, π]
    adjusted_angles = np.where(
        adjusted_angles <= -np.pi, adjusted_angles + 2 * np.pi, adjusted_angles
    )
    adjusted_angles = np.where(
        adjusted_angles > np.pi, adjusted_angles - 2 * np.pi, adjusted_angles
    )
    return adjusted_angles


def two_angles_are_close(angle1, angle2, **kwargs):
    """
    Check if two angles in [-pi, pi] are close on a circle.
    Due to the "wrapping" of the circle (pi==-pi), we first check if they're close,
    and then check if they're close after renotating to [0, 2pi]

    Arguments:
        angle1: in [-pi, pi]
        angle2: in [-pi, pi]
    """
    angle1, angle2 = force_angles_to_be_from_neg_pi_to_pi(np.array([angle1, angle2]))
    check1 = np.isclose(angle1, angle2, **kwargs)

    adjusted_angles = np.array([angle1, angle2])
    adjusted_angles = np.where(adjusted_angles <= 0, adjusted_angles + 2 * np.pi, adjusted_angles)
    check2 = np.isclose(adjusted_angles[0], adjusted_angles[1], **kwargs)
    return check1 or check2
