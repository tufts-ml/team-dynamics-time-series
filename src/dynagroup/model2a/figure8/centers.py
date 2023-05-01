from enum import Enum
from typing import List

import numpy as np

from dynagroup.params import ContinuousStateParameters_JAX
from dynagroup.types import NumpyArray3D


def compute_centers(CSP: ContinuousStateParameters_JAX) -> NumpyArray3D:
    """
    Overview
        If A is a rotation matrix rotating around a center, then the dynamics take the form
            x_{t+1} = A(x_{t} - center) + center   (*)
        However, the standard state dynamics is given by
            x_{t+1} = Ax_{t} + b  (~)
        Given a learned state dynamics in the form of Equation (~), we represent it in terms of Equation (*)
        by computing
            center = (I-A)^{-1} b
        Note that (A,b) can differ across entities j and entity-regimes k.

    Returns:
        np.array of shape (J,K,D) whose (j,k)-th subarray gives the center for the j-th entity
        and the k-th regime
    """
    J, K, D, _ = np.shape(CSP.As)

    centers = np.zeros((J, K, D))
    for j in range(J):
        for k in range(K):
            A = CSP.As[j, k]
            b = CSP.bs[j, k]
            centers[j, k, :] = np.linalg.inv(np.eye(D) - A) @ b
    return centers


class CircleLocation(Enum):
    """
    Represents circle locations, defined as
        UP := [0,1]
        DOWN := [0,-1]
    """

    UP = 1
    DOWN = 2
    NEITHER = 3


def compute_circle_locations(centers: NumpyArray3D) -> np.ndarray:
    """
    Arguments:
        centers: np.array of shape (J,K,D) whose (j,k)-th subarray gives the center for the j-th entity
            and the k-th regime
    Returns:
        circle_locations, np.array of shape (J,K) with dtype=CircleLocation whose (j,k)-th
            entry gives the center for the j-th entity and the k-th regime

        (In python 3.9, I could use the type annotation np.ndarray[CircleLocation])
    """

    J, K, _ = np.shape(centers)

    # TODO: Maybe compute the TRUE CENTERS from params_true instead of hard coding them.
    # They may change.
    TRUE_CENTER_OF_UP_CIRCLE = np.array([0, 1])
    TRUE_CENTER_OF_DOWN_CIRCLE = np.array([0, -1])

    # compute distances to up and down circle
    dists_to_up_circle = np.zeros((J, K))
    dists_to_down_circle = np.zeros((J, K))
    for j in range(J):
        for k in range(K):
            dists_to_up_circle[j, k] = np.linalg.norm(centers[j, k] - TRUE_CENTER_OF_UP_CIRCLE)
            dists_to_down_circle[j, k] = np.linalg.norm(centers[j, k] - TRUE_CENTER_OF_DOWN_CIRCLE)

    # compute entity regime indices by circle locations
    CLOSENESS_THRESHOLD = 0.25
    circle_locations = np.full((J, K), CircleLocation.NEITHER, dtype=CircleLocation)
    for j in range(J):
        for k in range(K):
            if dists_to_down_circle[j, k] <= CLOSENESS_THRESHOLD:
                circle_locations[j, k] = CircleLocation.DOWN
            elif dists_to_up_circle[j, k] <= CLOSENESS_THRESHOLD:
                circle_locations[j, k] = CircleLocation.UP

    return circle_locations


def compute_circle_locations_from_CSP(CSP: ContinuousStateParameters_JAX) -> np.ndarray:
    """
    Returns:
        circle_locations, np.array of shape (J,K) with dtype=CircleLocation whose (j,k)-th
            entry gives the center for the j-th entity and the k-th regime

        (In python 3.9, I could use the type annotation np.ndarray[CircleLocation])
    """
    centers = compute_centers(CSP)  # centers has shape (J,K,D)
    return compute_circle_locations(centers)


def compute_regime_labels_for_up_circle_by_entity(
    circle_locations: np.ndarray,
) -> List[int]:
    """
    Arguments:
        circle_locations, np.array of shape (J,K) with dtype=CircleLocation whose (j,k)-th
            entry gives the center for the j-th entity and the k-th regime

        (In python 3.9, I could use the type annotation np.ndarray[CircleLocation])
    """
    J, K = np.shape(circle_locations)

    INVALID_LABEL = -10000
    regime_labels_for_up_circle_by_entity = np.full(J, INVALID_LABEL)
    for j in range(J):
        for k in range(K):
            if circle_locations[j, k] == CircleLocation.UP:
                regime_labels_for_up_circle_by_entity[j] = k
    return regime_labels_for_up_circle_by_entity
