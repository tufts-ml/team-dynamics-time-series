from itertools import product
from typing import List

import jax.numpy as jnp

from dynagroup.types import JaxNumpyArray1D


"""
Construct anchor points on the normalized ([0,1] x [0,1]) basketball court 
"""


def find_midpoints_of_unit_interval(n: int) -> List[float]:
    """
    Find midpoints of cells of unit interval divided into n equal parts.

    Arguments:
        n: number of cells in partition of unit interval
    """
    # Calculate the width of each cell
    cell_width = 1 / n

    # Initialize a list to store midpoints
    midpoints = []

    # Calculate midpoints for each cell
    for i in range(n):
        midpoint = (i + 0.5) * cell_width
        midpoints.append(midpoint)

    return midpoints


def construct_anchor_points_on_normalized_court(n):
    """
    Arguments:
        n: number of cells in partition of unit interval
    """
    midpoints = find_midpoints_of_unit_interval(n)
    return jnp.array(list(product(midpoints, midpoints)))


def compute_normalized_distances_to_anchor_points(normalized_coord: JaxNumpyArray1D) -> JaxNumpyArray1D:
    """
    Arguments:
        normalized_coord: a point on the normalized basketball court [0,1]x[0,1]
    """

    # TODO: Don't hardcode the `n` which determines he number of anchor points
    ANCHOR_POINTS = construct_anchor_points_on_normalized_court(n=6)

    # TODO: Don't hardcode the temperature
    TEMPERATURE = 0.05

    distances = jnp.linalg.norm(ANCHOR_POINTS - normalized_coord, axis=1)
    potentials = jnp.exp(-distances / TEMPERATURE)
    potentials / jnp.sum(potentials)


def compute_one_hot_indicator_of_closest_anchor_point(
    normalized_coord: JaxNumpyArray1D,
    n_cells_in_partition_of_unit_interval: int,
) -> JaxNumpyArray1D:
    """
    Arguments:
        normalized_coord: a point on the normalized basketball court [0,1]x[0,1]
    """

    # TODO: Don't hardcode the `n` which determines he number of anchor points
    # TODO: Don't recompute the anchor points each time
    ANCHOR_POINTS = construct_anchor_points_on_normalized_court(n=n_cells_in_partition_of_unit_interval)
    distances = jnp.linalg.norm(ANCHOR_POINTS - normalized_coord, axis=1)
    one_hot_grid_location = jnp.zeros_like(distances)
    one_hot_grid_location = one_hot_grid_location.at[jnp.argmin(distances)].set(1.0)

    return one_hot_grid_location
