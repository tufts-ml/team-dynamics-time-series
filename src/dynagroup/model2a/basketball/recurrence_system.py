import jax.numpy as jnp

from dynagroup.model2a.basketball.anchors import (
    compute_one_hot_indicator_of_closest_anchor_point,
)
from dynagroup.types import JaxNumpyArray1D


"""
Module-level docstring:

    System recurrence transformations map (JD,) to (D_s,), where
        J: number of entities (for basketball J=10)
        D: dimension of continuous states (for basketball D=2)
        D_s: dimension of system recurrence information and system covariates after transformation

    The flattened JD vector scrolls through j's for each d.  I.e. is can be indexed as
    (j_1,0), (j_2, 0), .... (J,0), (j_1, 1), (j_2, 1), ..., (J,1), ... (J,D)

"""


def TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Computes the x-distances between centroids of the two teams.

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """
    ### WARNING: This function is experimental.
    return jnp.atleast_1d(jnp.mean(x_prevs_reshaped[:5]) - jnp.mean(x_prevs_reshaped[5:10]))


def ALL_PLAYER_LOCATIONS_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """
    return x_prevs_reshaped


def COURT_CONFIGURATION_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Give grid cell fullness indicators to see which locations on the court are inhabited.

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    Returns:
        A binary array of size (n_cells_in_partition_of_unit_interval**2), whose i-th element
        is a cell in a grid of the court which takes the value 1 if at least 1 player inhabits
        that grid cell
    """
    ### WARNING: This function is experimental.
    J = 10
    n_cells_in_partition_of_unit_interval = 6
    grid_cell_is_filled_by_someone = jnp.zeros(n_cells_in_partition_of_unit_interval**2, dtype=int)
    for j in range(5):
        entity_coord = jnp.array([x_prevs_reshaped[j], x_prevs_reshaped[j + J]])
        grid_cell_is_filled_by_this_entity = compute_one_hot_indicator_of_closest_anchor_point(
            entity_coord, n_cells_in_partition_of_unit_interval
        )
        grid_cell_is_filled_by_someone = jnp.logical_or(
            grid_cell_is_filled_by_someone, grid_cell_is_filled_by_this_entity
        )
    return jnp.array(grid_cell_is_filled_by_someone, dtype=int)


def COURT_CONFIGURATION_AND_ALL_PLAYER_LOCATION_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    Returns:
        A binary array of size (n_cells_in_partition_of_unit_interval**2), whose i-th element
        is a cell in a grid of the court which takes the value 1 if at least 1 player inhabits
        that grid cell
    """
    ### WARNING: This function is experimental.
    court_configuration = COURT_CONFIGURATION_system_recurrence_transformation(x_prevs_reshaped, None)
    return jnp.concatenate((court_configuration, x_prevs_reshaped))


LIST_OF_SYSTEM_RECURRENCES = [
    TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation,
    ALL_PLAYER_LOCATIONS_system_recurrence_transformation,
    COURT_CONFIGURATION_AND_ALL_PLAYER_LOCATION_system_recurrence_transformation,
]
