import jax.numpy as jnp

from dynagroup.types import JaxNumpyArray1D


"""
Module-level docstring:

    System recurrence transformations map (JD,) to (M_s,), where
        J: number of entities (for basketball J=10)
        D: dimension of continuous states (for basketball D=2)
        M_s: dimension of system recurrence information and system covariates after transformation

    The flattened JD vector scrolls through j's for each d.  I.e. is can be indexed as
    (j_1,0), (j_2, 0), .... (J,0), (j_1, 1), (j_2, 1), ..., (J,1), ... (J,D)

"""


def TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Computes the x-distances between centroids of the two teams.
    """
    return jnp.atleast_1d(jnp.mean(x_prevs_reshaped[:5]) - jnp.mean(x_prevs_reshaped[5:10]))


LIST_OF_SYSTEM_RECURRENCES = [
    TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation,
]
