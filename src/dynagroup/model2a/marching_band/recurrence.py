import jax.numpy as jnp
import numpy as np 

from dynagroup.types import JaxNumpyArray1D


"""
Module-level docstring:

    System recurrence transformations map (JD,) to (D_s,), where
        J: number of entities (for marching band J=64)
        D: dimension of continuous states (for marching band D=2)
        D_s: dimension of system recurrence information and system covariates after transformation

    The flattened JD vector scrolls through j's for each d.  I.e. is can be indexed as
    (j_1,0), (j_2, 0), .... (J,0), (j_1, 1), (j_2, 1), ..., (J,1), ... (J,D)

"""


def cluster_trigger_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Returns the scalar value associated with the probability that the cluster state should be triggered based on how many players are currently out of bounds.

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """
    count = 0
    for entity in x_prevs_reshaped[:64]: 
        if entity < 0 or entity > 1: 
            count +=1
   
    if count >= 7: 
        trigger = 1
    else:
        trigger = count/7   #What did you mean by probability? 
    return jnp.array([trigger]) 


def direction_entity_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Returns a flattened vector of how far/close the jth entity is to a certain gridpoint. There are 4 gridpoints at x= 0.2,0.4,0.6,0.8

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """

    grid1 = abs(x_prevs_reshaped[0] - 0.2) 
    grid2 = abs(x_prevs_reshaped[0] - 0.4) 
    grid3 = abs(x_prevs_reshaped[0] - 0.6) 
    grid4 = abs(x_prevs_reshaped[0] - 0.8) 

    return jnp.stack((grid1, grid2, grid3, grid4))
