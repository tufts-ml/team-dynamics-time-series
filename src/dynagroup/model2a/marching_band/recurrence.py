import jax.numpy as jnp
import numpy as np 
import jax

from dynagroup.types import JaxNumpyArray1D


def cluster_trigger_system_recurrence_transformation(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Returns the scalar value associated with the probability that the cluster state should be triggered based on how many players are currently out of bounds.

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's. Cite[ChatGPT]
    """
    
    condition1 = x_prevs_reshaped[0:64] > 1 
    condition2 = x_prevs_reshaped[0:64] < 0 

    count = jnp.sum(condition1) + jnp.sum(condition2)
    
    prob = jax.lax.cond(
        count >= 11,             
        lambda _: 1.0,           
        lambda _: count / 11.0,  
        operand=None             
    )

    return jnp.array([prob])


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

def identity_recurrence_entity(
    x_prevs_reshaped: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Returns the identity of the values. 

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """

    return x_prevs_reshaped


def identity_recurrence_system(
    x_prevs_reshaped: JaxNumpyArray1D,
    system_covariates: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Returns the identity of the values. 

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.
    """
    return x_prevs_reshaped
