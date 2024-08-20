import numpy as np

from dynagroup.model2a.figure8.diagnostics.fit_and_forecasting import (
    find_last_index_in_interval_where_array_value_is_close_to_desired_point,
)
from dynagroup.types import NumpyArray2D, NumpyArray3D


#WHAT DOES THE MASK MEAN? 

def make_mask_of_which_continuous_states_to_use(continuous_states: NumpyArray3D) -> NumpyArray2D:
    """
    Arguments:
        continuous_states: has shape (T,J,D)

    Returns
        boolean array of shape (T,J) which is False if the continuous states shouldn't be used for inference
    """

    T, J = np.shape(continuous_states)[:2]

    ENTITY_TO_MASK = 2
    continuous_states_for_entity = continuous_states[:, ENTITY_TO_MASK]
    last_training_idx_for_entity = (
        find_last_index_in_interval_where_array_value_is_close_to_desired_point(
            continuous_states_for_entity,
            desired_point=np.array([1, 1]),
            starting_index=0,
            ending_index=T,
        )
    )

    use_continuous_states = np.full((T, J), True)
    use_continuous_states[last_training_idx_for_entity:, ENTITY_TO_MASK] = False
    return use_continuous_states
