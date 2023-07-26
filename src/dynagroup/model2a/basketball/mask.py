import warnings
from typing import List, Optional, Union

import numpy as np

from dynagroup.types import NumpyArray2D, NumpyArray3D


def make_mask_of_which_continuous_states_to_use(
    continuous_states: NumpyArray3D,
    entities_to_mask: Optional[Union[int, List[int]]],
    forecast_horizon: int,
) -> Optional[NumpyArray2D]:
    """
    Arguments:
        continuous_states: has shape (T,J,D)
        entity_to_mask: Index in {0,...,J-1}, or None (if no masking)
        forecast_horizon: How many of the trailing observations to mask

    Returns:
        boolean array of shape (T,J) which is False if the continuous states shouldn't be used for inference,
            or None (which has equivalent functionality to the all-True matrix) if we are not using masking.
    """
    entities_to_mask = list(entities_to_mask)

    if entities_to_mask is None or forecast_horizon == 0:
        warnings.warn("Not masking data from any entity.")
        return None

    T, J = np.shape(continuous_states)[:2]

    last_training_idx_for_entity = T - forecast_horizon
    use_continuous_states = np.full((T, J), True)
    for entity_to_mask in entities_to_mask:
        use_continuous_states[last_training_idx_for_entity:, entity_to_mask] = False
    return use_continuous_states
