import copy
from typing import Optional

import numpy as np

from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D, NumpyArray1D


def make_sample_weights_which_mask_the_initial_timestep_for_each_event(
    continuous_states: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    use_continuous_states: Optional[JaxNumpyArray2D] = None,
) -> JaxNumpyArray2D:
    """
    Make sample weights (as a combo of `use_continuous_states` and `example_end_times`).
    The main idea is that `use_continuous_states` tells us whether a particular (t,j) index
    for timestep and entity should be used in inference (usually for partial forecasting purposes),
    whereas example end times tells us if a new example has started.

    Arguments:
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that
            the (t,j)-th element  is 1 if continuous_states[t,j] should be utilized
            and False otherwise.  For any (t,j) that shouldn't be utilized, we don't use
        example_end_times: optional, has shape (E+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across i.i.d "examples".
            An example might induce a largetime gap between timesteps, and a discontinuity in the continuous states x.

            If there are E examples, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th example ended.
            So to get the timesteps for the e-th example, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].
    """
    T, J, D = np.shape(continuous_states)

    if use_continuous_states is None:
        use_continuous_states = np.full((T, J), True)

    if example_end_times is None:
        T = len(continuous_states)
        example_end_times = np.array([-1, T])

    sample_weights = copy.deepcopy(use_continuous_states)
    for event_end_idx in example_end_times[:-1]:
        event_start_idx = event_end_idx + 1
        sample_weights[event_start_idx, :] = False
    return sample_weights
