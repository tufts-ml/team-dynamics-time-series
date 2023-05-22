import numpy as np

from dynagroup.types import NumpyArray1D, NumpyArray2D


def compute_running_vulnerability_to_north(squad_angles: NumpyArray2D) -> NumpyArray1D:
    forty_five_degrees_in_radians = np.pi / 4
    due_north_in_radians = -np.pi / 2
    north_quadrant_in_radians = (
        due_north_in_radians - forty_five_degrees_in_radians,
        due_north_in_radians + forty_five_degrees_in_radians,
    )

    T, J = np.shape(squad_angles)

    timesteps_since_someone_looked_in_north_quadrant = 0
    running_vulnerability_to_north_unnormalized = np.zeros(T)
    for t in range(T):
        running_vulnerability_to_north_unnormalized[
            t
        ] = timesteps_since_someone_looked_in_north_quadrant
        if np.array(
            [
                north_quadrant_in_radians[0] <= soldier_angle <= north_quadrant_in_radians[1]
                for soldier_angle in squad_angles[t, :]
            ]
        ).any():
            timesteps_since_someone_looked_in_north_quadrant = 0
        else:
            timesteps_since_someone_looked_in_north_quadrant += 1

    # we use a RELU normalization here.. after twenty seconds you are fully vulnerable.
    # ABOUT_TWENTY_SECONDS_IN_TIMESTEPS = 1290
    # return np.minimum(running_vulnerability_to_north_unnormalized/ABOUT_TWENTY_SECONDS_IN_TIMESTEPS,1)
    #
    # the below only works for short datasets
    return running_vulnerability_to_north_unnormalized / T
