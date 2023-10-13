import os
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from dynagroup.model2a.basketball.court import unnormalize_coords
from dynagroup.types import (
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
    NumpyArray5D,
)


###
# Load forecasts
###


def load_agentformer_forecasts(size: str) -> NumpyArray5D:
    """
    Argument:
        size: in [small, medium, large]
    Returns:
        array of shape (E,S,T_forecast,J,D)
    """

    FILEPATH_AGENTFORMER = (
        f"results/basketball/CLE_starters/artifacts_external/agentformer_sampled_trajectories_{size}_set.npy"
    )

    forecasts_normalized_and_collapsed = np.load(FILEPATH_AGENTFORMER)  # (S, ET, J, D)
    S, ET, J, D = np.shape(forecasts_normalized_and_collapsed)

    T_forecast = 30
    E = int(ET / T_forecast)

    forecasts_normalized_SETJD = np.reshape(forecasts_normalized_and_collapsed, (S, E, T_forecast, J, D))
    # Rk: We expect the `ET` dimension to scroll through all timesteps for each example.
    # I.e. examples is the "outer" loop.
    if not forecasts_normalized_SETJD[0, 1, 0, 0, 0] == forecasts_normalized_and_collapsed[0, T_forecast, 0, 0]:
        raise ValueError("This function is not implemented correctly")

    forecasts_normalized = forecasts_normalized_SETJD.swapaxes(0, 1)  # has shape (E,S,T,J,D)

    return unnormalize_coords(forecasts_normalized)


def load_dynagroup_forecasts(dir_forecasts_ours: str) -> Tuple[NumpyArray5D, NumpyArray4D, NumpyArray4D]:
    """
    Returns:
        forward_simulations_all_examples: array of shape (E,S,T_forecast,J,D)
        fixed_velocity_all_examples: array of shape (E,1,T_forecast,J,D)
        ground_truth_all_examples:   array of shape (E,T_forecast,J,D)
    """
    # List all items in the directory (both files and directories)
    all_items = os.listdir(dir_forecasts_ours)

    # Filter only directories
    subdirectories = [item for item in all_items if os.path.isdir(os.path.join(dir_forecasts_ours, item))]

    # TODO: ideally we'd infer these things from the data instead of hardcode it
    E, S, T_forecast, J, D = 78, 20, 30, 10, 2
    if len(subdirectories) != E:
        raise ValueError("The function is not implemented correctly.")

    forward_simulations_all_examples = np.zeros((E, S, T_forecast, J, D))
    fixed_velocity_all_examples = np.zeros((E, 1, T_forecast, J, D))
    ground_truth_all_examples = np.zeros((E, T_forecast, J, D))

    for e in range(E):
        prefix = f"example_idx_{e}_"
        subdirectories_for_example = [s for s in subdirectories if s.startswith(prefix)]
        if not len(subdirectories_for_example) == 1:
            breakpoint()
            raise ValueError("Found multiple forecasting subdirectories for a single example index. Error!")
        subdirectory_for_example = subdirectories_for_example[0]
        total_dir = os.path.join(dir_forecasts_ours, subdirectory_for_example)
        forward_simulations = np.load(os.path.join(total_dir, "forward_simulations.npy"))
        fixed_velocity = np.load(os.path.join(total_dir, "fixed_velocity.npy"))
        ground_truth = np.load(os.path.join(total_dir, "ground_truth.npy"))

        if np.shape(forward_simulations) != (S, T_forecast, J, D):
            raise ValueError("The function is not implemented correctly.")

        forward_simulations_all_examples[e] = forward_simulations
        fixed_velocity_all_examples[e, 0] = fixed_velocity
        try:
            ground_truth_all_examples[e] = ground_truth
        except:
            # some examples seem to have had context sizes that didn't leave enough room for a forecast
            ground_truth_all_examples[e] = np.full((T_forecast, J, D), np.nan)
    return forward_simulations_all_examples, fixed_velocity_all_examples, ground_truth_all_examples


###
# Metrics
###


@dataclass
class Metrics:
    BOTH_TEAMS__MSE_ESJ: NumpyArray3D
    BOTH_TEAMS__MSE_ES: NumpyArray2D
    BOTH_TEAMS__MSE_ES: NumpyArray2D
    BOTH_TEAMS__MSE_E: NumpyArray1D
    BOTH_TEAMS__MSE_T: NumpyArray1D
    BOTH_TEAMS__MSE: float
    BOTH_TEAMS__SE_MSE: float
    CLE__MSE_ESJ: NumpyArray3D
    CLE__MSE_ES: NumpyArray2D
    CLE__MSE_ES: NumpyArray2D
    CLE__MSE_E: NumpyArray1D
    CLE__MSE_T: NumpyArray1D
    CLE__MSE: float
    CLE__SE_MSE: float


def compute_squared_distances_from_forecasts_to_truth(
    forecasts: NumpyArray5D, ground_truth: NumpyArray4D
) -> NumpyArray4D:
    """
    Given forecasts with shape (E,S,T,J,D) and ground truth with shape (E,T,J,D),
    we want to compute the MSE along the T and D dimensions,
    to give an array of shape (E,S,J)?

    Arguments:
        forecasts: shape (E,S,T,J,D)
        ground_truth: shape (E,T,J,D)

    Returns:
        squared_distances: shape (E,S,T,J)

    Notation:
        E: number of examples
        S: number of probabilistic forecasts
        T: size of forecasting window
        J: number of entities
        D: dimensionality
    """

    if not np.ndim(forecasts) == 5:
        raise ValueError("Forecasts does not have the expected dimensionality.")

    (E_1, S_1, T_1, J_1, D_1) = np.shape(forecasts)
    (E_2, T_2, J_2, D_2) = np.shape(ground_truth)
    if not (E_1 == E_2 and T_1 == T_2 and J_1 == J_2 and D_1 == D_2):
        raise ValueError("Dimensionalities of forecasts and ground truth do not match where expected.")

    # Compute sum of square differences along D dimension
    diff = forecasts - ground_truth[:, None, :, :, :]
    return np.sum(diff**2, axis=4)


def compute_metrics(forecasts: NumpyArray5D, ground_truth: NumpyArray4D) -> Metrics:
    """
    Arguments:
        forecasts: shape (E,S,T,J,D)
        ground_truth: shape (E,T,J,D)
    """

    # I didn't construct the random context times correctly; for the AISTATS 2024 submission, I allowed the bounds to
    # be one timestep too large, so that for a small number of examples (3/78), there was not a way to construct
    # the desired forecasting window (30 timesteps) after the random context time.  For these examples, the ground truth
    # is given as np.nan.
    num_valid_examples = len([x for x in ground_truth[:, 0, 0, 0] if not np.isnan(x)])

    # we suppress warnings where we have a mean of empty slice.  This comes from cases
    # where the ground truth is all nan.
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")

    # Whole team
    BOTH_TEAMS__squared_distances_ESTJ = compute_squared_distances_from_forecasts_to_truth(forecasts, ground_truth)
    BOTH_TEAMS__MSE_ESJ = np.nanmean(BOTH_TEAMS__squared_distances_ESTJ, axis=3)
    BOTH_TEAMS__MSE_ES = np.nanmean(BOTH_TEAMS__squared_distances_ESTJ, axis=(2, 3))
    BOTH_TEAMS__MSE_E = np.nanmean(BOTH_TEAMS__squared_distances_ESTJ, axis=(1, 2, 3))
    BOTH_TEAMS__MSE_T = np.nanmean(BOTH_TEAMS__squared_distances_ESTJ, axis=(0, 1, 3))
    BOTH_TEAMS__MSE = np.nanmean(BOTH_TEAMS__squared_distances_ESTJ)
    BOTH_TEAMS__SE_MSE = np.nanstd(BOTH_TEAMS__MSE_E) / np.sqrt(num_valid_examples)

    # CLE only
    CLE__squared_distances_ESTJ = BOTH_TEAMS__squared_distances_ESTJ[:, :, :, :5]
    CLE__MSE_ESJ = np.nanmean(CLE__squared_distances_ESTJ, axis=3)
    CLE__MSE_ES = np.nanmean(CLE__squared_distances_ESTJ, axis=(2, 3))
    CLE__MSE_E = np.nanmean(CLE__squared_distances_ESTJ, axis=(1, 2, 3))
    CLE__MSE_T = np.nanmean(CLE__squared_distances_ESTJ, axis=(0, 1, 3))
    CLE__MSE = np.nanmean(CLE__squared_distances_ESTJ)
    CLE__SE_MSE = np.nanstd(CLE__MSE_E) / np.sqrt(num_valid_examples)
    return Metrics(
        BOTH_TEAMS__MSE_ESJ,
        BOTH_TEAMS__MSE_ES,
        BOTH_TEAMS__MSE_E,
        BOTH_TEAMS__MSE_T,
        BOTH_TEAMS__MSE,
        BOTH_TEAMS__SE_MSE,
        CLE__MSE_ESJ,
        CLE__MSE_ES,
        CLE__MSE_E,
        CLE__MSE_T,
        CLE__MSE,
        CLE__SE_MSE,
    )
