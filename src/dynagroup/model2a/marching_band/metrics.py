from dynagroup.params import AllParameters, dims_from_params
from dynagroup.types import NumpyArray2D
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from typing import List, Union


from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summaries_NUMPY,
    HMM_Posterior_Summary,
)
from dynagroup.types import (
    NumpyArray2D,
    NumpyArray3D,
)

def _get_an_expected_regime_from_VEZ_summaries(
    VEZ_summaries: Union[HMM_Posterior_Summaries_JAX, List[HMM_Posterior_Summary]],
    j: int,
    t: int,
    k: int,
):
    if isinstance(VEZ_summaries, HMM_Posterior_Summaries_JAX) or isinstance(
        VEZ_summaries, HMM_Posterior_Summaries_NUMPY
    ):
        return float(VEZ_summaries.expected_regimes[t, j, k])
    elif isinstance(VEZ_summaries, list):
        return VEZ_summaries[j].expected_regimes[t][k]
    else:
        raise ValueError(f"I don't understand the type of VEZ_summaries, {type(VEZ_summaries)}")


def compute_next_step_predictive_means(
    all_params: AllParameters,
    data: NumpyArray3D,
    VEZ_summaries: Union[HMM_Posterior_Summaries_JAX, List[HMM_Posterior_Summary]],
    after_learning: bool,
) -> NumpyArray2D:
    
    DIMS = dims_from_params(all_params)
    T = len(data)

    predictive_means = np.zeros((T, DIMS.J, DIMS.D))

    predictive_means[0] = np.ones((DIMS.J, DIMS.D)) * np.nan

    for j in range(DIMS.J):
        for t in range(1, T):
            predictive_means_under_entity_regime = np.zeros((DIMS.K, DIMS.D))
            probs_under_entity_regimes = np.zeros(DIMS.K)
            for k in range(DIMS.K):
                A = all_params.CSP.As[j, k]
                b = all_params.CSP.bs[j, k]
                predictive_means_under_entity_regime[k] = A @ data[t - 1, j] + b
                if after_learning:
                    probs_under_entity_regimes[k] = _get_an_expected_regime_from_VEZ_summaries(
                        VEZ_summaries, j, t, k
                    )
                else:
                    # TODO: grab the initialization that we actually used during inference.
                    init_dists_over_entity_regimes = np.ones((DIMS.J, DIMS.K)) / DIMS.K
                    probs_under_entity_regimes[k] = init_dists_over_entity_regimes[j, k]
            predictive_means[t, j] = (
                predictive_means_under_entity_regime.T @ probs_under_entity_regimes
            )
    return predictive_means








