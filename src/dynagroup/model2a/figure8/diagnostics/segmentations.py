import numpy as np
from ssm.util import find_permutation

from dynagroup.types import NumpyArray1D, NumpyArray2D


def compute_regime_labeling_accuracy(
    estimated_regime_seq: NumpyArray2D,
    true_regime_seq: NumpyArray1D,
) -> float:
    """
    Due to label-switching, need to find first find the permutation that best matches the truth
    in order to compute accuracy.
    """
    # Rk: The `find_permutation` function requires numpy arrays
    perm_of_estimated = find_permutation(
        np.array(estimated_regime_seq),
        np.array(true_regime_seq),
    )
    estimated_regime_seq_with_aligned_labels = np.array(
        [perm_of_estimated[x] for x in estimated_regime_seq]
    )
    pct_correct_regimes = np.mean(true_regime_seq == estimated_regime_seq_with_aligned_labels)
    return pct_correct_regimes
