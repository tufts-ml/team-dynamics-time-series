from collections import Counter

import numpy as np

from dynagroup.initialize import InitializationResults
from dynagroup.types import NumpyArray1D


def print_regime_occupancies(regime_sequence: NumpyArray1D) -> None:
    # count occurrences of each value
    counts = Counter(regime_sequence)

    # calculate percentages
    total_counts = sum(counts.values())
    percentages = {k: v / total_counts * 100 for k, v in counts.items()}

    # print percentages
    for k, v in percentages.items():
        print(f"{k}: {v:.2f}%")


def print_multi_level_regime_occupancies_after_init(results_init: InitializationResults) -> None:
    print(f"\n--- After init, the system level regime occupancies are ---")
    s_hat = np.array(results_init.record_of_most_likely_system_states[:, -1], dtype=int)
    print_regime_occupancies(s_hat)

    T, J, num_EM_iterations = np.shape(results_init.record_of_most_likely_entity_states)
    for j in range(J):
        zj_hat = np.array(results_init.record_of_most_likely_entity_states[:, j, -1], dtype=int)
        print(f"\n--- After init, the entity level regime occupancies for entity {j} are ---")
        print_regime_occupancies(zj_hat)
