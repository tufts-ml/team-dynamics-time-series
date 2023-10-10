import os
from typing import Callable, Optional

import numpy as np
from matplotlib import pyplot as plt

from dynagroup.params import EntityTransitionParameters
from dynagroup.util import normalize_log_potentials_by_axis


def investigate_entity_transition_probs_in_different_contexts(
    ETP: EntityTransitionParameters,
    xs: np.array,
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX: Callable,
    save_dir: Optional[str] = None,
):
    """
    Parameter investigation: Check entity transition probabilities
    under different system regimes and different closeness-to-origin statuses.

    We print out to screen. If save_dir is not None, we save the TPM to disk.
    """

    # TODO: Don't hardcode the max closeness. It depends on x run through the transformation.

    x_tildes = np.apply_along_axis(
        transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
        2,
        xs,
    )
    large_closeness = np.percentile(x_tildes, 95)
    small_closeness = np.percentile(x_tildes, 5)

    J, L, _, _ = np.shape(ETP.Psis)
    dummy_value = 0  # choice of d should be arbitrary.
    for j in range(J):
        print(f"\n----------- Now investigating entity {j}")
        for system_state in [0, 1]:
            for close_to_origin in [False, True]:
                input = large_closeness if close_to_origin else small_closeness
                potentials = ETP.Ps[j, system_state] + ETP.Psis[j, system_state, :, dummy_value] * input
                entity_trans_probs = np.exp(normalize_log_potentials_by_axis(potentials, axis=1))
                print(
                    f"For entity {j}, under system_state {system_state}, if close to origin is {close_to_origin}, the entity transition probs are \n {entity_trans_probs}"
                )
                if save_dir is not None:
                    filename_to_save = os.path.join(
                        save_dir,
                        f"tpm_L={L}_s={system_state}_entity={j}_close_to_origin={close_to_origin}.pdf",
                    )
                    plot_tpm_as_heatmap(entity_trans_probs, filename_to_save)


def plot_tpm_as_heatmap(P: np.array, filename_to_save: str) -> None:
    K = len(P)
    fig, ax = plt.subplots()
    ax.imshow(P, cmap="Reds")

    # # Set the tick labels to be the state names
    regimes = [f"{k+1}" for k in range(K)]
    # ax.set_xticks(np.arange(len(regimes)))
    # ax.set_yticks(np.arange(len(regimes)))
    # ax.set_xticklabels(regimes)
    # ax.set_yticklabels(regimes)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Loop over data dimensions and create text annotations
    for i in range(len(regimes)):
        for j in range(len(regimes)):
            ax.text(
                j,
                i,
                "{:.2f}".format(P[i, j]),
                ha="center",
                va="center",
                color="k",
                fontdict={"fontsize": 40, "weight": "bold"},
            )
    fig.tight_layout()
    fig.savefig(filename_to_save)

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())
