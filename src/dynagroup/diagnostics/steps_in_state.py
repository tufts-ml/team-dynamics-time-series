from typing import Optional

import matplotlib.pyplot as plt

from dynagroup.types import NumpyArray2D


def plot_steps_assigned_to_state(
    outcomes_jk: NumpyArray2D,
    predictors_jk: NumpyArray2D,
    j: int,
    k: int,
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Relative to the "kmeans" diagnostics plots:
        PRO: These show not only the discrete derivative, but also the location of origin.
        CON: There is a separate plot for each state.

    Arguments:
        outcomes_jk:  The outcomes for the j-th entity that were assigned to the k-th state.
        predictors_jk: The observations immediately before the outcomes.
        j: entity index, used for title
        k: state index, used for title
    """
    # Create a figure and axis
    plt.close("all")
    fig, ax = plt.subplots()

    # Add arrows between pairs of points
    for i in range(len(outcomes_jk)):
        ax.annotate(
            "",
            xytext=predictors_jk[i],
            xy=outcomes_jk[i],
            arrowprops=dict(arrowstyle="->", color="r"),
        )

    plt.title(f"Steps for entity {j} in state {k}.")
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(
            save_dir + f"{basename_prefix}_steps_assigned_to_state_for_entity_{j}_state_{k}.pdf"
        )
