from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


### Define a custom color palette

SYSTEM_REGIME_COLOR_NAMES = [
    "pastel yellow",
    "baby blue",
    "light lavender",
    "pale olive",
    "coral",
    "peach",
    "dusty purple",
    "light turquoise",
    "bubblegum pink",
    "pale pink",
    "lavender",
]

SYSTEM_REGIME_COLORS = sns.xkcd_palette(SYSTEM_REGIME_COLOR_NAMES)
# Note: I'm calling these `SYSTEM_REGIME_COLORS`, and use them as such elsewhere,
# but here I'm using them for entity colors... kind of confusing..
sns.set_style("white")
sns.set_context("talk")


def polar_plot_the_squad_headings(
    squad_angles,
    clock_times,
    save_dir: Optional[str] = None,
    basename_prefix: Optional[str] = "",
    show_plot=True,
):
    J = np.shape(squad_angles)[1]

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    for j in range(J):
        ax.plot(
            squad_angles[:, j] + np.pi, clock_times, color=SYSTEM_REGIME_COLORS[j], label=f"{j}"
        )
    ax.set_yticklabels([])

    # Set the tick locations and labels
    # ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    # ax.set_xticklabels(["0 (W)", "45", "90 (S)", "135", "180 (E)", "225", "270 (N)", "315"])

    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax.set_xticklabels(["0 (E)", "45", "90 (N)", "135", "180 (W)", "225", "270 (S)", "315"])

    ## legend
    # For polar axes, it may be useful to move the legend slightly away from the
    # axes center, to avoid overlap between the legend and the axes.  The following
    # snippet places the legend's lower left corner just outside the polar axes
    # at an angle of 67.5 degrees in polar coordinates.
    angle = np.deg2rad(340)
    ax.legend(loc="lower left", bbox_to_anchor=(0.5 + np.cos(angle) / 2, 0.5 + np.sin(angle) / 2))
    # ax.legend(loc="lower left", bbox_to_anchor=(1.15, 0.5))
    if save_dir is not None:
        fig.savefig(save_dir + f"{basename_prefix}_squad_headings_on_circle.pdf")
    if show_plot:
        plt.show()
