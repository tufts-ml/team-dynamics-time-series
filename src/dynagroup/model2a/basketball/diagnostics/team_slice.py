from typing import Optional, Tuple

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from dynagroup.types import JaxNumpyArray3D, NumpyArray1D


def plot_TOR_team_slice(
    continuous_states: JaxNumpyArray3D,
    t_0: int,
    T_slice_max: int,
    s_hats: Optional[NumpyArray1D] = None,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    save_dir: Optional[str] = None,
    show_plot: Optional[bool] = False,
    figsize: Optional[Tuple[int]] = (8, 4),
) -> None:
    """
    Arguments:
        continuous_states_for_entity: jnp.array with shape (T, D)
        T_slice_max: The attempted number of timesteps in the slice that we work with for fitting and forecasting.
            The actual T_slice could be less if there are not enough timesteps remaining in the sample.

        s_hats: If not None, we plot the numbers of the system regimes above the curve.

    Remarks:
        The player names and jersey numbers are:
            [('Luis Scola', '4'),
            ('DeMarre Carroll', '5'),
            ('Kyle Lowry', '7'),
            ('DeMar DeRozan', '10'),
            ('Jonas Valanciunas', '17')]

    """

    PLAYER_NAMES = [
        "Luis Scola",
        "DeMarre Carroll",
        "Kyle Lowry",
        "DeMar DeRozan",
        "Jonas Valanciunas",
    ]
    # PLAYER_LAST_NAMES = ["Scola", "Carroll", "Lowry", "DeRozan", "Valanciunas"]
    # JERSEY_NUMBERS = ["4", "5", "7", "10", "17"]

    # Define a list of colormaps
    colormap_names = ["Blues", "Greens", "Greys", "Purples", "Oranges"]

    T, J, _ = np.shape(continuous_states)
    t_end = np.min((t_0 + T_slice_max, T))

    print(f"Plotting the truth (clip) for team.")
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    for j in range(J):
        xs = continuous_states[t_0:t_end, j, 0]
        ys = continuous_states[t_0:t_end, j, 1]
        ax.scatter(
            xs,
            ys,
            c=[i for i in range(0, t_end - t_0)],
            cmap=colormap_names[j],
            alpha=1.0,
        )

        if s_hats is not None:
            # optionally plot entity regime on it.
            t_every = 10
            ss = s_hats[t_0:t_end]
            ss_sliced = ss[::t_every]
            xs_sliced = xs[::t_every]
            ys_sliced = ys[::t_every]

            for x, y, s in zip(xs_sliced, ys_sliced, ss_sliced):
                pl.text(x, y, str(s), color=cm.get_cmap(colormap_names[j])(0.75), fontsize=12)

    if y_lim:
        plt.ylim(y_lim)
    if x_lim:
        plt.xlim(x_lim)

    # Turn off x-axis and y-axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Turn off ticks on the x-axis and y-axis
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    # make legend after the fact
    # Create custom legend elements
    patches = [None] * J
    for j in range(J):
        patches[j] = mpatches.Patch(color=cm.get_cmap(colormap_names[j])(0.75), linestyle="-", label=PLAYER_NAMES[j])

    # Display the legend
    plt.legend(handles=patches, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if save_dir is not None:
        fig.savefig(save_dir + f"team_slice.pdf")

    if show_plot:
        plt.show()

    # An attempt to avoid inadventently retaining figures which consume too much memory.
    # References:
    # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
    plt.close(plt.gcf())
