# import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dynagroup.model2a.basketball.court import (
    X_MAX_COURT,
    X_MIN_COURT,
    Y_MAX_COURT,
    Y_MIN_COURT,
)
from dynagroup.model2a.basketball.data.baller2vec_format import Event
from dynagroup.params import ContinuousStateParameters_JAX
from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.util import compute_cartesian_product_of_two_1d_arrays


"""
Functions for animating an event (play) from a basketball game.
My starting point for this code was the references below.
I updated the code to do things like:
    - Provide general info learned by my model (e.g. the system state) as text below each fame
    - Draw vector fields for some player of interest.

References:
    https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/
    https://github.com/gmf05/nba/blob/master/scripts/notebooks/svmovie.ipynb
"""

###
# CONSTANTS
###

# Vector field colors
COLOR_NAMES = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink",
]
COLORS = sns.xkcd_palette(COLOR_NAMES)

# Coordinates after normalizing the court to the unit square [0,1]x[0,1]
EPSILON = 0.05
X_MIN_NORM, X_MAX_NORM = 0.0, 1.0
Y_MIN_NORM, Y_MAX_NORM = 0.0, 1.0
N_BINS_PER_AXIS = 10
X_NORM = np.linspace(X_MIN_NORM - EPSILON, X_MAX_NORM + EPSILON, N_BINS_PER_AXIS)
Y_NORM = np.linspace(Y_MIN_NORM - EPSILON, Y_MAX_NORM + EPSILON, N_BINS_PER_AXIS)
XY_NORM = compute_cartesian_product_of_two_1d_arrays(
    X_NORM, Y_NORM
)  # shape (N_BINS_PER_AXIS**2, D=2)


###
# HELPER FUNCTIONS
####


# Draw court.
# SIMPLE VERSION, IMAGE BASED
def draw_court(axis):
    # fig = plt.figure(figsize=(15,7.5))
    img = mpimg.imread("image/nba_court_T.png")
    plt.imshow(img, extent=axis, zorder=0)


def unnorm(coords: NumpyArray2D) -> NumpyArray2D:
    """
    Take normalized basketball coordinates on the unit square [0,1] x [0,1]
    and "unnormalize" them so that we're back in the original rectangle
        [X_MIN_COURT, X_MAX_COURT] x [Y_MIN_COURT, Y_MAX_COURT] = [0,100] x [0,50]
    """
    coords_unnorm = np.zeros_like(coords)
    coords_unnorm[:, 0] = coords[:, 0] * X_MAX_COURT
    coords_unnorm[:, 1] = coords[:, 1] * Y_MAX_COURT
    return coords_unnorm


# def gridify(vector : NumpyArray1D, n_reps: int) -> NumpyArray2D:
#     """
#     Repeats each element of vector `n_reps` times and makes it a row in a matrix.
#     """
#     return np.reshape(np.repeat(vector, n_reps), (n_reps, n_reps))


# ###
# # GRID CONSTANTS
# ###


# X_NORM_GRID = gridify(X_NORM, N_BINS_PER_AXIS)
# Y_NORM_GRID = gridify(Y_NORM, N_BINS_PER_AXIS).T


###
# MAIN ANIMATOR FUNCTIONS
###


def init(ax, info_text, player_text, player_circ, ball_circ, play_description: str, quiver_handle):
    # Draw court & zoom out slightly to give light buffer
    draw_court([X_MIN_COURT, X_MAX_COURT, Y_MIN_COURT, Y_MAX_COURT])

    # Initialize info text, player text, player circ, and ball circ.
    info_text[0].set_text("")
    for i in range(10):
        player_text[i].set_text("")
        ax.add_patch(player_circ[i])
    ax.add_patch(ball_circ)

    # Initialize quiver_handle (for vector fields, if used)
    quiver_handle = ax.quiver([], [])

    # Setup axis basis
    ax.axis("off")
    dx = 5
    plt.xlim([X_MIN_COURT - dx, X_MAX_COURT + dx])
    plt.ylim([Y_MIN_COURT - dx, Y_MAX_COURT + dx])
    plt.title(play_description)

    # return value tells the animator which objects on the plot to update after each frame
    return tuple(info_text) + tuple(player_text) + tuple(player_circ) + (ball_circ, quiver_handle)


# Animation function / loop
def update(
    n,
    ax,
    event,
    info_text,
    player_text,
    player_circ,
    ball_circ,
    vector_field_dict: Optional[Dict],
    model_dict: Optional[Dict[str, NumpyArray1D]],
    quiver_handle,
):
    ### 1. Draw players by team, with jersey numbers
    for i in range(10):
        player_circ[i].center = (event.moments[n].player_xs[i], event.moments[n].player_ys[i])
        player_text[i].set_text(
            str(event.moments[n].player_ids[i])
        )  # todo: update with jersey number
        player_text[i].set_x(event.moments[n].player_xs[i])
        player_text[i].set_y(event.moments[n].player_ys[i])

    ### 2. Draw ball
    ball_circ.center = (event.moments[n].ball_x, event.moments[n].ball_y)
    # Fluctuate ball radius to indicate Z position : helpful for shots
    ball_circ.radius = 1 + event.moments[n].ball_z / 17 * 2

    ### 3. Print game clock info
    info_str = (
        f"Period: {event.moments[n].period} "
        f"Elapsed secs in period: {event.moments[n].period_time_elapsed_secs:.02f} "
        f"Shot clock: {event.moments[n].shot_clock:.02f} "
    )

    ### 3. Make vector field
    if vector_field_dict is not None:
        ax.quiver([], [])
        k_hat = int(vector_field_dict["k_j_sequence"][n])
        A_j = vector_field_dict["A_j"][k_hat]
        b_j = vector_field_dict["b_j"][k_hat]
        dxydt_norm = XY_NORM.dot(A_j.T) + b_j - XY_NORM
        xy = unnorm(XY_NORM)
        dxydt = unnorm(dxydt_norm)
        quiver_handle = ax.quiver(
            xy[:, 0], xy[:, 1], dxydt[:, 0], dxydt[:, 1], color=COLORS[k_hat % len(COLORS)]
        )

    ### 4. Print game clock info
    info_str = (
        f"Period: {event.moments[n].period} "
        f"Elapsed secs in period: {event.moments[n].period_time_elapsed_secs:.02f} "
        f"Shot clock: {event.moments[n].shot_clock:.02f} "
        f"n: {n}."
    )

    if model_dict is not None:
        for i, (k, v) in enumerate(model_dict.items()):
            # TODO: make this dict ordered
            info_str += f" {k}:{v[n]}"
    info_text[0].set_text(info_str)

    plt.pause(0.04)  # Uncomment to watch movie as it's being made
    return tuple(info_text) + tuple(player_text) + tuple(player_circ) + (ball_circ, quiver_handle)


def animate_event(
    event: Event,
    save_dir: str,
    filename_postfix: Optional[str] = "",
    model_dict: Optional[Dict[str, NumpyArray1D]] = None,
    vector_field_dict: Optional[Dict] = None,
):
    """
    Animates an event (play) from a basketball game, where an event has type Event (baller2vec format).
    My starting point for this function is the code given in the references.
    However, I have updated the code to do things like:
        - Provide general textual info learned by my model (e.g. the most likely system state) below each fame
        - Draw vector fields for some player of interest.

    Arguments:
        model_dict: Optionally append info at the bottom of the animation about what the model
            learned at each timestep.  Each NumpyArray1D value should have the same length as the number
            of moments in the flattened_events.

        vector_field_dict: Provides information to optionally draw learned vector fields for some player of interest.
            The dict must look like this: {"k_j_sequence": k_hats_for_focal_entity, "A_j": A_j, "b_j": b_j}

    References:
        https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/
        https://github.com/gmf05/nba/blob/master/scripts/notebooks/svmovie.ipynb

    """

    # TODO:  Each NumpyArray1D value should have the same length as the number
    #        of moments in the flattened_events.

    if model_dict is not None:
        for v in model_dict.values():
            if len(v) != len(event.moments):
                raise ValueError(
                    f" Each NumpyArray1D value in the model_dict should have the same length as the number "
                    f" of moments in the event!"
                )

    # animation
    plt.close("all")
    fig = plt.figure(figsize=(15, 7.5))
    ax = plt.gca()

    # Animated elements
    info_text = [ax.text(0, -5, "")]
    player_text = [None] * 10
    player_circ = [None] * 10
    R = 2.2
    for i in range(10):
        player_text[i] = ax.text(0, 0, "", color="w", ha="center", va="center")
        if event.moments[0].player_hoop_sides[i] == 0:
            col = "b"
        else:
            col = "r"
        player_circ[i] = plt.Circle((0, 0), R, color=col)
    ball_circ = plt.Circle((0, 0), R, color=[1, 0.4, 0])

    # vector fields
    quiver_handle = ax.quiver([], [])

    # TODO: Add the `play_description`.  This would tell us if the event was a rebound, etc.
    # It should be obtainable from the 'y' file created by the baller2vec repo.
    play_description = f"Event: {event.idx}. Label: {event.label}"
    n_frames = len(event.moments)

    # Create partial functions default arguments
    init_partial = partial(
        init,
        ax=ax,
        info_text=info_text,
        player_text=player_text,
        player_circ=player_circ,
        ball_circ=ball_circ,
        play_description=play_description,
        quiver_handle=quiver_handle,
    )
    update_partial = partial(
        update,
        ax=ax,
        event=event,
        info_text=info_text,
        player_text=player_text,
        player_circ=player_circ,
        ball_circ=ball_circ,
        model_dict=model_dict,
        vector_field_dict=vector_field_dict,
        quiver_handle=quiver_handle,
    )
    ani = animation.FuncAnimation(  # noqa
        fig,
        update_partial,
        frames=n_frames,
        init_func=init_partial,
        blit=True,
        interval=1,
        repeat=False,
    )
    plt.show()
    # basename = play_description +"_" + filename_postfix + ".mp4"
    # filepath = os.path.join(save_dir, basename)
    # ani.save(filepath, dpi=100,fps=5)
    # plt.close('all') #close the plot


def animate_events_over_vector_field_for_one_player(
    events: List[Event],
    event_start_stop_idxs: List[Tuple[int]],
    most_likely_entity_states: NumpyArray2D,
    CSP: ContinuousStateParameters_JAX,
    j_focal: int,
    save_dir: str,
    filename_postfix: Optional[str] = "",
    s_maxes: Optional[NumpyArray1D] = None,
) -> None:
    """
    Arguments:
        s_maxes: An optional array giving the most likely system state for each timestep.
            If provided, we create a model_dict object to pass to the `animate_event` function.
    """
    for event_idx, event in enumerate(events):
        print(f"Now animating event idx {event_idx}, which has type {event.label}")
        event_start_idx, event_stop_idx = (
            event_start_stop_idxs[event_idx][0],
            event_start_stop_idxs[event_idx][1],
        )
        k_hats_for_focal_entity = most_likely_entity_states[event_start_idx:event_stop_idx, j_focal]
        A_j_init, b_j_init = CSP.As[j_focal], CSP.bs[j_focal]
        vector_field_dict_for_event = {
            "k_j_sequence": k_hats_for_focal_entity,
            "A_j": A_j_init,
            "b_j": b_j_init,
        }
        if s_maxes is not None:
            model_dict_for_event = {"System state": s_maxes[event_start_idx:event_stop_idx]}
        else:
            model_dict_for_event = None
        filename_postfix += f"_focal_player_{j_focal}"
        animate_event(
            event, save_dir, filename_postfix, model_dict_for_event, vector_field_dict_for_event
        )
