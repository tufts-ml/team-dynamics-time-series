### Reference: https://danvatterott.com/blog/2016/06/16/creating-videos-of-nba-action-with-sportsvu-data/
### Reference: https://github.com/gmf05/nba/blob/master/scripts/notebooks/svmovie.ipynb

from functools import partial

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from dynagroup.model2a.basketball.data.baller2vec_format import Event


###
# CONSTANTS
###
# Court dimensions
X_MIN = 0
X_MAX = 100
Y_MIN = 0
Y_MAX = 50


# Draw court.
# SIMPLE VERSION, IMAGE BASED
def draw_court(axis):
    # fig = plt.figure(figsize=(15,7.5))
    img = mpimg.imread("image/nba_court_T.png")
    plt.imshow(img, extent=axis, zorder=0)


def init(ax, info_text, player_text, player_circ, ball_circ, play_description: str):
    # Draw court & zoom out slightly to give light buffer
    draw_court([X_MIN, X_MAX, Y_MIN, Y_MAX])
    for i in range(3):
        info_text[i].set_text("")
    for i in range(10):
        player_text[i].set_text("")
        ax.add_patch(player_circ[i])
    ax.add_patch(ball_circ)
    ax.axis("off")
    dx = 5
    plt.xlim([X_MIN - dx, X_MAX + dx])
    plt.ylim([Y_MIN - dx, Y_MAX + dx])
    plt.title(play_description)
    return tuple(info_text) + tuple(player_text) + tuple(player_circ) + (ball_circ,)


# Animation function / loop
def animate(n, event, info_text, player_text, player_circ, ball_circ):
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
    info_text[0].set_text(f"Period: {event.moments[n].period}")
    info_text[1].set_text(
        f"Elapsed secs in period: {event.moments[n].period_time_elapsed_secs:.02f}"
    )
    info_text[2].set_text(f"Shot clock: {event.moments[n].shot_clock:.02f}")

    plt.pause(0.2)  # Uncomment to watch movie as it's being made
    return tuple(info_text) + tuple(player_text) + tuple(player_circ) + (ball_circ,)


def animate_event(event: Event):
    """
    Animates an event, where an event has type Event (baller2vec format)
    """

    # animation
    plt.close("all")
    fig = plt.figure(figsize=(15, 7.5))
    ax = plt.gca()

    # Animated elements
    info_text = [None] * 3
    info_text[0] = ax.text(0, -5, "")
    info_text[1] = ax.text(10, -5, "")
    info_text[2] = ax.text(40, -5, "")
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
    )
    animate_partial = partial(
        animate,
        event=event,
        info_text=info_text,
        player_text=player_text,
        player_circ=player_circ,
        ball_circ=ball_circ,
    )
    ani = animation.FuncAnimation(  # noqa
        fig,
        animate_partial,
        frames=n_frames,
        init_func=init_partial,
        blit=True,
        interval=1,
        repeat=False,
    )
    plt.show()
