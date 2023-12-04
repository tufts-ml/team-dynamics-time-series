import matplotlib.pyplot as plt
import numpy as np


###
# Computation helpers
###


def compute_average_starting_location(qS, xs):
    """
    Starting location is where players were located for the timestep
    BEFORE the system change (as the system change then instigates the
    forward movement; see the HSRDM probabilistic graphical model.)
    """
    weighted_starts = np.einsum("tl, tjd -> tjld", qS[1:], xs[:-1])
    prob_mass = np.sum(qS[1:], axis=0)
    return np.sum(weighted_starts, axis=0) / prob_mass[None, :, None]  # (J,L,D)


def compute_average_movement_vectors(qS, xs):
    vs = xs[1:] - xs[:-1]
    weighted_movements = np.einsum("tl, tjd -> tjld", qS[:-1], vs)
    prob_mass = np.sum(qS[:-1], axis=0)
    return np.sum(weighted_movements, axis=0) / prob_mass[None, :, None]  # (J,L,D)


####
# Plotting helpers
###


def _plot_object(ax, start, vector, label):
    # Function to plot starting location and movement vector
    ax.quiver(start[0], start[1], vector[0], vector[1], angles="xy", scale_units="xy", scale=1, label=label)


def plot_system_state(avg_starts, avg_vectors, ell):
    # Create a figure and axis
    fig, ax = plt.subplots()

    J, L, D = np.shape(avg_starts)
    for j in range(J):
        _plot_object(ax, avg_starts[j, ell], avg_vectors[j, ell], label=f"Object {j}")

    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Set labels and legend
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()

    # Show the plot
    plt.title("Objects and Movement Vectors")
    plt.grid(True)
    plt.show()


####
# Main function
###


def plot_system_states(VES_summary, xs_train):
    qS = np.array(VES_summary.expected_regimes)  # TxL
    xs_train = np.asarray(xs_train)  # TxJxD

    avg_starts = compute_average_starting_location(qS, xs_train)
    avg_vectors = compute_average_movement_vectors(qS, xs_train)

    num_system_states = np.shape(qS)[1]
    for ell in range(num_system_states):
        plot_system_state(avg_starts, avg_vectors, ell=ell)
