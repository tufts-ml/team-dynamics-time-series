from dynagroup.eda.show_trajectory_slices import plot_trajectory_slices
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.data.orig_format import (
    get_data_in_original_formatting,
)


###
# Configs
###

# Directories
save_dir = "/Users/mwojno01/Desktop/explore_basketball/"


###
# Get data
###
if not "DATA" in globals():
    DATA = get_data_in_original_formatting()


###
# I/O
###
ensure_dir(save_dir)

###
# Plot trajectory slices
###

for event_idx in range(10):
    input(f"Event idx is now {event_idx}")

    pct_event_to_skip = 0.0

    event_start = DATA.event_boundaries[event_idx] + 1
    event_end = DATA.event_boundaries[event_idx + 1]
    event_duration = event_end - event_start

    T_start = int(event_start + pct_event_to_skip * (event_duration))
    T_slice_max = event_end - T_start

    plot_trajectory_slices(
        DATA.positions,
        T_start,
        T_slice_max,
        x_lim=(0, 2),
        y_lim=(0, 1),
        save_dir=None,
        show_plot=True,
        figsize=(8, 4),
    )
