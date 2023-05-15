from dynagroup.io import ensure_dir
from dynagroup.model2a.supra.eda.show_squad_headings import (
    polar_plot_the_squad_headings,
)
from dynagroup.model2a.supra.get_data import (
    get_df,
    make_data_snippet,
    make_time_snippet_from_contact_start_with_desired_elapsed_secs,
)


###
# Configs
###

save_dir = "/Users/mwojno01/Desktop/TRY_supra_2023_05_15/"

###
# Procedural Stuff
###

if save_dir:
    ensure_dir(save_dir)

###
# Get Data Snippet
###

if not "df" in globals():
    df = get_df()


time_snippet = make_time_snippet_from_contact_start_with_desired_elapsed_secs(
    df, timestep_every=20, elapsed_secs_desired=60.0
)
snip = make_data_snippet(df, time_snippet)
polar_plot_the_squad_headings(snip.squad_angles, snip.clock_times, save_dir=None)
