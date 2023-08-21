from dynagroup.io import ensure_dir
from dynagroup.model2a.supra.eda.show_squad_headings import (
    polar_plot_the_squad_headings,
)
from dynagroup.model2a.supra.get_data import (
    get_df,
    make_data_snippet,
    make_time_snippet_based_on_desired_elapsed_secs,
)


###
# Configs
###

save_dir = "results/supra/analyses/TRY_supra_2023_05_15/"

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


for i in range(9):
    print(f"Showing the {i+1}-st minute")
    time_snippet = make_time_snippet_based_on_desired_elapsed_secs(
        df,
        elapsed_secs_after_contact_start_for_starting=60 * i,
        elapsed_secs_after_start_for_snipping=60,
        timestep_every=20,
    )

    snip = make_data_snippet(df, time_snippet)
    polar_plot_the_squad_headings(snip.squad_angles, snip.clock_times, save_dir=None)
