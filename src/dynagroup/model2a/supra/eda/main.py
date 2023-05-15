from dynagroup.io import ensure_dir
from dynagroup.model2a.supra.eda.show_squad_headings import (
    polar_plot_the_squad_headings,
)
from dynagroup.model2a.supra.get_data import (
    CONTACT_ALL,
    CONTACT_START,
    get_df,
    make_data_snippet,
)


### configs
save_dir = "/Users/mwojno01/Desktop/TRY_supra_2023_05_15/"

### upfront stuff
if save_dir:
    ensure_dir(save_dir)

### Get data (with clip of interest)
if not "df" in globals():
    df = get_df()

snip = make_data_snippet(df, CONTACT_START)
# snip=make_data_snippet(df, CONTACT_ALL)
polar_plot_the_squad_headings(snip.squad_angles, snip.clock_times, save_dir=None)
