import os

import numpy as np
from matplotlib import pyplot as plt

from dynagroup.io import ensure_dir


### Configs
load_dir = "/Users/mwojno01/Repos/dynagroupresults/DSARF_Results_Preetish_2/forecast_npy_files/"
save_dir = "/Users/mwojno01/Repos/dynagroupresults/DSARF_Results_Preetish_2/forecast_plots_by_mike/"

### Setup
ensure_dir(save_dir)

### Find all files to plot
# traverse the directory tree recursively using os.walk()
for dirpath, dirnames, filenames in os.walk(load_dir):
    # print the names of all files in the current directory
    for filename in filenames:
        # skip invisible files (files starting with a dot)
        if filename.startswith("."):
            continue

        load_path = os.path.join(dirpath, filename)
        print(load_path)
        xs_predicted = np.load(load_path)

        Y_LIM = (-2.5, 2.5)

        fig1 = plt.figure(figsize=(4, 6))
        im = plt.scatter(
            xs_predicted[:, 0],
            xs_predicted[:, 1],
            c=[i for i in range(len(xs_predicted))],
            cmap="cool",
            alpha=1.0,
        )
        if Y_LIM:
            plt.ylim(Y_LIM)
        plt.show()

        # plot_type=dirpath.split("/")[-1]
        save_path = os.path.join(save_dir, filename.split(".")[0] + ".pdf")
        fig1.savefig(save_path)

        # An attempt to avoid inadventently retaining figures which consume too much memory.
        # References:
        # 1) https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
        # 2) https://stackoverflow.com/questions/16334588/create-a-figure-that-is-reference-counted/16337909#16337909
        plt.close(plt.gcf())
