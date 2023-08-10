from collections import Counter
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn

from dynagroup.types import NumpyArray2D, NumpyArray3D


sns.set_style("whitegrid")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink",
    "slate grey",
    "bright cyan",
    "emerald green",
    "scarlet",
    "neon purple",
    "aqua blue",
    "hot pink",
    "teal",
    "dandelion",
    "sky blue",
]

colors = sns.xkcd_palette(color_names)


def plot_kmeans_on_2d_data(
    data: NumpyArray3D,
    weights: NumpyArray2D,
    kmeans_fits_by_entity: List[sklearn.cluster._kmeans.KMeans],
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Inspired by: https://gist.github.com/michaelchughes/aa182035f4de1ef03f09e3f616edc53b

    Arguments:
        kmeans_fits_by_entity: List of length J, where J is the number of entities.
    """
    J = np.shape(data)[1]
    K = kmeans_fits_by_entity[0].n_clusters

    for j in range(J):
        plt.figure(figsize=(8, 8))
        xs = [x for (t, x) in enumerate(data[:, j, 0])]
        ys = [y for (t, y) in enumerate(data[:, j, 1])]

        labels_for_entity = kmeans_fits_by_entity[j].labels_
        for k in range(K):
            xs_k = [
                x
                for (t, x) in enumerate(xs)
                if (labels_for_entity[t] == k and weights[t, j] == True)
            ]
            ys_k = [
                y
                for (t, y) in enumerate(ys)
                if (labels_for_entity[t] == k and weights[t, j] == True)
            ]
            plt.plot(xs_k, ys_k, ".", color=colors[k % len(colors)], label=str(k))

        cluster_usage_str = f"Cluster usage: {Counter(labels_for_entity)}".replace("Counter", "")

        # Set axes to have equal size
        plt.gca().set_aspect("equal")
        plt.xlabel("Court length (normalized)")
        plt.ylabel("Court width (normalized)")
        plt.legend(loc="upper right", ncol=2)
        plt.title(f"K-means assignments for entity {j}.\n")
        plt.title(cluster_usage_str, fontdict={"fontsize": 12})
        plt.tight_layout()

        if show_plot:
            plt.show()
        if save_dir is not None:
            plt.savefig(
                save_dir
                + f"{basename_prefix}_hexbin_plot_of_discrete_derivatives_for_entity_{j}.pdf"
            )
