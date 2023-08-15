from typing import List, Optional

import numpy as np
import seaborn as sns
import sklearn

from dynagroup.diagnostics.clusters import plot_clusters_on_2d_data
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
    binary_weights: NumpyArray2D,
    kmeans_fits_by_entity: List[sklearn.cluster._kmeans.KMeans],
    save_dir: Optional[str] = None,
    show_plot: bool = False,
    basename_prefix: str = "",
):
    """
    Inspired by: https://gist.github.com/michaelchughes/aa182035f4de1ef03f09e3f616edc53b

    Arguments:
        kmeans_fits_by_entity: List of length J, where J is the number of entities.
        binary_weights: has shape (T,J) and dtype=bool
    """
    T_data = np.shape(data)[0]
    J = np.shape(data)[1]
    num_clusters = kmeans_fits_by_entity[0].n_clusters

    cluster_labels = np.zeros((T_data, J), dtype=int)
    for j in range(J):
        cluster_labels[:, j] = kmeans_fits_by_entity[j].labels_

    return plot_clusters_on_2d_data(
        data,
        binary_weights,
        cluster_labels,
        num_clusters,
        save_dir,
        show_plot,
        basename_prefix=f"{basename_prefix}_kmeans_on_raw_data_or_derivatives",
    )
