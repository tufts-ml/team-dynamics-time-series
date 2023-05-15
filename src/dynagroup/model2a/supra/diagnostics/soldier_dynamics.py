import numpy as np
import pandas as pd

from dynagroup.model2a.circle.directions import (
    LABELS_OF_DIRECTIONS,
    RADIANS_OF_DIRECTIONS,
)
from dynagroup.model2a.supra.diagnostics.soldier_segmentations import COLOR_NAMES
from dynagroup.params import AllParameters_JAX


def report_on_directional_attractors(params: AllParameters_JAX):
    K = np.shape(params.CSP.ar_coefs)[-1]

    dests = params.CSP.drifts / (1 - params.CSP.ar_coefs)
    df_directional_attractors = pd.DataFrame(dests, columns=COLOR_NAMES[:K])

    print("Directional Attractors by Entity and Regime")
    print(df_directional_attractors)
    # print(f"\ndegrees: {DEGREES_OF_DIRECTIONS}")
    print(f"\nradians: {RADIANS_OF_DIRECTIONS}")
    print(f"labels: {LABELS_OF_DIRECTIONS}")
