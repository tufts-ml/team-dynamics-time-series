from dataclasses import dataclass
from typing import Dict, List

import ruptures as rpt

from dynagroup.types import NumpyArray1D


@dataclass
class SeriesAndChangePoints:
    series: NumpyArray1D
    changepoints: List[int]


def make_changepoints_dict(
    data_dict: Dict[str, NumpyArray1D],
    changepoint_penalty: float = 10,
) -> Dict[str, SeriesAndChangePoints]:
    """
    Given a data dict of (named) time series, we compute changepoints and return
    a changepoints dict of (named) time series with changepoint indices.
    """

    # TODO: What changepoint alg is being used? Is there some better alg?
    # TODO: Is there some way to require (or encourage) a certain number of timesteps
    # passing before identifying a changepoint?
    # TODO: What's a good way to set the penalty parameter?

    changepoints_dict = {}
    for data_name, data in data_dict.items():
        algo = rpt.Pelt(model="rbf").fit(data)
        changepoints = algo.predict(pen=changepoint_penalty)
        changepoints_dict[data_name] = SeriesAndChangePoints(data, changepoints)
    return changepoints_dict
