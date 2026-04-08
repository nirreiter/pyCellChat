from __future__ import annotations

from collections.abc import Callable

import numpy as np


def geometric_mean(values: np.ndarray) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if array.ndim == 1:
            return float(np.exp(np.mean(np.log(array))))
        return np.exp(np.mean(np.log(array), axis=0))


def tri_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    quartiles = np.quantile(array, [0.25, 0.5, 0.5, 0.75], method="linear")
    return float(np.mean(quartiles))


def thresholded_mean(values: np.ndarray, trim: float = 0.1) -> float:
    array = np.asarray(values, dtype=float)
    percent_expressing = float(np.count_nonzero(array) / array.size)
    if percent_expressing < trim:
        return 0.0
    return float(np.mean(array))


def truncated_mean(values: np.ndarray, trim: float = 0.1) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    n_obs = array.size
    n_trim = int(np.floor(trim * n_obs))
    if n_trim == 0:
        return float(np.mean(array))
    trimmed = array[n_trim : n_obs - n_trim]
    if trimmed.size == 0:
        return float(np.mean(array))
    return float(np.mean(trimmed))


def build_group_average(type_mean: str, trim: float = 0.1) -> Callable[[np.ndarray], float]:
    if type_mean == "triMean":
        return tri_mean
    if type_mean == "truncatedMean":
        return lambda values: truncated_mean(values, trim=trim)
    if type_mean == "thresholdedMean":
        return lambda values: thresholded_mean(values, trim=trim)
    if type_mean == "median":
        return lambda values: float(np.median(np.asarray(values, dtype=float)))
    raise ValueError(
        "type must be one of {'triMean', 'truncatedMean', 'thresholdedMean', 'median'}"
    )
