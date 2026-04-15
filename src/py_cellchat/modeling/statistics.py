from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _as_float_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def geometric_mean(values: np.ndarray) -> np.ndarray | float:
    array = _as_float_array(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        if array.ndim == 1:
            return float(np.exp(np.mean(np.log(array))))
        return np.exp(np.mean(np.log(array), axis=0))


def tri_mean(values: np.ndarray) -> np.ndarray | float:
    array = _as_float_array(values)
    if array.ndim == 1:
        quartiles = np.quantile(array, [0.25, 0.5, 0.5, 0.75], method="linear")
        return float(np.mean(quartiles))
    quartiles = np.quantile(array, [0.25, 0.5, 0.5, 0.75], axis=0, method="linear")
    return np.mean(quartiles, axis=0)


def thresholded_mean(values: np.ndarray, trim: float = 0.1) -> np.ndarray | float:
    array = _as_float_array(values)
    if array.ndim == 1:
        percent_expressing = float(np.count_nonzero(array) / array.size)
        if percent_expressing < trim:
            return 0.0
        return float(np.mean(array))

    percent_expressing = np.count_nonzero(array, axis=0) / array.shape[0]
    means = np.mean(array, axis=0)
    return np.where(percent_expressing < trim, 0.0, means)


def truncated_mean(values: np.ndarray, trim: float = 0.1) -> np.ndarray | float:
    array = np.sort(_as_float_array(values), axis=0)
    if array.ndim == 1:
        n_obs = array.size
        n_trim = int(np.floor(trim * n_obs))
        if n_trim == 0:
            return float(np.mean(array))
        trimmed = array[n_trim : n_obs - n_trim]
        if trimmed.size == 0:
            return float(np.mean(array))
        return float(np.mean(trimmed))

    n_obs = array.shape[0]
    n_trim = int(np.floor(trim * n_obs))
    if n_trim == 0:
        return np.mean(array, axis=0)
    trimmed = array[n_trim : n_obs - n_trim, :]
    if trimmed.shape[0] == 0:
        return np.mean(array, axis=0)
    return np.mean(trimmed, axis=0)


def build_group_average(type_mean: str, trim: float = 0.1) -> Callable[[np.ndarray], np.ndarray | float]:
    if type_mean == "triMean":
        return tri_mean
    if type_mean == "truncatedMean":
        return lambda values: truncated_mean(values, trim=trim)
    if type_mean == "thresholdedMean":
        return lambda values: thresholded_mean(values, trim=trim)
    if type_mean == "median":
        return lambda values: np.median(_as_float_array(values), axis=0)
    raise ValueError(
        "type must be one of {'triMean', 'truncatedMean', 'thresholdedMean', 'median'}"
    )
