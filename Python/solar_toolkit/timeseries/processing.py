"""Numerical processing for one-dimensional time series.

English: Apply a centered moving average and calculate shape-stable finite
differences.

中文：计算居中移动平均和平保持数组形状的有限差分。
"""

from __future__ import annotations

import operator

import numpy as np


def smooth_series(values, *, window_length: int = 5) -> np.ndarray:
    """Return an equally long centered average with zero-padded edges.

    ``window_length`` must be a positive integer. Even windows retain the
    historical adjustment to the next odd size; NaNs propagate through every
    neighborhood containing them.
    """

    try:
        window = operator.index(window_length)
    except TypeError as exc:
        raise ValueError("window_length must be a positive integer") from exc
    if isinstance(window_length, (bool, np.bool_)) or window <= 0:
        raise ValueError("window_length must be a positive integer")
    if window % 2 == 0:
        window += 1
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional series")
    if array.size == 0:
        return np.asarray([], dtype=float)
    kernel = np.ones(window, dtype=float) / window
    start = window // 2
    return np.convolve(array, kernel, mode="full")[start : start + array.size]


def derivative_series(values, *, spacing_seconds: float = 1.0) -> np.ndarray:
    """Return a one-dimensional derivative with finite positive scalar spacing.

    Empty input returns an empty array; a finite singleton returns zero because
    no finite difference is available. A non-finite singleton remains missing
    (NaN), and NaNs in longer series propagate through the finite differences.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional series")
    try:
        spacing_value = np.asarray(spacing_seconds)
        if spacing_value.ndim != 0 or isinstance(
            spacing_value.item(), (bool, np.bool_)
        ):
            raise ValueError("non-numeric or non-scalar spacing")
        spacing = float(spacing_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("spacing_seconds must be a finite positive scalar") from exc
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing_seconds must be a finite positive scalar")
    if array.size == 0:
        return np.asarray([], dtype=float)
    if array.size == 1:
        return np.where(np.isfinite(array), 0.0, np.nan)
    return np.gradient(array, spacing)


__all__ = ["derivative_series", "smooth_series"]
