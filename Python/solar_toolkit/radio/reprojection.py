"""Pure array reprojection and time-matching helpers for radio data.

Coordinate-system concerns stay outside this module: callers provide a
``pixel_mapper`` that converts radio pixels into target-image pixels.  This
keeps the numerical interpolation reusable without importing an overlay
workflow, plotting code, or a concrete AIA application.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..time import normalize_time, seconds_to_ns

__all__ = [
    "RadioReprojectionResult",
    "interpolate_scattered_to_grid",
    "nearest_time_index",
    "reproject_radio_array",
]

PixelMapper = Callable[[float, float], tuple[float, float]]


@dataclass(frozen=True)
class RadioReprojectionResult:
    """Numerical result of mapping a radio array onto a target pixel grid."""

    data: np.ndarray
    peak_pixel: tuple[float, float]
    amplitude: float
    mapped_sample_count: int


def interpolate_scattered_to_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    target_shape: tuple[int, int],
    *,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate scattered ``(x, y)`` samples onto a target image grid.

    Linear and cubic interpolation automatically fall back to nearest-neighbor
    interpolation when the input geometry is insufficient or degenerate.
    """

    height, width = _target_shape(target_shape)
    method = str(method).lower()
    if method not in {"linear", "nearest", "cubic"}:
        raise ValueError("method must be 'linear', 'nearest', or 'cubic'")

    points = np.asarray(points_xy, dtype=np.float64)
    sample_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (sample_count, 2)")
    if points.shape[0] != sample_values.size:
        raise ValueError("points_xy and values must contain the same sample count")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(sample_values)
    points = points[finite]
    sample_values = sample_values[finite]
    if sample_values.size == 0:
        return np.full((height, width), np.nan, dtype=np.float32)

    from scipy.interpolate import griddata

    grid_y, grid_x = np.mgrid[0:height, 0:width]
    selected_method = method
    if selected_method in {"linear", "cubic"} and sample_values.size < 3:
        selected_method = "nearest"
    try:
        result = griddata(
            points,
            sample_values,
            (grid_x, grid_y),
            method=selected_method,
            fill_value=np.nan,
        )
    except Exception:
        result = griddata(
            points,
            sample_values,
            (grid_x, grid_y),
            method="nearest",
            fill_value=np.nan,
        )
    if selected_method != "nearest" and np.all(~np.isfinite(result)):
        result = griddata(
            points,
            sample_values,
            (grid_x, grid_y),
            method="nearest",
            fill_value=np.nan,
        )
    return np.asarray(result, dtype=np.float32)


def reproject_radio_array(
    radio_data: np.ndarray,
    target_shape: tuple[int, int],
    pixel_mapper: PixelMapper,
    *,
    method: str = "linear",
) -> RadioReprojectionResult | None:
    """Map finite radio samples to a target pixel grid and interpolate them."""

    source = np.asarray(radio_data, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("radio_data must be a two-dimensional array")
    _target_shape(target_shape)
    if not callable(pixel_mapper):
        raise TypeError("pixel_mapper must be callable")

    source_y, source_x = np.nonzero(np.isfinite(source))
    points: list[tuple[float, float]] = []
    values: list[float] = []
    for y_index, x_index in zip(source_y, source_x, strict=True):
        target_x, target_y = pixel_mapper(float(x_index), float(y_index))
        if np.isfinite(target_x) and np.isfinite(target_y):
            points.append((float(target_x), float(target_y)))
            values.append(float(source[y_index, x_index]))
    if not points:
        return None

    data = interpolate_scattered_to_grid(
        np.asarray(points, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        target_shape,
        method=method,
    )
    if np.all(~np.isfinite(data)):
        return None
    peak_y, peak_x = np.unravel_index(np.nanargmax(data), data.shape)
    return RadioReprojectionResult(
        data=data,
        peak_pixel=(float(peak_x), float(peak_y)),
        amplitude=float(np.nanmax(data)),
        mapped_sample_count=len(points),
    )


def nearest_time_index(
    target_time: datetime,
    candidate_times: Sequence[datetime],
    *,
    max_delta_seconds: float | None = None,
) -> int | None:
    """Return the first nearest candidate index, optionally within seconds.

    Naive datetimes use UTC; aware datetimes and explicitly scaled Astropy
    times are compared as continuous integer TAI nanoseconds. Empty candidates
    or an exceeded inclusive tolerance return ``None``. Invalid timestamps or
    a negative/nonfinite tolerance raise ``ValueError``. Inputs are unchanged.
    """

    if max_delta_seconds is not None:
        max_delta_seconds = float(max_delta_seconds)
        if not np.isfinite(max_delta_seconds) or max_delta_seconds < 0:
            raise ValueError("max_delta_seconds must be finite and non-negative")
    if len(candidate_times) == 0:
        return None

    target_ns = normalize_time(target_time).tai_ns
    deltas = [
        abs(normalize_time(candidate).tai_ns - target_ns)
        for candidate in candidate_times
    ]
    index = min(range(len(deltas)), key=deltas.__getitem__)
    if max_delta_seconds is not None and deltas[index] > seconds_to_ns(
        max_delta_seconds
    ):
        return None
    return index


def _target_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("target_shape must contain (height, width)")
    height, width = shape
    if isinstance(height, bool) or isinstance(width, bool):
        raise TypeError("target dimensions must be integers")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise TypeError("target dimensions must be integers")
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("target dimensions must be positive")
    return height, width
