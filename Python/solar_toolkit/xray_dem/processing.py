"""Soft X-ray and Neupert-effect numerical helpers.

English: Smooth flux arrays and calculate finite-difference derivatives with
explicit numerical modes. The default modes preserve the lightweight public
API, while the Savitzky-Golay and forward-difference modes reproduce the
historical GOES/Neupert scripts.

中文：使用显式数值模式平滑软 X 射线通量并计算有限差分导数。默认模式保持
轻量公共 API 的既有行为；Savitzky-Golay 与前向差分模式用于复现历史 GOES/
Neupert 脚本。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from astropy import units as u

from solar_toolkit._utils.numeric import as_value, numeric_data
from solar_toolkit.time import normalize_time
from solar_toolkit.timeseries import derivative_series, smooth_series

SmoothingMethod = Literal["moving-average", "savgol"]
DerivativeMethod = Literal["median-gradient", "gradient", "forward"]


def smooth_flux_data(
    flux_data,
    window_length: int = 5,
    polyorder: int | None = None,
    *,
    method: SmoothingMethod = "moving-average",
    adjust_even_window: bool = False,
    clamp_to_data: bool = False,
) -> np.ndarray:
    """Return a smoothed flux array.

    Parameters
    ----------
    method
        ``"moving-average"`` keeps the original public-package behavior.
        ``"savgol"`` uses :func:`scipy.signal.savgol_filter`, matching the
        historical GOES analysis scripts. Masked and non-finite samples make
        their smoothing neighborhoods NaN, without filling or interpolation.
    adjust_even_window
        Increase an even Savitzky-Golay window by one. This opt-in behavior
        reproduces the dedicated Neupert comparison script without changing
        the other scripts, which historically passed an even window directly.
    clamp_to_data
        Limit the window and polynomial order to the available samples. This
        reproduces the historical defensive Neupert helper.
    """

    if method == "moving-average":
        return smooth_series(flux_data, window_length=window_length)
    if method != "savgol":
        raise ValueError(f"Unsupported smoothing method: {method!r}")

    array = numeric_data(flux_data)
    if array.ndim != 1:
        raise ValueError("flux_data must be a one-dimensional array")
    window = int(window_length)
    order = 3 if polyorder is None else int(polyorder)
    if window < 1:
        raise ValueError("window_length must be positive")
    if order < 0:
        raise ValueError("polyorder must be non-negative")

    if adjust_even_window and window % 2 == 0:
        window += 1

    if clamp_to_data:
        maximum = array.size if array.size % 2 == 1 else array.size - 1
        if maximum < 1:
            raise ValueError("At least one flux sample is required")
        window = min(window, maximum)
        order = min(order, window - 1)

    if order >= window:
        raise ValueError("polyorder must be less than window_length")

    from scipy.signal import savgol_filter

    if np.isfinite(array).all():
        return np.asarray(savgol_filter(array, window, order), dtype=float)
    if window > array.size:
        raise ValueError("window_length must not exceed the number of flux samples")
    array[~np.isfinite(array)] = np.nan
    # Use the same convolution coefficients as the finite path. The default
    # interp edge fits require finite data, so fit only complete edge windows.
    result = savgol_filter(array, window, order, mode="constant", cval=np.nan)
    half = window // 2
    if half and np.isfinite(array[:window]).all():
        result[:half] = savgol_filter(array[:window], window, order)[:half]
    if half and np.isfinite(array[-window:]).all():
        result[-half:] = savgol_filter(array[-window:], window, order)[-half:]
    return np.asarray(result, dtype=float)


def _time_seconds(time_data) -> np.ndarray:
    """Return SI seconds on a complete, strictly increasing integer TAI axis."""
    values = np.asanyarray(getattr(time_data, "values", time_data))
    if values.ndim != 1:
        raise ValueError("time_data must be a one-dimensional timestamp array")
    if not values.size:
        return np.asarray([], dtype=float)
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        raise ValueError("time_data requires timestamps, not unlabelled numeric times")
    mask = getattr(time_data, "mask", False)
    if not callable(mask) and np.any(mask):
        raise ValueError("time_data must not contain missing timestamps")
    try:
        nanoseconds = [normalize_time(value).tai_ns for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("time_data must contain valid timestamps") from exc
    if not all(
        right > left for left, right in zip(nanoseconds, nanoseconds[1:], strict=False)
    ):
        raise ValueError("time_data must increase strictly without duplicates")
    origin = int(nanoseconds[0])
    # Subtract in integer arithmetic before converting to floating seconds.
    # Python integers also avoid int64 overflow for widely separated dates.
    return np.asarray([int(value) - origin for value in nanoseconds], dtype=float) / 1e9


def calculate_derivative(
    time_data_or_flux,
    flux_data=None,
    *,
    spacing_seconds: float | None = None,
    method: DerivativeMethod = "median-gradient",
) -> np.ndarray:
    """Calculate a derivative for flux data.

    Flux arrays must be one-dimensional. If only one argument is supplied, unit
    spacing is used. Explicit spacing must be finite and positive. With a time
    axis, timestamps must be valid, equally numerous and strictly increasing;
    invalid axes raise ``ValueError`` even with a spacing override. Naive times
    mean UTC. Relative time differences use continuous TAI seconds, preserving
    submicrosecond cadence and elapsed SI seconds across UTC leap seconds.
    Masks are missing samples; singleton missing flux remains NaN. Bare spacing
    is in seconds, while time Quantities are converted explicitly.
    ``"median-gradient"`` preserves the original public-package behavior,
    ``"gradient"`` evaluates :func:`numpy.gradient` at exact sample times,
    and ``"forward"`` returns the historical ``diff(flux) / diff(time)`` result.
    """

    if method not in {"median-gradient", "gradient", "forward"}:
        raise ValueError(f"Unsupported derivative method: {method!r}")
    spacing = None
    if spacing_seconds is not None:
        try:
            if np.asarray(spacing_seconds).dtype.kind == "b":
                raise ValueError("boolean spacing")
            spacing_value = as_value(spacing_seconds, u.s, "spacing_seconds")
            if spacing_value.ndim:
                raise ValueError("invalid scalar spacing")
            spacing = float(spacing_value)
        except u.UnitConversionError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("spacing_seconds must be finite and positive") from exc
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing_seconds must be finite and positive")
    values = numeric_data(time_data_or_flux if flux_data is None else flux_data)
    if values.ndim != 1:
        raise ValueError("flux_data must be a one-dimensional array")
    if flux_data is None:
        if method == "forward":
            return np.diff(values) / (1.0 if spacing is None else spacing)
        return derivative_series(
            values,
            spacing_seconds=1.0 if spacing is None else spacing,
        )

    times = _time_seconds(time_data_or_flux)
    if values.size != times.size:
        raise ValueError("time_data and flux_data must have the same length")
    if values.size < 2:
        if method == "forward":
            return np.asarray([], dtype=float)
        return derivative_series(
            values, spacing_seconds=1.0 if spacing is None else spacing
        )

    if spacing is not None:
        if method == "forward":
            return np.diff(values) / spacing
        return derivative_series(values, spacing_seconds=spacing)

    if method == "forward":
        return np.diff(values) / np.diff(times)
    if method == "gradient":
        result = np.gradient(values, times)
        result[~np.isfinite(values)] = np.nan
        return result
    spacing = float(np.median(np.diff(times)))
    return derivative_series(values, spacing_seconds=spacing)


__all__ = [
    "DerivativeMethod",
    "SmoothingMethod",
    "calculate_derivative",
    "smooth_flux_data",
]
