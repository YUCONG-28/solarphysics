"""Pure numerical helpers for CSO spectrogram processing.

The functions in this module operate only on in-memory arrays.  They do not
read files, discover configuration, plot figures, or launch an application.
"""

from __future__ import annotations

import numpy as np

from solar_toolkit._utils.numeric import as_value, numeric_data

__all__ = [
    "block_mean_rebin",
    "calc_polarization_ratio",
    "finite_color_limits",
    "safe_log10",
]


def block_mean_rebin(
    data: np.ndarray,
    *,
    frequency_bin: int = 1,
    time_bin: int = 1,
    dtype=np.float64,
) -> np.ndarray:
    """Downsample a frequency-by-time array using non-overlapping block means.

    Samples at the high-index edge that do not fill a complete block are
    trimmed, matching the existing CSO workflow's rebinning behavior. Masked
    samples are NaN and propagate through their block. Means are accumulated in
    float64; a lower-precision floating output requires an explicit ``dtype``.
    """

    array = numeric_data(data)
    if array.ndim != 2:
        raise ValueError("data must be a two-dimensional frequency-by-time array")
    frequency_bin = _positive_integer("frequency_bin", frequency_bin)
    time_bin = _positive_integer("time_bin", time_bin)

    frequency_count = (array.shape[0] // frequency_bin) * frequency_bin
    time_count = (array.shape[1] // time_bin) * time_bin
    if frequency_count == 0 or time_count == 0:
        raise ValueError("data is too short for the requested bin sizes")

    output_dtype = np.dtype(dtype)
    if not np.issubdtype(output_dtype, np.floating):
        raise TypeError("dtype must be a floating-point dtype")
    trimmed = array[:frequency_count, :time_count]
    result = trimmed.reshape(
        frequency_count // frequency_bin,
        frequency_bin,
        time_count // time_bin,
        time_bin,
    ).mean(axis=(1, 3), dtype=np.float64)
    return result.astype(output_dtype, copy=False)


def calc_polarization_ratio(
    right: np.ndarray,
    left: np.ndarray,
    *,
    eps: float = 1e-30,
) -> np.ndarray:
    """Return the bounded circular-polarization ratio ``(R-L)/(R+L)``.

    Non-finite inputs and denominators whose absolute value is at most ``eps``
    produce ``NaN`` rather than an infinite or unstable ratio. Masks are missing
    samples. Arithmetic uses float64 to avoid integer overflow; the established
    float32 output is retained. If a side supplies a unit, both sides are
    converted to that unit (preferring the right side's unit); bare values are
    interpreted in the same unit. Incompatible dimensions are rejected.
    """

    unit = getattr(right, "unit", None)
    if unit is None:
        unit = getattr(left, "unit", None)
    if unit is None:
        right_array = numeric_data(right)
        left_array = numeric_data(left)
    else:
        right_array = as_value(right, unit, "right")
        left_array = as_value(left, unit, "left")
    if right_array.shape != left_array.shape:
        raise ValueError("right and left polarization arrays must have equal shapes")
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite non-negative value")

    with np.errstate(over="ignore", invalid="ignore"):
        denominator = right_array + left_array
    valid = (
        np.isfinite(right_array)
        & np.isfinite(left_array)
        & np.isfinite(denominator)
        & (np.abs(denominator) > eps)
    )
    ratio = np.full(right_array.shape, np.nan, dtype=np.float32)
    ratio[valid] = np.clip(
        (right_array[valid] - left_array[valid]) / denominator[valid],
        -1.0,
        1.0,
    ).astype(np.float32)
    return ratio


def safe_log10(data: np.ndarray) -> np.ndarray:
    """Compute base-10 logarithms while mapping non-positive values to ``NaN``."""

    array = numeric_data(data)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(np.where(array > 0, array, np.nan))


def finite_color_limits(
    data: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    symmetric: bool = False,
    fallback: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return percentile limits from finite values only.

    ``symmetric=True`` expands the limits symmetrically around zero, which is
    suitable for the signed polarization ratio.
    """

    lower = float(lower_percentile)
    upper = float(upper_percentile)
    if not (0.0 <= lower <= upper <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= lower <= upper <= 100")

    values = numeric_data(data)
    values = values[np.isfinite(values)]
    if values.size == 0:
        vmin, vmax = map(float, fallback)
    else:
        vmin, vmax = np.percentile(values, [lower, upper]).astype(float)
    if not np.isfinite([vmin, vmax]).all() or vmin > vmax:
        raise ValueError("fallback must contain finite ascending limits")
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    return float(vmin), float(vmax)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
