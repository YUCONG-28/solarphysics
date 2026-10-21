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

from .._utils.numeric import as_value, numeric_data
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
    footprint: np.ndarray | None = None
    method_used: str | None = None
    fallback_reason: str | None = None


def interpolate_scattered_to_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    target_shape: tuple[int, int],
    *,
    method: str = "linear",
    fallback: str | None = None,
    return_diagnostics: bool = False,
):
    """Interpolate scattered ``(x, y)`` samples onto a target image grid.

    The default leaves unsupported pixels NaN and never silently changes the
    requested method. Explicit ``fallback='nearest'`` permits nearest-neighbor
    values after a degenerate triangulation or outside the supported region.
    Their footprint remains zero outside the convex hull of finite mapped
    sample centers (a line segment/point for degenerate geometry). This is
    interpolation support, not measured pixel area or a flux-conservation model.

    Masks exclude samples; bare coordinates mean target pixels, and pixel
    Quantities are accepted. Values retain native numerical units. Optional
    diagnostics return ``(data, footprint, dict)`` and disclose the actual method
    and fallback reason. Inputs are unchanged; output data remains float32.
    """

    height, width = _target_shape(target_shape)
    method = str(method).lower()
    if method not in {"linear", "nearest", "cubic"}:
        raise ValueError("method must be 'linear', 'nearest', or 'cubic'")
    if fallback not in {None, "nearest"}:
        raise ValueError("fallback must be None or 'nearest'")
    if not isinstance(return_diagnostics, bool):
        raise TypeError("return_diagnostics must be a boolean")

    import astropy.units as u
    from scipy.interpolate import griddata
    from scipy.spatial import QhullError

    points = as_value(points_xy, u.pix, "points_xy")
    sample_values = numeric_data(values).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (sample_count, 2)")
    if points.shape[0] != sample_values.size:
        raise ValueError("points_xy and values must contain the same sample count")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(sample_values)
    points = points[finite]
    sample_values = sample_values[finite]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    result = np.full((height, width), np.nan)
    support = _sample_support(points, np.column_stack((grid_x.ravel(), grid_y.ravel())))
    support = support.reshape((height, width))
    selected_method = None
    reason = "no_finite_samples" if not sample_values.size else None
    if sample_values.size:
        try:
            result = griddata(points, sample_values, (grid_x, grid_y), method=method)
            selected_method = method
        except (QhullError, ValueError) as exc:
            # Shape/method validation happened above; a triangulation cannot be
            # constructed for coincident or collinear samples.
            reason = f"degenerate_geometry: {type(exc).__name__}"
        result[~support] = np.nan
        if fallback == "nearest" and np.any(~np.isfinite(result)):
            missing = ~np.isfinite(result)
            nearest = griddata(
                points, sample_values, (grid_x, grid_y), method="nearest"
            )
            result[missing] = nearest[missing]
            selected_method = (
                "nearest"
                if selected_method is None
                else (method if method == "nearest" else f"{method}+nearest")
            )
            reason = reason or "outside_support_or_missing_interpolation"
    footprint = (support & np.isfinite(result)).astype(float)
    result = np.asarray(result, dtype=np.float32)
    diagnostics = {
        "requested_method": method,
        "method_used": selected_method,
        "fallback": fallback,
        "fallback_reason": reason,
        "mapped_sample_count": int(sample_values.size),
    }
    return (result, footprint, diagnostics) if return_diagnostics else result


def _sample_support(points, targets):
    """Membership in a sample-center hull, including rank-zero/one support."""
    from scipy.spatial import Delaunay, QhullError

    if not len(points):
        return np.zeros(len(targets), dtype=bool)
    center = points[0]
    offsets = points - center
    _, singular, axes = np.linalg.svd(offsets, full_matrices=False)
    scale = max(1.0, float(np.max(np.abs(points))))
    tolerance = max(1e-9, 64 * np.finfo(float).eps * scale)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        return np.linalg.norm(targets - center, axis=1) <= tolerance
    if rank == 1:
        direction = axes[0]
        positions = offsets @ direction
        projected = (targets - center) @ direction
        distance = np.linalg.norm(
            targets - center - projected[:, None] * direction, axis=1
        )
        return (
            (distance <= tolerance)
            & (projected >= positions.min() - tolerance)
            & (projected <= positions.max() + tolerance)
        )
    try:
        return Delaunay(points).find_simplex(targets, tol=1e-10) >= 0
    except QhullError:
        return np.zeros(len(targets), dtype=bool)


def reproject_radio_array(
    radio_data: np.ndarray,
    target_shape: tuple[int, int],
    pixel_mapper: PixelMapper,
    *,
    method: str = "linear",
    fallback: str | None = None,
    mask=None,
) -> RadioReprojectionResult | None:
    """Map finite unmasked radio samples, with explicit support diagnostics.

    ``fallback=None`` leaves unsupported target pixels NaN. Explicit
    ``fallback='nearest'`` permits extrapolated values with zero footprint.
    Empty/entirely unsupported results return None. No units, uncertainties or
    observation metadata are inferred; native numerical values are retained.
    """

    source = numeric_data(radio_data, mask=mask)
    if source.ndim != 2:
        raise ValueError("radio_data must be a two-dimensional array")
    _target_shape(target_shape)
    if not callable(pixel_mapper):
        raise TypeError("pixel_mapper must be callable")
    method = str(method).lower()
    if method not in {"linear", "nearest", "cubic"}:
        raise ValueError("method must be 'linear', 'nearest', or 'cubic'")
    if fallback not in {None, "nearest"}:
        raise ValueError("fallback must be None or 'nearest'")

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

    data, footprint, diagnostics = interpolate_scattered_to_grid(
        np.asarray(points, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        target_shape,
        method=method,
        fallback=fallback,
        return_diagnostics=True,
    )
    if np.all(~np.isfinite(data)):
        return None
    peak_y, peak_x = np.unravel_index(np.nanargmax(data), data.shape)
    return RadioReprojectionResult(
        data=data,
        peak_pixel=(float(peak_x), float(peak_y)),
        amplitude=float(np.nanmax(data)),
        mapped_sample_count=len(points),
        footprint=footprint,
        method_used=diagnostics["method_used"],
        fallback_reason=diagnostics["fallback_reason"],
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
