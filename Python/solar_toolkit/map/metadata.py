"""Metadata helpers for SunPy-compatible maps and FITS headers.

English: Resolve observation times and display extents from map-like objects or
metadata mappings.

中文：从类 Map 对象或元数据映射中解析观测时间和显示范围。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from solar_toolkit.time import normalize_time

from .coordinates import calculate_fits_extent_from_header


def get_map_obs_time(map_or_header: Any):
    """Return naive UTC at microsecond resolution from a map or FITS header.

    Astropy time scales and FITS ``TIMESYS`` are converted, not discarded.
    Date-only headers require ``TIME-OBS``/``TIME_OBS``; incomplete or invalid
    times raise ``ValueError``. Missing observation metadata raises ``KeyError``.
    """

    return _observation_stamp(map_or_header).to_datetime()


def _observation_stamp(map_or_header: Any):
    """Read explicit metadata before consulting a Map's synthetic date fallback."""
    try:
        raw_header = _header_from(map_or_header)
    except TypeError:
        # Date-only lightweight records remain supported; SunPy Maps always
        # have metadata and therefore never reach this fallback.
        value = getattr(map_or_header, "date", None)
        if value is None:
            raise
        return normalize_time(value)
    header = {str(key).upper(): value for key, value in raw_header.items()}
    for key in (
        "DATE-OBS",
        "DATE_OBS",
        "DATE-BEG",
        "DATE_BEG",
        "DATE-AVG",
        "DATE_AVG",
        "DATE-END",
        "DATE_END",
    ):
        if key in header:
            value = header[key]
            if isinstance(value, str) and len(value.strip()) == 10:
                clock = header.get("TIME-OBS", header.get("TIME_OBS"))
                if clock is not None:
                    value = f"{value.strip()}T{str(clock).strip()}"
            return normalize_time(value, scale=header.get("TIMESYS"))
    raise KeyError("Missing explicit observation-time metadata (DATE-OBS/DATE_OBS)")


def get_display_extent(
    map_or_header: Any,
    image_shape: Sequence[int] | None = None,
    *,
    preserve_orientation: bool = True,
    legacy_linear: bool = False,
) -> tuple[float, float, float, float]:
    """Return an axis-aligned display approximation in native header units.

    Rotated/sheared/distorted WCS requires WCSAxes or a full coordinate grid.
    Explicit ``legacy_linear=True`` allows the historical CDELT approximation
    with a warning. A celestial extent is not a scientific world mapping.
    """

    if hasattr(map_or_header, "meta"):
        header = map_or_header.meta
        if image_shape is None and hasattr(map_or_header, "data"):
            image_shape = np.asarray(map_or_header.data).shape[-2:]
    else:
        header = _header_from(map_or_header)
    return calculate_fits_extent_from_header(
        header,
        image_shape=image_shape,
        preserve_orientation=preserve_orientation,
        legacy_linear=legacy_linear,
    )


def _header_from(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "header"):
        return value.header
    if hasattr(value, "meta"):
        return value.meta
    raise TypeError(
        "Expected a mapping, FITS HDU/header-like object, or SunPy Map-like object"
    )


__all__ = ["get_display_extent", "get_map_obs_time"]
