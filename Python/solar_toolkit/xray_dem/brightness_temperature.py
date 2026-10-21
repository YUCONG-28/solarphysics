"""Geometry and array helpers for precomputed brightness-temperature maps.

These helpers read derived products; they do not perform a DEM inversion.
Coordinates use the linear FITS axes, in the units of the supplied header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solar_toolkit.io import read_fits_data_header
from solar_toolkit.map import get_display_extent
from solar_toolkit.radio.time_matching import observation_stamp
from solar_toolkit.time import extract_timestamp_from_filename


class AiaMapGeometry:
    """AIA reference geometry, without loading an application or plotting.

    ``hdu_index=1`` preserves the AIA extension convention used by the original
    plotting recipes. ``extent`` retains their increasing-axis display order.
    ``obs_dt`` is naive UTC; ``obs_time`` retains available fractional seconds.
    Missing observation times are represented by ``None`` and ``"Unknown"``.
    """

    def __init__(self, path: str | Path, *, hdu_index: int = 1) -> None:
        _, self.header = read_fits_data_header(path, hdu_index=hdu_index)
        self.extent = list(get_display_extent(self.header, preserve_orientation=False))
        self.nx, self.ny = int(self.header["NAXIS1"]), int(self.header["NAXIS2"])
        self.rsun = float(
            self.header["RSUN_OBS"]
            if "RSUN_OBS" in self.header
            else self.header["R_SUN"] * abs(self.header["CDELT1"])
        )
        self.sun_center = (self.header["CRVAL1"], self.header["CRVAL2"])
        try:
            stamp = observation_stamp(
                self.header,
                path=path,
                fallback_parser=extract_timestamp_from_filename,
            )
        except (ValueError, TypeError):
            self.obs_time, self.obs_dt = "Unknown", None
        else:
            self.obs_time = format_observation_time(stamp.utc)
            self.obs_dt = stamp.to_datetime()

    def display_extent(
        self,
        mode: str,
        *,
        solar_padding_factor: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> list[float]:
        """Select full, padded solar-disk, or explicitly configured bounds.

        Unknown modes retain the original recipes' full-image fallback.
        """
        if mode == "solar_disk":
            cx, cy = self.sun_center
            radius = self.rsun * solar_padding_factor
            return [cx - radius, cx + radius, cy - radius, cy + radius]
        if mode == "custom":
            return [*x_range, *y_range]
        return list(self.extent)


def format_observation_time(utc: str) -> str:
    """Format normalized UTC as the plotting recipes' ``... UT`` label."""
    whole, _, fraction = utc.partition(".")
    fraction = fraction.rstrip("0")
    return whole.replace("T", " ") + (f".{fraction}" if fraction else "") + " UT"


def load_brightness_temperature(path: str | Path) -> np.ndarray:
    """Load a nonempty 2-D Kelvin array as MK, replacing NaN/Inf with zero.

    The historical display policy is explicit here: missing values are filled
    only for display. The source array on disk is never modified.
    """
    data = np.load(path, allow_pickle=False).astype(np.float64)
    if data.ndim != 2 or not data.size:
        raise ValueError("Brightness temperature must be a nonempty 2-D array")
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) / 1e6


def brightness_temperature_extent(
    *,
    pixel_size: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> list[float]:
    """Expand first/last pixel-center bounds by half a positive pixel size."""
    values = np.asarray([pixel_size, *x_range, *y_range], dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all() or pixel_size <= 0:
        raise ValueError("Grid bounds must be finite and pixel_size positive")
    xmin, xmax = x_range
    ymin, ymax = y_range
    if xmin > xmax or ymin > ymax:
        raise ValueError("Grid center bounds must be in increasing order")
    half = float(pixel_size) / 2
    return [xmin - half, xmax + half, ymin - half, ymax + half]


__all__ = [
    "AiaMapGeometry",
    "brightness_temperature_extent",
    "format_observation_time",
    "load_brightness_temperature",
]
