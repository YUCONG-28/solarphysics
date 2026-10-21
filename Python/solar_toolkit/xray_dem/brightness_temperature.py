"""Geometry and array helpers for precomputed brightness-temperature maps.

These helpers read derived products; they do not perform a DEM inversion.
Display bounds use the complete reference WCS and are expressed in arcseconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solar_toolkit.io import read_fits_data_header
from solar_toolkit.map.coordinates import hpc_pixel_coordinates, hpc_world_bounds
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
        self.nx, self.ny = int(self.header["NAXIS1"]), int(self.header["NAXIS2"])
        self.extent = list(hpc_world_bounds(self.header, (self.ny, self.nx)))
        if "RSUN_OBS" in self.header:
            self.rsun = float(self.header["RSUN_OBS"])
        else:
            x0, y0 = self.header["CRPIX1"] - 1, self.header["CRPIX2"] - 1
            tx, ty = hpc_pixel_coordinates(self.header, [x0, x0 + 1], [y0, y0])
            self.rsun = float(
                self.header["R_SUN"] * np.hypot(tx[1] - tx[0], ty[1] - ty[0])
            )
        if not np.isfinite(self.rsun) or self.rsun <= 0:
            raise ValueError(
                "The reference map must declare a finite positive solar radius"
            )
        # Solar disk center is HPC (0, 0), not an arbitrary WCS reference pixel.
        self.sun_center = (0.0, 0.0)
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


def load_brightness_temperature(
    path: str | Path, *, fill_missing: float | None = None
) -> np.ndarray:
    """Load a nonempty 2-D Kelvin array as MK, preserving missing samples.

    A finite ``fill_missing`` explicitly selects a display fill value in MK.
    The source array on disk is never modified. No inversion is performed.
    """
    data = np.load(path, allow_pickle=False).astype(np.float64)
    if data.ndim != 2 or not data.size:
        raise ValueError("Brightness temperature must be a nonempty 2-D array")
    data = np.where(np.isfinite(data), data / 1e6, np.nan)
    if fill_missing is not None:
        if not np.isfinite(fill_missing):
            raise ValueError("fill_missing must be finite, in MK")
        data = np.where(np.isfinite(data), data, float(fill_missing))
    return data


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
