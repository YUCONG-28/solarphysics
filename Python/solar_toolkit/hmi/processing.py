"""Reusable numerical processing for HMI magnetograms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import astropy.units as u
    import sunpy.map


def process_hmi_magnetic_field(
    hmi_map: sunpy.map.GenericMap,
    threshold: u.Quantity | None = None,
    sigma: float = 3,
) -> sunpy.map.GenericMap:
    """Threshold and smooth an HMI map without modifying its data or metadata.

    ``threshold`` is a finite, non-negative scalar magnetic-field quantity;
    ``sigma`` is a finite, non-negative Gaussian width in pixels. Missing BUNIT
    retains the historical Gauss assumption, applied only to the output map.
    Nonfinite and explicitly masked pixels remain NaN. The input mask is copied
    to the output. Smoothing is normalized by valid-pixel
    support, so missing measurements do not contaminate nearby valid pixels.
    """

    import astropy.units as u
    import numpy as np
    import sunpy.map
    from scipy.ndimage import gaussian_filter

    hmi_unit = hmi_map.unit if hmi_map.unit is not None else u.Gauss
    if not hmi_unit.is_equivalent(u.Gauss):
        raise u.UnitConversionError("HMI data must have magnetic-field units")
    level = _magnetic_level(0 * u.Gauss if threshold is None else threshold)
    threshold_value = level.to_value(hmi_unit)
    width = float(sigma)
    if not np.isfinite(width) or width < 0:
        raise ValueError("sigma must be finite and non-negative")

    values = np.array(hmi_map.data, dtype=float, copy=True)
    valid = np.isfinite(values)
    mask = getattr(hmi_map, "mask", None)
    if mask is not None:
        mask = np.array(mask, dtype=bool, copy=True)
        valid &= ~mask
    values[~valid | (np.abs(values) < threshold_value)] = 0.0
    smoothed = gaussian_filter(values, sigma=width)
    if not valid.all():
        support = gaussian_filter(valid.astype(float), sigma=width)
        np.divide(smoothed, support, out=smoothed, where=support > 0)
        smoothed[~valid | (support == 0)] = np.nan
    metadata = hmi_map.meta.copy()
    if hmi_map.unit is None:
        metadata["bunit"] = "G"
    return sunpy.map.Map(smoothed * hmi_unit, metadata, mask=mask)


def create_magnetic_contour_levels(
    base_level: u.Quantity | None = None,
) -> u.Quantity:
    """Return symmetric negative and positive magnetic contour levels."""

    import astropy.units as u

    level = _magnetic_level(50 * u.Gauss if base_level is None else base_level)
    return u.Quantity([-level.value, level.value], level.unit)


def _magnetic_level(value):
    """Validate a scalar field magnitude while retaining its original unit."""
    import astropy.units as u
    import numpy as np

    if not isinstance(value, u.Quantity):
        raise TypeError("A magnetic-field quantity is required")
    if not value.unit.is_equivalent(u.Gauss):
        raise u.UnitConversionError("Magnetic-field units are required")
    if not value.isscalar or not np.isfinite(value.value) or value.value < 0:
        raise ValueError("Magnetic-field magnitude must be finite and non-negative")
    return value


__all__ = ["create_magnetic_contour_levels", "process_hmi_magnetic_field"]
