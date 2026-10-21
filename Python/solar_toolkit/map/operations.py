"""SunPy map operations shared by AIA and HMI workflows.

The functions import SunPy lazily so importing this module remains lightweight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sunpy.map


def create_aia_submap(
    aia_map: sunpy.map.GenericMap,
    roi_bounds: tuple[float, float, float, float],
    *,
    legacy_roi: bool = False,
) -> sunpy.map.GenericMap:
    """Crop four finite arcsecond bounds ``(xmin, xmax, ymin, ymax)``.

    Each axis must increase strictly. SunPy includes pixels intersecting the
    requested boundary. Explicit ``legacy_roi=True`` reproduces the historical
    bug that used ``ymin`` for both corners, yielding a horizontal strip.
    """

    import astropy.units as u
    import numpy as np
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import propagate_with_solar_surface

    if not isinstance(legacy_roi, bool):
        raise TypeError("legacy_roi must be a boolean")
    try:
        bounds = np.asarray(roi_bounds, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "roi_bounds must contain four finite arcsecond values"
        ) from exc
    if bounds.shape != (4,) or not np.isfinite(bounds).all():
        raise ValueError("roi_bounds must contain four finite arcsecond values")
    tx1, tx2, ty1, ty2 = bounds
    if tx1 >= tx2 or ty1 >= ty2:
        raise ValueError("roi_bounds must increase strictly along both axes")

    with propagate_with_solar_surface():
        frame = aia_map.coordinate_frame
        bottom_left = SkyCoord(Tx=tx1 * u.arcsec, Ty=ty1 * u.arcsec, frame=frame)
        top_right = SkyCoord(
            Tx=tx2 * u.arcsec, Ty=(ty1 if legacy_roi else ty2) * u.arcsec, frame=frame
        )
        return aia_map.submap(bottom_left, top_right=top_right)


def normalize_aia_exposure(aia_map: sunpy.map.GenericMap) -> sunpy.map.GenericMap:
    """Convert integrated detector values to rates without losing the mask.

    Explicit DN, count or photon units (optionally per pixel) are required.
    Existing rate units are returned unchanged. Integrated values require a
    finite positive exposure in seconds; invalid units/exposures raise
    ``ValueError``. Original exposure metadata retains the acquisition duration,
    while BUNIT and a history entry record the division. Inputs are not modified.
    """

    import astropy.units as u
    import numpy as np
    import sunpy.map

    unit = aia_map.unit
    if unit is None:
        raise ValueError("AIA BUNIT must explicitly declare detector units")
    detector_units = tuple(
        base / pixel for base in (u.DN, u.ct, u.photon) for pixel in (u.one, u.pix)
    )
    if any(unit.is_equivalent(base / u.s) for base in detector_units):
        return aia_map
    if not any(unit.is_equivalent(base) for base in detector_units):
        raise ValueError("AIA BUNIT must declare integrated detector units or rates")
    exposure_time = aia_map.exposure_time
    try:
        exposure = float(exposure_time.to_value(u.s))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("AIA exposure must be finite and positive") from exc
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("AIA exposure must be finite and positive")
    metadata = aia_map.meta.copy()
    # Solar FITS commonly use DN, which is not in Astropy's strict FITS unit set.
    metadata["bunit"] = (unit / u.s).to_string()
    history = metadata.get("history", [])
    history = [history] if isinstance(history, str) else list(history)
    history.append(
        f"Exposure normalization: divided {unit} values by {exposure:g} s; original exposure retained"
    )
    metadata["history"] = history
    mask = getattr(aia_map, "mask", None)
    return sunpy.map.Map(
        np.asarray(aia_map.data, dtype=float) / exposure,
        metadata,
        mask=None if mask is None else mask.copy(),
    )


def align_maps_to_reference(
    source_map: sunpy.map.GenericMap,
    target_wcs,
) -> sunpy.map.GenericMap:
    """Reproject ``source_map`` to ``target_wcs`` with solar-surface propagation."""

    from sunpy.coordinates import propagate_with_solar_surface

    with propagate_with_solar_surface():
        return source_map.reproject_to(target_wcs)


__all__ = ["align_maps_to_reference", "create_aia_submap", "normalize_aia_exposure"]
