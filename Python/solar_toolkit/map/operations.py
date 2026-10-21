"""SunPy map operations shared by AIA and HMI workflows.

The functions import SunPy lazily so importing this module remains lightweight.
"""

from __future__ import annotations

from copy import deepcopy
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
    Existing rates are not divided again; missing-value masks are merged.
    Integrated values require a
    finite positive exposure in seconds; invalid units/exposures raise
    ``ValueError``. Original exposure metadata retains the acquisition duration,
    while BUNIT and a history entry record the division. Inputs are not modified.
    """

    import astropy.units as u
    import numpy as np
    import sunpy.map
    from astropy.nddata import (
        CCDData,
        InverseVariance,
        StdDevUncertainty,
        VarianceUncertainty,
    )

    unit = aia_map.unit
    if unit is None:
        raise ValueError("AIA BUNIT must explicitly declare detector units")
    detector_units = tuple(
        base / pixel for base in (u.DN, u.ct, u.photon) for pixel in (u.one, u.pix)
    )
    raw = np.array(np.ma.getdata(aia_map.data), dtype=float, copy=True)
    mask = np.ma.getmaskarray(aia_map.data).copy() | ~np.isfinite(raw)
    explicit_mask = getattr(aia_map, "mask", None)
    if explicit_mask is not None:
        mask |= np.asarray(explicit_mask, dtype=bool)
    raw[~np.isfinite(raw)] = np.nan
    if any(unit.is_equivalent(base / u.s) for base in detector_units):
        previous = (
            np.zeros(raw.shape, dtype=bool)
            if explicit_mask is None
            else np.asarray(explicit_mask, dtype=bool)
        )
        if np.array_equal(mask, previous):
            return aia_map
        return sunpy.map.Map(
            raw,
            deepcopy(aia_map.meta),
            mask=mask,
            uncertainty=deepcopy(getattr(aia_map, "uncertainty", None)),
            plot_settings=deepcopy(aia_map.plot_settings),
        )
    if not any(unit.is_equivalent(base) for base in detector_units):
        raise ValueError("AIA BUNIT must declare integrated detector units or rates")
    exposure_time = aia_map.exposure_time
    try:
        exposure = float(exposure_time.to_value(u.s))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("AIA exposure must be finite and positive") from exc
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("AIA exposure must be finite and positive")
    uncertainty = getattr(aia_map, "uncertainty", None)
    if uncertainty is not None and not isinstance(
        uncertainty, (StdDevUncertainty, VarianceUncertainty, InverseVariance)
    ):
        raise ValueError("Unsupported exposure uncertainty type")
    if uncertainty is not None and uncertainty.array.shape != aia_map.data.shape:
        raise ValueError("Exposure uncertainty must match the image shape")
    detector = CCDData(
        raw,
        unit=unit,
        mask=mask,
        uncertainty=deepcopy(uncertainty),
    )
    rate = detector.divide(exposure * u.s)
    metadata = deepcopy(aia_map.meta)
    # Solar FITS commonly use DN, which is not in Astropy's strict FITS unit set.
    metadata["bunit"] = (unit / u.s).to_string()
    history = metadata.get("history", [])
    history = [history] if isinstance(history, str) else list(history)
    history.append(
        f"Exposure normalization: divided {unit} values by {exposure:g} s; original exposure retained"
    )
    metadata["history"] = history
    return sunpy.map.Map(
        rate.data,
        metadata,
        mask=rate.mask,
        uncertainty=rate.uncertainty,
        plot_settings=deepcopy(aia_map.plot_settings),
    )


def align_maps_to_reference(
    source_map: sunpy.map.GenericMap,
    target_wcs,
    *,
    return_footprint: bool = False,
    uncertainty_policy: str = "raise",
):
    """Reproject a masked solar image while retaining acquisition metadata.

    The target WCS controls geometry, observer and reference time. Source
    observation time, units, exposure and instrument metadata remain attached
    to the sampled measurements. Invalid source pixels are NaN before
    interpolation; output mask and optional footprint mark unsupported pixels.

    Correlated interpolation uncertainties are not modeled: an input uncertainty
    raises by default. Explicit ``uncertainty_policy='drop'`` drops it and adds
    a history entry. Returns a Map, or ``(Map, footprint)`` when requested.
    Inputs are unchanged. Solar-surface propagation is the retained assumption.
    """

    import numpy as np
    import sunpy.map
    from astropy.time import Time
    from astropy.wcs import WCS
    from reproject import reproject_interp
    from sunpy.coordinates import propagate_with_solar_surface

    from .._utils.numeric import numeric_data
    from ..time import normalize_time
    from .metadata import _observation_stamp

    if not isinstance(return_footprint, bool):
        raise TypeError("return_footprint must be a boolean")
    if uncertainty_policy not in {"raise", "drop"}:
        raise ValueError("uncertainty_policy must be 'raise' or 'drop'")
    uncertainty = getattr(source_map, "uncertainty", None)
    if uncertainty is not None and uncertainty_policy == "raise":
        raise ValueError(
            "Reprojection uncertainty is unsupported; explicitly choose 'drop'"
        )
    observed = _observation_stamp(source_map)
    reference = deepcopy(target_wcs) if isinstance(target_wcs, WCS) else WCS(target_wcs)
    if reference.pixel_n_dim != 2 or reference.array_shape is None:
        raise ValueError("target_wcs must describe a two-dimensional output shape")
    data = numeric_data(source_map)
    with propagate_with_solar_surface():
        output, footprint = reproject_interp(
            (data, source_map.wcs),
            reference,
            shape_out=reference.array_shape,
            return_footprint=True,
        )
    output = np.array(output, copy=True)
    footprint = np.array(footprint, dtype=float, copy=True)
    invalid = ~np.isfinite(output) | ~np.isfinite(footprint) | (footprint <= 0)
    output[invalid] = np.nan
    footprint[invalid] = 0.0
    metadata = reference.to_header(relax=True)
    for key in (
        "bunit",
        "exptime",
        "exposure",
        "telescop",
        "instrume",
        "detector",
        "obsrvtry",
        "wavelnth",
        "waveunit",
        "content",
        "origin",
    ):
        if key in source_map.meta:
            metadata[key] = deepcopy(source_map.meta[key])
    target_time = metadata.get("DATE-AVG", metadata.get("DATE-OBS"))
    if target_time is not None:
        metadata["DATE-AVG"] = normalize_time(
            target_time, scale=metadata.get("TIMESYS")
        ).utc
        metadata["MJD-AVG"] = Time(metadata["DATE-AVG"], scale="utc").mjd
    # WCS serialization can retain the target's observation MJD. It must not
    # contradict the source acquisition timestamp after resampling.
    for key in ("DATE-BEG", "DATE-END", "MJD-BEG", "MJD-END"):
        metadata.pop(key, None)
    metadata["DATE-OBS"] = observed.utc
    metadata["MJD-OBS"] = Time(observed.utc, scale="utc").mjd
    metadata["TIMESYS"] = "UTC"
    history = source_map.meta.get("history", [])
    history = [history] if isinstance(history, str) else list(history)
    history.append("Reprojected to target WCS; masked source samples excluded")
    if uncertainty is not None:
        history.append(
            "Reprojection uncertainty explicitly dropped; no propagation model"
        )
    for entry in history:
        metadata.add_history(entry)
    result = sunpy.map.Map(
        output, metadata, mask=invalid, plot_settings=deepcopy(source_map.plot_settings)
    )
    return (result, footprint) if return_footprint else result


__all__ = ["align_maps_to_reference", "create_aia_submap", "normalize_aia_exposure"]
