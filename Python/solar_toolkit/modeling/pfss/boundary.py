"""Validate global radial boundaries without repairing missing observations."""

import hashlib
from pathlib import Path

import numpy as np


def _overlap_weights(old, new):
    """Cell-overlap averages on an equal-area axis, including noninteger ratios."""
    lo = np.arange(new)[:, None] * old / new
    hi = (np.arange(new)[:, None] + 1) * old / new
    weights = np.maximum(
        0, np.minimum(hi, np.arange(old) + 1) - np.maximum(lo, np.arange(old))
    )
    return weights / (old / new)


def prepare_boundary(source, *, nphi=360, ns=180, component=None):
    """Return a canonical CEA Br map and explicit processing diagnostics.

    ``component='Br'`` is a caller declaration, retained as provenance; it does
    not override a known LOS product. Only Carrington CEA/CAR maps are accepted.
    NaNs, incomplete coverage and noncanonical rotated grids fail closed.
    """
    import astropy.units as u
    import sunpy.map

    m = sunpy.map.Map(source) if isinstance(source, (str, Path)) else source
    if int(nphi) != nphi or int(ns) != ns or min(nphi, ns) < 4:
        raise ValueError("Output dimensions must be integer and at least four")
    data = np.asarray(m.data, dtype=float)
    meta = m.meta.copy()
    series = str(meta.get("series", "")) + " " + str(meta.get("content", ""))
    known_los = any(
        s in series.lower()
        for s in ["hmi.m_", "synoptic_ml", "line-of-sight", "line of sight"]
    )
    radial = component == "Br" or any(
        s in series.lower() for s in ["synoptic_mr", "radial"]
    )
    if known_los or not radial:
        raise ValueError(
            "A documented global Br product or explicit Br declaration is required"
        )
    ctypes = tuple(str(x).upper() for x in m.wcs.wcs.ctype)
    projection = ctypes[0][-3:]
    if not (
        ctypes[0].startswith("CRLN")
        and ctypes[1].startswith("CRLT")
        and projection in {"CEA", "CAR"}
        and ctypes[1].endswith(projection)
    ):
        raise ValueError(
            "PFSS requires a Carrington global CEA/CAR map; a disk LOS/SHARP map is invalid"
        )
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(
            "Global boundary contains missing/invalid cells; no zero filling is performed"
        )
    if m.unit is None or not m.unit.is_equivalent(u.G):
        raise ValueError("Boundary BUNIT must be a magnetic field unit")
    data = (data * m.unit).to_value(u.G)
    pc = m.wcs.wcs.get_pc()
    if not np.allclose(pc, np.diag(np.diag(pc)), atol=1e-12) or not np.allclose(
        np.abs(np.diag(pc)), 1
    ):
        raise ValueError(
            "Rotated/skewed global grids must be reprojected explicitly before PFSS"
        )
    scales = m.wcs.wcs.cdelt * np.diag(pc)
    ny, nx = data.shape
    if not np.isclose(abs(scales[0]) * nx, 360, atol=1e-3):
        raise ValueError("Longitude coverage must be exactly one 360-degree period")
    if abs(float(meta.get("crval2", 0))) > 1e-8:
        raise ValueError("Latitude reference is not centered on the equator")
    expected_lat = 360 / np.pi if projection == "CEA" else 180
    if projection == "CEA" and not np.isclose(float(meta.get("pv2_1", 1)), 1):
        raise ValueError("CEA latitude scale parameter must be one")
    if not np.isclose(abs(scales[1]) * ny, expected_lat, atol=1e-3) or not np.isclose(
        float(meta.get("crpix2", 0)), (ny + 1) / 2, atol=1e-5
    ):
        raise ValueError(
            "Latitude coverage must include both poles without missing strips"
        )
    if not any(meta.get(k) for k in ("date-obs", "date_obs", "t_obs", "t_start")):
        raise ValueError("Boundary observation/assembly time is missing")
    if nx < nphi or ny < ns:
        raise ValueError("Requested grid upsamples the boundary; choose a smaller grid")
    original_hash = hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
    flips = []
    for axis, scale in enumerate(scales):
        if scale < 0:
            data = np.flip(data, axis=1 - axis)
            key = f"crpix{axis+1}"
            meta[key] = (nx if axis == 0 else ny) + 1 - float(meta[key])
            flips.append(axis)
        meta[f"cdelt{axis+1}"] = abs(scale)
    for key in list(meta):
        if str(key).lower().startswith(("pc1_", "pc2_", "cd1_", "cd2_")):
            del meta[key]
    meta["bunit"] = "G"
    # ``scales`` are WCS degrees already. HMI's raw Sine Latitude marker
    # must not survive, or its Map subclass applies 180/pi a second time.
    meta["cunit1"] = "deg"
    meta["cunit2"] = "deg"
    m = sunpy.map.Map(data, meta)
    car_reprojected = False
    if projection == "CAR":
        # For this validated, axis-aligned global grid, conservative overlap
        # in sin(latitude) is an exact cell-area CAR -> CEA reprojection.
        # Generic spherical-polygon reprojection can leave polar seam holes.
        target = meta.copy()
        target.update(
            dict(ctype1="CRLN-CEA", ctype2="CRLT-CEA", cdelt2=360 / np.pi / ny, pv2_1=1)
        )
        original = data
        lat_edges = np.linspace(-np.pi / 2, np.pi / 2, ny + 1)
        original_weights = np.diff(np.sin(lat_edges))[:, None] * (2 * np.pi / nx)
        before_signed = float(np.sum(original * original_weights))
        before_unsigned = float(np.sum(abs(original) * original_weights))
        old_edges = np.sin(lat_edges)
        new_edges = np.linspace(-1, 1, ny + 1)
        overlap = np.maximum(
            0,
            np.minimum(new_edges[1:, None], old_edges[None, 1:])
            - np.maximum(new_edges[:-1, None], old_edges[None, :-1]),
        )
        weights = overlap / np.diff(new_edges)[:, None]
        if not np.allclose(weights.sum(axis=1), 1, atol=1e-12):
            raise ValueError("CAR-to-CEA reprojection leaves incomplete coverage")
        data = weights @ data
        m = sunpy.map.Map(data, target)
        meta = m.meta.copy()
        car_reprojected = True
    else:
        area = 4 * np.pi / data.size
        before_signed = float(np.sum(data) * area)
        before_unsigned = float(np.sum(abs(data)) * area)
    rebinned = _overlap_weights(ny, ns) @ data @ _overlap_weights(nx, nphi).T
    out_meta = meta.copy()
    for axis, old, new in [(1, nx, nphi), (2, ny, ns)]:
        out_meta[f"crpix{axis}"] = (float(meta[f"crpix{axis}"]) - 0.5) * new / old + 0.5
        out_meta[f"cdelt{axis}"] = float(meta[f"cdelt{axis}"]) * old / new
    after_signed = float(np.sum(rebinned) * 4 * np.pi / rebinned.size)
    after_unsigned = float(np.sum(abs(rebinned)) * 4 * np.pi / rebinned.size)
    report = dict(
        schema_version=1,
        component="Br",
        component_declaration=component,
        input_shape=[ny, nx],
        output_shape=[ns, nphi],
        input_array_sha256=original_hash,
        projection=projection,
        car_reprojected=car_reprojected,
        flipped_axes=flips,
        algorithm="equal-area cell-overlap average",
        net_flux_correction="none",
        signed_flux_before_G_sr=before_signed,
        signed_flux_after_G_sr=after_signed,
        unsigned_flux_before_G_sr=before_unsigned,
        unsigned_flux_after_G_sr=after_unsigned,
        net_flux_fraction=abs(after_signed) / after_unsigned if after_unsigned else 0,
        signed_flux_change_over_unsigned=(
            abs(after_signed - before_signed) / before_unsigned
            if before_unsigned
            else 0
        ),
        boundary_date_utc=str(m.date.utc.isot),
        time_start=meta.get("t_start"),
        time_stop=meta.get("t_stop"),
        observer_assumption="Earth ephemeris if global synoptic product has no observer metadata",
        time_assumption="synoptic boundary is not an event-time snapshot",
    )
    if report["signed_flux_change_over_unsigned"] > 0.01:
        raise ValueError(
            "Reprojection/rebinning changes net flux by >1% of original unsigned flux"
        )
    return sunpy.map.Map(rebinned, out_meta), report


__all__ = ["prepare_boundary"]
