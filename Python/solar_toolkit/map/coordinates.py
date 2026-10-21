"""Pure coordinate helpers for FITS image display geometry.

The functions here do not read files, plot figures, or depend on local data.
They provide a tested baseline for future refactoring of radio FITS
``extent``/``origin`` handling.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

Extent = tuple[float, float, float, float]

__all__ = [
    "Extent",
    "hpc_pixel_coordinates",
    "hpc_pixel_grid",
    "hpc_world_to_pixel",
    "hpc_world_bounds",
    "calculate_fits_extent_from_header",
    "infer_image_origin_from_header",
    "normalize_radio_extent",
    "validate_extent_orientation",
]


def _header_float(header: Mapping[str, Any], key: str) -> float:
    try:
        return float(header[key])
    except KeyError as exc:
        raise KeyError(f"Missing FITS header keyword: {key}") from exc


def _header_for_wcs(header):
    """Copy scalar FITS metadata without mapping-only commentary containers.

    ``dict(fits.Header)`` retains commentary wrapper objects that stringify to
    multiline values. They do not belong to WCS, and rebuilding Header from
    that mapping can otherwise fail despite the original cards being valid.
    """
    from astropy.io.fits import Header

    result = Header()
    for key, value in header.items():
        name = str(key).upper()
        if name not in {"", "COMMENT", "HISTORY", "KEYCOMMENTS"}:
            result[name] = value
    return result


def _resolve_image_shape(
    header: Mapping[str, Any], image_shape: Sequence[int] | None
) -> tuple[int, int]:
    if image_shape is not None:
        if len(image_shape) != 2:
            raise ValueError("image_shape must be a two-item sequence: (ny, nx)")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in image_shape
        ):
            raise ValueError("image_shape dimensions must be integers")
        ny, nx = int(image_shape[0]), int(image_shape[1])
    else:
        nx = int(_header_float(header, "NAXIS1"))
        ny = int(_header_float(header, "NAXIS2"))

    if nx <= 0 or ny <= 0:
        raise ValueError("image dimensions must be positive")
    return ny, nx


def calculate_fits_extent_from_header(
    header: Mapping[str, Any],
    image_shape: Sequence[int] | None = None,
    *,
    preserve_orientation: bool = True,
    legacy_linear: bool = False,
) -> Extent:
    """Return axis-aligned pixel-edge extents in the declared native units.

    CD, PC and CROTA are interpreted by Astropy WCS. Rotated/sheared axes and
    distortion are not representable by four imshow bounds and raise by default.
    ``legacy_linear=True`` explicitly retains the old CDELT-only approximation
    with a warning. For a celestial projection these are *projection-plane*
    coordinates around CRVAL, not a substitute for its complete spherical WCS;
    scientific world coordinates use :func:`hpc_pixel_coordinates` instead.
    """
    import astropy.units as u
    from astropy.wcs import WCS

    header = {str(key).upper(): value for key, value in header.items()}
    ny, nx = _resolve_image_shape(header, image_shape)
    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        _header_float(header, key)
    has_cd = any(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2))
    if not has_cd:
        for key in ("CDELT1", "CDELT2"):
            _header_float(header, key)
    wcs = WCS(_header_for_wcs(header), naxis=2)
    matrix = np.array(wcs.pixel_scale_matrix, dtype=float, copy=True)
    reference = np.array(wcs.wcs.crval, dtype=float, copy=True)
    for axis in range(2):
        native = header.get(f"CUNIT{axis + 1}", "")
        actual = wcs.world_axis_units[axis]
        if native and actual:
            factor = u.Unit(actual).to(u.Unit(native))
            matrix[axis] *= factor
            reference[axis] *= factor
    diagonal = np.diag(matrix)
    off_diagonal = matrix - np.diag(diagonal)
    rotated = np.any(np.abs(off_diagonal) > 1e-12 * max(1.0, np.max(np.abs(matrix))))
    if (rotated or wcs.has_distortion) and not legacy_linear:
        raise ValueError(
            "Rotated, sheared or distorted WCS requires WCSAxes or a full WCS grid"
        )
    if legacy_linear:
        warnings.warn(
            "legacy_linear uses a CDELT-only display approximation, not the full WCS",
            UserWarning,
            stacklevel=2,
        )
        diagonal = np.array(
            [_header_float(header, "CDELT1"), _header_float(header, "CDELT2")]
        )
        reference = np.array(
            [_header_float(header, "CRVAL1"), _header_float(header, "CRVAL2")]
        )
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(reference).all()
        or np.any(diagonal == 0)
    ):
        raise ValueError("WCS display geometry must be finite and nondegenerate")
    if wcs.has_celestial:
        warnings.warn(
            "Celestial extent is a projection-plane display approximation; use WCSAxes for world-coordinate display",
            UserWarning,
            stacklevel=2,
        )
    x0, y0 = reference + (0.5 - wcs.wcs.crpix) * diagonal
    x1, y1 = reference + (np.array([nx, ny]) + 0.5 - wcs.wcs.crpix) * diagonal
    if preserve_orientation:
        return float(x0), float(x1), float(y0), float(y1)
    return (*sorted((float(x0), float(x1))), *sorted((float(y0), float(y1))))


def hpc_pixel_coordinates(header, x_pix, y_pix, *, legacy_linear=False):
    """Map broadcast pixel centers to HPLN/HPLT arcseconds with the full WCS.

    Zero-based pixels, FITS matrix conventions, projection, inline distortion,
    angular units and reversed celestial world-axis order use Astropy. Longitudes
    are wrapped about zero. Observer/time are not guessed: these are angular
    coordinates in the header's own frame, not a cross-observer transformation.
    Explicit ``legacy_linear=True`` preserves the historical projected-plane
    approximation (including its CROTA convention) for compatibility only.
    """
    import astropy.units as u
    from astropy.coordinates import Longitude

    from .._utils.numeric import as_value

    if not isinstance(legacy_linear, bool):
        raise TypeError("legacy_linear must be a boolean")
    header = {str(key).upper(): value for key, value in header.items()}
    x, y = np.broadcast_arrays(
        as_value(x_pix, u.pix, "x_pix"), as_value(y_pix, u.pix, "y_pix")
    )
    if legacy_linear:
        return _legacy_hpc_coordinates(header, x, y)
    wcs, lon, lat = _hpc_wcs(header)
    world = wcs.pixel_to_world_values(x, y)
    x_arcsec = Longitude(
        np.asarray(world[lon]) * u.Unit(wcs.world_axis_units[lon]),
        wrap_angle=180 * u.deg,
    ).to_value(u.arcsec)
    y_arcsec = np.asarray(world[lat]) * u.Unit(wcs.world_axis_units[lat]).to(u.arcsec)
    return x_arcsec, y_arcsec


def _hpc_wcs(header):
    import astropy.units as u
    from astropy.wcs import WCS

    header = {str(key).upper(): value for key, value in header.items()}
    for key in ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"):
        _header_float(header, key)
    if not any(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2)):
        for key in ("CDELT1", "CDELT2"):
            _header_float(header, key)
    ctype = [str(header.get(f"CTYPE{i}", "")).upper() for i in (1, 2)]
    longitude = [
        i for i, value in enumerate(ctype) if "HPLN" in value or "SOLX" in value
    ]
    latitude = [
        i for i, value in enumerate(ctype) if "HPLT" in value or "SOLY" in value
    ]
    if len(longitude) != 1 or len(latitude) != 1:
        raise ValueError("Explicit HPLN/HPLT or SOLX/SOLY world axes are required")
    for axis in (1, 2):
        unit = header.get(f"CUNIT{axis}")
        aliases = {
            "asec": "arcsec",
            "arcsecs": "arcsec",
            "arcsecond": "arcsec",
            "arcseconds": "arcsec",
            "degree": "deg",
            "degrees": "deg",
            "amin": "arcmin",
            "arcminute": "arcmin",
            "arcminutes": "arcmin",
            "radian": "rad",
            "radians": "rad",
        }
        if str(unit).lower() in aliases:
            unit = header[f"CUNIT{axis}"] = aliases[str(unit).lower()]
        if unit is None or not u.Unit(unit).is_equivalent(u.arcsec):
            raise ValueError(f"CUNIT{axis} must explicitly declare angular units")
    wcs = WCS(_header_for_wcs(header), naxis=2)
    return wcs, longitude[0], latitude[0]


def hpc_world_to_pixel(header, x_arcsec, y_arcsec):
    """Invert the complete angular WCS into zero-based pixel coordinates.

    Bare values mean arcseconds; angular Quantities are converted explicitly.
    Inputs broadcast, and the header's celestial-axis order is respected.
    Coordinates are in the header's own frame, without an observer transform.
    """
    import astropy.units as u

    from .._utils.numeric import as_value

    wcs, lon, lat = _hpc_wcs(header)
    x, y = np.broadcast_arrays(
        as_value(x_arcsec, u.arcsec, "x_arcsec"),
        as_value(y_arcsec, u.arcsec, "y_arcsec"),
    )
    world = [None, None]
    world[lon] = x * u.arcsec.to(u.Unit(wcs.world_axis_units[lon]))
    world[lat] = y * u.arcsec.to(u.Unit(wcs.world_axis_units[lat]))
    px, py = wcs.world_to_pixel_values(*world)
    return np.asarray(px), np.asarray(py)


def hpc_world_bounds(header, shape):
    """Return sampled pixel-edge bounds ``(xmin, xmax, ymin, ymax)`` in arcsec.

    This display envelope samples every boundary pixel edge at quarter-pixel
    intervals through the complete WCS. It must not replace world-to-pixel
    mapping or scientific ROI membership. Singular/nonfinite edges raise.
    """
    ny, nx = _resolve_image_shape(header, shape)
    x = np.linspace(-0.5, nx - 0.5, 4 * nx + 1)
    y = np.linspace(-0.5, ny - 0.5, 4 * ny + 1)
    bx = np.concatenate((x, x, np.full_like(y, -0.5), np.full_like(y, nx - 0.5)))
    by = np.concatenate((np.full_like(x, -0.5), np.full_like(x, ny - 0.5), y, y))
    wx, wy = hpc_pixel_coordinates(header, bx, by)
    if not np.isfinite(wx).all() or not np.isfinite(wy).all():
        raise ValueError("Image boundary contains nonfinite WCS coordinates")
    return float(wx.min()), float(wx.max()), float(wy.min()), float(wy.max())


def hpc_pixel_grid(header, shape):
    """Return full-WCS HPLN/HPLT arcsecond grids shaped ``(ny, nx)``."""
    ny, nx = _resolve_image_shape(header, shape)
    y, x = np.indices((ny, nx), dtype=float)
    return hpc_pixel_coordinates(header, x, y)


def _legacy_hpc_coordinates(header, x, y):
    import astropy.units as u

    dx = x + 1 - float(header.get("CRPIX1", 1))
    dy = y + 1 - float(header.get("CRPIX2", 1))
    sx, sy = float(header.get("CDELT1", 1)), float(header.get("CDELT2", 1))
    if all(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2)):
        wx = float(header["CD1_1"]) * dx + float(header["CD1_2"]) * dy
        wy = float(header["CD2_1"]) * dx + float(header["CD2_2"]) * dy
    elif any(f"PC{i}_{j}" in header for i in (1, 2) for j in (1, 2)):
        wx = sx * (
            float(header.get("PC1_1", 1)) * dx + float(header.get("PC1_2", 0)) * dy
        )
        wy = sy * (
            float(header.get("PC2_1", 0)) * dx + float(header.get("PC2_2", 1)) * dy
        )
    else:
        angle = np.deg2rad(float(header.get("CROTA2", 0)))
        wx = np.cos(angle) * sx * dx - np.sin(angle) * sy * dy
        wy = np.sin(angle) * sx * dx + np.cos(angle) * sy * dy
    return (
        (float(header.get("CRVAL1", 0)) + wx)
        * u.Unit(header.get("CUNIT1", "arcsec")).to(u.arcsec),
        (float(header.get("CRVAL2", 0)) + wy)
        * u.Unit(header.get("CUNIT2", "arcsec")).to(u.arcsec),
    )


def infer_image_origin_from_header(
    header: Mapping[str, Any] | None = None,
    *,
    preserve_orientation: bool = True,
    mode: str = "auto",
) -> str:
    """Infer the matplotlib image origin for a radio FITS display.

    ``mode`` may be ``"upper"``, ``"lower"``, or ``"auto"``. The ``header`` is
    accepted for API symmetry and future validation; the current baseline
    mirrors the advanced radio script's rule:

    - preserved FITS orientation uses ``origin="lower"``;
    - normalized legacy display uses ``origin="upper"``.
    """

    _ = header
    normalized_mode = str(mode or "auto").lower()
    if normalized_mode in {"upper", "lower"}:
        return normalized_mode
    if normalized_mode != "auto":
        raise ValueError("mode must be 'auto', 'upper', or 'lower'")
    return "lower" if preserve_orientation else "upper"


def normalize_radio_extent(extent: Sequence[float]) -> Extent:
    """Return ``extent`` sorted into increasing x and y order."""

    if len(extent) != 4:
        raise ValueError("extent must contain four values")
    left, right, bottom, top = map(float, extent)
    x0, x1 = sorted((left, right))
    y0, y1 = sorted((bottom, top))
    return (x0, x1, y0, y1)


def validate_extent_orientation(extent: Sequence[float]) -> dict[str, bool]:
    """Describe whether an extent is normal or inverted on each axis."""

    if len(extent) != 4:
        raise ValueError("extent must contain four values")
    left, right, bottom, top = map(float, extent)
    x_increases = right > left
    y_increases = top > bottom
    return {
        "x_increases": x_increases,
        "y_increases": y_increases,
        "x_inverted": right < left,
        "y_inverted": top < bottom,
        "is_normalized": x_increases and y_increases,
    }
