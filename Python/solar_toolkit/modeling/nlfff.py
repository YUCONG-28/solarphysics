"""Explicit inputs for local magnetic-field analysis, without an NLFFF solver.

SHARP components are observations on their native CEA grid. Imported volumes
are caller-supplied model results; reading one establishes no force-free,
solenoidal, convergence or observational-validation claim.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile
from zlib import error as CompressionError

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS, WcsError

from solar_toolkit._utils.numeric import numeric_data
from solar_toolkit.io import read_fits_data_header, resolve_fits_path
from solar_toolkit.time import normalize_time

__all__ = [
    "NLFFFGrid",
    "SharpCEABoundary",
    "read_nlfff_grid",
    "read_sharp_cea_boundary",
]


@dataclass(frozen=True)
class SharpCEABoundary:
    """Native CEA samples in G, ordered west/north/radial ``(Bp, -Bt, Br)``.

    ``field_G`` has shape ``(3, ny, nx)``; ``valid_mask`` is True only where
    all three components are finite. Individual missing samples remain NaN.
    ``header`` is an independent copy of the original Br header, retaining its
    CEA WCS. Local component directions do not turn that WCS into a Cartesian
    plane, and no projection or field extrapolation has been performed.
    """

    field_G: np.ndarray
    valid_mask: np.ndarray
    header: fits.Header
    metadata: dict[str, Any]

    @property
    def bx_G(self):
        """Westward component in G."""
        return self.field_G[0]

    @property
    def by_G(self):
        """Northward component in G."""
        return self.field_G[1]

    @property
    def bz_G(self):
        """Radially outward component in G."""
        return self.field_G[2]


@dataclass(frozen=True)
class NLFFFGrid:
    """Detached imported arrays in G/Mm; no solver or physical certification.

    Volume and optional reference use ``(component, z, y, x)`` axis order.
    ``spacing_Mm`` is positive ``[dx, dy, dz]``; optional ``observed_G`` is
    ``(3, ny, nx)``. Coordinates are local west/north/radial. Spacing alone
    supplies neither an absolute solar origin nor an observer/WCS transform.
    """

    volume_G: np.ndarray
    spacing_Mm: np.ndarray
    reference_G: np.ndarray | None
    observed_G: np.ndarray | None
    metadata: dict[str, Any]


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _source(path):
    before = _identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    _unchanged(path, before)
    return before, {
        "path": str(path),
        "size_bytes": before[2],
        "sha256": digest.hexdigest(),
    }


def _unchanged(path, identity):
    if _identity(path) != identity:
        raise RuntimeError(f"Magnetic input changed during access: {path}")


def _integer(header, name):
    value = header.get(name)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be an explicit non-negative integer")
    return int(value)


def _field_factor(unit, name):
    try:
        return u.Unit(unit).to(u.G)
    except (ValueError, TypeError, u.UnitConversionError) as exc:
        raise ValueError(f"{name} must declare magnetic-field units") from exc


def _record_time(value):
    text = str(value).strip()
    explicit_scale = any(
        text.upper().endswith(suffix)
        for suffix in ("_UTC", " UTC", "_TAI", " TAI", "Z")
    )
    return normalize_time(text, scale=None if explicit_scale else "tai")


def _component_identity(header, path, component):
    known = {"br", "bt", "bp"}
    filename = {token.lower() for token in path.name.split(".")} & known
    declared = str(header.get("SEGMENT", "")).strip().lower()
    if declared and declared != component.lower():
        raise ValueError(
            f"SEGMENT does not identify the supplied {component} component"
        )
    if filename and filename != {component.lower()}:
        raise ValueError(f"Filename contradicts the supplied {component} component")
    if not declared and not filename:
        raise ValueError(
            f"{component} identity requires SEGMENT or an exact filename component token"
        )
    return "SEGMENT" if declared else "filename"


def _cea_grid(header, shape):
    grid = WCS(header, naxis=2)
    if tuple(grid.wcs.ctype) != ("CRLN-CEA", "CRLT-CEA"):
        raise ValueError(
            "SHARP inputs must use a Carrington longitude/latitude CEA WCS"
        )
    scales = np.array([unit.to(u.deg) for unit in grid.wcs.cunit])
    matrix = grid.pixel_scale_matrix * scales[:, None]
    reference = grid.wcs.crval * scales
    reference[0] %= 360
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(reference).all()
        or not np.isfinite(grid.wcs.crpix).all()
        or abs(np.linalg.det(matrix)) < np.finfo(float).tiny
        or grid.has_distortion
    ):
        raise ValueError("SHARP CEA WCS must be finite, nonsingular and undistorted")
    ny, nx = shape
    pixels = np.array(
        [
            [0, 0],
            [nx - 1, 0],
            [0, ny - 1],
            [nx - 1, ny - 1],
            [(nx - 1) / 2, (ny - 1) / 2],
        ]
    )
    world = grid.all_pix2world(pixels, 0) * scales
    if not np.isfinite(world).all():
        raise ValueError(
            "SHARP CEA WCS does not map its image grid to finite coordinates"
        )
    # Unit vectors avoid the longitude wrap at 0/360 degrees. Canonical pole
    # values and actual world points also compare implicit projection defaults.
    lon, lat = np.deg2rad(world).T
    sphere = np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )
    return (
        grid.wcs.crpix.copy(),
        reference,
        matrix,
        np.array([grid.wcs.lonpole, grid.wcs.latpole]),
        sphere,
    )


def _same_grid(left, right):
    return all(
        np.shape(a) == np.shape(b) and np.allclose(a, b, rtol=0, atol=1e-10)
        for a, b in zip(left, right, strict=True)
    )


def read_sharp_cea_boundary(br_path, bt_path, bp_path) -> SharpCEABoundary:
    """Read one co-registered SHARP CEA record, preserving missing pixels.

    Each path may be FITS or gzip; a missing plain path uses only its same-name
    ``.gz`` counterpart. The first nonempty image is read via shared FITS I/O.
    HARPNUM and T_REC are required and must identify one record (T_REC is TAI
    unless it explicitly declares UTC). QUALITY is required and must be zero.
    SERIES, when present, must match the definitive ``hmi.sharp_cea_720s``
    series. A missing SERIES card is retained as unverified, never inferred
    from a filename. Component identity requires SEGMENT or an exact Br/Bt/Bp
    filename token; a known contradictory filename token is always rejected.
    Present T_OBS timestamps must agree with one another, independently of T_REC.

    Shapes and CEA WCS must agree; magnetic units convert explicitly to G.
    Nonfinite values become NaN, with no filling, cropping, binning or boundary
    preprocessing. All-invalid boundaries fail. Error/confidence maps are not
    inferred, so a zero QUALITY flag alone does not establish scientific
    suitability for an NLFFF reconstruction.
    """
    paths = [Path(resolve_fits_path(p)).resolve() for p in (br_path, bt_path, bp_path)]
    if len(set(paths)) != 3:
        raise ValueError("Br, Bt and Bp must refer to three distinct component files")
    sources = [_source(path) for path in paths]
    values, headers, grids, records, series, stamps = [], [], [], [], [], []
    observed_times, identities = [], []
    for component, path in zip(("Br", "Bt", "Bp"), paths, strict=True):
        data, header = read_fits_data_header(path, hdu_index=None)
        if data.ndim != 2 or min(data.shape) < 1:
            raise ValueError(f"{component} must be a nonempty two-dimensional image")
        harp = _integer(header, "HARPNUM")
        if harp == 0:
            raise ValueError("HARPNUM must be positive")
        if _integer(header, "QUALITY") != 0:
            raise ValueError(
                f"{component} QUALITY must be zero; flagged data require review"
            )
        if "T_REC" not in header:
            raise ValueError("SHARP record requires T_REC")
        stamp = _record_time(header["T_REC"])
        stamps.append(stamp)
        records.append((harp, stamp.tai_ns))
        if "T_OBS" in header:
            observed_times.append(_record_time(header["T_OBS"]))
        value = str(header.get("SERIES", "")).strip().lower()
        if value and value != "hmi.sharp_cea_720s":
            raise ValueError("Only definitive hmi.sharp_cea_720s SERIES is supported")
        series.append(value or None)
        identities.append(_component_identity(header, path, component))
        try:
            grids.append(_cea_grid(header, data.shape))
        except WcsError as exc:
            raise ValueError(f"{component} has an invalid SHARP CEA WCS") from exc
        factor = _field_factor(header.get("BUNIT"), f"{component} BUNIT")
        array = numeric_data(data)
        with np.errstate(over="ignore", invalid="ignore"):
            array *= factor
        array[~np.isfinite(array)] = np.nan
        values.append(array)
        headers.append(header.copy())
    if len(set(records)) != 1:
        raise ValueError("SHARP components must share HARPNUM and T_REC")
    if len({stamp.tai_ns for stamp in observed_times}) > 1:
        raise ValueError("Present SHARP T_OBS timestamps must agree")
    if any(v.shape != values[0].shape for v in values[1:]):
        raise ValueError("SHARP component image shapes disagree")
    if any(not _same_grid(grids[0], grid) for grid in grids[1:]):
        raise ValueError("SHARP components must share the same CEA WCS grid")
    field = np.stack((values[2], -values[1], values[0]))
    valid = np.isfinite(field).all(axis=0)
    if not valid.any():
        raise ValueError("SHARP boundary has no pixel with three valid components")
    for path, (identity, _) in zip(paths, sources, strict=True):
        _unchanged(path, identity)
    metadata = {
        "schema_version": 1,
        "frame": "local_west_north_radial",
        "array_axis_order": "component,y,x",
        "field_unit": "G",
        "component_order": ["Bp", "-Bt", "Br"],
        "component_identity_source": identities,
        "harpnum": records[0][0],
        "record_tai_ns": records[0][1],
        "record_utc": stamps[0].utc,
        "observation_utc": observed_times[0].utc if observed_times else None,
        "series": series,
        "series_verified": all(series),
        "quality": [0, 0, 0],
        "shape": list(field.shape),
        "valid_pixels": int(valid.sum()),
        "missing_pixels": int((~valid).sum()),
        "sources": {
            name: source
            for name, (_, source) in zip(("Br", "Bt", "Bp"), sources, strict=True)
        },
        "limitations": [
            "No error/confidence maps supplied",
            "No boundary preprocessing or extrapolation",
            "Local component directions retain native CEA sample positions",
        ],
    }
    return SharpCEABoundary(field, valid, headers[0], metadata)


def _finite_array(value, name, shape=None):
    if value.dtype.kind not in "iuf" or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} must be a real numerical array with shape {shape}")
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _json_constant(value):
    raise ValueError(f"Nonfinite JSON constant is unsupported: {value}")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON metadata key: {key}")
        result[key] = value
    return result


@contextmanager
def _npz_archive(source):
    try:
        loaded = np.load(source, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ValueError("NLFFF grid must be an NPZ archive")
        with loaded as data:
            yield data
    except (BadZipFile, EOFError, CompressionError) as exc:
        raise ValueError("Damaged NLFFF NPZ archive") from exc


def read_nlfff_grid(path) -> NLFFFGrid:
    """Read an explicit local model NPZ; never load pickle or run a solver.

    Required keys are ``volume`` (3,nz,ny,nx), ``spacing`` ([dx,dy,dz]) and
    ``metadata`` (scalar Unicode JSON). Metadata requires schema_version=1,
    frame='local_west_north_radial', array_axis_order='component,z,y,x',
    field_unit and length_unit. Optional ``reference`` matches volume;
    ``observed`` has shape (3,ny,nx). All arrays must be finite, real, nonempty,
    and non-object. Spacing must be strictly positive. All field arrays use
    the same declared field unit; outputs convert to G and Mm.

    Extra JSON metadata is retained as source_metadata but no other frame,
    axis convention, data-array key or implicit unit is inferred. This format
    contains no absolute position/observer transformation and cannot by itself
    establish registration to AIA or another solar image.
    """
    source = Path(path).expanduser().resolve()
    identity, provenance = _source(source)
    with _npz_archive(source) as data:
        keys = set(data.files)
        if (
            len(keys) != len(data.files)
            or not {"volume", "spacing", "metadata"} <= keys
            or keys - {"volume", "spacing", "metadata", "reference", "observed"}
        ):
            raise ValueError(
                "Unsupported NLFFF NPZ keys; volume, spacing and metadata are required"
            )
        text = data["metadata"]
        if text.shape != () or text.dtype.kind != "U":
            raise ValueError(
                "metadata must be scalar Unicode JSON, never an object array"
            )
        meta = json.loads(
            text.item(), parse_constant=_json_constant, object_pairs_hook=_json_object
        )
        json.dumps(meta, allow_nan=False)
        if (
            not isinstance(meta, dict)
            or type(meta.get("schema_version")) is not int
            or meta["schema_version"] != 1
        ):
            raise ValueError("Unsupported NLFFF metadata schema_version")
        if (
            meta.get("frame") != "local_west_north_radial"
            or meta.get("array_axis_order") != "component,z,y,x"
        ):
            raise ValueError("Unsupported NLFFF frame or array_axis_order")
        factor = _field_factor(meta.get("field_unit"), "field_unit")
        try:
            length_factor = u.Unit(meta.get("length_unit")).to(u.Mm)
        except (ValueError, TypeError, u.UnitConversionError) as exc:
            raise ValueError("length_unit must declare length units") from exc
        volume = _finite_array(data["volume"], "volume")
        if volume.ndim != 4 or volume.shape[0] != 3 or min(volume.shape[1:]) < 1:
            raise ValueError("volume must have shape (3,nz,ny,nx) with nonempty axes")
        spacing = _finite_array(data["spacing"], "spacing", (3,)) * length_factor
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing must be finite and strictly positive in Mm")
        arrays = [volume]
        for name, shape in (
            ("reference", volume.shape),
            ("observed", (3, *volume.shape[-2:])),
        ):
            arrays.append(
                _finite_array(data[name], name, shape) if name in keys else None
            )
        with np.errstate(over="ignore", invalid="ignore"):
            arrays = [None if array is None else array * factor for array in arrays]
        if any(array is not None and not np.isfinite(array).all() for array in arrays):
            raise ValueError("Magnetic unit conversion overflows float64")
    _unchanged(source, identity)
    return NLFFFGrid(
        arrays[0],
        spacing,
        arrays[1],
        arrays[2],
        metadata={
            "schema_version": 1,
            "frame": "local_west_north_radial",
            "array_axis_order": "component,z,y,x",
            "field_unit": "G",
            "length_unit": "Mm",
            "source_metadata": meta,
            "source": provenance,
            "model_validated": False,
            "limitations": [
                "Imported model only; no force-free or divergence validation",
                "No absolute solar origin or observer transform inferred",
            ],
        },
    )
