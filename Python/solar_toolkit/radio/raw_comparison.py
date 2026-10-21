"""Independent, provenance-preserving image matches for raw comparison figures.

Observation timestamps are read only from explicit observation-time keywords.
Neither a file creation DATE nor a filename is used to invent a missing time.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .muser import hpc_grid
from .muser_comparison import FrameRef, load_dart_sum, pair_dart

__all__ = [
    "PreparedImage",
    "observation_seconds",
    "prepare_muser",
    "prepare_aia",
    "prepare_dart",
]


@dataclass(frozen=True)
class PreparedImage:
    data: np.ndarray | None
    header: fits.Header | None
    metadata: dict


@dataclass(frozen=True)
class _Entry:
    path: Path
    seconds: float
    time_source: str
    hdu: int
    header: fits.Header
    stamp: object = None


def observation_seconds(header):
    """Legacy seconds view of the exact canonical FITS observation timestamp."""
    from .time_matching import observation_stamp

    try:
        stamp = observation_stamp(header)
    except ValueError as exc:
        raise ValueError(
            "No reliable DATE-OBS, DATE_OBS or T_OBS observation timestamp"
        ) from exc
    return float(Time(stamp.utc, scale="utc").unix), stamp.source


def _utc(seconds):
    return Time(float(seconds), format="unix", scale="utc", precision=6).isot


def _header(path):
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        for i, hdu in enumerate(hdus):
            h = hdu.header
            if (
                int(h.get("NAXIS", 0)) >= 2
                and int(h.get("NAXIS1", 0)) > 0
                and int(h.get("NAXIS2", 0)) > 0
            ):
                combined = hdus[0].header.copy()
                combined.update(h)
                return i, combined
    raise ValueError("No image HDU")


def _index(directory):
    directory = Path(directory)
    paths = (
        [directory]
        if directory.is_file()
        else sorted(
            p
            for p in directory.rglob("*")
            if p.is_file()
            and p.name.lower().endswith((".fits", ".fit", ".fts", ".fits.gz"))
        )
    )
    entries, excluded = [], []
    for path in paths:
        try:
            hdu, header = _header(path)
            from .time_matching import observation_stamp

            stamp = observation_stamp(header)
            seconds = float(Time(stamp.utc, scale="utc").unix)
            entries.append(_Entry(path, seconds, stamp.source, hdu, header, stamp))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            excluded.append({"path": str(path), "reason": str(exc)})
    entries.sort(key=lambda entry: (entry.stamp.tai_ns, str(entry.path)))
    return entries, {"indexed_files": len(entries), "excluded_files": excluded}


def _metadata(target, entries, **kwargs):
    from solar_toolkit.time import TimeStamp, normalize_time

    stamps = [
        entry.stamp or normalize_time(entry.seconds, format="unix") for entry in entries
    ]
    reference = normalize_time(target, format="unix")
    observed = TimeStamp.from_tai_ns(
        sum(stamp.tai_ns for stamp in stamps) // len(stamps)
    )
    delta_ns = observed.tai_ns - reference.tai_ns
    return {
        "status": "ok",
        "reference_utc": reference.utc + "Z",
        "utc": observed.utc + "Z",
        "time_tai_ns": observed.tai_ns,
        "delta_seconds": delta_ns / 1_000_000_000,
        "delta_ns": delta_ns,
        "paths": [str(entry.path) for entry in entries],
        "time_source": [entry.time_source for entry in entries],
        "source_times": [
            {
                "utc": stamp.utc + "Z",
                "time_tai_ns": stamp.tai_ns,
                "delta_ns": stamp.tai_ns - reference.tai_ns,
                "delta_seconds": (stamp.tai_ns - reference.tai_ns) / 1_000_000_000,
                "keyword": entry.time_source,
                "path": str(entry.path),
            }
            for entry, stamp in zip(entries, stamps, strict=True)
        ],
        **kwargs,
    }


def _missing(target, reason, **kwargs):
    return PreparedImage(
        None,
        None,
        {
            "status": "no_data",
            "reason": reason,
            "reference_utc": _utc(target),
            "paths": [],
            **kwargs,
        },
    )


def _nearest(entries, target, tolerance):
    from solar_toolkit.time import normalize_time
    from solar_toolkit.time.precision import seconds_to_ns

    target_ns = normalize_time(target, format="unix").tai_ns
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not entries:
        return None
    entry = min(entries, key=lambda item: abs(item.stamp.tai_ns - target_ns))
    return (
        entry
        if abs(entry.stamp.tai_ns - target_ns) <= seconds_to_ns(tolerance)
        else None
    )


def _validate_frequency(frequency_mhz):
    if not np.isfinite(frequency_mhz) or frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be finite and positive")


def _frequency_candidates(entries, frequency_mhz, diagnostics):
    candidates = []
    for entry in entries:
        try:
            frequency, _ = _frequency(entry.header, entry.path)
            if not np.isfinite(frequency) or abs(frequency - frequency_mhz) > 0.01:
                raise ValueError(
                    "DART image frequency does not match the selected band"
                )
            candidates.append(entry)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostics["excluded_files"].append(
                {"path": str(entry.path), "reason": str(exc)}
            )
    return candidates


def _validate_image(data, header):
    if data.ndim != 2 or not np.any(np.isfinite(data)):
        raise ValueError("A two-dimensional image with finite pixels is required")
    # Validate the actual WCS without interpolating or resampling image pixels.
    grid = hpc_grid(header, (2, 2))
    if not all(np.isfinite(part).all() for part in grid):
        raise ValueError("Invalid helioprojective coordinates")
    required = (
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
    )
    if any(key not in header for key in required):
        raise ValueError("Incomplete helioprojective WCS")
    if not (
        all(k in header for k in ("CDELT1", "CDELT2"))
        or all(k in header for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
    ):
        raise ValueError("Missing explicit helioprojective pixel scale")
    matrix = WCS(header, naxis=2).pixel_scale_matrix
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ValueError("Singular helioprojective WCS")


def _plane(entry):
    with fits.open(entry.path, memmap=False) as hdus:
        data = np.asarray(hdus[entry.hdu].data, dtype=float).squeeze()
    _validate_image(data, entry.header)
    return data, entry.header.copy()


def _frequency(header, path):
    # A pixel's FREQ coordinate takes precedence over a reference/rest-frequency
    # shortcut. Even a singleton axis need not have CRPIX=1.
    for axis in range(3, int(header.get("NAXIS", 2)) + 1):
        if str(header.get(f"CTYPE{axis}", "")).upper().startswith("FREQ"):
            value = float(header[f"CRVAL{axis}"]) + (
                1 - float(header.get(f"CRPIX{axis}", 1))
            ) * float(header.get(f"CDELT{axis}", 1))
            unit = str(header.get(f"CUNIT{axis}", "Hz")).lower()
            factor = {"hz": 1e-6, "khz": 0.001, "mhz": 1.0, "ghz": 1000.0}.get(unit)
            if factor is None:
                raise ValueError(f"Unrecognized frequency unit: {unit}")
            return value * factor, f"FREQ axis {axis} at first pixel"
    for key in ("FREQ", "FREQUENCY", "RESTFRQ", "RESTFREQ"):
        if key in header:
            value = float(header[key])
            unit = str(
                header.get(
                    "FREQUNIT", "Hz" if key.startswith("REST") or value > 1e5 else "MHz"
                )
            ).lower()
            factor = {"hz": 1e-6, "khz": 0.001, "mhz": 1.0, "ghz": 1000.0}.get(unit)
            if factor is None:
                raise ValueError(f"Unrecognized frequency unit: {unit}")
            return value * factor, key
    for part in reversed(Path(path).parts):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)MHz", part, re.IGNORECASE)
        if match:
            return float(match[1]), "frequency_directory"
    raise ValueError("Missing observing frequency")


def prepare_muser(directory, target_seconds, *, frequency_mhz, tolerance=0.25):
    """Select one nominal-I frequency plane, retaining raw pixels and beam."""
    _validate_frequency(frequency_mhz)
    directory = Path(directory)
    if (directory / "img_fits").is_dir():
        directory /= "img_fits"
    entries, diagnostics = _index(directory)
    entry = _nearest(entries, target_seconds, tolerance)
    if entry is None:
        return _missing(
            target_seconds,
            "No MUSER image within the time tolerance",
            diagnostics=diagnostics,
        )
    try:
        h = entry.header
        if str(h.get("CTYPE4", "")).strip().upper() != "STOKES":
            raise ValueError("MUSER requires an explicit Stokes axis")
        if any(k not in h for k in ("CRVAL4", "CRPIX4", "CDELT4")):
            raise ValueError("MUSER Stokes reference metadata is incomplete")
        code = float(h["CRVAL4"]) + (1 - float(h["CRPIX4"])) * float(h["CDELT4"])
        if not np.isclose(code, 1.0, rtol=0, atol=1e-8):
            raise ValueError(f"MUSER first-plane Stokes code is {code:g}, not I=1")
        with fits.open(entry.path, memmap=False) as hdus:
            array = hdus[entry.hdu].data
            if array.ndim != 5 or array.shape[1:3] != (1, 1):
                raise ValueError("Expected MUSER (frequency, 1, 1, y, x) image")
            frequency = np.asarray(hdus["FMHZ"].data, dtype=float).ravel()
            if len(frequency) != array.shape[0] or not np.isfinite(frequency).all():
                raise ValueError("Invalid MUSER frequency planes")
            i = int(np.argmin(abs(frequency - frequency_mhz)))
            if abs(frequency[i] - frequency_mhz) > 5:
                raise ValueError(
                    "No MUSER plane within 5 MHz of the selected frequency"
                )
            data = np.array(array[i, 0, 0], dtype=float)
            beam = {
                key: float(np.asarray(hdus[key].data).ravel()[i])
                for key in ("BMAJ", "BMIN", "BPA")
            }
        _validate_image(data, h)
        header = WCS(h, naxis=2).to_header()
        for key in (
            "DATE-OBS",
            "HGLN_OBS",
            "HGLT_OBS",
            "DSUN_OBS",
            "RSUN_OBS",
            "RSUN_REF",
            "BUNIT",
            "TELESCOP",
        ):
            if key in h:
                header[key] = h[key]
        header.update(beam)
        _validate_image(data, header)
        metadata = _metadata(
            target_seconds,
            [entry],
            frequency_mhz=float(frequency[i]),
            frequency_source="FMHZ",
            requested_frequency_mhz=float(frequency_mhz),
            unit=str(h.get("BUNIT", "unspecified")),
            nominal_polarization="file-labelled Stokes I; upstream synthesis unverified",
            stokes_axis={
                key: h[key] for key in ("CTYPE4", "CRVAL4", "CRPIX4", "CDELT4")
            },
            first_plane_stokes_code=code,
            beam=beam,
            diagnostics=diagnostics,
        )
        return PreparedImage(data, header, metadata)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return _missing(
            target_seconds,
            str(exc),
            diagnostics=diagnostics,
            rejected_paths=[str(entry.path)],
        )


def prepare_aia(directory, target_seconds, wavelength=171, tolerance=12.0):
    """Read the independently nearest AIA image without changing its pixel grid."""
    entries, diagnostics = _index(directory)
    valid = []
    for entry in entries:
        try:
            if float(entry.header.get("WAVELNTH", -1)) == wavelength:
                valid.append(entry)
        except (TypeError, ValueError):
            diagnostics["excluded_files"].append(
                {"path": str(entry.path), "reason": "Invalid AIA wavelength"}
            )
    entries = valid
    entry = _nearest(entries, target_seconds, tolerance)
    if entry is None:
        return _missing(
            target_seconds,
            f"No AIA {wavelength} image within the time tolerance",
            diagnostics=diagnostics,
        )
    try:
        data, header = _plane(entry)
        return PreparedImage(
            data,
            header,
            _metadata(
                target_seconds,
                [entry],
                wavelength_angstrom=int(wavelength),
                unit=str(header.get("BUNIT", "unspecified")),
                diagnostics=diagnostics,
            ),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _missing(
            target_seconds,
            str(exc),
            diagnostics=diagnostics,
            rejected_paths=[str(entry.path)],
        )


def prepare_dart(
    directory,
    target_seconds,
    mode="I",
    *,
    frequency_mhz,
    tolerance=0.25,
    pair_tolerance=0.1,
):
    """Prepare native I or a strict, coordinate-checked LL+RR pair independently."""
    _validate_frequency(frequency_mhz)
    if not np.isfinite(pair_tolerance) or not 0 <= pair_tolerance <= 0.1:
        raise ValueError("pair_tolerance must be finite and between 0 and 0.1 seconds")
    directory = Path(directory)
    if mode not in ("I", "RR+LL"):
        raise ValueError("DART mode must be I or RR+LL")
    if mode == "I":
        folder = directory / "I" if (directory / "I").is_dir() else directory
        entries, diagnostics = _index(folder)
        entries = _frequency_candidates(entries, frequency_mhz, diagnostics)
        entry = _nearest(entries, target_seconds, tolerance)
        selected = [] if entry is None else [entry]
    else:
        left, dl = _index(directory / "LL")
        right, dr = _index(directory / "RR")
        diagnostics = {"LL": dl, "RR": dr}
        left = _frequency_candidates(left, frequency_mhz, dl)
        right = _frequency_candidates(right, frequency_mhz, dr)
        # Filter each original stream first; excluded frames must not consume
        # eligible opposite-polarization observations during pairing.
        left = [entry for entry in left if _nearest([entry], target_seconds, tolerance)]
        right = [
            entry for entry in right if _nearest([entry], target_seconds, tolerance)
        ]
        pairs = pair_dart(
            [FrameRef(e.seconds, (e.path,), stamp=e.stamp) for e in left],
            [FrameRef(e.seconds, (e.path,), stamp=e.stamp) for e in right],
            pair_tolerance,
        )
        lookup = {e.path: e for e in left + right}
        nearest = (
            min(pairs, key=lambda p: abs(p.seconds - target_seconds)) if pairs else None
        )
        selected = [] if nearest is None else [lookup[p] for p in nearest.paths]
    if not selected:
        return _missing(
            target_seconds,
            f"No DART {mode} match within the time tolerance",
            diagnostics=diagnostics,
        )
    try:
        frequencies = [_frequency(e.header, e.path) for e in selected]
        if any(
            not np.isfinite(f) or abs(f - frequency_mhz) > 0.01 for f, _ in frequencies
        ):
            raise ValueError("DART image frequency does not match the selected band")
        if mode == "I":
            entry = selected[0]
            if any(
                part.upper() in ("LL", "RR", "LCP", "RCP") for part in entry.path.parts
            ):
                raise ValueError(
                    "A single circular-polarization image cannot be used as native I"
                )
            labelled_i = "I" in [part.upper() for part in entry.path.parent.parts]
            for axis in range(3, int(entry.header.get("NAXIS", 2)) + 1):
                if str(entry.header.get(f"CTYPE{axis}", "")).upper() == "STOKES":
                    code = float(entry.header.get(f"CRVAL{axis}", np.nan)) + (
                        1 - float(entry.header.get(f"CRPIX{axis}", 1))
                    ) * float(entry.header.get(f"CDELT{axis}", 1))
                    if not np.isclose(code, 1):
                        raise ValueError("Native DART Stokes axis does not label I")
                    labelled_i = True
            if not labelled_i:
                raise ValueError(
                    "DART native I requires an I product directory or explicit I Stokes axis"
                )
            data, header = _plane(entry)
        else:
            for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
                if selected[0].header.get(key) != selected[1].header.get(key):
                    raise ValueError(f"LL/RR metadata mismatch: {key}")
            # Reuse the existing sum reader's shape, WCS, unit and polarization checks.
            ref = FrameRef(
                float(np.mean([e.seconds for e in selected])),
                tuple(e.path for e in selected),
            )
            summed = load_dart_sum(ref)
            data, header = summed.image, summed.header.copy()
            _validate_image(data, header)
        return PreparedImage(
            data,
            header,
            _metadata(
                target_seconds,
                selected,
                frequency_mhz=float(frequencies[0][0]),
                frequency_source=[source for _, source in frequencies],
                unit=str(header.get("BUNIT", "unspecified")),
                nominal_polarization=(
                    "native I product" if mode == "I" else "RR+LL (no factor of 1/2)"
                ),
                pair_delta_seconds=abs(selected[0].seconds - selected[-1].seconds),
                diagnostics=diagnostics,
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return _missing(
            target_seconds,
            str(exc),
            diagnostics=diagnostics,
            rejected_paths=[str(e.path) for e in selected],
        )
