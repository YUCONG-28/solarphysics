"""Memory-conscious readers for four-file DART dynamic spectra.

``SpecDataIdB.fits`` already contains logarithmic Stokes I values in dB.  This
module preserves those values: display downsampling and narrowband extraction
only calculate finite arithmetic means and never apply another logarithm.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

__all__ = [
    "DartNarrowbandCurve",
    "DartNarrowbandResult",
    "DartSpectrogramFiles",
    "DartSpectrogramWindow",
    "discover_dart_spectrogram_files",
    "extract_dart_narrowband_lightcurves",
    "read_dart_spectrogram_window",
]

_LOGGER = logging.getLogger(__name__)
UTC = timezone.utc
_DEFAULT_CHUNK_MEMORY_MB = 64
_FILE_SUFFIXES = {
    "stokes_i_db": "specdataidb.fits",
    "stokes_v_over_i": "specdatavp.fits",
    "frequency": "specfrequency.fits",
    "time": "spectime.fits",
}


@dataclass(frozen=True)
class DartSpectrogramFiles:
    """Paths to one complete four-file DART spectrogram observation."""

    stokes_i_db: Path
    stokes_v_over_i: Path
    frequency: Path
    time: Path


@dataclass(frozen=True)
class DartSpectrogramWindow:
    """Downsampled Stokes I and V/I arrays with aligned physical axes."""

    stokes_i_db: np.ndarray
    stokes_v_over_i: np.ndarray
    frequency_mhz: np.ndarray
    time_utc: tuple[datetime, ...]


@dataclass(frozen=True)
class DartNarrowbandCurve:
    """One original-channel narrowband Stokes I dB light curve."""

    center_frequency_mhz: float
    bandwidth_mhz: float
    requested_frequency_range_mhz: tuple[float, float]
    sampled_frequency_range_mhz: tuple[float, float]
    channel_count: int
    stokes_i_db: np.ndarray


@dataclass(frozen=True)
class DartNarrowbandResult:
    """Narrowband curves sharing one UTC time axis."""

    time_utc: tuple[datetime, ...]
    curves: tuple[DartNarrowbandCurve, ...]


def discover_dart_spectrogram_files(
    directory: str | Path,
) -> DartSpectrogramFiles:
    """Discover exactly one file for each standard DART FITS suffix."""

    folder = Path(directory).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"DART spectrogram directory does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"DART spectrogram path is not a directory: {folder}")

    files = [path for path in folder.iterdir() if path.is_file()]
    matches: dict[str, Path] = {}
    for field_name, suffix in _FILE_SUFFIXES.items():
        candidates = sorted(
            (path for path in files if path.name.casefold().endswith(suffix)),
            key=lambda path: path.name.casefold(),
        )
        if not candidates:
            raise FileNotFoundError(
                f"Missing DART FITS file ending with {suffix!r} in {folder}"
            )
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(
                f"Multiple DART FITS files end with {suffix!r} in {folder}: {names}"
            )
        matches[field_name] = candidates[0]

    return DartSpectrogramFiles(**matches)


def read_dart_spectrogram_window(
    files_or_directory: DartSpectrogramFiles | str | Path,
    *,
    frequency_range_mhz: tuple[float, float] | None = None,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    max_frequency_samples: int = 1600,
    max_time_samples: int = 1600,
    chunk_memory_mb: int = _DEFAULT_CHUNK_MEMORY_MB,
) -> DartSpectrogramWindow:
    """Read and downsample an aligned DART display window using FITS memmaps.

    Stokes I is already logarithmic dB data.  The block means calculated here
    are direct finite means of those supplied values.
    """

    files = _coerce_files(files_or_directory)
    frequency_mhz, time_utc, data_shape = _read_and_validate_metadata(files)
    fi0, fi1 = _frequency_index_range(frequency_mhz, frequency_range_mhz)
    ti0, ti1 = _time_index_range(time_utc, time_range_utc)

    max_frequency_samples = _positive_integer(
        max_frequency_samples, "max_frequency_samples"
    )
    max_time_samples = _positive_integer(max_time_samples, "max_time_samples")
    chunk_memory_mb = _positive_integer(chunk_memory_mb, "chunk_memory_mb")

    n_frequency = fi1 - fi0 + 1
    n_time = ti1 - ti0 + 1
    frequency_bin = max(1, math.ceil(n_frequency / max_frequency_samples))
    time_bin = max(1, math.ceil(n_time / max_time_samples))
    _LOGGER.info(
        "Reading DART display window shape=%s slice=(%d:%d, %d:%d) bins=(%d, %d)",
        data_shape,
        fi0,
        fi1,
        ti0,
        ti1,
        frequency_bin,
        time_bin,
    )

    stokes_i_db = _read_downsampled_plane(
        files.stokes_i_db,
        fi0=fi0,
        fi1=fi1,
        ti0=ti0,
        ti1=ti1,
        frequency_bin=frequency_bin,
        time_bin=time_bin,
        chunk_memory_mb=chunk_memory_mb,
    )
    stokes_v_over_i = _read_downsampled_plane(
        files.stokes_v_over_i,
        fi0=fi0,
        fi1=fi1,
        ti0=ti0,
        ti1=ti1,
        frequency_bin=frequency_bin,
        time_bin=time_bin,
        chunk_memory_mb=chunk_memory_mb,
    )
    frequency_out = _block_mean_axis(
        frequency_mhz[fi0 : fi1 + 1], frequency_bin
    ).astype(np.float64, copy=False)
    time_out = _block_mean_times(time_utc[ti0 : ti1 + 1], time_bin)

    if frequency_out[0] > frequency_out[-1]:
        frequency_out = frequency_out[::-1].copy()
        stokes_i_db = stokes_i_db[::-1, :].copy()
        stokes_v_over_i = stokes_v_over_i[::-1, :].copy()

    return DartSpectrogramWindow(
        stokes_i_db=stokes_i_db,
        stokes_v_over_i=stokes_v_over_i,
        frequency_mhz=frequency_out,
        time_utc=time_out,
    )


def extract_dart_narrowband_lightcurves(
    files_or_directory: DartSpectrogramFiles | str | Path,
    center_frequencies_mhz: Sequence[float],
    bandwidth_mhz: float,
    *,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
) -> DartNarrowbandResult:
    """Average original IdB channels inside each requested total bandwidth.

    No linear-domain conversion, smoothing, normalization, ``10*log10`` or
    additional ``log10`` operation is performed.
    """

    files = _coerce_files(files_or_directory)
    frequency_mhz, time_utc, _ = _read_and_validate_metadata(files)
    centers = _validate_centers(center_frequencies_mhz)
    bandwidth = float(bandwidth_mhz)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth_mhz must be a finite value greater than zero")
    ti0, ti1 = _time_index_range(time_utc, time_range_utc)
    observed_min = float(np.min(frequency_mhz))
    observed_max = float(np.max(frequency_mhz))

    curve_specs: list[tuple[float, float, float, np.ndarray]] = []
    half_width = bandwidth / 2.0
    for center in centers:
        requested_low = center - half_width
        requested_high = center + half_width
        if requested_low < observed_min or requested_high > observed_max:
            raise ValueError(
                "Requested narrowband is outside the observed frequency range: "
                f"{requested_low:.9g}-{requested_high:.9g} MHz versus "
                f"{observed_min:.9g}-{observed_max:.9g} MHz"
            )
        indices = np.flatnonzero(
            (frequency_mhz >= requested_low) & (frequency_mhz <= requested_high)
        )
        if not indices.size:
            raise ValueError(
                "Requested narrowband contains no sampled frequency channel: "
                f"center={center:.9g} MHz, bandwidth={bandwidth:.9g} MHz"
            )
        curve_specs.append((center, requested_low, requested_high, indices))

    curves: list[DartNarrowbandCurve] = []
    with fits.open(files.stokes_i_db, memmap=True) as hdul:
        plane = hdul[0].data
        for center, requested_low, requested_high, indices in curve_specs:
            values = _finite_frequency_mean(
                plane,
                frequency_indices=indices,
                ti0=ti0,
                ti1=ti1,
                chunk_memory_mb=_DEFAULT_CHUNK_MEMORY_MB,
            )
            sampled = frequency_mhz[indices]
            curves.append(
                DartNarrowbandCurve(
                    center_frequency_mhz=center,
                    bandwidth_mhz=bandwidth,
                    requested_frequency_range_mhz=(
                        float(requested_low),
                        float(requested_high),
                    ),
                    sampled_frequency_range_mhz=(
                        float(np.min(sampled)),
                        float(np.max(sampled)),
                    ),
                    channel_count=int(indices.size),
                    stokes_i_db=values,
                )
            )

    return DartNarrowbandResult(
        time_utc=tuple(time_utc[ti0 : ti1 + 1]),
        curves=tuple(curves),
    )


def _coerce_files(
    files_or_directory: DartSpectrogramFiles | str | Path,
) -> DartSpectrogramFiles:
    if isinstance(files_or_directory, DartSpectrogramFiles):
        values = {
            name: Path(getattr(files_or_directory, name)).expanduser().resolve()
            for name in _FILE_SUFFIXES
        }
        for name, path in values.items():
            if not path.is_file():
                raise FileNotFoundError(f"DART {name} FITS file does not exist: {path}")
        return DartSpectrogramFiles(**values)
    return discover_dart_spectrogram_files(files_or_directory)


def _read_and_validate_metadata(
    files: DartSpectrogramFiles,
) -> tuple[np.ndarray, tuple[datetime, ...], tuple[int, int]]:
    frequency_mhz = _read_frequency_axis(files.frequency)
    time_utc = _read_time_axis(files.time)
    expected_shape = (frequency_mhz.size, len(time_utc))
    shapes = {
        "SpecDataIdB": _primary_data_shape(files.stokes_i_db),
        "SpecDataVP": _primary_data_shape(files.stokes_v_over_i),
    }
    for label, shape in shapes.items():
        if shape != expected_shape:
            raise ValueError(
                f"{label} shape {shape} does not match frequency/time axes "
                f"{expected_shape}"
            )
    return frequency_mhz, time_utc, expected_shape


def _read_frequency_axis(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"DART frequency FITS contains no primary data: {path}")
        shape = tuple(int(value) for value in data.shape)
        if len(shape) == 1:
            values = np.asarray(data, dtype=np.float64)
        elif len(shape) == 2 and 1 in shape:
            values = np.asarray(data, dtype=np.float64).reshape(-1)
        else:
            raise ValueError(
                "DART frequency FITS must be a vector or singleton 2-D array; "
                f"found shape {shape} in {path}"
            )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"DART frequency axis must contain finite values: {path}")
    differences = np.diff(values)
    if differences.size and not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"DART frequency axis must be strictly monotonic: {path}")
    return values


def _read_time_axis(path: Path) -> tuple[datetime, ...]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"DART time FITS contains no primary data: {path}")
        values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] == 0:
        raise ValueError(
            "DART time FITS must have shape (time, 6) for "
            f"year/month/day/hour/minute/second; found {values.shape} in {path}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"DART time axis contains non-finite components: {path}")

    parsed: list[datetime] = []
    for row_index, row in enumerate(values):
        try:
            year = _integer_time_component(row[0], "year")
            if 0 <= year < 100:
                year += 2000
            month = _integer_time_component(row[1], "month")
            day = _integer_time_component(row[2], "day")
            hour = _integer_time_component(row[3], "hour")
            minute = _integer_time_component(row[4], "minute")
            second = float(row[5])
            if second < 0 or second >= 61:
                raise ValueError(f"second must be in [0, 61), found {second!r}")
            value = datetime(
                year,
                month,
                day,
                hour,
                minute,
                tzinfo=UTC,
            ) + timedelta(seconds=second)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"Invalid DART time row {row_index} in {path}: {row.tolist()} ({exc})"
            ) from exc
        parsed.append(value)

    timestamps = np.asarray([value.timestamp() for value in parsed])
    if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"DART UTC time axis must be strictly increasing: {path}")
    return tuple(parsed)


def _integer_time_component(value: float, label: str) -> int:
    rounded = int(round(float(value)))
    if not math.isclose(float(value), rounded, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{label} must be an integer, found {value!r}")
    return rounded


def _primary_data_shape(path: Path) -> tuple[int, int]:
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"DART data FITS contains no primary data: {path}")
        shape = tuple(int(value) for value in data.shape)
    if len(shape) != 2:
        raise ValueError(f"DART data FITS must be 2-D; found shape {shape} in {path}")
    return shape


def _frequency_index_range(
    frequency_mhz: np.ndarray,
    requested: tuple[float, float] | None,
) -> tuple[int, int]:
    observed_min = float(np.min(frequency_mhz))
    observed_max = float(np.max(frequency_mhz))
    if requested is None:
        low, high = observed_min, observed_max
    else:
        if len(requested) != 2:
            raise ValueError("frequency_range_mhz must contain exactly two values")
        low, high = (float(requested[0]), float(requested[1]))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("frequency_range_mhz values must be finite")
        if low > high:
            raise ValueError("frequency_range_mhz start must not exceed its end")
        if low < observed_min or high > observed_max:
            raise ValueError(
                "Requested frequency display range is outside the observation: "
                f"{low:.9g}-{high:.9g} MHz versus "
                f"{observed_min:.9g}-{observed_max:.9g} MHz"
            )
    indices = np.flatnonzero((frequency_mhz >= low) & (frequency_mhz <= high))
    if not indices.size:
        raise ValueError(f"No frequency samples fall inside {low:.9g}-{high:.9g} MHz")
    return int(indices[0]), int(indices[-1])


def _time_index_range(
    time_utc: tuple[datetime, ...],
    requested: tuple[datetime | str, datetime | str] | None,
) -> tuple[int, int]:
    observed_start = time_utc[0]
    observed_end = time_utc[-1]
    if requested is None:
        start, end = observed_start, observed_end
    else:
        if len(requested) != 2:
            raise ValueError("time_range_utc must contain exactly two values")
        start = _coerce_utc_datetime(requested[0])
        end = _coerce_utc_datetime(requested[1])
        if start > end:
            raise ValueError("time_range_utc start must not exceed its end")
        if start < observed_start or end > observed_end:
            raise ValueError(
                "Requested UTC time range is outside the observation: "
                f"{start.isoformat()} to {end.isoformat()} versus "
                f"{observed_start.isoformat()} to {observed_end.isoformat()}"
            )
    timestamps = np.asarray([value.timestamp() for value in time_utc])
    indices = np.flatnonzero(
        (timestamps >= start.timestamp()) & (timestamps <= end.timestamp())
    )
    if not indices.size:
        raise ValueError(
            "No DART time samples fall inside the requested UTC time range"
        )
    return int(indices[0]), int(indices[-1])


def _coerce_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("UTC time range values must not be blank")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid UTC datetime value: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_integer(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _block_mean_axis(values: np.ndarray, bin_size: int) -> np.ndarray:
    starts = np.arange(0, values.size, bin_size, dtype=int)
    return np.asarray(
        [float(np.mean(values[start : start + bin_size])) for start in starts],
        dtype=np.float64,
    )


def _block_mean_times(
    values: tuple[datetime, ...], bin_size: int
) -> tuple[datetime, ...]:
    timestamps = np.asarray([value.timestamp() for value in values])
    starts = np.arange(0, timestamps.size, bin_size, dtype=int)
    return tuple(
        datetime.fromtimestamp(
            float(np.mean(timestamps[start : start + bin_size])), tz=UTC
        )
        for start in starts
    )


def _read_downsampled_plane(
    path: Path,
    *,
    fi0: int,
    fi1: int,
    ti0: int,
    ti1: int,
    frequency_bin: int,
    time_bin: int,
    chunk_memory_mb: int,
) -> np.ndarray:
    n_frequency = fi1 - fi0 + 1
    n_time = ti1 - ti0 + 1
    n_frequency_out = math.ceil(n_frequency / frequency_bin)
    n_time_out = math.ceil(n_time / time_bin)
    output = np.empty((n_frequency_out, n_time_out), dtype=np.float32)

    budget_bytes = int(chunk_memory_mb) * 1024 * 1024
    max_raw_cells = max(1, budget_bytes // 16)
    raw_columns = max(time_bin, max_raw_cells // max(1, n_frequency))
    output_columns_per_chunk = max(1, raw_columns // time_bin)

    with fits.open(path, memmap=True) as hdul:
        plane = hdul[0].data
        for out_t0 in range(0, n_time_out, output_columns_per_chunk):
            out_t1 = min(n_time_out, out_t0 + output_columns_per_chunk)
            raw_t0 = out_t0 * time_bin
            raw_t1 = min(n_time, out_t1 * time_bin)
            chunk = np.array(
                plane[fi0 : fi1 + 1, ti0 + raw_t0 : ti0 + raw_t1],
                dtype=np.float32,
                copy=True,
            )
            reduced = _finite_block_mean(
                chunk,
                frequency_bin=frequency_bin,
                time_bin=time_bin,
            )
            output[:, out_t0:out_t1] = reduced
    return output


def _finite_block_mean(
    values: np.ndarray,
    *,
    frequency_bin: int,
    time_bin: int,
) -> np.ndarray:
    n_frequency, n_time = values.shape
    n_frequency_out = math.ceil(n_frequency / frequency_bin)
    n_time_out = math.ceil(n_time / time_bin)
    padded_shape = (
        n_frequency_out * frequency_bin,
        n_time_out * time_bin,
    )
    padded = np.zeros(padded_shape, dtype=np.float32)
    padded[:n_frequency, :n_time] = values
    finite = np.zeros(padded_shape, dtype=bool)
    finite[:n_frequency, :n_time] = np.isfinite(values)
    padded[~finite] = 0.0

    reshaped = padded.reshape(
        n_frequency_out,
        frequency_bin,
        n_time_out,
        time_bin,
    )
    finite_reshaped = finite.reshape(
        n_frequency_out,
        frequency_bin,
        n_time_out,
        time_bin,
    )
    sums = np.sum(reshaped, axis=(1, 3), dtype=np.float64)
    counts = np.sum(finite_reshaped, axis=(1, 3), dtype=np.int64)
    result = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=result, where=counts > 0, casting="unsafe")
    return result


def _validate_centers(values: Sequence[float]) -> tuple[float, ...]:
    try:
        centers = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("center_frequencies_mhz must contain numeric values") from exc
    if not centers:
        raise ValueError("At least one center frequency is required")
    if len(centers) > 20:
        raise ValueError("At most 20 center frequencies may be requested")
    if not all(np.isfinite(value) for value in centers):
        raise ValueError("Center frequencies must be finite")
    if len(set(centers)) != len(centers):
        raise ValueError("Center frequencies must not contain duplicates")
    return centers


def _finite_frequency_mean(
    plane,
    *,
    frequency_indices: np.ndarray,
    ti0: int,
    ti1: int,
    chunk_memory_mb: int,
) -> np.ndarray:
    first = int(frequency_indices[0])
    last = int(frequency_indices[-1])
    expected = np.arange(first, last + 1)
    if not np.array_equal(frequency_indices, expected):
        raise ValueError("DART narrowband frequency channels must be contiguous")

    n_frequency = int(frequency_indices.size)
    n_time = ti1 - ti0 + 1
    output = np.full(n_time, np.nan, dtype=np.float64)
    bytes_per_time = max(1, n_frequency * np.dtype(np.float32).itemsize)
    columns_per_chunk = max(1, int(chunk_memory_mb * 1024 * 1024 // bytes_per_time))
    for local_t0 in range(0, n_time, columns_per_chunk):
        local_t1 = min(n_time, local_t0 + columns_per_chunk)
        chunk = np.array(
            plane[first : last + 1, ti0 + local_t0 : ti0 + local_t1],
            dtype=np.float64,
            copy=True,
        )
        finite = np.isfinite(chunk)
        sums = np.sum(np.where(finite, chunk, 0.0), axis=0, dtype=np.float64)
        counts = np.sum(finite, axis=0, dtype=np.int64)
        np.divide(
            sums,
            counts,
            out=output[local_t0:local_t1],
            where=counts > 0,
        )
    return output
