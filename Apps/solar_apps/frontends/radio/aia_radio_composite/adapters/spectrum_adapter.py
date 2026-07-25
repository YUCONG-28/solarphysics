"""Normalize existing DART and CSO readers into the frontend spectrum schema."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from solar_toolkit.radio.cso import CSOSpectrogram, read_cso_spectrogram
from solar_toolkit.radio.dart_spectrogram import (
    discover_dart_spectrogram_files,
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)

from ..models import (
    CompositeRequest,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumWindow,
)

__all__ = [
    "load_cso_spectrum_window",
    "load_dart_spectrum_window",
    "load_spectrum_window",
    "extract_spectrum_flux_curve",
    "extract_spectrum_flux_curves",
]


def extract_spectrum_flux_curve(
    request: CompositeRequest,
    band: SpectrumBand,
    *,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
) -> SpectrumFluxCurve:
    """Extract one native-unit finite-channel-mean CSO or DART curve."""

    return extract_spectrum_flux_curves(
        request,
        (band,),
        time_range_utc=time_range_utc,
        cso_polarization=cso_polarization,
    )[0]


def extract_spectrum_flux_curves(
    request: CompositeRequest,
    bands: tuple[SpectrumBand, ...] | list[SpectrumBand],
    *,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
) -> tuple[SpectrumFluxCurve, ...]:
    """Extract multiple matched native-channel spectrum curves efficiently."""

    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    normalized = tuple(bands)
    if not normalized:
        raise ValueError("bands must not be empty")
    if any(not isinstance(band, SpectrumBand) for band in normalized):
        raise TypeError("bands must contain SpectrumBand values")
    if request.spectrum_type == "dart":
        widths = {band.bandwidth_mhz for band in normalized}
        if len(widths) == 1:
            result = extract_dart_narrowband_lightcurves(
                request.spectrum_path,
                [band.center_mhz for band in normalized],
                normalized[0].bandwidth_mhz,
                time_range_utc=time_range_utc,
            )
            if len(result.curves) != len(normalized):
                raise RuntimeError(
                    "DART narrowband extraction returned an unexpected curve count"
                )
            return tuple(
                SpectrumFluxCurve(
                    time_utc=result.time_utc,
                    values=curve.stokes_i_db,
                    source="DART",
                    polarization="Stokes I",
                    unit="dB",
                    requested_band=band,
                    sampled_frequency_range_mhz=curve.sampled_frequency_range_mhz,
                    channel_count=curve.channel_count,
                )
                for band, curve in zip(normalized, result.curves, strict=True)
            )
        return tuple(
            extract_spectrum_flux_curves(
                request,
                (band,),
                time_range_utc=time_range_utc,
                cso_polarization=cso_polarization,
            )[0]
            for band in normalized
        )

    window = load_cso_spectrum_window(
        request.spectrum_path,
        polarization=cso_polarization or request.polarization,
        time_range_utc=time_range_utc,
    )
    curves: list[SpectrumFluxCurve] = []
    for band in normalized:
        indices = band.observed_indices(window.frequency_mhz)
        selected = np.asarray(window.data[indices, :], dtype=float)
        finite = np.isfinite(selected)
        counts = finite.sum(axis=0)
        sums = np.where(finite, selected, 0.0).sum(axis=0)
        values = np.full(selected.shape[1], np.nan, dtype=float)
        np.divide(sums, counts, out=values, where=counts > 0)
        sampled = window.frequency_mhz[indices]
        curves.append(
            SpectrumFluxCurve(
                time_utc=window.time_utc,
                values=values,
                source="CSO",
                polarization=window.polarization,
                unit=window.unit,
                requested_band=band,
                sampled_frequency_range_mhz=(
                    float(np.min(sampled)),
                    float(np.max(sampled)),
                ),
                channel_count=int(indices.size),
            )
        )
    return tuple(curves)


def load_spectrum_window(
    request: CompositeRequest,
    *,
    frequency_range_mhz: tuple[float, float] | None = None,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    max_frequency_samples: int = 1600,
    max_time_samples: int = 1600,
    cso_polarization: str | None = None,
) -> SpectrumWindow:
    """Load the request's DART or CSO data through its canonical reader."""

    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    if request.spectrum_type == "dart":
        return load_dart_spectrum_window(
            request.spectrum_path,
            frequency_range_mhz=frequency_range_mhz,
            time_range_utc=time_range_utc,
            max_frequency_samples=max_frequency_samples,
            max_time_samples=max_time_samples,
        )
    return load_cso_spectrum_window(
        request.spectrum_path,
        polarization=cso_polarization or request.polarization,
        frequency_range_mhz=frequency_range_mhz,
        time_range_utc=time_range_utc,
        max_frequency_samples=max_frequency_samples,
        max_time_samples=max_time_samples,
    )


def load_dart_spectrum_window(
    path: str | Path,
    *,
    frequency_range_mhz: tuple[float, float] | None = None,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    max_frequency_samples: int = 1600,
    max_time_samples: int = 1600,
) -> SpectrumWindow:
    """Map the canonical four-file DART window to ``SpectrumWindow``."""

    directory = Path(path).expanduser().resolve(strict=False)
    files = discover_dart_spectrogram_files(directory)
    window = read_dart_spectrogram_window(
        files,
        frequency_range_mhz=frequency_range_mhz,
        time_range_utc=time_range_utc,
        max_frequency_samples=max_frequency_samples,
        max_time_samples=max_time_samples,
    )
    return SpectrumWindow(
        data=window.stokes_i_db,
        frequency_mhz=window.frequency_mhz,
        time_utc=window.time_utc,
        polarization="Stokes I",
        unit="dB",
        source="DART",
        metadata={
            "directory": str(directory),
            "files": {
                "SpecDataIdB": str(files.stokes_i_db.resolve(strict=False)),
                "SpecDataVP": str(files.stokes_v_over_i.resolve(strict=False)),
                "SpecFrequency": str(files.frequency.resolve(strict=False)),
                "SpecTime": str(files.time.resolve(strict=False)),
            },
            "reader": (
                "solar_toolkit.radio.dart_spectrogram." "read_dart_spectrogram_window"
            ),
            "display_plane": "stokes_i_db",
        },
    )


def load_cso_spectrum_window(
    path: str | Path,
    *,
    polarization: str,
    frequency_range_mhz: tuple[float, float] | None = None,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    max_frequency_samples: int | None = None,
    max_time_samples: int | None = None,
) -> SpectrumWindow:
    """Read and normalize one CSO polarization through the shared FITS reader."""

    source = Path(path).expanduser().resolve(strict=False)
    candidates: list[tuple[Path, CSOSpectrogram]] = []
    for fits_path in _cso_paths(source):
        candidates.extend(
            (fits_path, spectrum) for spectrum in read_cso_spectrogram(fits_path)
        )
    selected_path, selected = _select_cso_polarization(
        candidates,
        polarization=polarization,
    )
    data = np.asarray(selected.data)
    frequencies = np.asarray(selected.freq, dtype=float)
    seconds = np.asarray(selected.time, dtype=float)
    if data.shape != (frequencies.size, seconds.size):
        raise ValueError(
            "CSO reader returned data whose shape does not match frequency/time axes"
        )
    if selected.dt_base is None:
        raise ValueError("CSO reader returned no DATE-OBS base time")
    base = _as_utc(selected.dt_base)
    times = tuple(base + timedelta(seconds=float(value)) for value in seconds)

    frequency_indices = _range_indices(
        frequencies,
        frequency_range_mhz,
        label="CSO frequency",
    )
    time_indices = _time_range_indices(times, time_range_utc)
    data = data[np.ix_(frequency_indices, time_indices)]
    frequencies = frequencies[frequency_indices]
    times = tuple(times[index] for index in time_indices)
    if frequencies.size > 1 and frequencies[0] > frequencies[-1]:
        frequencies = frequencies[::-1].copy()
        data = data[::-1, :].copy()
    if len(times) > 1 and times[0] > times[-1]:
        times = tuple(reversed(times))
        data = data[:, ::-1].copy()
    original_shape = tuple(int(value) for value in data.shape)
    frequency_preview_indices = _preview_indices(
        frequencies.size,
        max_frequency_samples,
        label="max_frequency_samples",
    )
    time_preview_indices = _preview_indices(
        len(times),
        max_time_samples,
        label="max_time_samples",
    )
    data = data[np.ix_(frequency_preview_indices, time_preview_indices)]
    frequencies = frequencies[frequency_preview_indices]
    times = tuple(times[index] for index in time_preview_indices)

    return SpectrumWindow(
        data=data,
        frequency_mhz=frequencies,
        time_utc=times,
        polarization=_normalized_polarization(selected.polar),
        unit=selected.unit or "unknown",
        source="CSO",
        metadata={
            "path": str(selected_path.resolve(strict=False)),
            "date_obs": str(selected.dateobs),
            "reader": "solar_toolkit.radio.cso.read_cso_spectrogram",
            "time_coordinate": "DATE-OBS + seconds",
            "original_selected_shape": list(original_shape),
            "preview_downsampled": tuple(data.shape) != original_shape,
        },
    )


def _cso_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"CSO spectrum path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"CSO spectrum path is not a directory: {path}")
    paths = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.casefold() in {".fits", ".fit", ".fts"}
    )
    if not paths:
        raise FileNotFoundError(f"No CSO FITS files found in {path}")
    return paths


def _select_cso_polarization(
    candidates: list[tuple[Path, CSOSpectrogram]],
    *,
    polarization: str,
) -> tuple[Path, CSOSpectrogram]:
    if not candidates:
        raise RuntimeError("No CSO spectrogram was read")
    target = _normalized_polarization(polarization)
    matching = [
        item for item in candidates if _normalized_polarization(item[1].polar) == target
    ]
    if not matching:
        available = sorted(
            {_normalized_polarization(item[1].polar) for item in candidates}
        )
        raise ValueError(
            f"CSO polarization {target!r} is unavailable; "
            f"available: {', '.join(available)}"
        )
    return matching[0]


def _normalized_polarization(value: Any) -> str:
    text = str(value).strip().upper().replace(" ", "")
    aliases = {
        "RCP": "RR",
        "LCP": "LL",
        "RCP+LCP": "RR+LL",
        "LCP+RCP": "RR+LL",
    }
    return aliases.get(text, text)


def _range_indices(
    values: np.ndarray,
    limits: tuple[float, float] | None,
    *,
    label: str,
) -> np.ndarray:
    if limits is None:
        return np.arange(values.size)
    low, high = sorted(float(value) for value in limits)
    indices = np.flatnonzero((values >= low) & (values <= high))
    if not indices.size:
        raise ValueError(f"{label} range contains no samples")
    return indices


def _time_range_indices(
    values: tuple[datetime, ...],
    limits: tuple[datetime | str, datetime | str] | None,
) -> np.ndarray:
    if limits is None:
        return np.arange(len(values))
    start, end = (_coerce_utc(value) for value in limits)
    if start > end:
        start, end = end, start
    indices = np.asarray(
        [index for index, value in enumerate(values) if start <= value <= end],
        dtype=int,
    )
    if not indices.size:
        raise ValueError("CSO time range contains no samples")
    return indices


def _preview_indices(
    size: int,
    maximum: int | None,
    *,
    label: str,
) -> np.ndarray:
    if maximum is None:
        return np.arange(size, dtype=int)
    limit = int(maximum)
    if limit <= 0:
        raise ValueError(f"{label} must be greater than zero")
    if size <= limit:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, limit, dtype=int))


def _coerce_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return _as_utc(value)
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    return _as_utc(datetime.fromisoformat(text))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
