"""Validated data contracts for the AIA radio composite frontend."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AIA_RADIO_COMPOSITE_SCHEMA_VERSION = 3
AIA_WAVELENGTHS = frozenset({94, 131, 171, 193, 211, 304, 335, 1600})
POLARIZATIONS = frozenset({"RR", "LL", "RR+LL"})
ROI_TYPES = frozenset({"box", "lasso"})
SPECTRUM_TYPES = frozenset({"dart", "cso"})
SPECTRUM_SOURCES = frozenset({"DART", "CSO"})
ROI_CURVE_COLUMNS = (
    "time",
    "frequency",
    "raw_sum",
    "raw_mean",
    "raw_peak",
    "quality_flag",
)

__all__ = [
    "AIA_RADIO_COMPOSITE_SCHEMA_VERSION",
    "AIA_WAVELENGTHS",
    "CompositeRequest",
    "CompositeResult",
    "POLARIZATIONS",
    "ROI_CURVE_COLUMNS",
    "ROI_TYPES",
    "SPECTRUM_SOURCES",
    "SPECTRUM_TYPES",
    "SpectrumBand",
    "SpectrumFluxCurve",
    "SpectrumWindow",
    "parse_roi_curve_times",
]


def _normalized_path(value: str | Path, *, label: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is required")
    return Path(text).expanduser().resolve(strict=False)


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _utc_datetime(value: datetime | str, *, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} is required")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _normalized_roi_vertices(
    values: Sequence[Sequence[float]],
    *,
    roi_type: str,
) -> tuple[tuple[float, float], ...]:
    try:
        vertices = tuple(
            (
                _finite_float(vertex[0], label="ROI HPLN"),
                _finite_float(vertex[1], label="ROI HPLT"),
            )
            for vertex in values
        )
    except (IndexError, TypeError) as exc:
        raise TypeError("roi_vertices_arcsec must contain HPLN/HPLT pairs") from exc
    required = 4 if roi_type == "box" else 3
    if len(vertices) < required:
        raise ValueError(f"{roi_type} ROI requires at least {required} vertices")
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    if min(xs) >= max(xs) or min(ys) >= max(ys):
        raise ValueError("ROI vertices define a degenerate region")
    return vertices


def _metadata_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} keys must be strings")
    return {
        key: _json_safe(item, label=f"{label}.{key}") for key, item in value.items()
    }


def _json_safe(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite values")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), label=label)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _utc_datetime(value, label=label).isoformat()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{label} object keys must be strings")
        return {
            key: _json_safe(item, label=f"{label}.{key}") for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{label} contains a non-JSON value: {type(value).__name__}")


def parse_roi_curve_times(curve: pd.DataFrame) -> pd.Series:
    """Parse mixed UTC timestamps while retaining untimed failed-quality rows.

    The canonical Radio extractor emits an empty ``obs_time`` when a FITS row
    cannot supply an observation time. Such rows are useful quality
    diagnostics but cannot be placed on a time axis. A row marked ``ok`` must
    always have a valid UTC timestamp.
    """

    if not isinstance(curve, pd.DataFrame):
        raise TypeError("curve must be a pandas DataFrame")
    if "time" not in curve or "quality_flag" not in curve:
        raise ValueError("curve must contain time and quality_flag columns")
    parsed = pd.to_datetime(
        curve["time"],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    quality_ok = curve["quality_flag"].astype(str).str.strip().str.lower().eq("ok")
    if (parsed.isna() & quality_ok).any():
        raise ValueError("curve time contains invalid UTC values on ok-quality rows")
    return parsed


@dataclass(frozen=True, slots=True)
class CompositeRequest:
    """One validated request for an AIA/radio/spectrum composite product."""

    aia_directory: Path
    aia_wave: int
    aia_time: datetime
    radio_directory: Path
    radio_frequency: float
    polarization: str
    roi_type: str
    roi_vertices_arcsec: tuple[tuple[float, float], ...]
    spectrum_type: str
    spectrum_path: Path

    def __post_init__(self) -> None:
        """Normalize paths, UTC time, selectors, and HPLN/HPLT vertices."""

        object.__setattr__(
            self,
            "aia_directory",
            _normalized_path(self.aia_directory, label="aia_directory"),
        )
        try:
            wave = int(self.aia_wave)
            wave_value = float(self.aia_wave)
        except (TypeError, ValueError) as exc:
            raise TypeError("aia_wave must be an integer wavelength") from exc
        if not math.isfinite(wave_value) or wave_value != wave:
            raise ValueError("aia_wave must be an integer wavelength")
        if wave not in AIA_WAVELENGTHS:
            supported = ", ".join(str(value) for value in sorted(AIA_WAVELENGTHS))
            raise ValueError(f"aia_wave must be one of: {supported}")
        object.__setattr__(self, "aia_wave", wave)
        object.__setattr__(
            self,
            "aia_time",
            _utc_datetime(self.aia_time, label="aia_time"),
        )
        object.__setattr__(
            self,
            "radio_directory",
            _normalized_path(self.radio_directory, label="radio_directory"),
        )
        frequency = _finite_float(
            self.radio_frequency,
            label="radio_frequency",
        )
        if frequency <= 0:
            raise ValueError("radio_frequency must be greater than zero")
        object.__setattr__(self, "radio_frequency", frequency)
        polarization = str(self.polarization).strip().upper()
        if polarization not in POLARIZATIONS:
            raise ValueError("polarization must be RR, LL, or RR+LL")
        object.__setattr__(self, "polarization", polarization)
        roi_type = str(self.roi_type).strip().lower()
        if roi_type not in ROI_TYPES:
            raise ValueError("roi_type must be box or lasso")
        object.__setattr__(self, "roi_type", roi_type)
        object.__setattr__(
            self,
            "roi_vertices_arcsec",
            _normalized_roi_vertices(
                self.roi_vertices_arcsec,
                roi_type=roi_type,
            ),
        )
        spectrum_type = str(self.spectrum_type).strip().lower()
        if spectrum_type not in SPECTRUM_TYPES:
            raise ValueError("spectrum_type must be dart or cso")
        object.__setattr__(self, "spectrum_type", spectrum_type)
        object.__setattr__(
            self,
            "spectrum_path",
            _normalized_path(self.spectrum_path, label="spectrum_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe request representation."""

        return {
            "schema_version": AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
            "aia": {
                "directory": str(self.aia_directory),
                "wave": self.aia_wave,
                "time_utc": self.aia_time.isoformat(),
            },
            "radio": {
                "directory": str(self.radio_directory),
                "frequency_mhz": self.radio_frequency,
                "polarization": self.polarization,
            },
            "roi": {
                "type": self.roi_type,
                "coordinate_system": "HPLN/HPLT arcsec",
                "vertices_arcsec": [
                    {"hpln": x, "hplt": y} for x, y in self.roi_vertices_arcsec
                ],
            },
            "spectrum": {
                "type": self.spectrum_type,
                "path": str(self.spectrum_path),
            },
        }


@dataclass(frozen=True, slots=True)
class SpectrumWindow:
    """One normalized frequency-by-time DART or CSO display window."""

    data: np.ndarray
    frequency_mhz: np.ndarray
    time_utc: tuple[datetime, ...]
    polarization: str
    unit: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate aligned axes while leaving scientific sample values intact."""

        data = np.asarray(self.data)
        if data.ndim != 2 or not np.issubdtype(data.dtype, np.number):
            raise ValueError("spectrum data must be a numeric two-dimensional array")
        frequency = np.asarray(self.frequency_mhz, dtype=float)
        if frequency.ndim != 1 or not frequency.size:
            raise ValueError("frequency_mhz must be a non-empty one-dimensional axis")
        if not np.isfinite(frequency).all():
            raise ValueError("frequency_mhz must contain only finite values")
        if frequency.size > 1 and not np.all(np.diff(frequency) > 0):
            raise ValueError("frequency_mhz must be strictly increasing")
        times = tuple(
            _utc_datetime(value, label=f"time_utc[{index}]")
            for index, value in enumerate(self.time_utc)
        )
        if not times:
            raise ValueError("time_utc must not be empty")
        if any(left >= right for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("time_utc must be strictly increasing")
        expected_shape = (frequency.size, len(times))
        if data.shape != expected_shape:
            raise ValueError(
                f"spectrum data shape {data.shape} does not match axes {expected_shape}"
            )
        polarization = str(self.polarization).strip()
        if not polarization:
            raise ValueError("spectrum polarization is required")
        unit = str(self.unit).strip()
        if not unit:
            raise ValueError("spectrum unit is required")
        source = str(self.source).strip().upper()
        if source not in SPECTRUM_SOURCES:
            raise ValueError("spectrum source must be DART or CSO")

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "frequency_mhz", frequency)
        object.__setattr__(self, "time_utc", times)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "metadata",
            _metadata_mapping(self.metadata, label="spectrum metadata"),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe window metadata without embedding the data matrix."""

        return {
            "source": self.source,
            "polarization": self.polarization,
            "unit": self.unit,
            "shape": [int(value) for value in self.data.shape],
            "frequency_range_mhz": [
                float(self.frequency_mhz[0]),
                float(self.frequency_mhz[-1]),
            ],
            "time_range_utc": [
                self.time_utc[0].isoformat(),
                self.time_utc[-1].isoformat(),
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SpectrumBand:
    """One requested and observed frequency interval in MHz."""

    low_mhz: float
    high_mhz: float

    def __post_init__(self) -> None:
        low = _finite_float(self.low_mhz, label="spectrum band lower bound")
        high = _finite_float(self.high_mhz, label="spectrum band upper bound")
        if low >= high:
            raise ValueError("spectrum band lower bound must be below the upper bound")
        object.__setattr__(self, "low_mhz", low)
        object.__setattr__(self, "high_mhz", high)

    @property
    def center_mhz(self) -> float:
        return (self.low_mhz + self.high_mhz) / 2.0

    @property
    def bandwidth_mhz(self) -> float:
        return self.high_mhz - self.low_mhz

    def observed_indices(self, frequency_mhz: Sequence[float]) -> np.ndarray:
        """Return original-channel indices after strict range validation."""

        frequency = np.asarray(frequency_mhz, dtype=float)
        if frequency.ndim != 1 or not frequency.size:
            raise ValueError("spectrum frequency axis must be non-empty")
        if not np.isfinite(frequency).all():
            raise ValueError("spectrum frequency axis must contain finite values")
        observed_low = float(np.min(frequency))
        observed_high = float(np.max(frequency))
        tolerance = max(1e-9, abs(observed_high - observed_low) * 1e-12)
        if (
            self.low_mhz < observed_low - tolerance
            or self.high_mhz > observed_high + tolerance
        ):
            raise ValueError(
                "Selected spectrum band is outside the observed frequency range: "
                f"{observed_low:g}-{observed_high:g} MHz"
            )
        indices = np.flatnonzero(
            (frequency >= self.low_mhz) & (frequency <= self.high_mhz)
        )
        if not indices.size:
            raise ValueError(
                "Selected spectrum band contains no original frequency channel"
            )
        return indices

    def to_dict(self) -> dict[str, float]:
        return {
            "low_mhz": self.low_mhz,
            "high_mhz": self.high_mhz,
            "center_mhz": self.center_mhz,
            "bandwidth_mhz": self.bandwidth_mhz,
        }


@dataclass(frozen=True, slots=True)
class SpectrumFluxCurve:
    """One native-unit, finite-channel-mean spectrum light curve."""

    time_utc: tuple[datetime, ...]
    values: np.ndarray
    source: str
    polarization: str
    unit: str
    requested_band: SpectrumBand
    sampled_frequency_range_mhz: tuple[float, float]
    channel_count: int
    aggregation: str = "finite_channel_mean"

    def __post_init__(self) -> None:
        times = tuple(
            _utc_datetime(value, label=f"spectrum flux time_utc[{index}]")
            for index, value in enumerate(self.time_utc)
        )
        if not times:
            raise ValueError("spectrum flux time_utc must not be empty")
        if any(left >= right for left, right in zip(times, times[1:], strict=False)):
            raise ValueError("spectrum flux time_utc must be strictly increasing")
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or values.size != len(times):
            raise ValueError("spectrum flux values must align with time_utc")
        if np.isinf(values).any():
            raise ValueError("spectrum flux values must not contain infinity")
        source = str(self.source).strip().upper()
        if source not in SPECTRUM_SOURCES:
            raise ValueError("spectrum flux source must be DART or CSO")
        polarization = str(self.polarization).strip()
        unit = str(self.unit).strip()
        if not polarization or not unit:
            raise ValueError("spectrum flux polarization and unit are required")
        if not isinstance(self.requested_band, SpectrumBand):
            raise TypeError("requested_band must be a SpectrumBand")
        sampled = tuple(
            _finite_float(value, label="sampled frequency bound")
            for value in self.sampled_frequency_range_mhz
        )
        if len(sampled) != 2 or sampled[0] > sampled[1]:
            raise ValueError("sampled frequency range must be ordered")
        channel_count = int(self.channel_count)
        if channel_count <= 0:
            raise ValueError("spectrum flux channel_count must be greater than zero")
        aggregation = str(self.aggregation).strip()
        if aggregation != "finite_channel_mean":
            raise ValueError("spectrum flux aggregation must be finite_channel_mean")

        object.__setattr__(self, "time_utc", times)
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "sampled_frequency_range_mhz", sampled)
        object.__setattr__(self, "channel_count", channel_count)
        object.__setattr__(self, "aggregation", aggregation)

    def to_frame(self) -> pd.DataFrame:
        """Return a reproducible long-form CSV table."""

        return pd.DataFrame(
            {
                "time_utc": [value.isoformat() for value in self.time_utc],
                "value": self.values,
                "source": self.source,
                "polarization": self.polarization,
                "unit": self.unit,
                "requested_low_mhz": self.requested_band.low_mhz,
                "requested_high_mhz": self.requested_band.high_mhz,
                "sampled_low_mhz": self.sampled_frequency_range_mhz[0],
                "sampled_high_mhz": self.sampled_frequency_range_mhz[1],
                "center_mhz": self.requested_band.center_mhz,
                "bandwidth_mhz": self.requested_band.bandwidth_mhz,
                "channel_count": self.channel_count,
                "aggregation": self.aggregation,
            }
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "polarization": self.polarization,
            "unit": self.unit,
            "requested_band": self.requested_band.to_dict(),
            "sampled_frequency_range_mhz": [
                self.sampled_frequency_range_mhz[0],
                self.sampled_frequency_range_mhz[1],
            ],
            "channel_count": self.channel_count,
            "aggregation": self.aggregation,
            "sample_count": len(self.time_utc),
            "finite_sample_count": int(np.isfinite(self.values).sum()),
            "time_range_utc": [
                self.time_utc[0].isoformat(),
                self.time_utc[-1].isoformat(),
            ],
        }


@dataclass(frozen=True, slots=True)
class CompositeResult:
    """In-memory top panel, ROI curve, spectrum window, and result metadata."""

    top_image: bytes
    roi_curve: pd.DataFrame
    spectrum: SpectrumWindow
    metadata: Mapping[str, Any] = field(default_factory=dict)
    spectrum_flux_curve: SpectrumFluxCurve | None = None
    spectrum_flux_curves: tuple[SpectrumFluxCurve, ...] = ()

    def __post_init__(self) -> None:
        """Validate result components and preserve a private curve snapshot."""

        if not isinstance(self.top_image, bytes) or not self.top_image:
            raise ValueError("top_image must be non-empty bytes")
        if not isinstance(self.roi_curve, pd.DataFrame):
            raise TypeError("roi_curve must be a pandas DataFrame")
        missing = [name for name in ROI_CURVE_COLUMNS if name not in self.roi_curve]
        if missing:
            raise ValueError(f"roi_curve is missing required columns: {missing}")
        curve = self.roi_curve.copy(deep=True)
        if not curve.empty:
            parse_roi_curve_times(curve)
            frequencies = pd.to_numeric(curve["frequency"], errors="coerce")
            if not np.isfinite(frequencies.to_numpy(dtype=float)).all():
                raise ValueError("roi_curve frequency must contain finite values")
            if curve["quality_flag"].astype(str).str.strip().eq("").any():
                raise ValueError("roi_curve quality_flag must not be empty")
        if not isinstance(self.spectrum, SpectrumWindow):
            raise TypeError("spectrum must be a SpectrumWindow")
        if self.spectrum_flux_curve is not None and not isinstance(
            self.spectrum_flux_curve,
            SpectrumFluxCurve,
        ):
            raise TypeError("spectrum_flux_curve must be a SpectrumFluxCurve")
        curves = tuple(self.spectrum_flux_curves)
        if any(not isinstance(item, SpectrumFluxCurve) for item in curves):
            raise TypeError(
                "spectrum_flux_curves must contain SpectrumFluxCurve values"
            )
        if self.spectrum_flux_curve is not None:
            if not curves:
                curves = (self.spectrum_flux_curve,)
            elif curves[0] is not self.spectrum_flux_curve:
                raise ValueError(
                    "spectrum_flux_curve must match the first spectrum_flux_curves item"
                )
        elif curves:
            object.__setattr__(self, "spectrum_flux_curve", curves[0])
        centers = [curve.requested_band.center_mhz for curve in curves]
        if len(set(centers)) != len(centers):
            raise ValueError("spectrum_flux_curves must have unique band centers")

        object.__setattr__(self, "roi_curve", curve)
        object.__setattr__(self, "spectrum_flux_curves", curves)
        object.__setattr__(
            self,
            "metadata",
            _metadata_mapping(self.metadata, label="result metadata"),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe result inventory without embedding binary/table data."""

        return {
            "schema_version": AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
            "top_image_bytes": len(self.top_image),
            "roi_curve_rows": int(len(self.roi_curve)),
            "roi_curve_columns": [str(name) for name in self.roi_curve.columns],
            "spectrum": self.spectrum.to_metadata_dict(),
            "spectrum_flux_curve": (
                self.spectrum_flux_curve.to_metadata_dict()
                if self.spectrum_flux_curve is not None
                else None
            ),
            "spectrum_flux_curves": [
                curve.to_metadata_dict() for curve in self.spectrum_flux_curves
            ],
            "metadata": dict(self.metadata),
        }
