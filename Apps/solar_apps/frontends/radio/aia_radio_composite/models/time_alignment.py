"""Display-only DART time alignment against a matched radio observation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .schema import SpectrumWindow

__all__ = [
    "SpectrumTimeAlignment",
    "build_spectrum_time_alignment",
]


@dataclass(frozen=True, slots=True)
class SpectrumTimeAlignment:
    """One constant display offset preserving native spectrum sampling."""

    reference_radio_time_utc: datetime
    nearest_spectrum_time_utc: datetime
    display_offset_seconds: float
    native_cadence_seconds: float
    nearest_delta_seconds: float

    def __post_init__(self) -> None:
        reference = _utc_datetime(
            self.reference_radio_time_utc,
            label="reference radio time",
        )
        nearest = _utc_datetime(
            self.nearest_spectrum_time_utc,
            label="nearest spectrum time",
        )
        offset = _finite_float(
            self.display_offset_seconds,
            label="spectrum display offset",
        )
        cadence = _finite_float(
            self.native_cadence_seconds,
            label="spectrum native cadence",
        )
        delta = _finite_float(
            self.nearest_delta_seconds,
            label="nearest spectrum delta",
        )
        if cadence <= 0:
            raise ValueError("spectrum native cadence must be greater than zero")
        if delta < 0:
            raise ValueError("nearest spectrum delta must not be negative")
        expected_offset = (reference - nearest).total_seconds()
        if not np.isclose(offset, expected_offset, rtol=0.0, atol=1e-9):
            raise ValueError(
                "spectrum display offset must equal reference radio time minus "
                "nearest spectrum time"
            )
        if not np.isclose(delta, abs(offset), rtol=0.0, atol=1e-9):
            raise ValueError(
                "nearest spectrum delta must equal the absolute display offset"
            )
        object.__setattr__(self, "reference_radio_time_utc", reference)
        object.__setattr__(self, "nearest_spectrum_time_utc", nearest)
        object.__setattr__(self, "display_offset_seconds", offset)
        object.__setattr__(self, "native_cadence_seconds", cadence)
        object.__setattr__(self, "nearest_delta_seconds", delta)

    def align_times(self, values: Sequence[datetime]) -> tuple[datetime, ...]:
        """Apply the fixed display offset without changing sample ordering."""

        offset = timedelta(seconds=self.display_offset_seconds)
        return tuple(
            _utc_datetime(value, label=f"spectrum display time[{index}]") + offset
            for index, value in enumerate(values)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance for static and video exports."""

        return {
            "mode": "constant_display_offset_no_interpolation",
            "source": "DART",
            "reference_radio_time_utc": self.reference_radio_time_utc.isoformat(),
            "nearest_spectrum_time_utc": self.nearest_spectrum_time_utc.isoformat(),
            "display_offset_seconds": self.display_offset_seconds,
            "native_cadence_seconds": self.native_cadence_seconds,
            "nearest_delta_seconds": self.nearest_delta_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SpectrumTimeAlignment:
        """Restore alignment metadata stored in a composite result."""

        if str(value.get("source", "")).upper() != "DART":
            raise ValueError("spectrum time alignment source must be DART")
        return cls(
            reference_radio_time_utc=value["reference_radio_time_utc"],
            nearest_spectrum_time_utc=value["nearest_spectrum_time_utc"],
            display_offset_seconds=value["display_offset_seconds"],
            native_cadence_seconds=value["native_cadence_seconds"],
            nearest_delta_seconds=value["nearest_delta_seconds"],
        )


def build_spectrum_time_alignment(
    spectrum: SpectrumWindow,
    reference_radio_time_utc: datetime | str,
) -> SpectrumTimeAlignment | None:
    """Align DART display UTC to the nearest matched radio observation.

    CSO remains on its native time axis. DART values are not reordered,
    interpolated, or resampled; callers apply only the returned constant offset
    while rendering.
    """

    if not isinstance(spectrum, SpectrumWindow):
        raise TypeError("spectrum must be a SpectrumWindow")
    if spectrum.source.upper() != "DART":
        return None
    times = tuple(spectrum.time_utc)
    if len(times) < 2:
        raise ValueError(
            "DART time alignment requires at least two original time samples"
        )
    timestamps = np.asarray([value.timestamp() for value in times], dtype=float)
    differences = np.diff(timestamps)
    finite_positive = differences[np.isfinite(differences) & (differences > 0)]
    if not finite_positive.size:
        raise ValueError("DART time axis has no valid positive sampling interval")
    cadence = float(np.median(finite_positive))
    reference = _utc_datetime(
        reference_radio_time_utc,
        label="reference radio time",
    )
    nearest_index = int(np.argmin(np.abs(timestamps - reference.timestamp())))
    nearest = times[nearest_index]
    offset = float((reference - nearest).total_seconds())
    delta = abs(offset)
    tolerance = max(1e-9, cadence * 1e-6)
    if delta > cadence + tolerance:
        raise ValueError(
            "No DART sample lies within one native sampling period of the "
            f"matched radio time {reference.isoformat()}; adjust the spectrum "
            "UTC window"
        )
    return SpectrumTimeAlignment(
        reference_radio_time_utc=reference,
        nearest_spectrum_time_utc=nearest,
        display_offset_seconds=offset,
        native_cadence_seconds=cadence,
        nearest_delta_seconds=delta,
    )


def _utc_datetime(value: datetime | str, *, label: str) -> datetime:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"{label} must be a valid UTC datetime")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    else:
        parsed = parsed.tz_convert(UTC)
    return parsed.to_pydatetime()


def _finite_float(value: Any, *, label: str) -> float:
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized
