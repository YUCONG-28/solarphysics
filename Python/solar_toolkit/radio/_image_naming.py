"""Shared filename derivation for persisted radio image products."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
)

from ._time import table_timestamp

_TIME_COLUMNS = ("obs_time", "time", "datetime", "timestamp")
_FREQUENCY_COLUMNS = ("freq_mhz", "frequency_mhz", "freq", "frequency")
_POLARIZATION_COLUMNS = ("polarization", "pol", "stokes")
_POLARIZATION_ALIASES = {
    "ll": "lcp",
    "lcp": "lcp",
    "rr": "rcp",
    "rcp": "rcp",
    "ll+rr": "lcp_plus_rcp",
    "rr+ll": "lcp_plus_rcp",
    "lcp+rcp": "lcp_plus_rcp",
    "i": "stokes_i",
    "stokes_i": "stokes_i",
    "v/i": "stokes_v_over_i",
    "stokes_v_over_i": "stokes_v_over_i",
}


def build_radio_image_filename(
    data,
    *,
    sequence: int,
    product: str | Sequence[str],
    generated_at: dt.datetime,
    qualifiers: str | Sequence[str] = (),
    frequency_mhz: float | None = None,
    polarization: str | Sequence[str] | None = None,
    instrument: str = "radio",
    extension: str = ".png",
) -> str:
    """Build a radio image name from tabular observation metadata."""

    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    start_time, end_time = dataframe_time_bounds(frame)
    time_source = "observation"
    if start_time is None:
        start_time = generated_at
        end_time = None
        time_source = "generated"
    if frequency_mhz is None:
        frequency_mhz = dataframe_single_frequency(frame)
    if polarization is None:
        polarization = dataframe_polarization(frame)
    channel = None if frequency_mhz is None else f"{float(frequency_mhz):g}mhz"
    return build_image_filename(
        ImageFilenameSpec(
            sequence=sequence,
            start_time=start_time,
            end_time=end_time,
            instrument=instrument,
            channel=channel,
            polarization=polarization,
            product=product,
            qualifiers=qualifiers,
            extension=extension,
            time_source=time_source,
        )
    )


def dataframe_time_bounds(frame: pd.DataFrame):
    """Return earliest and latest valid UTC times from a known time column."""

    for column in _TIME_COLUMNS:
        if column not in frame.columns:
            continue
        values = frame[column].map(_coerce_datetime).dropna()
        if values.empty:
            continue
        start = values.min().to_pydatetime()
        end = values.max().to_pydatetime()
        return start, None if end == start else end
    return None, None


def _coerce_datetime(value):
    """Parse radio table times without treating compact timestamps as epochs."""
    return table_timestamp(value, utc=True)


def dataframe_single_frequency(frame: pd.DataFrame) -> float | None:
    """Return one shared frequency, or ``None`` for mixed/unknown data."""

    for column in _FREQUENCY_COLUMNS:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
        if len(values) == 1:
            return float(values[0])
        if len(values) > 1:
            return None
    return None


def dataframe_polarization(frame: pd.DataFrame) -> str | None:
    """Return one canonical polarization token for the represented product."""

    for column in _POLARIZATION_COLUMNS:
        if column not in frame.columns:
            continue
        values = {
            _canonical_polarization(value)
            for value in frame[column].dropna().tolist()
            if str(value).strip()
        }
        values.discard(None)
        if not values:
            continue
        if values == {"lcp", "rcp"}:
            return "lcp_plus_rcp"
        if len(values) == 1:
            return next(iter(values))
        return None
    return None


def _canonical_polarization(value: Any) -> str | None:
    text = str(value).strip().casefold().replace(" ", "_")
    return _POLARIZATION_ALIASES.get(text)


def chronological_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic rows ordered by valid start time then input order."""

    indexed = list(enumerate(rows))

    def key(item):
        index, row = item
        value = table_timestamp(row.get("t_start"), utc=True)
        return (1, index) if pd.isna(value) else (0, value, index)

    return [row for _index, row in sorted(indexed, key=key)]
