"""Tests for deterministic generated scientific-image filenames."""

from __future__ import annotations

import datetime as dt
import re
from types import SimpleNamespace

import numpy as np
import pytest

from solar_toolkit.visualization import image_naming
from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
    format_utc_filename_time,
)


def test_format_utc_time_converts_timezone_and_truncates_subseconds() -> None:
    value = dt.datetime(
        2025,
        1,
        24,
        12,
        48,
        30,
        987654,
        tzinfo=dt.timezone(dt.timedelta(hours=8)),
    )

    assert format_utc_filename_time(value) == "20250124T044830Z"


def test_format_utc_time_accepts_numpy_datetime64() -> None:
    value = np.datetime64("2025-01-24T04:48:30.999999999")
    assert format_utc_filename_time(value) == "20250124T044830Z"


@pytest.mark.parametrize(
    "fraction", ["9", "99", "999", "9999", "99999", "999999", "999999999"]
)
@pytest.mark.parametrize("zone,hour", [("Z", "04"), ("+08:00", "12")])
def test_fractional_iso_times_use_python310_parser_surface(
    monkeypatch, fraction: str, zone: str, hour: str
) -> None:
    class Python310Datetime(dt.datetime):
        @classmethod
        def fromisoformat(cls, text):
            for digits in re.findall(r":\d{2}\.(\d+)", text):
                if len(digits) not in (3, 6):
                    raise ValueError(
                        "Python 3.10 accepts three or six fractional digits"
                    )
            return super().fromisoformat(text)

    monkeypatch.setattr(
        image_naming,
        "dt",
        SimpleNamespace(
            datetime=Python310Datetime, date=dt.date, time=dt.time, timezone=dt.timezone
        ),
    )
    assert (
        format_utc_filename_time(f"2025-01-24T{hour}:48:30.{fraction}{zone}")
        == "20250124T044830Z"
    )


def test_numpy_nanoseconds_use_python310_parser_surface(monkeypatch) -> None:
    class Python310Datetime(dt.datetime):
        @classmethod
        def fromisoformat(cls, text):
            digits = text.split(".")[-1]
            if "." in text and len(digits) not in (3, 6):
                raise ValueError("Python 3.10 rejects nanosecond ISO text")
            return super().fromisoformat(text)

    monkeypatch.setattr(
        image_naming,
        "dt",
        SimpleNamespace(
            datetime=Python310Datetime, date=dt.date, time=dt.time, timezone=dt.timezone
        ),
    )
    assert (
        format_utc_filename_time(np.datetime64("2025-01-24T04:48:30.999999999"))
        == "20250124T044830Z"
    )


def test_format_utc_time_range_uses_start_and_end() -> None:
    assert (
        format_utc_filename_time(
            "2025-01-24T04:48:00Z",
            "2025-01-24T04:50:00.999Z",
        )
        == "20250124T044800Z-20250124T045000Z"
    )
    with pytest.raises(ValueError, match="precedes"):
        format_utc_filename_time(
            "2025-01-24T04:50:00Z",
            "2025-01-24T04:48:00Z",
        )


def test_build_filename_normalizes_channel_polarization_and_products() -> None:
    assert (
        build_image_filename(
            ImageFilenameSpec(
                sequence=2,
                start_time="2025-01-24T04:48:31.500Z",
                instrument="Radio",
                channel="223.5 MHz",
                polarization="LL",
                product="Source Map",
            )
        )
        == "0002_20250124T044831Z_radio_223p5mhz_lcp_source_map.png"
    )


def test_build_filename_supports_composite_stokes_token() -> None:
    assert build_image_filename(
        ImageFilenameSpec(
            sequence=1,
            start_time="2025-01-24T04:48:00Z",
            end_time="2025-01-24T04:50:00Z",
            instrument="DART",
            polarization="Stokes I V over I",
            product="Dynamic Spectrum",
        )
    ) == (
        "0001_20250124T044800Z-20250124T045000Z_"
        "dart_stokes_i_v_over_i_dynamic_spectrum.png"
    )


def test_generated_time_source_is_explicit_and_deterministic() -> None:
    batch_time = dt.datetime(2026, 7, 17, 10, 11, 12, tzinfo=dt.timezone.utc)
    spec = ImageFilenameSpec(
        sequence=1,
        start_time=batch_time,
        instrument="AIA",
        channel="171 Angstrom",
        product="Intensity",
        time_source="generated",
    )

    assert build_image_filename(spec) == (
        "0001_20260717T101112Z_generated_aia_171a_intensity.png"
    )
    assert build_image_filename(spec) == build_image_filename(spec)


@pytest.mark.parametrize("sequence", [0, -1, 10000])
def test_invalid_sequence_is_rejected(sequence: int) -> None:
    with pytest.raises(ValueError, match="sequence"):
        build_image_filename(
            ImageFilenameSpec(
                sequence=sequence,
                start_time="2025-01-24T04:48:00Z",
                instrument="AIA",
                product="Intensity",
            )
        )


def test_invalid_extension_and_empty_ascii_token_are_rejected() -> None:
    with pytest.raises(ValueError, match="extension"):
        build_image_filename(
            ImageFilenameSpec(
                sequence=1,
                start_time="2025-01-24T04:48:00Z",
                instrument="AIA",
                product="Intensity",
                extension=".mp4",
            )
        )
    with pytest.raises(ValueError, match="instrument"):
        build_image_filename(
            ImageFilenameSpec(
                sequence=1,
                start_time="2025-01-24T04:48:00Z",
                instrument="太阳",
                product="Intensity",
            )
        )
