from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from solar_apps.frontends.image_composer.catalog import (
    discover_images,
    extract_timestamp,
    scan_folder,
    timestamp_from_filename,
)


def _image(path: Path, color: str = "red", *, exif=None) -> None:
    image = Image.new("RGB", (20, 12), color)
    if exif is None:
        image.save(path)
    else:
        image.save(path, exif=exif)


def test_discovery_is_non_recursive_and_naturally_sorted(tmp_path: Path) -> None:
    _image(tmp_path / "frame10.png")
    _image(tmp_path / "frame2.png")
    _image(tmp_path / "frame1.jpg")
    nested = tmp_path / "nested"
    nested.mkdir()
    _image(nested / "frame0.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in discover_images(tmp_path)] == [
        "frame1.jpg",
        "frame2.png",
        "frame10.png",
    ]
    assert [record.ordinal for record in scan_folder(tmp_path)] == [1, 2, 3]


def test_filename_timestamp_patterns_include_fractional_seconds() -> None:
    expected = datetime(2026, 7, 17, 14, 30, 25, 250_000, tzinfo=UTC)
    assert timestamp_from_filename("CameraA_20260717_143025.250.png") == expected
    assert timestamp_from_filename("CameraB_2026-07-17_14-30-25-250.jpg") == expected
    assert timestamp_from_filename("bad_20260230_120000.png") is None


def test_exif_priority_and_subseconds_win_over_filename_and_mtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "camera_20260717_143025.900.jpg"
    exif = Image.Exif()
    exif[36867] = "2026:07:17 14:30:24"
    exif[37521] = "125"
    exif[36868] = "2026:07:17 14:30:23"
    exif[306] = "2026:07:17 14:30:22"
    _image(path, exif=exif)
    os.utime(path, (1_700_000_000, 1_700_000_000))

    timestamp, source = extract_timestamp(path)

    assert timestamp == datetime(2026, 7, 17, 14, 30, 24, 125_000, tzinfo=UTC)
    assert source == "exif:DateTimeOriginal"


@pytest.mark.parametrize(
    ("time_tag", "subsecond_tag", "expected_source"),
    [
        (36868, 37522, "exif:DateTimeDigitized"),
        (306, 37520, "exif:DateTime"),
    ],
)
def test_secondary_exif_time_sources(
    tmp_path: Path,
    time_tag: int,
    subsecond_tag: int,
    expected_source: str,
) -> None:
    path = tmp_path / f"source_{time_tag}.jpg"
    exif = Image.Exif()
    exif[time_tag] = "2026:07:17 14:30:24"
    exif[subsecond_tag] = "75"
    _image(path, exif=exif)

    timestamp, source = extract_timestamp(path)

    assert timestamp == datetime(2026, 7, 17, 14, 30, 24, 750_000, tzinfo=UTC)
    assert source == expected_source


def test_filename_then_mtime_fallback(tmp_path: Path) -> None:
    named = tmp_path / "image_20260717_143025.png"
    _image(named)
    assert extract_timestamp(named) == (
        datetime(2026, 7, 17, 14, 30, 25, tzinfo=UTC),
        "filename",
    )

    fallback = tmp_path / "plain.png"
    _image(fallback)
    os.utime(fallback, (1_700_000_000, 1_700_000_000))
    timestamp, source = extract_timestamp(fallback)
    assert timestamp == datetime.fromtimestamp(1_700_000_000, UTC)
    assert source == "mtime"


@pytest.mark.parametrize("prefer_system_zone", [False, True])
def test_exif_offset_filename_utc_and_mtime_agree_independent_of_host_zone(
    tmp_path: Path,
    monkeypatch,
    prefer_system_zone: bool,
) -> None:
    import time
    from solar_toolkit.time import normalize_time

    before = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    from solar_apps.frontends.image_composer import catalog

    tzset = getattr(time, "tzset", None) if prefer_system_zone else None
    if tzset is not None:
        tzset()
    else:
        # Windows has no tzset; exercise the same non-UTC local-clock trap.
        class ShanghaiDateTime(datetime):
            @classmethod
            def fromtimestamp(cls, value, tz=None):
                if tz is None:
                    return datetime.fromtimestamp(
                        value, timezone(timedelta(hours=8))
                    ).replace(tzinfo=None)
                return datetime.fromtimestamp(value, tz)

        monkeypatch.setattr(catalog, "datetime", ShanghaiDateTime)
    try:
        exif_path = tmp_path / "camera.jpg"
        exif = Image.Exif()
        exif[36867] = "2025:01:24 12:48:30"
        exif[37521] = "123456700"
        exif[36881] = "+08:00"
        _image(exif_path, exif=exif)
        named = tmp_path / "image_20250124_044830.123456700Z.png"
        _image(named)
        fallback = tmp_path / "plain.png"
        _image(fallback)
        # NTFS represents 100 ns ticks; use an exact physical timestamp on every OS.
        epoch_ns = 1_737_694_110_123_456_700
        os.utime(fallback, ns=(epoch_ns, epoch_ns))
        records = scan_folder(tmp_path)
        times = [normalize_time(record.timestamp_text).tai_ns for record in records]
        assert len(set(times)) == 1
        assert all(record.timestamp.tzinfo is UTC for record in records)
        assert all(
            normalize_time(record.timestamp_text).utc.endswith(".123456700")
            for record in records
        )
    finally:
        if before is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = before
        if tzset is not None:
            tzset()


def test_filename_explicit_offsets_convert_to_utc() -> None:
    expected = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    assert timestamp_from_filename("image_20250124_124830+0800.png") == expected
    assert timestamp_from_filename("image_20250124_004830-04:00.png") == expected


def test_mtime_reader_preserves_integer_nanoseconds(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from solar_apps.frontends.image_composer import catalog

    fallback = tmp_path / "plain.png"
    _image(fallback)
    epoch_ns = 1_737_694_110_123_456_789

    class ExactNanosecondPath(type(tmp_path)):
        def stat(self, *args, **kwargs):
            # Supply stat's exact integer clock independently of FS resolution.
            return SimpleNamespace(st_mtime_ns=epoch_ns)

    monkeypatch.setattr(catalog, "Path", ExactNanosecondPath)
    text, source = catalog.extract_timestamp_text(fallback)
    assert source == "mtime"
    assert text == "2025-01-24T04:48:30.123456789Z"
