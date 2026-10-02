from __future__ import annotations

import datetime as dt

import pytest


def test_extract_time_from_common_solar_filenames():
    from solar_toolkit.time import extract_time_from_filename

    assert extract_time_from_filename("hmi_20000101_044837_TAI.fits") == dt.datetime(
        2000, 1, 1, 4, 48, 37
    )
    assert extract_time_from_filename(
        "aia_2000-01-01T04:48:37Z_211.fits"
    ) == dt.datetime(2000, 1, 1, 4, 48, 37)
    assert extract_time_from_filename("cso_20000101044837_LL.fits") == dt.datetime(
        2000, 1, 1, 4, 48, 37
    )
    assert extract_time_from_filename("context_20000101.png") == dt.datetime(
        2000, 1, 1
    )


def test_parse_time_nearest_and_range_filter():
    from solar_toolkit.time import filter_by_time_range, nearest_by_time, parse_time

    rows = [
        ("a", parse_time("2000-01-01T04:48:00Z")),
        ("b", parse_time("2000-01-01 04:49:00")),
        ("c", parse_time(dt.datetime(2000, 1, 1, 4, 51, 0))),
    ]

    assert parse_time("2000-01-01T04:48:37Z") == dt.datetime(2000, 1, 1, 4, 48, 37)
    assert nearest_by_time(parse_time("2000-01-01T04:48:40"), rows) == rows[1]
    assert (
        nearest_by_time(parse_time("2000-01-01T05:48:40"), rows, max_diff_seconds=60)
        is None
    )
    assert (
        filter_by_time_range(
            rows,
            parse_time("2000-01-01T04:48:30"),
            parse_time("2000-01-01T04:51:00"),
        )
        == rows[1:]
    )


def test_extract_time_raises_when_no_timestamp_exists():
    from solar_toolkit.time import extract_time_from_filename

    with pytest.raises(ValueError, match="Could not extract time"):
        extract_time_from_filename("no_timestamp_here.fits")
