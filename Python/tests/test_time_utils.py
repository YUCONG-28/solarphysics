from __future__ import annotations

import datetime as dt

import pytest


def test_extract_time_from_common_solar_filenames():
    from solar_toolkit.time import extract_time_from_filename

    assert extract_time_from_filename("hmi_20000101_044837_TAI.fits") == dt.datetime(
        2000, 1, 1, 4, 48, 5
    )
    assert extract_time_from_filename(
        "aia_2000-01-01T04:48:37Z_211.fits"
    ) == dt.datetime(2000, 1, 1, 4, 48, 37)
    assert extract_time_from_filename("cso_20000101044837_LL.fits") == dt.datetime(
        2000, 1, 1, 4, 48, 37
    )
    assert extract_time_from_filename("context_20000101.png") == dt.datetime(2000, 1, 1)


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


def test_legacy_time_helpers_convert_tai_timezone_and_fraction():
    from solar_toolkit.time import extract_time_from_filename, parse_time
    from solar_toolkit.time.parsing import parse_isot_time

    expected = dt.datetime(2020, 1, 1, 4, 48, 0, 123456)
    assert extract_time_from_filename("hmi_20200101_044837.123456_TAI.fits") == expected
    assert parse_time("2020-01-01T04:48:37.123456_TAI") == expected
    assert parse_isot_time("2020-01-01T12:48:00.123456+08:00") == expected
    assert extract_time_from_filename("cso_20200101044800123456_LL.fits") == expected


@pytest.mark.parametrize("zone", ["Z", "+08:00", "+0800"])
def test_legacy_parser_accepts_nanosecond_inputs_at_microsecond_boundary(zone):
    from solar_toolkit.time import parse_time

    hour = "04" if zone == "Z" else "12"
    assert parse_time(f"2020-01-01T{hour}:48:00.123456789{zone}") == dt.datetime(
        2020,
        1,
        1,
        4,
        48,
        0,
        123456,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "aia_2020-01-01T99:48:30Z.fits",
        "hmi_20200101_994830_TAI.fits",
        "cso_20200230044830_LL.fits",
        "aia_2020-01-01Tbroken.fits",
    ],
)
def test_invalid_full_timestamp_never_falls_back_to_date_only(filename):
    from solar_toolkit.time import extract_time_from_filename

    with pytest.raises(ValueError, match="Could not extract time"):
        extract_time_from_filename(filename)


def test_generic_nearest_and_range_keep_nanosecond_string_precision():
    from solar_toolkit.time import filter_by_time_range, nearest_by_time

    rows = [
        ("one", "2020-01-01T04:48:00.000000001"),
        ("two", "2020-01-01T04:48:00.000000002"),
    ]
    assert nearest_by_time("2020-01-01T04:48:00.000000002", rows) == rows[1]
    assert (
        nearest_by_time(
            "2020-01-01T04:48:00.000000003", rows, max_diff_seconds="0.000000001"
        )
        == rows[1]
    )
    assert (
        nearest_by_time(
            "2020-01-01T04:48:00.000000003", rows, max_diff_seconds="0.000000000"
        )
        is None
    )
    assert filter_by_time_range(rows, rows[1][1], rows[1][1]) == [rows[1]]
    assert filter_by_time_range(rows, dt.date(2020, 1, 1), "2020-01-02") == rows
    with pytest.raises(ValueError, match="start_time"):
        filter_by_time_range(rows, "2020-01-02", "2020-01-01")


@pytest.mark.parametrize(
    "threshold", [float("nan"), float("inf"), -1, "-0.000000000001"]
)
def test_generic_nearest_rejects_invalid_tolerance_even_for_empty_input(threshold):
    from solar_toolkit.time import nearest_by_time

    with pytest.raises(ValueError):
        nearest_by_time("2020-01-01T04:48:00", [], max_diff_seconds=threshold)


@pytest.mark.parametrize(
    "fraction,microseconds",
    [("6", 600000), ("681", 681000), ("123456", 123456), ("123456789", 123456)],
)
def test_explicit_radio_underscore_fraction_is_not_discarded(fraction, microseconds):
    from solar_toolkit.time import extract_time_from_filename, normalize_time

    assert extract_time_from_filename(
        f"radio_20200101_044837_{fraction}.fits"
    ) == dt.datetime(
        2020,
        1,
        1,
        4,
        48,
        37,
        microseconds,
    )
    stamp = normalize_time(f"20200101_044837_{fraction}")
    assert stamp.utc == "2020-01-01T04:48:37." + fraction.ljust(9, "0")


def test_filename_fraction_rules_distinguish_aia_band_and_ambiguous_suffix():
    from solar_toolkit.time import extract_time_from_filename

    expected = dt.datetime(2020, 1, 1, 4, 48, 37)
    assert extract_time_from_filename("aia_20200101_044837Z_211.fits") == expected
    assert extract_time_from_filename("aia_20200101_044837_211.fits") == expected
    assert (
        extract_time_from_filename("cso_20200101_044837_681_LL.fits").microsecond
        == 681000
    )
    assert extract_time_from_filename(
        "hmi_20200101_044837_681_TAI.fits"
    ) == dt.datetime(
        2020,
        1,
        1,
        4,
        48,
        0,
        681000,
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        extract_time_from_filename("generic_20200101_044837_681.fits")
    with pytest.raises(ValueError, match="exceeds nanoseconds"):
        extract_time_from_filename("radio_20200101_044837_1234567890.fits")


def test_scientific_filename_extraction_retains_nanoseconds_and_rejects_date_only():
    from solar_toolkit.time import (
        extract_time_from_filename,
        extract_timestamp_from_filename,
    )

    one = extract_timestamp_from_filename("aia_2020-01-01T04:48:00.000000001Z_211.fits")
    two = extract_timestamp_from_filename("aia_2020-01-01T04:48:00.000000002Z_211.fits")
    assert two.tai_ns - one.tai_ns == 1
    assert one.source == "filename"
    assert one.raw == "2020-01-01T04:48:00.000000001Z"
    radio = extract_timestamp_from_filename("radio_20200101_044837_123456789.fits")
    assert radio.utc == "2020-01-01T04:48:37.123456789"
    assert radio.raw == "20200101_044837_123456789"
    tai = extract_timestamp_from_filename("hmi_20200101_044837.123456789_TAI.fits")
    assert tai.utc == "2020-01-01T04:48:00.123456789"
    assert extract_time_from_filename("context_20200101.png") == dt.datetime(2020, 1, 1)
    for filename in [
        "context_20200101.png",
        "aia_2020-01-01Tbroken.fits",
        "aia_2020-01-01T04:48:00.1234567890Z.fits",
    ]:
        with pytest.raises(ValueError):
            extract_timestamp_from_filename(filename)
