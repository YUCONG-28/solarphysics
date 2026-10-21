"""Astronomical time inputs retain their scale, units and valid calendar."""

from datetime import datetime

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from solar_toolkit.time import (
    extract_timestamp_from_filename,
    nearest_by_time,
    normalize_time,
    parse_time,
    seconds_to_ns,
)


@pytest.mark.parametrize("scale", ["utc", "tai", "tt"])
@pytest.mark.parametrize("clock", ["00:00:60", "23:59:60", "23:59:61"])
def test_invalid_calendar_seconds_are_not_rolled_forward(scale, clock):
    with pytest.raises(ValueError, match="second|time"):
        normalize_time(f"2000-01-01T{clock}", scale=scale)


@pytest.mark.parametrize(
    "text",
    [
        "2016-12-31T23:59:60.123456789Z",
        "2017-01-01T07:59:60.123456789+08:00",
    ],
)
def test_real_utc_leap_seconds_preserve_all_digits(text):
    result = normalize_time(text)
    assert result.utc == "2016-12-31T23:59:60.123456789"
    assert result.shift(1).utc == "2017-01-01T00:00:00.123456789"
    with pytest.raises(ValueError, match="second"):
        result.to_datetime()


def test_legacy_parser_converts_astropy_time_instead_of_stringifying_scale():
    original = Time("2000-01-01T00:00:32", scale="tai")
    assert parse_time(original) == datetime(2000, 1, 1)
    assert original.scale == "tai"


def test_filename_uses_basename_time_before_timestamped_parent_directory():
    result = extract_timestamp_from_filename(
        "/synthetic/20000101_000000/radio_20000102_030405.123.fits"
    )
    assert result.utc == "2000-01-02T03:04:05.123000000"


@pytest.mark.parametrize("value", [TimeDelta(1 / 86400, format="jd"), 1000 * u.ms])
def test_explicit_duration_units_are_seconds(value):
    assert seconds_to_ns(value) == 1_000_000_000


def test_masked_astropy_time_is_rejected_without_filling_an_observation():
    value = Time(np.ma.array([2451545.0], mask=[True]), format="jd")[0]
    with pytest.raises(ValueError, match="masked"):
        normalize_time(value)


def test_duration_sign_is_checked_before_nanosecond_rounding():
    with pytest.raises(ValueError, match="nonnegative"):
        seconds_to_ns(-0.1 * u.ns, nonnegative=True)
    assert seconds_to_ns(-0.1 * u.ns) == 0


def test_nearest_tolerance_accepts_explicit_duration_units():
    observation = "2000-01-01T00:00:00.001Z"
    assert (
        nearest_by_time(
            "2000-01-01T00:00:00Z", [observation], max_diff_seconds=1 * u.ms
        )
        == observation
    )


def test_explicit_astropy_calendar_format_cannot_bypass_second_validation():
    with pytest.raises(ValueError, match="second"):
        normalize_time("2000:001:23:59:60", format="yday")
    assert normalize_time("2016:366:23:59:60", format="yday").utc == (
        "2016-12-31T23:59:60.000000000"
    )


@pytest.mark.parametrize("offset", ["+01:60", "-0160", "+24:00", "-24:00"])
def test_invalid_timezone_fields_are_not_normalized(offset):
    with pytest.raises(ValueError, match="Timezone"):
        normalize_time("2000-01-01T00:00:00" + offset)


def test_explicit_calendar_format_does_not_bypass_leap_precision_validation():
    with pytest.raises(ValueError, match="nanosecond"):
        normalize_time("2000:001:00:00:60.0000000001", format="yday")


@pytest.mark.parametrize("value", ["2000", "2000-01", "2000-01-01"])
def test_datetime64_requires_a_clock(value):
    with pytest.raises(ValueError, match="including time"):
        normalize_time(np.datetime64(value))


@pytest.mark.parametrize("unit", ["h", "m", "s", "ms", "us", "ns"])
def test_datetime64_clock_units_preserve_the_instant_without_integer_cast(unit):
    assert (
        normalize_time(np.datetime64("2000-01-01T00", unit)).utc
        == "2000-01-01T00:00:00.000000000"
    )


def test_legacy_parser_uses_the_same_strict_timezone_validation():
    with pytest.raises(ValueError, match="Timezone"):
        parse_time("2000-01-01T00:00:00+01:60")
