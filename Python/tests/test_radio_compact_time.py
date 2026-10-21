"""Compact CSV calendar times must not become Unix nanoseconds."""

from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from solar_toolkit.radio.centers import parse_datetime_value as parse_center_time
from solar_toolkit.radio.io import parse_datetime_value as parse_radio_time
from solar_toolkit.radio.trajectory import (
    load_centers_table,
    normalize_centers_dataframe,
    summarize_motion,
)


@pytest.mark.parametrize("parse", [parse_center_time, parse_radio_time])
@pytest.mark.parametrize(
    "value",
    ["20000101044829509", 20000101044829509, np.int64(20000101044829509)],
)
def test_compact_integer_and_text_have_the_same_utc_instant(parse, value):
    assert parse(value) == datetime(2000, 1, 1, 4, 48, 29, 509000)


@pytest.mark.parametrize("parse", [parse_center_time, parse_radio_time])
def test_compact_integer_seconds_do_not_require_a_fraction(parse):
    assert parse(20000101044829) == datetime(2000, 1, 1, 4, 48, 29)


@pytest.mark.parametrize("parse", [parse_center_time, parse_radio_time])
@pytest.mark.parametrize(
    "value", [float(20000101044829509), np.float64(20000101044829509), True, 1]
)
def test_ambiguous_numeric_scalars_have_no_implicit_epoch(parse, value):
    assert parse(value) is None


def _center(time):
    return {
        "time": time,
        "freq": 150.0,
        "center_x_arcsec": 10.0,
        "center_y_arcsec": 20.0,
        "quality_flag": "ok",
        "trajectory_valid": True,
        "overlay_valid": True,
        "observation_label": "synthetic",
    }


def test_integer_dataframe_times_keep_calendar_precision_and_input():
    original = pd.DataFrame([_center(20000101044829509), _center(20000101044829510)])
    before = original.copy(deep=True)
    result = normalize_centers_dataframe(original)
    assert result.obs_time.tolist() == [
        pd.Timestamp("2000-01-01T04:48:29.509"),
        pd.Timestamp("2000-01-01T04:48:29.510"),
    ]
    assert summarize_motion(result).iloc[0].duration_sec == pytest.approx(0.001)
    assert result.observation_label.tolist() == ["synthetic", "synthetic"]
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("time_column", ["time", "obs_time"])
@pytest.mark.parametrize("missing_row", [False, True])
def test_csv_loader_preserves_compact_text_before_numeric_inference(
    tmp_path, time_column, missing_row
):
    rows = [
        f"{time_column},freq,center_x_arcsec,center_y_arcsec,source_label",
        "20000101044829509,150,10,20,first",
        "20000101044829510,150,11,21,second",
    ]
    if missing_row:
        rows.append(",150,12,22,missing-time")
    text = "\n".join(rows) + "\n"
    csv_path = tmp_path / "synthetic-centers.csv"
    csv_path.write_text(text, encoding="utf-8")
    before = csv_path.read_bytes()
    file_result = load_centers_table(csv_path)
    buffer_result = load_centers_table(StringIO(text))
    expected = [
        pd.Timestamp("2000-01-01T04:48:29.509"),
        pd.Timestamp("2000-01-01T04:48:29.510"),
    ]
    assert file_result.obs_time.tolist() == expected
    assert file_result.source_label.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(file_result, buffer_result)
    assert csv_path.read_bytes() == before


@pytest.mark.parametrize(
    "value",
    [float(20000101044829509), np.float64(20000101044829509), 1.0, True, None, pd.NA],
)
def test_unrecoverable_numeric_or_missing_times_are_excluded(value):
    original = pd.DataFrame([_center(value)], dtype=object)
    before = original.copy(deep=True)
    for valid_only in (False, True):
        assert normalize_centers_dataframe(original, valid_only=valid_only).empty
    pd.testing.assert_frame_equal(original, before)


def test_numpy_datetime64_retains_legacy_timestamp_support():
    value = np.datetime64("2000-01-01T04:48:29.509")
    result = normalize_centers_dataframe(pd.DataFrame([_center(value)], dtype=object))
    assert result.iloc[0].obs_time == pd.Timestamp("2000-01-01T04:48:29.509")
