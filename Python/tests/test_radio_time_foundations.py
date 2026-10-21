"""Radio consumers share exact clocks and explicit invalid-sample policies."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from solar_toolkit.radio._image_naming import dataframe_time_bounds
from solar_toolkit.radio.dart_spectrogram import (
    _block_mean_times,
    _read_time_axis,
    _time_index_range,
)
from solar_toolkit.radio.drift_rate import calculate_drift_rate_from_line
from solar_toolkit.radio.frequency_priority_diagnostics import (
    _drift_rate_from_row,
    _frequency_on_drift_line,
    _parse_radio_datetime,
)
from solar_toolkit.radio.height_comparison import _frequency_on_drift_selection
from solar_toolkit.radio.height_plots import _clean_height_df
from solar_toolkit.radio.io import parse_datetime_value, parse_timestamp_value
from solar_toolkit.radio.source_geometry import normalize_source_observations
from solar_toolkit.radio.trajectory import normalize_centers_dataframe, summarize_motion


def _line(**updates):
    return {
        "t_start": "2016-12-31T23:59:59Z",
        "t_end": "2017-01-01T00:00:00Z",
        "f_start_mhz": 180.0,
        "f_end_mhz": 120.0,
        **updates,
    }


def _center(**updates):
    return {
        "time": "2000-01-01T00:00:00Z",
        "freq": 150.0,
        "center_x_arcsec": 1000.0,
        "center_y_arcsec": 10.0,
        **updates,
    }


def test_shared_exact_parser_retains_leap_seconds_and_datetime_view_rejects():
    stamp = parse_timestamp_value("2016-12-31T23:59:60.123456789Z")
    assert stamp.utc == "2016-12-31T23:59:60.123456789"
    with pytest.raises(ValueError, match="second"):
        parse_datetime_value(stamp)


def test_drift_duration_counts_the_leap_second_without_mutating_input():
    line = _line()
    before = dict(line)
    result = calculate_drift_rate_from_line(line)
    assert result.duration_s == 2
    assert result.drift_rate_mhz_s == -30
    assert line == before


def test_drift_duration_keeps_submicrosecond_endpoints():
    result = calculate_drift_rate_from_line(
        _line(
            t_start="2000-01-01T00:00:00.000000001Z",
            t_end="2000-01-01T00:00:00.000000011Z",
        )
    )
    assert result.duration_s == 1e-8
    assert result.drift_rate_mhz_s == -6e9
    assert result.quality_flag == "ok"


@pytest.mark.parametrize(
    "interpolate", [_frequency_on_drift_line, _frequency_on_drift_selection]
)
def test_drift_interpolation_uses_si_time_at_a_leap_second(interpolate):
    stamp = parse_timestamp_value("2016-12-31T23:59:60Z")
    assert interpolate(stamp, _line(), 0.0) == 150


def test_derived_drift_rate_uses_the_exact_shared_clock():
    assert _drift_rate_from_row(_line()) == -30


@pytest.mark.parametrize(
    "value", [float(20000101044829509), "bad20000101044829509", "20000101044829509junk"]
)
def test_diagnostic_parser_does_not_clean_invalid_samples_into_dates(value):
    assert pd.isna(_parse_radio_datetime(value))


def test_diagnostic_parser_keeps_nanoseconds():
    expected = pd.Timestamp("2000-01-01T00:00:00.000000001")
    assert _parse_radio_datetime(expected.isoformat()) == expected


@pytest.mark.parametrize("flag", [None, np.nan, pd.NA, "", "unknown"])
def test_explicit_unknown_geometry_quality_is_invalid_but_row_is_retained(flag):
    original = pd.DataFrame([_center(quality_flag=flag)])
    before = original.copy(deep=True)
    result = normalize_source_observations(original)
    assert len(result) == 1 and not result.iloc[0].source_valid
    pd.testing.assert_frame_equal(original, before)


def test_absent_geometry_quality_remains_a_legacy_table():
    assert normalize_source_observations(pd.DataFrame([_center()])).iloc[0].source_valid


def test_trajectory_times_keep_nanoseconds_and_si_duration():
    fine = normalize_centers_dataframe(
        pd.DataFrame([_center(time="2000-01-01T00:00:00.000000001Z")])
    )
    assert fine.iloc[0].obs_time.nanosecond == 1
    across_leap = normalize_centers_dataframe(
        pd.DataFrame([_center(time=_line()[key]) for key in ("t_start", "t_end")])
    )
    assert summarize_motion(across_leap).iloc[0].duration_sec == 2


def test_radio_naming_preserves_compact_fractional_time_bounds():
    start, end = dataframe_time_bounds(
        pd.DataFrame({"time": [20000101044829509, 20000101044829809]})
    )
    assert start == datetime(2000, 1, 1, 4, 48, 29, 509000, tzinfo=timezone.utc)
    assert end == datetime(2000, 1, 1, 4, 48, 29, 809000, tzinfo=timezone.utc)


def test_height_plot_time_is_calendar_time_without_implicit_numeric_epoch():
    frame = _clean_height_df(
        pd.DataFrame([{"time": 20000101044829509, "frequency_mhz": 150}])
    )
    assert frame.iloc[0].time_dt == pd.Timestamp("2000-01-01T04:48:29.509")


@pytest.mark.parametrize("date", [(2000, 1, 1), (2016, 12, 31)])
def test_dart_datetime_reader_never_rolls_second_60_into_next_day(monkeypatch, date):
    hdus = fits.HDUList([fits.PrimaryHDU(np.array([[*date, 23, 59, 60]], dtype=float))])
    monkeypatch.setattr(
        "solar_toolkit.radio.dart_spectrogram.fits.open", lambda *a, **k: hdus
    )
    with pytest.raises(ValueError, match="second"):
        _read_time_axis(Path("in-memory-synthetic"))


def test_dart_block_time_mean_uses_tai_before_returning_legacy_datetime():
    values = (
        datetime(2016, 12, 31, 23, 59, 58, tzinfo=timezone.utc),
        datetime(2017, 1, 1, tzinfo=timezone.utc),
    )
    assert _block_mean_times(values, 2) == (
        datetime(2016, 12, 31, 23, 59, 59, 500000, tzinfo=timezone.utc),
    )


def test_dart_window_keeps_submicrosecond_requested_bounds():
    values = (
        datetime(2000, 1, 1, tzinfo=timezone.utc),
        datetime(2000, 1, 1, microsecond=1, tzinfo=timezone.utc),
    )
    assert _time_index_range(
        values, ("2000-01-01T00:00:00.000000001Z", "2000-01-01T00:00:00.000001Z")
    ) == (1, 1)


def test_trajectory_playback_uses_the_shared_compact_calendar_parser():
    from solar_toolkit.radio.trajectory import filter_time_range, frame_times

    raw = pd.DataFrame({"obs_time": [20000101044829509, "invalid"]})
    assert frame_times(raw) == [pd.Timestamp("2000-01-01T04:48:29.509")]
    selected = filter_time_range(raw, start=20000101044829509, end=20000101044829509)
    assert len(selected) == 1
    with pytest.raises(ValueError, match="calendar"):
        filter_time_range(raw, start=1.5)


def test_aia_nearest_uses_si_seconds_and_nanosecond_ordering():
    from solar_toolkit.aia.background import find_nearest_aia

    table = pd.DataFrame({"path": ["synthetic.fits"], "obs_time": [_line()["t_start"]]})
    match = find_nearest_aia(table, _line()["t_end"], max_dt_seconds=1.5)
    assert match.status == "too_far" and match.delta_seconds == 2
    table["obs_time"] = ["2000-01-01T00:00:00.000000001Z"]
    match = find_nearest_aia(
        table, "2000-01-01T00:00:00.000000011Z", max_dt_seconds=1e-8
    )
    assert match.status == "matched" and match.delta_seconds == 1e-8


def test_roi_active_bucket_distance_counts_the_leap_second():
    from solar_toolkit.radio.roi_lightcurve import _ActiveTimeIndex

    index = _ActiveTimeIndex(
        [(0, SimpleNamespace(obs_time=datetime(2016, 12, 31, 23, 59, 59)))]
    )
    assert index.nearest(datetime(2017, 1, 1))[0] == 2


def test_roi_pair_threshold_and_midpoint_use_si_time(monkeypatch):
    from solar_toolkit.radio import roi_lightcurve as roi
    from solar_toolkit.radio.centers import POL_LCP, POL_RCP, RadioImage

    def image(time, pol):
        return RadioImage(
            Path("synthetic.fits"), 0, np.ones((2, 2)), fits.Header(), pol, 150, time
        )

    monkeypatch.setattr(roi, "_pair_incompatibility", lambda *args: "")
    left = image(datetime(2016, 12, 31, 23, 59, 59), POL_LCP)
    right = image(datetime(2017, 1, 1), POL_RCP)
    paired, skipped = roi._make_paired_sum_images([left, right], tolerance_sec=1.5)
    assert not paired and len(skipped) == 2
    left.obs_time = datetime(2016, 12, 31, 23, 59, 58)
    paired, skipped = roi._make_paired_sum_images([left, right], tolerance_sec=3)
    assert not skipped
    assert paired[0].obs_time == datetime(2016, 12, 31, 23, 59, 59, 500000)


def test_cso_lazy_and_overlay_windows_use_si_offset_axis():
    from solar_toolkit.radio._lazy_spectrogram import LazySpectrogram
    from solar_toolkit.radio.spectrogram import _spectrogram_overlap_segments

    base, end = datetime(2016, 12, 31, 23, 59, 59), datetime(2017, 1, 1)
    axis = np.array([0.0, 1.0, 2.0])
    data = np.arange(6.0).reshape(2, 3)
    lazy = LazySpectrogram(
        data, axis, np.array([100, 200]), "LL", "2016-12-31", "K", base
    )
    rebinned, times, _ = lazy.read_slice_rebinned(base, end, 100, 200, 1, 1)
    np.testing.assert_array_equal(rebinned, data)
    np.testing.assert_array_equal(times, axis)
    meta = {
        "dt_base": base,
        "file_start": base,
        "file_end": end,
        "time_arr": axis,
        "path": "synthetic.fits",
    }
    segments = _spectrogram_overlap_segments({}, base, end, [meta])
    assert (segments[0]["ti0"], segments[0]["ti1"]) == (0, 2)


def test_hmi_overlay_rejects_match_outside_si_tolerance(monkeypatch, tmp_path):
    import astropy.coordinates
    import sunpy.map

    from solar_toolkit.hmi.overlay import run_overlay_workflow

    base, end = datetime(2016, 12, 31, 23, 59, 59), datetime(2017, 1, 1)
    monkeypatch.setattr(
        "solar_toolkit.io.discovery.get_sorted_fits_files",
        lambda path, **kw: (
            [("hmi", base)] if path.endswith("hmi") else [("aia1", end), ("aia2", end)]
        ),
    )
    monkeypatch.setattr(
        "solar_toolkit.visualization.plotting.setup_chinese_font", lambda: None
    )
    monkeypatch.setattr(
        "solar_toolkit._utils.memory.monitor_memory_usage", lambda *args: None
    )
    monkeypatch.setattr(
        "solar_toolkit._utils.memory.optimized_gc_collect", lambda: None
    )
    monkeypatch.setattr(astropy.coordinates, "SkyCoord", lambda **kwargs: None)
    calls = []
    reference = SimpleNamespace(
        coordinate_frame=None, submap=lambda *args, **kw: SimpleNamespace(wcs=None)
    )
    monkeypatch.setattr(sunpy.map, "Map", lambda path: calls.append(path) or reference)
    assert (
        run_overlay_workflow(
            tmp_path / "aia",
            tmp_path / "hmi",
            tmp_path / "out",
            roi_bounds=(-1, -1, 1, 1),
            max_time_diff_seconds=1.5,
            show_progress=False,
        )
        == []
    )
    assert calls == ["aia2"]  # No science image reads occur for rejected matches.
