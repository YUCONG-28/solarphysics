"""App time buckets retain calendar policy while distances use SI seconds."""

from datetime import datetime

import numpy as np
import pytest


def test_dem_radio_match_counts_a_leap_second_without_changing_bucket_policy(
    monkeypatch,
):
    from solar_apps.workflows.xray_dem import dem_radio_source_overlay as workflow

    monkeypatch.setitem(workflow.CONFIG, "time_match_level", "any")
    monkeypatch.setattr(workflow, "get_radio_time", lambda path: "2016-12-31T23:59:59")
    path, _, distance = workflow.find_matching_radio(
        ["synthetic.fits"], datetime(2017, 1, 1)
    )
    assert path == "synthetic.fits" and distance == 2


@pytest.mark.parametrize(
    "end", ["2017-01-01T00:00:00Z", "2016-12-31T23:59:59.000000010Z"]
)
def test_app_drift_rate_delegates_exact_elapsed_time(end):
    from solar_apps.workflows.radio.drift_selection_application import (
        calculate_drift_rate_from_line,
    )

    result = calculate_drift_rate_from_line(
        {
            "t_start": "2016-12-31T23:59:59Z",
            "t_end": end,
            "f_start_mhz": 180,
            "f_end_mhz": 120,
        }
    )
    assert result.duration_s == (2 if end.startswith("2017") else 1e-8)


def test_app_cso_bin_selection_includes_full_si_offset_window():
    from solar_apps.workflows.radio.cso_workflow import (
        LazySpectrogram,
        PlotConfig,
        calc_bin_sizes,
        calc_bin_sizes_for_specs,
    )

    base, end = datetime(2016, 12, 31, 23, 59, 59), datetime(2017, 1, 1)
    spec = LazySpectrogram(
        np.ones((2, 3)),
        np.array([0.0, 1.0, 2.0]),
        np.array([100.0, 200.0]),
        "LL",
        "2016-12-31",
        "K",
        base,
    )
    config = PlotConfig(
        t_start=base,
        t_end=end,
        f_start=100,
        f_end=200,
        rebin_t_target=1,
        rebin_f_target=1,
    )
    assert calc_bin_sizes(spec, config) == (3, 2)
    assert calc_bin_sizes_for_specs([spec], config, base, end) == (3, 2)


def test_overlay_nearest_does_not_accept_a_two_second_gap_under_one_point_five():
    from solar_apps.workflows.radio._overlay_selection import _nearest_file_by_time

    assert (
        _nearest_file_by_time(
            ["synthetic.fits"],
            lambda _: datetime(2016, 12, 31, 23, 59, 59),
            datetime(2017, 1, 1),
            1.5,
        )
        is None
    )


def test_overlay_parser_never_cleans_invalid_trailing_time_text():
    from solar_apps.workflows.radio._overlay_selection import _parse_flexible_datetime

    assert _parse_flexible_datetime("20000101000000junk") is None
    assert _parse_flexible_datetime("2000-01-01T00:00:00.123456789Z").nanosecond == 789
    assert _parse_flexible_datetime("2000124_120000_123").day == 24


def test_source_background_window_counts_the_leap_second(monkeypatch):
    from solar_apps.workflows.radio import source_map_workflow  # noqa: F401
    from solar_apps.workflows.radio import _source_map_workflow_core as core

    monkeypatch.setattr(core.os, "listdir", lambda path: ["before.fits"])
    monkeypatch.setattr(
        core,
        "_radio_datetime_from_filename",
        lambda path, cfg: (
            datetime(2017, 1, 1)
            if path.endswith("current.fits")
            else datetime(2016, 12, 31, 23, 59, 59)
        ),
    )
    assert (
        core.find_background_files_for_current_radio(
            "/synthetic/current.fits", {"background_pre_event_seconds": 1.5}
        )
        == []
    )
