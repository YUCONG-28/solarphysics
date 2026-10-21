"""Legacy DEM plotting commands share the scientific geometry implementation."""

from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.xray_dem import aia_dem_inversion as temperature
from solar_apps.workflows.xray_dem import dem_radio_source_overlay as overlay
from solar_toolkit.xray_dem.brightness_temperature import AiaMapGeometry


@pytest.mark.parametrize("workflow", [temperature, overlay])
def test_dem_recipes_keep_legacy_helpers_and_independent_configuration(
    workflow, tmp_path, monkeypatch
):
    assert workflow.SolarMap is AiaMapGeometry
    path = tmp_path / "temperature.npy"
    np.save(path, [[1e6, np.nan], [2e6, np.inf]])
    np.testing.assert_array_equal(workflow.load_tb(str(path)), [[1, 0], [2, 0]])
    monkeypatch.setattr(
        workflow,
        "CONFIG",
        {
            **workflow.CONFIG,
            "tb_pixel_size": 4,
            "tb_xmin": -4,
            "tb_xmax": 4,
            "tb_ymin": -6,
            "tb_ymax": 6,
        },
    )
    assert workflow.get_tb_extent() == [-6, 6, -8, 8]
    assert workflow.build_parser().parse_args([]) is not None


def test_radio_time_keeps_fraction_for_matching_and_converts_tai(tmp_path):
    path = tmp_path / "radio.fits"
    fits.PrimaryHDU(
        np.ones((2, 2)),
        header=fits.Header({"DATE-OBS": "2000-01-01T00:00:32.375", "TIMESYS": "TAI"}),
    ).writeto(path)

    label = overlay.get_radio_time(str(path))

    assert label == "2000-01-01 00:00:00.375 UT"
    assert overlay._str_to_datetime(label) == datetime(2000, 1, 1, 0, 0, 0, 375000)
    assert overlay._str_to_datetime("Unknown") is None


def test_radio_filename_fallback_preserves_legacy_single_digit_month():
    assert overlay.get_radio_time("149MHz_2000101_000000_125.fits") == (
        "2000-01-01 00:00:00.125 UT"
    )


def test_radio_match_distinguishes_subsecond_frames(tmp_path, monkeypatch):
    paths = []
    for index, fraction in enumerate((".1", ".8")):
        path = tmp_path / f"radio-{index}.fits"
        fits.PrimaryHDU(
            np.ones((2, 2)),
            header=fits.Header({"DATE-OBS": f"2000-01-01T00:00:00{fraction}"}),
        ).writeto(path)
        paths.append(str(path))
    monkeypatch.setitem(overlay.CONFIG, "time_match_level", "minute")

    best_path, _, difference = overlay.find_matching_radio(
        paths, datetime(2000, 1, 1, 0, 0, 0, 700000)
    )

    assert best_path == paths[1]
    assert difference == pytest.approx(0.1)
