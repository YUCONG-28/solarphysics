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


def test_radio_overlay_uses_world_grid_and_keeps_missing_pixels(tmp_path, monkeypatch):
    from astropy.wcs import WCS
    from solar_toolkit.map.coordinates import hpc_pixel_grid

    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2,
            "CRPIX2": 2,
            "CRVAL1": 0,
            "CRVAL2": 0,
            "CD1_1": 0.001,
            "CD1_2": -0.002,
            "CD2_1": 0.002,
            "CD2_2": 0.001,
        }
    )
    data = np.arange(9, dtype=float).reshape(3, 3)
    data[1, 1] = np.nan
    path = tmp_path / "radio.fits"
    fits.PrimaryHDU(data, header).writeto(path)
    before = path.read_bytes()
    values, returned_header = overlay.load_radio(str(path), return_wcs=True)
    coordinates = hpc_pixel_grid(returned_header, values.shape)
    y, x = np.indices(values.shape)
    longitude, latitude = WCS(returned_header).pixel_to_world_values(x, y)
    expected_x = ((longitude + 180) % 360 - 180) * 3600
    np.testing.assert_allclose(coordinates[0], expected_x, atol=1e-7)
    np.testing.assert_allclose(coordinates[1], latitude * 3600, atol=1e-7)

    class Axes:
        def contour(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    ax = Axes()
    monkeypatch.setitem(overlay.CONFIG, "radio_smooth_sigma", 0)
    overlay._draw_radio_gradient(ax, values, [], radio_coordinates=coordinates)
    np.testing.assert_array_equal(ax.args[0], coordinates[0])
    np.testing.assert_array_equal(ax.args[1], coordinates[1])
    np.testing.assert_array_equal(ax.args[2], data)
    assert "extent" not in ax.kwargs
    assert path.read_bytes() == before
    # The legacy rectangular display must reject rotated coordinates.
    with pytest.raises(ValueError):
        overlay.load_radio(str(path))


def test_radio_gradient_excludes_masked_extreme_and_constant_image(monkeypatch):
    class Axes:
        def contour(self, *args, **kwargs):
            self.values = args[-1]

    ax = Axes()
    data = np.ma.array([[1, 2, 3], [4, 1e20, 6], [7, 8, 9]], mask=False)
    data.mask[1, 1] = True
    original = data.copy()
    monkeypatch.setitem(overlay.CONFIG, "radio_smooth_sigma", 1)
    overlay._draw_radio_gradient(ax, data, [0, 3, 0, 3])
    assert np.isnan(ax.values[1, 1])
    assert np.nanmax(ax.values) < 9
    np.testing.assert_array_equal(data.data, original.data)
    np.testing.assert_array_equal(data.mask, original.mask)
    assert overlay._draw_radio_gradient(ax, np.ones((3, 3)), [0, 3, 0, 3]) == []
    assert overlay._draw_radio_gradient(ax, np.full((3, 3), np.nan), [0, 3, 0, 3]) == []
