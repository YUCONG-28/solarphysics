"""Small deterministic inputs for precomputed brightness-temperature helpers."""

import datetime as dt

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.xray_dem.brightness_temperature import (
    AiaMapGeometry,
    brightness_temperature_extent,
    load_brightness_temperature,
)


def reference_fits(tmp_path, *, time="2000-01-01T00:00:01.123456", **cards):
    header = fits.Header(
        {
            "CRPIX1": 2,
            "CRPIX2": 1,
            "CRVAL1": 10,
            "CRVAL2": -20,
            "CDELT1": -2,
            "CDELT2": 4,
            "RSUN_OBS": 960,
            **cards,
        }
    )
    if time is not None:
        header["T_OBS"] = time
    path = tmp_path / "aia_20000101T000002_171.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 3)), header=header)]
    ).writeto(path)
    return path


def test_reference_geometry_uses_pixel_edges_and_preserves_fractional_utc(tmp_path):
    geometry = AiaMapGeometry(reference_fits(tmp_path))
    assert (geometry.nx, geometry.ny) == (3, 2)
    assert geometry.extent == [7, 13, -22, -14]
    assert geometry.sun_center == (10, -20)
    assert geometry.rsun == 960
    assert geometry.obs_time == "2000-01-01 00:00:01.123456 UT"
    assert geometry.obs_dt == dt.datetime(2000, 1, 1, 0, 0, 1, 123456)


def test_reference_time_converts_tai_metadata_to_utc(tmp_path):
    geometry = AiaMapGeometry(
        reference_fits(tmp_path, time="2000-01-01T00:00:00", TIMESYS="TAI")
    )
    assert geometry.obs_dt == dt.datetime(1999, 12, 31, 23, 59, 28)
    assert geometry.obs_time == "1999-12-31 23:59:28 UT"


def test_reference_uses_filename_only_when_observation_header_is_missing(tmp_path):
    path = reference_fits(tmp_path, time=None)
    assert AiaMapGeometry(path).obs_time == "2000-01-01 00:00:02 UT"
    unknown = tmp_path / "unlabelled.fits"
    path.rename(unknown)
    geometry = AiaMapGeometry(unknown)
    assert geometry.obs_time == "Unknown"
    assert geometry.obs_dt is None


def test_display_modes_are_explicit_and_do_not_modify_full_extent(tmp_path):
    geometry = AiaMapGeometry(reference_fits(tmp_path))
    options = dict(solar_padding_factor=1.25, x_range=(-5, 6), y_range=(-7, 8))
    assert geometry.display_extent("custom", **options) == [-5, 6, -7, 8]
    assert geometry.display_extent("solar_disk", **options) == [
        -1190,
        1210,
        -1220,
        1180,
    ]
    extent = geometry.display_extent("full", **options)
    extent[0] = 100
    assert geometry.display_extent("unknown", **options) == [7, 13, -22, -14]


def test_brightness_temperature_converts_kelvin_and_fills_only_display_values(tmp_path):
    path = tmp_path / "temperature.npy"
    source = np.array([[1e6, np.nan], [np.inf, -np.inf]])
    np.save(path, source)

    result = load_brightness_temperature(path)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[1, 0], [0, 0]])
    np.testing.assert_array_equal(np.load(path), source)


@pytest.mark.parametrize("shape", [(2,), (1, 2, 3), (0, 2)])
def test_brightness_temperature_requires_a_nonempty_image(tmp_path, shape):
    path = tmp_path / "temperature.npy"
    np.save(path, np.zeros(shape))
    with pytest.raises(ValueError, match="2-D"):
        load_brightness_temperature(path)


def test_grid_extent_uses_first_and_last_pixel_centers():
    assert brightness_temperature_extent(
        pixel_size=2, x_range=(-2, 2), y_range=(-3, 3)
    ) == [-3, 3, -4, 4]


@pytest.mark.parametrize("pixel_size", [0, -1, np.nan, np.inf])
def test_grid_extent_rejects_invalid_pixel_size(pixel_size):
    with pytest.raises(ValueError, match="pixel_size"):
        brightness_temperature_extent(
            pixel_size=pixel_size, x_range=(0, 1), y_range=(0, 1)
        )
