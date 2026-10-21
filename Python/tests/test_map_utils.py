from __future__ import annotations

import datetime as dt

import numpy as np
import pytest


def test_map_extent_obs_time_and_normalization_from_header():
    from solar_toolkit.map import get_display_extent, get_map_obs_time, normalize_image

    header = {
        "DATE-OBS": "2000-01-01T04:48:37Z",
        "NAXIS1": 4,
        "NAXIS2": 2,
        "CRVAL1": 0.0,
        "CRPIX1": 2.0,
        "CDELT1": 1.0,
        "CRVAL2": 10.0,
        "CRPIX2": 1.0,
        "CDELT2": 2.0,
    }

    assert get_map_obs_time(header) == dt.datetime(2000, 1, 1, 4, 48, 37)
    assert get_display_extent(header) == (-1.5, 2.5, 9.0, 13.0)
    assert np.allclose(normalize_image(np.array([1.0, 2.0, 3.0])), [0.0, 0.5, 1.0])


def test_crop_roi_uses_pixel_bounds():
    from solar_toolkit.map import crop_roi

    data = np.arange(25).reshape(5, 5)
    cropped = crop_roi(data, x_range=(1, 4), y_range=(2, 5))

    assert cropped.tolist() == [[11, 12, 13], [16, 17, 18], [21, 22, 23]]


def test_map_observation_time_converts_astropy_tai_without_losing_precision():
    from types import SimpleNamespace

    from astropy.time import Time

    from solar_toolkit.map import get_map_obs_time

    value = Time("2000-01-01T00:00:32.123456", scale="tai")
    assert get_map_obs_time(SimpleNamespace(date=value)) == dt.datetime(
        2000, 1, 1, microsecond=123456
    )


@pytest.mark.parametrize("key", ["DATE-OBS", "DATE_OBS", "date-obs"])
def test_header_observation_time_honors_timesys_and_separate_clock(key):
    from solar_toolkit.map import get_map_obs_time

    assert get_map_obs_time(
        {key: "2000-01-01T00:00:32", "TIMESYS": "TAI"}
    ) == dt.datetime(2000, 1, 1)
    assert get_map_obs_time(
        {key: "2000-01-01", "TIME-OBS": "00:00:32", "TIMESYS": "TAI"}
    ) == dt.datetime(2000, 1, 1)


def test_header_observation_time_rejects_invalid_or_incomplete_time():
    from solar_toolkit.map import get_map_obs_time

    with pytest.raises(ValueError):
        get_map_obs_time({"DATE-OBS": "2000-01-01"})
    with pytest.raises(KeyError):
        get_map_obs_time({})
