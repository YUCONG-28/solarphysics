from __future__ import annotations

import datetime as dt

import numpy as np


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
