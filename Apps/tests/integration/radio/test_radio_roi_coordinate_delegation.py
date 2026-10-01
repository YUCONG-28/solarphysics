"""Synthetic WCS regression for the application ROI compatibility path."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.roi_lightcurve import roi_lightcurve_application as app
from solar_toolkit.radio import roi_lightcurve as scientific


def _header(kind: str) -> fits.Header:
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 3.0,
            "CRPIX2": 2.0,
            "CRVAL1": 10.0,
            "CRVAL2": -7.0,
            "CDELT1": 2.0,
            "CDELT2": 5.0,
        }
    )
    if kind == "pc":
        header.update({"PC1_1": 0.0, "PC1_2": -1.0, "PC2_1": 1.0, "PC2_2": 0.0})
    elif kind == "cd":
        header.update({"CD1_1": 0.0, "CD1_2": -2.0, "CD2_1": 5.0, "CD2_2": 0.0})
    else:
        header["CROTA2"] = 90.0
    return header


@pytest.mark.parametrize("kind", ["pc", "cd", "crota"])
def test_roi_coordinates_masks_and_statistics_match_scientific_library(kind) -> None:
    header = _header(kind)
    image = np.arange(63, dtype=float).reshape(7, 9)
    yy, xx = np.indices(image.shape, dtype=float)
    if kind == "crota":
        expected_x = 10.0 - 5.0 * (yy - 1.0)
        expected_y = -7.0 + 2.0 * (xx - 2.0)
    else:
        expected_x = 10.0 - 2.0 * (yy - 1.0)
        expected_y = -7.0 + 5.0 * (xx - 2.0)
    actual_x, actual_y = app._pixel_coordinates_hpc_arcsec(header, xx, yy)
    np.testing.assert_allclose(actual_x, expected_x, atol=1e-12)
    np.testing.assert_allclose(actual_y, expected_y, atol=1e-12)

    roi = scientific.RadioRoi.from_box(5.0, -10.0, 9.0, 5.0)
    expected_mask = (
        (expected_x >= 5.0)
        & (expected_x <= 9.0)
        & (expected_y >= -10.0)
        & (expected_y <= 5.0)
    )
    np.testing.assert_array_equal(
        app.build_radio_roi_mask(header, image.shape, roi), expected_mask
    )
    application_measurement = app.measure_radio_roi(image, header, roi)
    library_measurement = scientific.measure_radio_roi(image, header, roi)
    assert application_measurement == library_measurement
    assert application_measurement["roi_pixel_count"] == int(expected_mask.sum())
    assert application_measurement["raw_sum"] == float(image[expected_mask].sum())
    assert application_measurement["raw_mean"] == float(image[expected_mask].mean())
    assert application_measurement["raw_peak"] == float(image[expected_mask].max())
