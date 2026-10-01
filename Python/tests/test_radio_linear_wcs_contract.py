"""Independent FITS linear-WCS oracles for centers and ROI selection."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from solar_toolkit.radio.centers import compute_source_center, pixel_to_hpc_arcsec
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    _pixel_coordinates_hpc_arcsec,
    build_radio_roi_mask,
    measure_radio_roi,
)


def _skew_header(units, representation):
    factors = [u.Unit(unit).to(u.arcsec) for unit in units]
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": units[0],
            "CUNIT2": units[1],
            "CRPIX1": 3.5,
            "CRPIX2": 1.75,
            "CRVAL1": 10.0 / factors[0],
            "CRVAL2": -20.0 / factors[1],
        }
    )
    # FITS PC acts on pixel offsets; CDELT then scales each world-coordinate row.
    pc = np.array([[1.0, 0.35], [-0.4, 0.8]])
    scales = np.array([2.0 / factors[0], -3.0 / factors[1]])
    if representation == "pc":
        header["CDELT1"], header["CDELT2"] = scales
        for row in range(2):
            for column in range(2):
                header[f"PC{row + 1}_{column + 1}"] = pc[row, column]
    else:
        cd = scales[:, None] * pc
        for row in range(2):
            for column in range(2):
                header[f"CD{row + 1}_{column + 1}"] = cd[row, column]
    return header


def _linear_oracle_arcsec(header, pixels):
    # These APIs intentionally approximate the projected plane as linear. Let
    # wcslib handle FITS matrix/reference-pixel semantics without TAN projection.
    linear = header.copy()
    linear["CTYPE1"] = linear["CTYPE2"] = "LINEAR"
    world = WCS(linear).all_pix2world(np.asarray(pixels, dtype=float), 0)
    factors = [u.Unit(header[f"CUNIT{axis}"]).to(u.arcsec) for axis in (1, 2)]
    return world * factors


@pytest.mark.parametrize(
    "units", [("arcsec", "arcsec"), ("deg", "arcmin"), ("rad", "deg")]
)
@pytest.mark.parametrize("representation", ["pc", "cd"])
@pytest.mark.parametrize("consumer", ["scalar", "vector", "center", "roi"])
def test_scalar_centers_and_vector_roi_match_independent_linear_wcs(
    units, representation, consumer
):
    header = _skew_header(units, representation)
    pixels = np.array([[0.0, 0.0], [4.0, 2.0], [2.5, 0.75]])
    expected = _linear_oracle_arcsec(header, pixels)
    if consumer == "scalar":
        scalar = np.array([pixel_to_hpc_arcsec(header, x, y) for x, y in pixels])
        np.testing.assert_allclose(scalar, expected, atol=1e-10)
        return
    if consumer == "vector":
        vector = np.column_stack(
            _pixel_coordinates_hpc_arcsec(header, pixels[:, 0], pixels[:, 1])
        )
        np.testing.assert_allclose(vector, expected, atol=1e-10)
        return

    image = np.zeros((5, 6))
    image[2, 4] = 10.0
    if consumer == "center":
        center = compute_source_center(image, header, centroid="geometric")
        np.testing.assert_allclose(
            [center["center_x_arcsec"], center["center_y_arcsec"]],
            expected[1],
            atol=1e-10,
        )
        return
    x, y = expected[1]
    roi = RadioRoi.from_box(x - 0.15, y - 0.15, x + 0.15, y + 0.15)
    mask = build_radio_roi_mask(header, image.shape, roi)
    expected_mask = np.zeros_like(image, dtype=bool)
    expected_mask[2, 4] = True
    np.testing.assert_array_equal(mask, expected_mask)
    measured = measure_radio_roi(image, header, roi)
    assert measured["quality_flag"] == "ok"
    assert measured["raw_sum"] == pytest.approx(10.0)
    assert measured["valid_pixel_count"] == 1


def test_legacy_crota_keeps_rotation_of_already_scaled_pixel_offsets():
    header = fits.Header(
        {
            "CRPIX1": 2.25,
            "CRPIX2": 3.5,
            "CRVAL1": 7.0,
            "CRVAL2": -11.0,
            "CDELT1": 2.0,
            "CDELT2": -5.0,
            "CROTA2": 30.0,
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
        }
    )
    pixels = np.array([[0.0, 1.0], [3.0, 4.0]])
    theta = math.radians(30.0)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offsets = (pixels + 1.0 - [2.25, 3.5]) * [2.0, -5.0]
    expected = offsets @ rotation.T + [7.0, -11.0]
    np.testing.assert_allclose(
        [pixel_to_hpc_arcsec(header, x, y) for x, y in pixels], expected
    )
    np.testing.assert_allclose(
        np.column_stack(
            _pixel_coordinates_hpc_arcsec(header, pixels[:, 0], pixels[:, 1])
        ),
        expected,
    )
