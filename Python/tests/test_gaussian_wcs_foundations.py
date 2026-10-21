"""Independent synthetic WCS geometry checks for native-pixel Gaussian fits."""

from copy import deepcopy

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d
from solar_toolkit.radio.gaussian import (
    GaussianFitResult,
    fit_elliptical_gaussian_on_radio_image,
)
from solar_toolkit.radio.gaussian_coordinates import attach_wcs_geometry, wcs_fwhm_curve

FWHM = 2 * np.sqrt(2 * np.log(2))


def _header(*, angle=0.0, swapped=False, offset=False, scale=1.0):
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    matrix = rotation @ np.diag([-7.0, 3.5]) * scale
    reference = [1600.0, -1100.0] if offset else [0.0, 0.0]
    axes = ["HPLN-TAN", "HPLT-TAN"]
    if swapped:
        matrix = matrix[::-1]
        axes = axes[::-1]
        reference = reference[::-1]
    return {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 60,
        "CTYPE1": axes[0],
        "CTYPE2": axes[1],
        "CUNIT1": "arcsec",
        "CUNIT2": "arcsec",
        "CRPIX1": 31.0,
        "CRPIX2": 29.0,
        "CRVAL1": reference[0],
        "CRVAL2": reference[1],
        "CD1_1": matrix[0, 0],
        "CD1_2": matrix[0, 1],
        "CD2_1": matrix[1, 0],
        "CD2_2": matrix[1, 1],
    }


def _world(header, x, y):
    """Independent Astropy reference, with explicit HPLN world-axis selection."""
    wcs = WCS(header)
    world = wcs.all_pix2world(x, y, 0)
    lon = 0 if "HPLN" in header["CTYPE1"] else 1
    lat = 1 - lon
    # Astropy celestial low-level outputs use degrees by default.
    longitude = (np.asarray(world[lon]) + 180) % 360 - 180
    return longitude * 3600, np.asarray(world[lat]) * 3600


def _pixel(header, x_arcsec, y_arcsec):
    world = [x_arcsec / 3600, y_arcsec / 3600]
    if "HPLN" not in header["CTYPE1"]:
        world = world[::-1]
    return WCS(header).all_world2pix(*world, 0)


def _result(*, center=(30.0, 28.0), sigma=(3.8, 1.2), theta=0.43):
    y, x = np.indices((60, 64), dtype=float)
    model = elliptical_gaussian_2d((x, y), 15, *center, *sigma, theta)
    result = GaussianFitResult(
        model=model + 2,
        gaussian_only_model=model,
        center_pixel=center,
        center_arcsec=(999.0, 999.0),
        sigma_pixel=sigma,
        theta_rad=theta,
        amplitude=15,
        background_level=2,
        noise_sigma=1,
        snr=15,
        residual_rms=0,
        quality_flag="ok",
        covariance=np.diag(np.arange(1, 7)),
        mask_pixel_count=300,
    )
    result.initial_center_pixel_tuple = (center[0] - 0.4, center[1] + 0.2)
    return result


def test_wcs_geometry_curve_accepts_multiple_fits_history_and_comment_cards():
    header = fits.Header(_header(angle=0.7, offset=True))
    header.add_history("First observation processing step")
    header.add_history("Second observation processing step")
    header.add_comment("First explanatory comment")
    header.add_comment("Second explanatory comment")
    before = header.tostring()
    result = _result()
    attach_wcs_geometry(result, header)
    wx, wy = wcs_fwhm_curve(result)
    px, py = _pixel(header, wx, wy)
    values = elliptical_gaussian_2d(
        (px, py),
        result.amplitude,
        *result.center_pixel,
        *result.sigma_pixel,
        result.theta_rad,
    )
    np.testing.assert_allclose(values, result.amplitude / 2, rtol=1e-10)
    assert header.tostring() == before
    assert result.coordinate_header is not header


@pytest.mark.parametrize("angle", [0.0, 0.63])
@pytest.mark.parametrize("swapped", [False, True])
def test_wcs_center_local_ellipse_and_native_model_are_consistent(angle, swapped):
    header = _header(angle=angle, swapped=swapped)
    original_header = deepcopy(header)
    result = _result()
    native_model = result.model.copy()
    gaussian_model = result.gaussian_only_model.copy()
    formal_covariance = result.covariance.copy()
    native_parameters = (
        result.center_pixel,
        result.sigma_pixel,
        result.theta_rad,
        result.amplitude,
    )
    attach_wcs_geometry(result, header, raw_peak_pixel=(30, 28))

    np.testing.assert_allclose(
        result.center_arcsec, _world(header, *result.center_pixel), atol=1e-7
    )
    np.testing.assert_allclose(
        result.initial_center_arcsec,
        _world(header, *result.initial_center_pixel_tuple),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.raw_center_arcsec, _world(header, 30, 28), atol=1e-7
    )
    assert result.coordinate_roundtrip_error_pixel < 1e-7
    assert header == original_header
    assert result.coordinate_header is not header
    np.testing.assert_array_equal(result.model, native_model)
    np.testing.assert_array_equal(result.gaussian_only_model, gaussian_model)
    np.testing.assert_array_equal(result.covariance, formal_covariance)
    assert (
        result.center_pixel,
        result.sigma_pixel,
        result.theta_rad,
        result.amplitude,
    ) == native_parameters

    # At this TAN reference pixel with zero reference latitude, the exact local
    # angular Jacobian is the header's CD matrix, including axis order and sign.
    matrix = np.array(
        [[header["CD1_1"], header["CD1_2"]], [header["CD2_1"], header["CD2_2"]]]
    )
    if swapped:
        matrix = matrix[::-1]
    c, s = np.cos(result.theta_rad), np.sin(result.theta_rad)
    rotation = np.array([[c, -s], [s, c]])
    expected = (
        matrix
        @ rotation
        @ np.diag(np.square(result.sigma_pixel))
        @ rotation.T
        @ matrix.T
    )
    phi = result.theta_world_rad
    world_rotation = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    measured = (
        world_rotation
        @ np.diag((np.asarray(result.wcs_fwhm_arcsec) / FWHM) ** 2)
        @ world_rotation.T
    )
    np.testing.assert_allclose(measured, expected, rtol=2e-6, atol=1e-6)
    assert result.fwhm_major_arcsec == result.fwhm_width_arcsec
    assert result.fwhm_minor_arcsec == result.fwhm_height_arcsec
    assert result.width_coordinate_model == "local_wcs_jacobian"


@pytest.mark.parametrize("swapped", [False, True])
def test_projected_half_maximum_curve_has_exact_native_half_amplitude(swapped):
    header = _header(angle=0.7, swapped=swapped, offset=True, scale=30)
    result = _result(center=(40.3, 21.8), sigma=(8, 2.3), theta=-0.32)
    attach_wcs_geometry(result, header)
    x_world, y_world = wcs_fwhm_curve(result, samples=241)
    x_pixel, y_pixel = _pixel(header, x_world, y_world)
    actual = elliptical_gaussian_2d(
        (x_pixel, y_pixel),
        result.amplitude,
        *result.center_pixel,
        *result.sigma_pixel,
        result.theta_rad,
    )
    np.testing.assert_allclose(actual, result.amplitude / 2, rtol=1e-10)
    np.testing.assert_allclose(
        [x_world[0], y_world[0]], [x_world[-1], y_world[-1]], atol=1e-7
    )
    # Check a non-reference center with TAN distortion against independent WCS,
    # so an affine extent cannot accidentally satisfy the center assertion.
    np.testing.assert_allclose(
        result.center_arcsec, _world(header, *result.center_pixel), atol=1e-7
    )


@pytest.mark.parametrize("reject_width", [False, True])
def test_native_fit_preserves_pixels_and_publishes_wcs_geometry_before_overlay(
    reject_width,
):
    y, x = np.indices((60, 64), dtype=float)
    data = 3 + elliptical_gaussian_2d((x, y), 20, 30.2, 28.4, 4.5, 2.0, 0.35)
    original = data.copy()
    header = _header(angle=0.65, offset=True)
    cfg = {
        "fit_use_source_mask": False,
        "skip_low_quality_fit": False,
        "gaussian_fit_use_roi": False,
        "fit_background_model": "constant",
        "fit_snr_threshold": 0,
        "max_fwhm_arcsec": 10 if reject_width else 1e6,
    }
    # A deliberately wrong affine extent must not affect native fitted pixels
    # or supersede the explicit WCS calibration.
    extent = (-1, 1, -1, 1)
    legacy = fit_elliptical_gaussian_on_radio_image(data, extent, dict(cfg))
    fitted = fit_elliptical_gaussian_on_radio_image(
        data, extent, dict(cfg), coordinate_header=header
    )
    assert legacy is not None and fitted is not None
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(fitted.model, legacy.model)
    np.testing.assert_array_equal(
        fitted.gaussian_only_model, legacy.gaussian_only_model
    )
    np.testing.assert_allclose(fitted.center_pixel, legacy.center_pixel, atol=1e-12)
    np.testing.assert_allclose(fitted.sigma_pixel, legacy.sigma_pixel, atol=1e-12)
    np.testing.assert_allclose(
        fitted.center_arcsec, _world(header, *fitted.center_pixel), atol=1e-7
    )
    raw_y, raw_x = np.unravel_index(np.argmax(data), data.shape)
    assert fitted.raw_center_pixel == (raw_x, raw_y)
    np.testing.assert_allclose(
        fitted.raw_center_arcsec, _world(header, raw_x, raw_y), atol=1e-7
    )
    expected_distance = np.hypot(
        *(np.asarray(fitted.center_arcsec) - fitted.raw_center_arcsec)
    )
    assert fitted.center_peak_distance_arcsec == pytest.approx(expected_distance)
    assert fitted.fwhm_width_arcsec == fitted.wcs_fwhm_arcsec[0]
    assert fitted.fwhm_valid is (not reject_width)
    if reject_width:
        assert fitted.quality_flag_detail == "skipped_large_fwhm"
        assert not fitted.overlay_valid
        assert not fitted.trajectory_valid
