"""Synthetic checks for native radio localization, not event confirmation."""

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d
from solar_toolkit.radio.muser_localization import (
    evaluate_radio_model,
    fit_radio_plane,
    nearest_time_matches,
)


def test_all_21_planes_keep_frequency_and_beam_identity(tmp_path):
    from solar_toolkit.radio.muser import read_muser_images

    h = header()
    h["CTYPE4"], h["CRVAL4"] = "STOKES", 1
    frequencies = 148.477 + np.arange(21) * 4.88275
    cube = np.broadcast_to(
        np.arange(21)[:, None, None, None, None], (21, 1, 1, 7, 9)
    ).copy()
    hdus = [fits.PrimaryHDU(cube, h)]
    arrays = {
        "FMHZ": frequencies,
        "BMAJ": np.arange(21) + 1.0,
        "BMIN": np.arange(21) + 0.5,
        "BPA": np.arange(21) - 10.0,
    }
    hdus += [fits.ImageHDU(values, name=name) for name, values in arrays.items()]
    path = tmp_path / "21_plane_images.fits"
    fits.HDUList(hdus).writeto(path)
    planes = read_muser_images(path)
    assert len(planes) == 21
    for index, (frequency, image, plane_header, beam) in enumerate(planes):
        assert frequency == frequencies[index]
        assert (image == index).all()
        assert beam["BMAJ"] == arrays["BMAJ"][index]
        assert beam["BPA"] == arrays["BPA"][index]
        assert plane_header["CTYPE1"] == "HPLN-TAN"
        assert "CTYPE4" not in plane_header


def header(scale_x=42.0, scale_y=42.0, rotation=0.0):
    angle = np.deg2rad(rotation)
    return fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 45.0,
            "CRPIX2": 35.0,
            "CRVAL1": 700.0,
            "CRVAL2": -200.0,
            "CDELT1": scale_x,
            "CDELT2": scale_y,
            "PC1_1": np.cos(angle),
            "PC1_2": -np.sin(angle),
            "PC2_1": np.sin(angle),
            "PC2_2": np.cos(angle),
        }
    )


def source(
    shape=(72, 92),
    x=44.7,
    y=32.4,
    sx=6.0,
    sy=3.0,
    theta=0.6,
    amplitude=10.0,
    noise=0.0,
    plane=False,
):
    yy, xx = np.indices(shape, dtype=float)
    image = elliptical_gaussian_2d((xx, yy), amplitude, x, y, sx, sy, theta) + 2.0
    if plane:
        image += 0.015 * (xx - 44.0) - 0.012 * (yy - 35.0)
    image += np.random.default_rng(981).normal(0, noise, shape)
    return image


@pytest.mark.parametrize(
    "scale_x,scale_y,rotation", [(42, 42, 0), (-12, 21, 37), (1.1, 0.6, -75)]
)
def test_center_native_wcs_and_shape(scale_x, scale_y, rotation):
    h = header(scale_x, scale_y, rotation)
    data = source()
    result = fit_radio_plane(data, h, (18, 12, 73, 58))
    assert result["fit_success"]
    np.testing.assert_allclose(
        [result["center_x_pixel"], result["center_y_pixel"]], [44.7, 32.4], atol=1e-6
    )
    world = WCS(h).all_pix2world([[44.7, 32.4]], 0)[0]
    expected = [((world[0] + 180) % 360 - 180) * 3600, world[1] * 3600]
    np.testing.assert_allclose(
        [result["center_hpc_x_arcsec"], result["center_hpc_y_arcsec"]],
        expected,
        atol=1e-5,
    )
    assert result["wcs_roundtrip_error_pixel"] < 0.1
    np.testing.assert_allclose(
        evaluate_radio_model(data.shape, result), data, atol=1e-7
    )
    assert result["fwhm_major_arcsec"] >= result["fwhm_minor_arcsec"] > 0
    if scale_x == scale_y:
        np.testing.assert_allclose(
            result["fwhm_major_arcsec"], 6 * 2.354820045 * scale_x, rtol=0.002
        )
    assert result["quality_status"] == "ok"
    json.dumps(result, allow_nan=False)


def test_plane_background_missing_pixels_covariance():
    data = source(plane=True, noise=0.03)
    data[27:31, 38:41] = np.nan
    unchanged = data.copy()
    result = fit_radio_plane(data, header(), (18, 12, 73, 58), background="plane")
    assert result["fit_success"]
    np.testing.assert_allclose(
        [result["center_x_pixel"], result["center_y_pixel"]], [44.7, 32.4], atol=0.04
    )
    np.testing.assert_allclose(result["parameters"][-2:], [0.015, -0.012], atol=0.0002)
    assert result["formal_center_sigma_x_pixel"] > 0
    assert "not full localization" in result["formal_covariance_caveat"]
    np.testing.assert_equal(data, unchanged)
    residual = data - evaluate_radio_model(data.shape, result)
    assert np.isnan(residual[27:31, 38:41]).all()
    json.dumps(result, allow_nan=False)


def test_multi_peak_is_diagnosed_without_implicit_multi_fit():
    data = source(x=34, y=35, sx=3, sy=3, theta=0)
    data += source(x=58, y=35, sx=3, sy=3, theta=0, amplitude=9) - 2
    result = fit_radio_plane(data, header(), (15, 10, 79, 62))
    assert result["fit_success"]
    assert result["significant_peak_count"] >= 2
    assert "multiple_bright_peaks_detected" in result["quality_reasons"]
    assert result["model_adequacy"] == "diagnostic_concern"
    assert len(result["parameters"]) == 7


@pytest.mark.parametrize(
    "data,roi,reason",
    [
        (np.full((40, 40), np.nan), (0, 0, 40, 40), "insufficient_finite_pixels"),
        (np.ones((40, 40)), (0, 0, 40, 40), "no_positive_source_contrast"),
        (np.ones((40, 40)), (-1, 0, 40, 40), "roi_outside_image_or_empty"),
        (np.ones((40, 40)), (0, 0, 40.5, 40), "roi_bounds_must_be_finite_integers"),
    ],
)
def test_failed_record_retained_and_strict_json(data, roi, reason):
    result = fit_radio_plane(data, header(), roi)
    assert not result["fit_success"]
    assert reason in result["quality_reasons"]
    assert result["center_x_pixel"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError):
        evaluate_radio_model(data.shape, result)


def test_invalid_wcs_does_not_erase_numerical_center():
    h = header()
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    result = fit_radio_plane(source(), h, (18, 12, 73, 58))
    assert result["fit_success"]
    assert result["center_x_pixel"] == pytest.approx(44.7)
    assert result["center_hpc_x_arcsec"] is None
    assert "invalid_or_unsupported_hpc_wcs" in result["quality_reasons"]


def test_numpy_roi_and_nonfinite_failed_roi_still_serialize():
    result = fit_radio_plane(source(), header(), np.array([18, 12, 73, 58]))
    json.dumps(result, allow_nan=False)
    invalid = fit_radio_plane(source(), header(), [18, 12, np.nan, 58])
    assert not invalid["fit_success"]
    assert invalid["roi_xyxy"][2] is None
    json.dumps(invalid, allow_nan=False)


def test_boundary_truncation_and_maximum_evaluations():
    result = fit_radio_plane(source(x=25), header(), (22, 10, 75, 60))
    assert result["fit_success"]
    assert "two_sigma_source_reaches_roi_boundary" in result["quality_reasons"]
    failed = fit_radio_plane(source(noise=0.1), header(), (18, 12, 73, 58), max_nfev=1)
    assert not failed["fit_success"]
    assert "optimizer_not_converged" in failed["quality_reasons"]


def test_nearest_matches_retains_gaps_reuse_duplicates_and_unsorted_identity():
    rows = nearest_time_matches(
        [10.01, 10.02, 11.0, 12.0], [12.0, 10.0, 10.0, 15.0], 0.05
    )
    assert len(rows) == 4
    assert rows[0]["status"] == "ambiguous_nearest"
    assert rows[0]["candidate_indices"] == [1, 2]
    assert rows[0]["reference_reused"] and rows[1]["reference_reused"]
    assert rows[0]["reference_use_count"] == 2
    assert rows[2]["status"] == "unmatched"
    assert rows[3]["reference_index"] == 0
    assert rows[3]["time_delta_sec"] == 0
    json.dumps(rows, allow_nan=False)


def test_nearest_tie_large_utc_and_invalid_times():
    timestamp = 1737694110.0
    rows = nearest_time_matches(
        [timestamp + 0.05, timestamp + 1, np.nan],
        [timestamp, timestamp + 0.1, np.nan],
        0.05,
    )
    assert rows[0]["status"] == "ambiguous_nearest"
    assert rows[1]["status"] == "unmatched"
    assert rows[2]["status"] == "invalid_query_time"
    assert nearest_time_matches([10], [], 0.05)[0]["status"] == "unmatched"
    with pytest.raises(ValueError):
        nearest_time_matches([0], [0], -0.01)


@pytest.mark.parametrize("roi", [None, 4, ["invalid", 0, 3, 4], [[0, 1], [2, 3]]])
def test_malformed_roi_retains_serializable_failure(roi):
    result = fit_radio_plane(source(), header(), roi)
    assert not result["fit_success"]
    assert result["quality_reasons"]
    json.dumps(result, allow_nan=False)
