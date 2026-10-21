"""Full-WCS display resampling keeps regular axes scientifically meaningful."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from solar_toolkit.aia.background import read_aia_background


def _header(*, swapped=False):
    matrix = np.array([[-600, 180], [120, 900]])
    axes = ["HPLN-TAN", "HPLT-TAN"]
    reference = [18000, -9000]
    if swapped:
        matrix, axes, reference = matrix[::-1], axes[::-1], reference[::-1]
    return fits.Header(
        {
            "CTYPE1": axes[0],
            "CTYPE2": axes[1],
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 8,
            "CRPIX2": 7,
            "CRVAL1": reference[0],
            "CRVAL2": reference[1],
            "CD1_1": matrix[0, 0],
            "CD1_2": matrix[0, 1],
            "CD2_1": matrix[1, 0],
            "CD2_2": matrix[1, 1],
            "DATE-OBS": "2000-01-01T00:00:00",
            "WAVELNTH": 171,
        }
    )


def _inverse(header, background):
    gx, gy = np.meshgrid(background.x_arcsec / 3600, background.y_arcsec / 3600)
    world = (gy, gx) if "HPLN" not in header["CTYPE1"] else (gx, gy)
    return WCS(header).all_world2pix(*world, 0)


@pytest.mark.parametrize("swapped", [False, True])
def test_regular_background_samples_match_independent_inverse_wcs(tmp_path, swapped):
    py, px = np.indices((13, 15))
    image = 7 + 2 * px + 3 * py
    original = image.copy()
    header = _header(swapped=swapped)
    path = tmp_path / "synthetic.fits"
    fits.writeto(path, image, header)
    background = read_aia_background(path, log_scale=False, percentile_limits=(0, 100))
    x, y = _inverse(header, background)
    interior = (x >= 0) & (x <= 14) & (y >= 0) & (y <= 12)
    outside = (x < -0.5) | (x > 14.5) | (y < -0.5) | (y > 12.5)
    assert interior.sum() > 50 and outside.sum() > 0
    np.testing.assert_allclose(
        background.z[interior], (7 + 2 * x + 3 * y)[interior], atol=2e-9
    )
    assert np.isnan(background.z[outside]).all()
    np.testing.assert_array_equal(image, original)
    np.testing.assert_allclose(
        np.diff(background.x_arcsec), np.diff(background.x_arcsec)[0]
    )
    np.testing.assert_allclose(
        np.diff(background.y_arcsec), np.diff(background.y_arcsec)[0]
    )


@pytest.mark.parametrize("all_missing", [False, True])
def test_masked_or_missing_samples_never_become_background_signal(
    monkeypatch, all_missing
):
    from solar_toolkit.aia import background as module

    image = np.ma.array(np.ones((13, 15)), mask=all_missing)
    image.data[6, 7] = 999
    image.mask[6, 7] = True
    original = image.copy()
    header = _header()
    monkeypatch.setattr(
        module, "_read_first_image_hdu", lambda *_a, **_k: (image, header)
    )
    background = read_aia_background(
        "synthetic.fits", log_scale=False, percentile_limits=(0, 100)
    )
    x, y = _inverse(header, background)
    touching_hole = (np.abs(x - 7) < 1) & (np.abs(y - 6) < 1)
    assert touching_hole.any()
    assert np.isnan(background.z[touching_hole]).all()
    if all_missing:
        assert np.isnan(background.z).all()
    else:
        np.testing.assert_allclose(background.z[np.isfinite(background.z)], 1)
    np.testing.assert_array_equal(image.data, original.data)
    np.testing.assert_array_equal(image.mask, original.mask)
