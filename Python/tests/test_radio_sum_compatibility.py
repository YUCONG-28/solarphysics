"""L/R summation must reject mismatched units or spatial grids."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
    RadioImage,
    maybe_make_sum_images,
)
from solar_toolkit.radio.roi_lightcurve import _make_paired_sum_images


def _pair():
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 2.0,
            "CDELT2": -3.0,
            "BUNIT": "Jy/beam",
        }
    )
    return [
        RadioImage(
            Path(f"{pol}.fits"),
            0,
            np.full((3, 3), value),
            header.copy(),
            pol,
            149.0,
            datetime(2000, 1, 1, 4, 48),
        )
        for pol, value in [(POL_LCP, 2.0), (POL_RCP, 5.0)]
    ]


@pytest.mark.parametrize(
    ("keyword", "value", "reason"),
    [
        ("BUNIT", "K", "BUNIT"),
        ("CRPIX1", 2.5, "WCS"),
        ("CRVAL2", 1.0, "WCS"),
        ("CDELT1", -2.0, "WCS"),
        ("PC1_2", 0.1, "WCS"),
        ("CROTA2", 30.0, "WCS"),
    ],
)
@pytest.mark.parametrize("consumer", ["centers", "roi"])
def test_sum_rejects_equal_shape_with_incompatible_calibration(
    keyword, value, reason, consumer
):
    pair = _pair()
    pair[1].header[keyword] = value
    if consumer == "centers":
        with pytest.warns(UserWarning, match=reason):
            assert maybe_make_sum_images(pair) == []
    else:
        summed, skipped = _make_paired_sum_images(pair, tolerance_sec=0.5)
        assert summed == []
        assert any(reason in row[2] for row in skipped)
    np.testing.assert_array_equal(pair[0].image, np.full((3, 3), 2.0))
    np.testing.assert_array_equal(pair[1].image, np.full((3, 3), 5.0))


def test_compatible_pair_preserves_sum_and_normalizes_bunit_whitespace():
    pair = _pair()
    pair[1].header["BUNIT"] = " jy/BEAM "
    summed = maybe_make_sum_images(pair)
    assert len(summed) == 1
    assert summed[0].pol == POL_SUM
    np.testing.assert_array_equal(summed[0].image, np.full((3, 3), 7.0))
    assert summed[0].header["BUNIT"] == "Jy/beam"
