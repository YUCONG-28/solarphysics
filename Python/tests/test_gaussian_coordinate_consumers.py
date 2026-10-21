"""Multiple native fits and debug overlays must share complete WCS geometry."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.wcs import WCS

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d
from solar_toolkit.radio.gaussian import (
    fit_multiple_gaussians_on_radio_image,
    overlay_gaussian_fit_on_axis,
)


def _header():
    return dict(
        CTYPE1="HPLN-TAN",
        CTYPE2="HPLT-TAN",
        CUNIT1="arcsec",
        CUNIT2="arcsec",
        CRPIX1=31,
        CRPIX2=29,
        CRVAL1=1600,
        CRVAL2=-1100,
        CD1_1=-6,
        CD1_2=2,
        CD2_1=3,
        CD2_2=4,
    )


def _world(header, point):
    x, y = WCS(header).all_pix2world(*point, 0)
    return (((x + 180) % 360 - 180) * 3600, y * 3600)


@pytest.mark.parametrize("count", [1, 2])
def test_all_gaussian_components_and_candidates_use_header_not_extent(count):
    y, x = np.indices((60, 64))
    data = 3 + elliptical_gaussian_2d((x, y), 20, 16.3, 20.2, 3, 2, 0.2)
    if count == 2:
        data += elliptical_gaussian_2d((x, y), 15, 46.3, 40.2, 3, 2, -0.2)
    data = np.ma.array(data, mask=False)
    data.data[1, 1] = 999
    data.mask[1, 1] = True
    cfg = dict(
        multi_gaussian_source_count=count,
        fit_use_source_mask=False,
        skip_low_quality_fit=False,
        gaussian_fit_use_roi=False,
        fit_background_model="constant",
        fit_snr_threshold=0,
        fit_grow_snr_threshold=0,
        fit_min_mask_pixels=5,
        multi_gaussian_use_watershed=False,
        max_fwhm_arcsec=1e6,
    )
    result = fit_multiple_gaussians_on_radio_image(
        data, (-1, 1, -1, 1), cfg, coordinate_header=_header()
    )
    assert len(result.fit_results) == count
    assert len(result.candidates) == count
    for candidate in result.candidates:
        assert candidate.peak_pixel != (1, 1)
        np.testing.assert_allclose(
            candidate.peak_arcsec, _world(_header(), candidate.peak_pixel), atol=1e-7
        )
    for fitted in result.fit_results:
        assert fitted.width_coordinate_model == "local_wcs_jacobian"
        np.testing.assert_allclose(
            fitted.center_arcsec, _world(_header(), fitted.center_pixel), atol=1e-7
        )
        assert fitted.coordinate_roundtrip_error_pixel < 1e-8
    assert data.data[1, 1] == 999 and data.mask[1, 1]

    if count == 1:
        fitted = result.fit_results[0]
        fitted.source_mask = np.zeros(data.shape, dtype=bool)
        fitted.source_mask[18:23, 14:19] = True
        fig, axis = plt.subplots()
        try:
            overlay_gaussian_fit_on_axis(
                axis,
                fitted,
                (-1, 1, -1, 1),
                data.shape,
                {**cfg, "draw_coordinate_debug": True},
            )
            debug = axis.collections[-1]
            vertices = np.concatenate([path.vertices for path in debug.get_paths()])
            # The deliberately wrong extent is near zero; the full WCS mask
            # contours must instead surround the source's ~1600,-1100 arcsec.
            assert np.nanmedian(vertices[:, 0]) > 1000
            assert np.nanmedian(vertices[:, 1]) < -500
        finally:
            plt.close(fig)
