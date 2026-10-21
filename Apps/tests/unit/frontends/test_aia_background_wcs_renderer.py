"""Composite display consumes the regular full-WCS-resampled AIA grid."""

from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.aia_radio_composite.adapters.aia_adapter import (
    AiaSelection,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering.composite_renderer import (
    _draw_aia_radio_axis,
)
from solar_toolkit.aia.background import read_aia_background


@pytest.mark.parametrize("canvas", ["black", "white"])
def test_rotated_native_image_is_resampled_before_regular_imshow(tmp_path, canvas):
    data = np.arange(63.0).reshape(7, 9)
    header = fits.Header(
        dict(
            CTYPE1="HPLN-TAN",
            CTYPE2="HPLT-TAN",
            CUNIT1="arcsec",
            CUNIT2="arcsec",
            CRPIX1=5,
            CRPIX2=4,
            CRVAL1=1600,
            CRVAL2=-1100,
            CD1_1=-6,
            CD1_2=2,
            CD2_1=3,
            CD2_2=4,
            WAVELNTH=171,
        )
    )
    header["DATE-OBS"] = "2000-01-01T00:00:00"
    path = tmp_path / "rotated.fits"
    fits.writeto(path, data, header)
    background = read_aia_background(path, log_scale=False, percentile_limits=(0, 100))
    observed = datetime(2000, 1, 1, tzinfo=UTC)
    selection = AiaSelection(background, observed, observed, 0, 1)
    fig, axis = plt.subplots()
    try:
        _draw_aia_radio_axis(
            axis,
            selection,
            (),
            display_extent_arcsec=None,
            extended_canvas_color=canvas,
            overlay_colors=("red",),
            show_legend=False,
        )
        rendered = axis.images[0]
        dx = np.diff(background.x_arcsec)[0]
        dy = np.diff(background.y_arcsec)[0]
        np.testing.assert_allclose(
            rendered.get_extent(),
            [
                background.x_arcsec[0] - dx / 2,
                background.x_arcsec[-1] + dx / 2,
                background.y_arcsec[0] - dy / 2,
                background.y_arcsec[-1] + dy / 2,
            ],
        )
        np.testing.assert_allclose(
            rendered.get_array().filled(np.nan), background.z, equal_nan=True
        )
        assert np.ma.getmaskarray(rendered.get_array()).any()
    finally:
        plt.close(fig)
