import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_toolkit.radio.cso_window import read_cso_total_window


def test_sum_and_empty_time_bins(tmp_path):
    p = tmp_path / "cso.fits"
    h = fits.Header(
        {
            "DATE-OBS": "2000-01-01T00:00:00",
            "TIMESYS": "UTC",
            "POLARIZA": "RCP and LCP",
            "BUNIT": "SFU",
        }
    )
    data = np.array(
        [[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]]]
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format="3D", array=[[0.01, 0.11, 0.41]]),
            fits.Column(name="frequency", format="2D", array=[[100.0, 100.1]]),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(data, header=h), table]).writeto(p)
    base = Time("2000-01-01").unix
    t, f, out, meta = read_cso_total_window([p], [base, base + 0.5], [100.0, 100.25])
    np.testing.assert_allclose(out[0, [0, 1, 4]], [5, 7, 9])
    assert np.isnan(out[0, 2:4]).all()


def test_legacy_window_rejects_overlap_and_unknown_units(tmp_path):
    """The strict reader must not silently adopt extract_spectrum deduplication."""
    header = fits.Header(
        {
            "DATE-OBS": "2000-01-01T00:00:00",
            "TIMESYS": "UTC",
            "POLARIZA": "RCP and LCP",
            "BUNIT": "SFU",
        }
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format="2D", array=[[0.01, 0.11]]),
            fits.Column(name="frequency", format="2D", array=[[100.0, 100.1]]),
        ]
    )
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    for path in (a, b):
        fits.HDUList(
            [fits.PrimaryHDU(np.ones((2, 2, 2)), header), table.copy()]
        ).writeto(path)
    base = float(Time("2000-01-01").unix)
    with pytest.raises(ValueError, match="Overlapping"):
        read_cso_total_window([a, b], (base, base + 0.2), (100.0, 100.25))
    fits.setval(a, "BUNIT", value="K")
    with pytest.raises(ValueError, match="linear SFU"):
        read_cso_total_window([a], (base, base + 0.2), (100.0, 100.25))
