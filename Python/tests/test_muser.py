from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.radio.muser import (
    hpc_grid,
    peak_level,
    read_muser_images,
    read_muser_spectrum,
)
from solar_toolkit.radio.muser_comparison import (
    FrameRef,
    load_dart_sum,
    nearest_index,
    pair_dart,
)


def header():
    return fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 10.0,
            "CDELT2": 10.0,
            "DATE-OBS": "2000-01-01T00:00:00",
            "FREQ": 100.0,
        }
    )


def test_combined_frequency_and_beam(tmp_path):
    h = header()
    h["CTYPE4"] = "STOKES"
    h["CRVAL4"] = 1
    data = np.arange(18.0).reshape(2, 1, 1, 3, 3)
    hs = [fits.PrimaryHDU(data, h)]
    for k, a in [
        ("FMHZ", [100.0, 200.0]),
        ("BMAJ", [0.5, 0.4]),
        ("BMIN", [0.3, 0.2]),
        ("BPA", [10.0, 20.0]),
    ]:
        hs.append(fits.ImageHDU(np.array(a), name=k))
    p = tmp_path / "image.fits"
    fits.HDUList(hs).writeto(p)
    result = read_muser_images(p)
    assert result[1][0] == 200.0
    np.testing.assert_equal(result[1][1], data[1, 0, 0])
    assert result[1][3]["BPA"] == 20.0
    assert "CTYPE4" not in result[1][2]


def test_spectrum_explicit_i(tmp_path):
    h = fits.Header({"POLARIZA": "PWR_I"})
    hs = [fits.PrimaryHDU(np.ones((2, 3)), h)]
    for name, arr in [
        ("FMHZ", [200.0, 100.0]),
        ("TIME", 2451544.5 + np.arange(3) / 86400),
    ]:
        hs.append(
            fits.BinTableHDU.from_columns(
                [fits.Column(name=name, format="D", array=arr)], name=name
            )
        )
    p = tmp_path / "s.fits"
    fits.HDUList(hs).writeto(p)
    assert read_muser_spectrum(p).data.shape == (2, 3)
    np.testing.assert_equal(read_muser_spectrum(p).frequency_mhz, [100.0, 200.0])
    fits.setval(p, "POLARIZA", value="CORDATA_XX")
    with pytest.raises(ValueError, match="PWR_I"):
        read_muser_spectrum(p)


def test_peak_and_wcs():
    assert peak_level(np.array([[np.nan, -10], [5, np.inf]])) == 4.5
    with pytest.raises(ValueError):
        peak_level(np.zeros((2, 2)))
    x, y = hpc_grid(header(), (3, 3))
    assert x[1, 1] == pytest.approx(0, abs=1e-6)
    assert y[1, 1] == pytest.approx(0, abs=1e-6)
    assert x[1, 2] == pytest.approx(10, abs=1e-5)
    assert y[2, 1] == pytest.approx(10, abs=1e-5)
    h = header()
    h["CTYPE1"] = "RA---TAN"
    with pytest.raises(ValueError):
        hpc_grid(h, (3, 3))


def test_pairing_tolerance():
    a = [FrameRef(10.0, (Path("l"),))]
    b = [FrameRef(10.09, (Path("r"),))]
    assert len(pair_dart(a, b)) == 1
    assert pair_dart(a, [FrameRef(10.2, (Path("r"),))]) == []
    assert nearest_index([1.0, 2.0], 1.11, 0.1) is None
    assert nearest_index([], 1.0, 0.1) is None


def test_dart_sum_and_grid_guard(tmp_path):
    paths = []
    for pol, val in [("LL", 2.0), ("RR", 3.0)]:
        d = tmp_path / pol
        d.mkdir()
        p = d / "100MHz.fits"
        fits.writeto(p, np.full((3, 3), val), header())
        paths.append(p)
    ref = FrameRef(0.0, tuple(paths))
    np.testing.assert_equal(load_dart_sum(ref).image, np.full((3, 3), 5.0))
    fits.setval(paths[1], "CDELT1", value=20.0)
    with pytest.raises(ValueError, match="mismatch"):
        load_dart_sum(ref)


def test_strict_peak_selection_skips_unmatched_brightest(tmp_path):
    from astropy.time import Time

    from solar_toolkit.radio.muser_comparison import select_peak_frame

    t = float(Time("2000-01-01T00:00:00").unix)
    paths = []
    for pol in ["LL", "RR"]:
        d = tmp_path / pol
        d.mkdir()
        p = d / "100MHz.fits"
        fits.writeto(p, np.ones((3, 3)), header())
        paths.append(p)
    single = [FrameRef(t, (paths[0],))]
    paired = [FrameRef(t, tuple(paths))]
    target, _, i = select_peak_frame(
        [t, t + 2], [5, 100], single, [paired], [single], (t, t + 2)
    )
    assert target == t and i == 0
    with pytest.raises(ValueError, match="No complete"):
        select_peak_frame([t + 2], [100], single, [paired], [single], (t, t + 2))
