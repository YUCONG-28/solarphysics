import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_toolkit.radio.raw_comparison import (
    observation_seconds,
    prepare_aia,
)
from solar_toolkit.radio.raw_comparison import (
    prepare_dart as _prepare_dart,
)
from solar_toolkit.radio.raw_comparison import (
    prepare_muser as _prepare_muser,
)


def prepare_dart(*args, **kwargs):
    kwargs.setdefault("frequency_mhz", 238.0)
    return _prepare_dart(*args, **kwargs)


def prepare_muser(*args, **kwargs):
    kwargs.setdefault("frequency_mhz", 238.0)
    return _prepare_muser(*args, **kwargs)


TARGET = float(Time("2020-01-02T04:14:38.436").unix)


def header(seconds=TARGET):
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
            "DATE-OBS": Time(seconds, format="unix", precision=6).isot,
            "FREQ": 238.0,
            "BUNIT": "K",
        }
    )


def image(path, seconds=TARGET, value=1.0, **updates):
    path.parent.mkdir(parents=True, exist_ok=True)
    h = header(seconds)
    h.update(updates)
    fits.writeto(path, np.full((3, 3), value), h)
    return path


def muser(path, **updates):
    h = header()
    h.update({"CTYPE4": "STOKES", "CRVAL4": 1.0, "CRPIX4": 1.0, "CDELT4": 1.0})
    h.update(updates)
    values = np.arange(18.0).reshape(2, 1, 1, 3, 3)
    hdus = [fits.PrimaryHDU(values, h)]
    for key, values in [
        ("FMHZ", [219.0, 238.7109375]),
        ("BMAJ", [0.5, 0.4]),
        ("BMIN", [0.3, 0.2]),
        ("BPA", [10.0, 20.0]),
    ]:
        hdus.append(fits.ImageHDU(np.array(values), name=key))
    fits.HDUList(hdus).writeto(path)


def test_time_is_observation_header_and_rejects_ambiguous_padding():
    for value in [
        "20200102041438436",
        "20200102041438.436",
        "2020-01-02T04:14:38.436",
        "20200102041438 436",
    ]:
        t, key = observation_seconds(fits.Header({"DATE-OBS": value}))
        assert t == pytest.approx(TARGET, abs=1e-6)
        assert key == "DATE-OBS"
    for h in [
        fits.Header({"DATE": "2020-01-02T04:14:38.436"}),
        fits.Header({"DATE-OBS": "20200102044747 6334"}),
        fits.Header({"DATE-OBS": "2020-01-02"}),
    ]:
        with pytest.raises(ValueError, match="No reliable"):
            observation_seconds(h)


def test_muser_preserves_pixels_frequency_beam_and_nominal_i(tmp_path):
    muser(tmp_path / "muser.fits")
    result = prepare_muser(tmp_path, TARGET)
    assert result.metadata["status"] == "ok"
    assert result.metadata["frequency_mhz"] == 238.7109375
    assert result.metadata["first_plane_stokes_code"] == 1
    assert result.metadata["beam"]["BPA"] == 20
    np.testing.assert_equal(result.data, np.arange(9.0, 18.0).reshape(3, 3))
    assert result.header["BUNIT"] == "K"
    assert prepare_muser(tmp_path, TARGET + 0.251).data is None
    assert prepare_muser(tmp_path, TARGET, frequency_mhz=270).data is None


def test_muser_stokes_reference_pixel_is_used(tmp_path):
    # CRVAL4=1 alone must not be mistaken for the first plane's code.
    path = tmp_path / "muser.fits"
    muser(path, CRPIX4=2.0)
    result = prepare_muser(tmp_path, TARGET)
    assert result.data is None
    assert "code is 0" in result.metadata["reason"]
    fits.setval(path, "CRVAL4", value=2.0)
    result = prepare_muser(tmp_path, TARGET)
    assert result.metadata["first_plane_stokes_code"] == 1
    assert result.data is not None


def test_dart_native_i_and_single_polarization_guard(tmp_path):
    image(tmp_path / "238MHz" / "I" / "native.fits", value=7)
    result = prepare_dart(tmp_path / "238MHz", TARGET)
    np.testing.assert_equal(result.data, np.full((3, 3), 7))
    assert result.metadata["nominal_polarization"] == "native I product"
    image(tmp_path / "LL" / "left.fits")
    result = prepare_dart(tmp_path / "LL", TARGET)
    assert result.data is None
    assert "single circular" in result.metadata["reason"]


def test_dart_pair_checks_both_times_and_missing_side(tmp_path):
    image(tmp_path / "LL" / "left.fits", TARGET + 0.21, 2)
    right = image(tmp_path / "RR" / "right.fits", TARGET + 0.26, 3)
    assert prepare_dart(tmp_path, TARGET, mode="RR+LL").data is None
    fits.setval(
        right, "DATE-OBS", value=Time(TARGET + 0.24, format="unix", precision=6).isot
    )
    result = prepare_dart(tmp_path, TARGET, mode="RR+LL")
    np.testing.assert_equal(result.data, np.full((3, 3), 5))
    assert result.metadata["delta_seconds"] == pytest.approx(0.225, abs=1e-6)
    assert result.metadata["pair_delta_seconds"] == pytest.approx(0.03, abs=1e-6)
    right.unlink()
    assert prepare_dart(tmp_path, TARGET, mode="RR+LL").data is None


@pytest.mark.parametrize(
    "key,value", [("CDELT1", 20.0), ("BUNIT", "Jy/beam"), ("CD1_1", 0.5)]
)
def test_dart_pair_rejects_mismatched_units_and_wcs(tmp_path, key, value):
    image(tmp_path / "LL" / "left.fits")
    image(tmp_path / "RR" / "right.fits", **{key: value})
    result = prepare_dart(tmp_path, TARGET, mode="RR+LL")
    assert result.data is None
    assert "mismatch" in result.metadata["reason"]


def test_aia_extension_and_independent_matching(tmp_path):
    h = header(TARGET + 11.5)
    h["WAVELNTH"] = 171
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((3, 3)), h)]).writeto(
        tmp_path / "aia.fits"
    )
    result = prepare_aia(tmp_path, TARGET, wavelength=171)
    assert result.data is not None
    assert result.metadata["delta_seconds"] == pytest.approx(11.5, abs=1e-6)
    assert prepare_aia(tmp_path, TARGET - 1.0, wavelength=171).data is None
    assert prepare_aia(tmp_path, TARGET, wavelength=304).data is None


def test_unreadable_timestamp_reports_no_data_without_filename_fallback(tmp_path):
    image(
        tmp_path / "I" / "20200102041438436.fits", **{"DATE-OBS": "20200102044747 6334"}
    )
    result = prepare_dart(tmp_path, TARGET)
    assert result.data is None
    assert len(result.metadata["diagnostics"]["excluded_files"]) == 1


def test_unreliable_coordinates_and_frequency_are_no_data(tmp_path):
    path = image(tmp_path / "I" / "source.fits", CTYPE1="RA---TAN")
    result = prepare_dart(tmp_path, TARGET)
    assert result.data is None
    assert "helioprojective" in result.metadata["reason"]
    fits.setval(path, "CTYPE1", value="HPLN-TAN")
    fits.setval(path, "FREQ", value=285.0)
    assert prepare_dart(tmp_path, TARGET).data is None


def test_dart_filters_frequency_before_selecting_nearest(tmp_path):
    image(tmp_path / "I" / "wrong-band.fits", FREQ=285.0)
    image(tmp_path / "I" / "selected-band.fits", TARGET + 0.05, value=9.0)
    result = prepare_dart(tmp_path, TARGET)
    np.testing.assert_equal(result.data, np.full((3, 3), 9.0))
    assert len(result.metadata["diagnostics"]["excluded_files"]) == 1


def test_dart_filters_each_hand_before_pairing(tmp_path):
    image(tmp_path / "LL" / "left.fits", TARGET + 0.24, 2.0)
    image(tmp_path / "RR" / "outside.fits", TARGET + 0.251, 100.0)
    image(tmp_path / "RR" / "inside.fits", TARGET + 0.20, 3.0)
    result = prepare_dart(tmp_path, TARGET, mode="RR+LL")
    np.testing.assert_equal(result.data, np.full((3, 3), 5.0))
    assert all("outside.fits" not in path for path in result.metadata["paths"])


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.nan, np.inf])
def test_invalid_requested_frequency_is_rejected(tmp_path, frequency):
    for prepare in (_prepare_muser, _prepare_dart):
        with pytest.raises(ValueError, match="frequency_mhz"):
            prepare(tmp_path, TARGET, frequency_mhz=frequency)


@pytest.mark.parametrize("shortcut", [False, True])
def test_singleton_frequency_axis_uses_actual_first_pixel(tmp_path, shortcut):
    from solar_toolkit.radio.time_matching import index_radio_frames

    folder = tmp_path / "I"
    folder.mkdir()
    h = header()
    del h["FREQ"]
    if shortcut:
        h["FREQ"] = 238.0
    h.update(CTYPE3="FREQ", CUNIT3="Hz", CRVAL3=238e6, CRPIX3=2.0, CDELT3=10e6)
    path = folder / "singleton.fits"
    fits.writeto(path, np.ones((1, 3, 3)), h)
    assert index_radio_frames([path])[0].payload["frequency_mhz"] == 228.0
    assert _prepare_dart(folder, TARGET, frequency_mhz=238.0).data is None
    result = _prepare_dart(folder, TARGET, frequency_mhz=228.0)
    assert result.metadata["status"] == "ok"
    assert result.metadata["frequency_mhz"] == 228.0


def test_unsupported_pair_tolerance_is_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="pair_tolerance"):
        prepare_dart(tmp_path, TARGET, mode="RR+LL", pair_tolerance=0.2)
