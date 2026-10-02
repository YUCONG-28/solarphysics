"""Synthetic FITS contracts for bounded CSO extraction."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from solar_toolkit.radio.cso_extract import extract_spectrum

BASE = float(Time("2000-01-01T00:00:00", scale="utc").unix)


def write_cso(path, times, data, *, unit="SFU", timesys="UTC"):
    times = np.asarray(times, dtype=float)
    frequency = np.array([100.0, 101.0])
    header = fits.Header(
        {
            "DATE_BEG": Time(BASE + times[0], format="unix").isot,
            "DATE_END": Time(BASE + times[-1], format="unix").isot,
            "DATE-OBS": "2000-01-01T00:00:00",
            "TIMESYS": timesys,
            "POLARIZA": "RCP and LCP",
            "BUNIT": unit,
        }
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format=f"{len(times)}D", unit="s", array=[times]),
            fits.Column(name="frequency", format="2D", unit="MHz", array=[frequency]),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(data, header), table]).writeto(path)
    return str(path)


def test_pair_before_finite_weighted_binning_and_keep_empty_bins(tmp_path):
    data = np.array(
        [[[1.0, np.nan, 5.0], [3.0, 7.0, 9.0]], [[2.0, 4.0, 6.0], [4.0, 8.0, 10.0]]]
    )
    source = write_cso(tmp_path / "spectrum.fits", [0.01, 0.02, 0.21], data)
    arrays, meta = extract_spectrum(
        [source],
        (BASE, BASE + 0.3),
        quantized=False,
        time_bin_seconds=0.1,
        frequency_bounds=(100.0, 102.0),
        frequency_bin_mhz=2.0,
    )
    # Each native finite RCP/LCP pair contributes one sample, even when one
    # timestamp has only one valid frequency. Averaging timestamps would differ.
    np.testing.assert_allclose(arrays["intensity"][0, [0, 2]], [25 / 3, 15])
    np.testing.assert_equal(arrays["sample_count"], [[3, 0, 2]])
    assert np.isnan(arrays["intensity"][0, 1])
    assert meta["unit"] == "SFU"


def test_stable_native_order_and_cross_file_time_deduplication(tmp_path):
    first = np.array(
        [
            [[20.0, 10.0, 99.0, 30.0], [20.0, 10.0, 99.0, 30.0]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    )
    second = np.full((2, 2, 2), 200.0)
    a = write_cso(tmp_path / "a.fits", [0.2, 0.1, 0.1, 0.3], first)
    b = write_cso(tmp_path / "b.fits", [0.3, 0.4], second)
    arrays, meta = extract_spectrum(
        [b, a],
        (BASE, BASE + 0.5),
        quantized=False,
        time_bin_seconds=0.1,
        frequency_bounds=(100.0, 101.0),
        frequency_bin_mhz=None,
    )
    observed = arrays["intensity"][0, arrays["sample_count"][0] > 0]
    np.testing.assert_allclose(observed, [11.0, 21.0, 31.0, 400.0])
    np.testing.assert_equal(arrays["frequency_centers_mhz"], [100.0, 101.0])
    assert meta["duplicate_time_samples_removed"] == 2
    assert meta["sources"][0]["native_time_axis_inversion_count"] == 1


def test_encoded_polarization_sum_converts_before_addition(tmp_path):
    raw = np.full((2, 2, 2), 200, dtype=np.uint8)
    source = write_cso(tmp_path / "encoded.fits", [0.01, 0.11], raw)
    arrays, meta = extract_spectrum(
        [source],
        (BASE, BASE + 0.2),
        quantized=True,
        encoded_polarization="RCP+LCP",
        time_bin_seconds=0.1,
        frequency_bounds=(100.0, 101.0),
        frequency_bin_mhz=None,
    )
    np.testing.assert_equal(arrays["intensity"], np.full((2, 2), 400.0))
    assert meta["unit"] == "relative encoded intensity"
    assert "not establish an invertible calibration" in meta["unit_caveat"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"unit": "K"}, "explicitly declare SFU"),
        ({"timesys": "TAI"}, "TIMESYS"),
    ],
)
def test_reject_untrusted_units_or_time_scale(tmp_path, changes, message):
    source = write_cso(
        tmp_path / "invalid.fits", [0.01, 0.11], np.ones((2, 2, 2)), **changes
    )
    with pytest.raises(ValueError, match=message):
        extract_spectrum(
            [source],
            (BASE, BASE + 0.2),
            quantized=False,
            frequency_bounds=(100.0, 101.0),
        )
