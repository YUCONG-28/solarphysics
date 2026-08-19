"""Regression tests for the CSO lazy spectrogram reader in cso_workflow.

These tests use synthetic CSO-like FITS files so the lazy memmap reading and
chunked rebinning can be verified without real observation data.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from astropy.io import fits

from solar_apps.workflows.radio.cso_workflow import (
    LazySpectrogram,
    _find_range,
    read_cso_fits,
)


def _write_synthetic_cso(
    tmp_path,
    *,
    n_freq: int = 40,
    n_time: int = 40,
    n_pol: int = 2,
    base: str = "2025-01-24T00:00:00",
) -> str:
    """Write a synthetic CSO FITS and return its path."""
    time_arr = np.arange(n_time, dtype=np.float64)  # seconds since DATE-OBS
    freq_arr = np.linspace(80.0, 340.0, n_freq, dtype=np.float32)
    data = np.arange(n_pol * n_freq * n_time, dtype=np.float32).reshape(
        n_pol, n_freq, n_time
    )
    header = fits.Header()
    header["DATE-OBS"] = base
    header["POLARIZA"] = "RCP and LCP" if n_pol == 2 else "I"
    header["NAXIS"] = 3 if n_pol == 2 else 2
    header["BUNIT"] = "K"
    primary = fits.PrimaryHDU(data=data, header=header)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format="D", array=time_arr),
            fits.Column(name="frequency", format="E", array=freq_arr),
        ]
    )
    path = tmp_path / "synthetic_cso.fits"
    fits.HDUList([primary, table]).writeto(path, overwrite=True)
    return str(path)


def test_read_cso_fits_returns_two_lazy_spectrograms(tmp_path) -> None:
    path = _write_synthetic_cso(tmp_path)

    results, hdu = read_cso_fits(path)
    try:
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(spec, LazySpectrogram) for spec in results)
        assert [spec.polar for spec in results] == ["RR", "LL"]
        assert results[0].freq.shape == (40,)
        assert results[0].time.shape == (40,)
    finally:
        hdu.close()


def test_read_cso_fits_single_polarization(tmp_path) -> None:
    path = _write_synthetic_cso(tmp_path, n_pol=1)

    results, hdu = read_cso_fits(path)
    try:
        assert len(results) == 1
    finally:
        hdu.close()


def test_lazy_rebin_matches_manual_block_mean(tmp_path) -> None:
    path = _write_synthetic_cso(tmp_path)
    results, hdu = read_cso_fits(path)
    try:
        spec = results[0]
        base = dt.datetime(2025, 1, 24, 0, 0, 0)
        f_bin, t_bin = 2, 4

        z_out, time_out, freq_out = spec.read_slice_rebinned(
            base,
            base + dt.timedelta(seconds=39),
            80.0,
            340.0,
            t_bin,
            f_bin,
            chunk_mem_mb=1,
        )

        n_freq_trim = (spec.freq.size // f_bin) * f_bin
        n_time_trim = (spec.time.size // t_bin) * t_bin
        raw = np.asarray(spec._raw, dtype=np.float32)
        manual = (
            raw[:n_freq_trim, :n_time_trim]
            .reshape(n_freq_trim // f_bin, f_bin, n_time_trim // t_bin, t_bin)
            .mean(axis=(1, 3), dtype=np.float32)
        )

        assert z_out.shape == manual.shape
        assert np.allclose(z_out, manual, rtol=0, atol=1e-6)
        assert time_out.shape == (n_time_trim // t_bin,)
        assert freq_out.shape == (n_freq_trim // f_bin,)
    finally:
        hdu.close()


def test_find_range_ascending_and_descending() -> None:
    ascending = np.linspace(0.0, 100.0, 101)
    assert _find_range(ascending, 20.0, 80.0) == (20, 80)

    descending = ascending[::-1].copy()
    i0, i1 = _find_range(descending, 20.0, 80.0)
    assert ascending[i0] >= 20.0 and ascending[i1] <= 80.0
    assert i0 <= i1

    # Reversed (lo > hi) inputs are normalized.
    assert _find_range(ascending, 80.0, 20.0) == _find_range(ascending, 20.0, 80.0)
