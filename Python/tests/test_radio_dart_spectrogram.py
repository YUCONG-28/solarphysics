"""Tests for four-file DART dynamic-spectrogram processing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.radio.dart_spectrogram import (
    DartSpectrogramFiles,
    discover_dart_spectrogram_files,
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)

UTC = timezone.utc


def _write_dataset(
    folder: Path,
    *,
    frequency_mhz: np.ndarray,
    time_rows: np.ndarray,
    stokes_i_db: np.ndarray,
    stokes_v_over_i: np.ndarray | None = None,
    prefix: str = "2000-01-01_",
) -> DartSpectrogramFiles:
    folder.mkdir(parents=True, exist_ok=True)
    vp = (
        np.asarray(stokes_v_over_i, dtype=np.float64)
        if stokes_v_over_i is not None
        else np.asarray(stokes_i_db, dtype=np.float64) / 100.0
    )
    payloads = {
        "SpecDataIdB.fits": np.asarray(stokes_i_db, dtype=np.float64),
        "SpecDataVP.fits": vp,
        "SpecFrequency.fits": np.asarray(frequency_mhz, dtype=np.float64)[None, :],
        "SpecTime.fits": np.asarray(time_rows, dtype=np.float64),
    }
    for suffix, payload in payloads.items():
        fits.PrimaryHDU(data=payload).writeto(folder / f"{prefix}{suffix}")
    return discover_dart_spectrogram_files(folder)


def _time_rows(*seconds: float) -> np.ndarray:
    return np.asarray(
        [[0, 1, 1, 4, 45, second] for second in seconds],
        dtype=np.float64,
    )


def test_discover_dart_spectrogram_files_requires_one_of_each_suffix(
    tmp_path: Path,
) -> None:
    files = _write_dataset(
        tmp_path,
        frequency_mhz=np.asarray([148.0, 149.0]),
        time_rows=_time_rows(0.25, 1.25),
        stokes_i_db=np.ones((2, 2)),
        prefix="event_without_hardcoded_date_",
    )

    assert files.stokes_i_db.name.endswith("SpecDataIdB.fits")
    assert files.stokes_v_over_i.name.endswith("SpecDataVP.fits")
    assert files.frequency.name.endswith("SpecFrequency.fits")
    assert files.time.name.endswith("SpecTime.fits")

    files.time.unlink()
    with pytest.raises(FileNotFoundError, match="spectime.fits"):
        discover_dart_spectrogram_files(tmp_path)


def test_read_window_parses_utc_and_keeps_partial_downsampling_bins(
    tmp_path: Path,
) -> None:
    frequency = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    stokes_i = frequency[:, None] + np.asarray([0.0, 1.0, 2.0, 3.0])[None, :]
    stokes_i[0, 0] = np.nan
    files = _write_dataset(
        tmp_path,
        frequency_mhz=frequency,
        time_rows=_time_rows(0.25, 1.25, 2.25, 3.25),
        stokes_i_db=stokes_i,
        stokes_v_over_i=stokes_i / 100.0,
    )

    result = read_dart_spectrogram_window(
        files,
        max_frequency_samples=3,
        max_time_samples=2,
        chunk_memory_mb=1,
    )

    assert result.stokes_i_db.shape == (3, 2)
    assert result.stokes_v_over_i.shape == (3, 2)
    np.testing.assert_allclose(result.frequency_mhz, [100.5, 102.5, 104.0])
    expected_first_block = np.mean([101.0, 101.0, 102.0])
    assert result.stokes_i_db[0, 0] == pytest.approx(expected_first_block)
    assert result.stokes_i_db[-1, -1] == pytest.approx(106.5)
    assert result.time_utc[0] == datetime(2000, 1, 1, 4, 45, 0, 750000, tzinfo=UTC)
    assert result.time_utc[1] == datetime(2000, 1, 1, 4, 45, 2, 750000, tzinfo=UTC)


def test_read_window_normalizes_descending_frequency_orientation(
    tmp_path: Path,
) -> None:
    frequency = np.asarray([104.0, 103.0, 102.0, 101.0, 100.0])
    stokes_i = np.repeat(frequency[:, None], 2, axis=1)
    files = _write_dataset(
        tmp_path,
        frequency_mhz=frequency,
        time_rows=_time_rows(0.0, 1.0),
        stokes_i_db=stokes_i,
    )

    result = read_dart_spectrogram_window(
        files,
        frequency_range_mhz=(101.0, 103.0),
        max_frequency_samples=10,
        max_time_samples=10,
    )

    np.testing.assert_allclose(result.frequency_mhz, [101.0, 102.0, 103.0])
    np.testing.assert_allclose(result.stokes_i_db[:, 0], [101.0, 102.0, 103.0])


def test_metadata_validation_rejects_shape_and_time_axis_errors(
    tmp_path: Path,
) -> None:
    files = _write_dataset(
        tmp_path / "shape",
        frequency_mhz=np.asarray([148.0, 149.0, 150.0]),
        time_rows=_time_rows(0.0, 1.0),
        stokes_i_db=np.ones((3, 2)),
        stokes_v_over_i=np.ones((2, 3)),
    )
    with pytest.raises(ValueError, match="SpecDataVP shape"):
        read_dart_spectrogram_window(files)

    files = _write_dataset(
        tmp_path / "time",
        frequency_mhz=np.asarray([148.0, 149.0, 150.0]),
        time_rows=_time_rows(0.0, 0.0),
        stokes_i_db=np.ones((3, 2)),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_dart_spectrogram_window(files)


def test_narrowband_curve_is_direct_original_idb_mean_without_second_log(
    tmp_path: Path,
) -> None:
    frequency = np.asarray([147.0, 148.0, 149.0, 150.0, 151.0])
    stokes_i = np.asarray(
        [
            [5.0, 6.0],
            [10.0, 11.0],
            [20.0, np.nan],
            [30.0, 31.0],
            [40.0, 41.0],
        ]
    )
    files = _write_dataset(
        tmp_path,
        frequency_mhz=frequency,
        time_rows=_time_rows(0.0, 1.0),
        stokes_i_db=stokes_i,
    )

    result = extract_dart_narrowband_lightcurves(files, [149.0], 2.0)
    curve = result.curves[0]

    assert curve.requested_frequency_range_mhz == (148.0, 150.0)
    assert curve.sampled_frequency_range_mhz == (148.0, 150.0)
    assert curve.channel_count == 3
    np.testing.assert_allclose(curve.stokes_i_db, [20.0, 21.0])
    assert curve.stokes_i_db[0] != pytest.approx(np.log10(20.0))
    assert curve.stokes_i_db[0] != pytest.approx(10.0 * np.log10(20.0))


def test_narrowband_validation_rejects_duplicates_and_out_of_range_bands(
    tmp_path: Path,
) -> None:
    files = _write_dataset(
        tmp_path,
        frequency_mhz=np.asarray([148.0, 149.0, 150.0]),
        time_rows=_time_rows(0.0, 1.0),
        stokes_i_db=np.ones((3, 2)),
    )

    with pytest.raises(ValueError, match="duplicates"):
        extract_dart_narrowband_lightcurves(files, [149.0, 149.0], 1.0)
    with pytest.raises(ValueError, match="outside"):
        extract_dart_narrowband_lightcurves(files, [148.0], 2.0)
    with pytest.raises(ValueError, match="outside"):
        read_dart_spectrogram_window(files, frequency_range_mhz=(147.0, 149.0))


def test_narrowband_time_selection_is_inclusive_and_utc_aware(
    tmp_path: Path,
) -> None:
    files = _write_dataset(
        tmp_path,
        frequency_mhz=np.asarray([148.0, 149.0, 150.0]),
        time_rows=_time_rows(0.0, 1.0, 2.0),
        stokes_i_db=np.asarray(
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]
        ),
    )

    result = extract_dart_narrowband_lightcurves(
        files,
        [149.0],
        2.0,
        time_range_utc=(
            "2000-01-01T04:45:01Z",
            "2000-01-01T04:45:02+00:00",
        ),
    )

    assert len(result.time_utc) == 2
    assert result.time_utc[0].tzinfo is UTC
    np.testing.assert_allclose(result.curves[0].stokes_i_db, [21.0, 22.0])
