"""Independent synthetic contracts for native-time paired CSO extraction."""

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta

from solar_toolkit.radio.cso_extract import (
    extract_native_polarization,
    extract_spectrum,
)


def source_file(
    path,
    times,
    right,
    left,
    *,
    frequency=(100.0, 101.0),
    base="2000-01-01T00:00:00",
    scale=1.0,
    zero=0.0,
    unit="SFU",
    timesys="UTC",
):
    offsets = np.asarray(times, dtype=float)
    origin = Time(base, scale="utc")
    header = fits.Header(
        {
            "DATE-OBS": base,
            "DATE_BEG": (origin + TimeDelta(offsets[0], format="sec")).utc.isot,
            "DATE_END": (origin + TimeDelta(offsets[-1], format="sec")).utc.isot,
            "TIMESYS": timesys,
            "POLARIZA": "RCP and LCP",
            "BUNIT": unit,
            "BSCALE": scale,
            "BZERO": zero,
        }
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="time", format=f"{len(times)}D", unit="s", array=[offsets]
            ),
            fits.Column(
                name="frequency",
                format=f"{len(frequency)}D",
                unit="MHz",
                array=[frequency],
            ),
        ]
    )
    image = fits.PrimaryHDU(np.array([right, left], dtype=float), header)
    image.header["BSCALE"], image.header["BZERO"] = scale, zero
    fits.HDUList([image, table]).writeto(path)
    return path


def extract(paths, bounds=("2000-01-01T00:00:00", "2000-01-01T00:00:01"), **kwargs):
    return extract_native_polarization(
        paths,
        bounds,
        kwargs.pop("frequency_bounds", (100.0, 102.0)),
        frequency_bin_mhz=kwargs.pop("frequency_bin_mhz", 2.0),
        **kwargs,
    )


def test_pair_weighted_ratio_is_not_mean_of_native_ratios_and_chunks_agree(tmp_path):
    path = source_file(
        tmp_path / "paired.fits",
        [0.01, 0.02, 0.21],
        [[100, np.nan, 3], [1, 7, 9]],
        [[0, 4, 1], [1, 8, 10]],
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    a, metadata = extract([path], chunk_columns=1)
    b, _ = extract([path], chunk_columns=200)
    for key in a:
        np.testing.assert_equal(a[key], b[key])
    np.testing.assert_equal(a["sample_count"], [[2, 1, 2]])
    np.testing.assert_allclose(a["sum_rcp"], [[101, 7, 12]])
    np.testing.assert_allclose(a["intensity"], [[51, 15, 11.5]])
    assert a["polarization_percent"][0, 0] == pytest.approx(10000 / 102)
    assert a["polarization_percent"][0, 0] != pytest.approx(50)
    assert metadata["native_samples_aggregated"] == 5
    json.dumps(metadata, allow_nan=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_native_clock_keeps_sub_unix_float_resolution_samples(tmp_path):
    path = source_file(
        tmp_path / "precision.fits", [0, 1e-7, 2e-7], np.ones((2, 3)), np.ones((2, 3))
    )
    a, meta = extract([path])
    np.testing.assert_equal(np.diff(a["time_tai_ns"]), [100, 100])
    assert a["intensity"].shape == (1, 3)
    assert meta["sources"][0]["last_actual_utc"] == "2000-01-01T00:00:00.000000200"


def test_si_offsets_cross_the_utc_leap_second(tmp_path):
    path = source_file(
        tmp_path / "leap.fits",
        [86399, 86400, 86401],
        np.ones((2, 3)),
        np.ones((2, 3)),
        base="2016-12-31T00:00:00",
    )
    a, meta = extract([path], ("2016-12-31T23:59:59", "2017-01-01T00:00:00"))
    np.testing.assert_equal(np.diff(a["time_tai_ns"]), [1_000_000_000] * 2)
    assert a["sample_count"].shape == (1, 3)
    assert (
        meta["requested_time_bounds_tai_ns"][1]
        - meta["requested_time_bounds_tai_ns"][0]
        == 2_000_000_000
    )
    assert meta["sources"][0]["first_actual_utc"] == "2016-12-31T23:59:59.000000000"
    assert meta["sources"][0]["last_actual_utc"] == "2017-01-01T00:00:00.000000000"


def test_missing_bins_negative_polarization_and_scaled_values(tmp_path):
    path = source_file(
        tmp_path / "scaled.fits",
        [0, 1],
        [[1, 2], [np.nan, 3]],
        [[2, 3], [5, 4]],
        frequency=(100, 102),
        scale=2,
        zero=10,
    )
    a, _ = extract([path], frequency_bounds=(100, 103), frequency_bin_mhz=1 * u.MHz)
    np.testing.assert_equal(a["intensity"][[0, 2]], [[26, 30], [np.nan, 34]])
    assert np.isnan(a["intensity"][1]).all()
    np.testing.assert_equal(a["sample_count"][1], [0, 0])
    negative = source_file(
        tmp_path / "negative.fits", [0, 1], [[-2, -2], [1, 1]], [[10, 10], [1, 1]]
    )
    b, _ = extract([negative])
    assert np.isnan(b["polarization_percent"]).all()
    np.testing.assert_allclose(b["raw_polarization_percent"], -120)


def test_gap_segments_singleton_edges_and_missing_boundaries_are_explicit(tmp_path):
    times = [0, 1, 2, 5, 8, 9, 10]
    path = source_file(tmp_path / "gaps.fits", times, np.ones((2, 7)), np.ones((2, 7)))
    a, meta = extract([path], ("1999-12-31T23:59:59", "2000-01-01T00:00:11"))
    assert a["intensity"].shape == (1, 7)
    assert [(s["start_index"], s["stop_index"]) for s in meta["segments"]] == [
        (0, 3),
        (3, 4),
        (4, 7),
    ]
    assert len(meta["segments"][1]["time_edges_unix"]) == 2
    assert np.diff(meta["segments"][1]["time_edges_tai_ns"])[0] == 1_000_000_000
    assert len(meta["gaps_unix"]) == len(meta["boundary_missing_support_unix"]) == 2
    assert [s[1] - s[0] for s in meta["missing_support_tai_ns"]] == [
        500_000_000,
        2_000_000_000,
        2_000_000_000,
        500_000_000,
    ]


def test_reordering_identical_duplicates_and_cross_source_conflicts(tmp_path):
    right = np.array([[20.0, 10, 10, 30], [20, 10, 10, 30]])
    a = source_file(tmp_path / "a.fits", [0.2, 0.1, 0.1, 0.3], right, np.ones((2, 4)))
    arrays, meta = extract([a], frequency_bin_mhz=None)
    np.testing.assert_equal(arrays["intensity"], [[11, 21, 31], [11, 21, 31]])
    assert meta["duplicate_time_samples_removed"] == 1
    b = source_file(
        tmp_path / "b.fits", [0.3, 0.4], np.full((2, 2), 200), np.ones((2, 2))
    )
    with pytest.raises(ValueError, match="Conflicting paired") as error:
        extract([a, b])
    message = str(error.value)
    assert (
        str(a) in message
        and str(b) in message
        and "tai_ns=" in message
        and "column" in message
        and "UTC=" in message
    )


def test_gzip_paths_fallback_and_cleanup_keep_provenance(tmp_path):
    path = source_file(
        tmp_path / "compressed.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    expected, _ = extract([path])
    compressed = path.with_suffix(".fits.gz")
    with gzip.open(compressed, "wb") as output:
        output.write(path.read_bytes())
    path.unlink()
    cache = tmp_path / "cache"
    actual, meta = extract([path], cache_root=cache)
    for key in expected:
        np.testing.assert_equal(actual[key], expected[key])
    assert meta["sources"][0]["path"] == str(compressed)
    assert not list(cache.iterdir())
    old, _ = extract_spectrum(
        [path],
        (946684800.0, 946684801.0),
        quantized=False,
        frequency_bounds=(100, 102),
        cache_root=cache,
    )
    assert np.nanmax(old["intensity"]) == 2
    assert not list(cache.iterdir())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_columns": 0},
        {"chunk_columns": True},
        {"chunk_columns": 1.5},
        {"gap_factor": np.nan},
        {"gap_factor": 1},
        {"frequency_bin_mhz": np.inf},
        {"frequency_bounds": (100, np.nan)},
    ],
)
def test_invalid_configuration_rejected_before_reading_sources(kwargs):
    with pytest.raises(ValueError):
        extract(["nonexistent.fits"], **kwargs)


def test_empty_time_selection_is_missing_and_fractional_last_frequency_bin_is_clipped(
    tmp_path,
):
    path = source_file(
        tmp_path / "empty.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    a, meta = extract(
        [path],
        ("2000-01-01T00:01:00", "2000-01-01T00:01:01"),
        frequency_bounds=(100, 101.000005),
        frequency_bin_mhz=1,
    )
    assert a["intensity"].shape == (2, 0)
    np.testing.assert_equal(a["frequency_edges_mhz"], [100, 101, 101.000005])
    assert not meta["segments"] and len(meta["missing_support_unix"]) == 1


@pytest.mark.parametrize(
    "order", [("cso_extract", "_cso_native"), ("_cso_native", "cso_extract")]
)
def test_direct_module_imports_work_in_both_orders(order):
    code = f"from importlib import import_module; a=import_module('solar_toolkit.radio.{order[0]}'); b=import_module('solar_toolkit.radio.{order[1]}'); assert a.extract_native_polarization is b.extract_native_polarization"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_full_legacy_reader_plain_and_gzip_preserve_physical_scaling(tmp_path):
    from solar_toolkit.radio.cso import read_cso_spectrogram

    path = source_file(
        tmp_path / "legacy.fits",
        [0, 1],
        [[1, 2], [3, 4]],
        [[2, 3], [4, 5]],
        scale=2,
        zero=10,
    )
    expected = read_cso_spectrogram(path)
    np.testing.assert_equal(expected[0].data, [[12, 14], [16, 18]])
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()
    cache = tmp_path / "cache"
    actual = read_cso_spectrogram(path, cache_root=cache)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_equal(a.data, b.data)
        np.testing.assert_equal(a.time, b.time)
    assert not list(cache.iterdir())


@pytest.mark.parametrize("reader", ["comparison", "window"])
def test_legacy_bounded_readers_expand_once_and_keep_provenance(
    tmp_path, monkeypatch, reader
):
    from contextlib import contextmanager

    from solar_toolkit.radio import cso_extract
    from solar_toolkit.radio.cso_comparison import read_cso_comparison
    from solar_toolkit.radio.cso_window import read_cso_total_window

    path = source_file(
        tmp_path / "legacy.fits",
        [0, 0.1],
        np.ones((2, 2)),
        2 * np.ones((2, 2)),
        scale=2,
        zero=10,
    )
    bounds = [946684800.0, 946684800.2]

    def read():
        if reader == "comparison":
            return read_cso_comparison(
                [path], bounds, bands_mhz=[100], cache_root=tmp_path / "cache"
            ).spectrum_sfu
        return read_cso_total_window(
            [path], bounds, [100, 102], cache_root=tmp_path / "cache"
        )[2]

    expected = read()
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()
    calls = []
    actual_materialize = cso_extract.materialize_fits

    @contextmanager
    def counted(source, **kwargs):
        calls.append(source)
        with actual_materialize(source, **kwargs) as readable:
            yield readable

    monkeypatch.setattr(cso_extract, "materialize_fits", counted)
    np.testing.assert_equal(read(), expected)
    assert len(calls) == 1 and str(calls[0]).endswith(".fits.gz")
    assert not list((tmp_path / "cache").iterdir())


def test_lazy_compressed_handle_lifetime_and_failure_cleanup(tmp_path):
    from datetime import datetime, timedelta

    from solar_toolkit.radio._lazy_spectrogram import read_cso_fits

    path = source_file(
        tmp_path / "lazy.fits",
        [0, 1],
        np.ones((2, 2)),
        np.ones((2, 2)),
        scale=2,
        zero=10,
    )
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()
    cache = tmp_path / "cache"
    spectra, hdus = read_cso_fits(str(path), cache_root=cache)
    assert list(cache.iterdir())
    start = datetime(2000, 1, 1)
    args = start, start + timedelta(seconds=1), 100, 101, 1, 1
    np.testing.assert_equal(
        spectra[0].read_slice_rebinned(*args)[0], np.full((2, 2), 12)
    )
    assert spectra[0].source_path == str(path) + ".gz"
    hdus.close()
    hdus.close()
    assert not list(cache.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        spectra[0].read_slice_rebinned(*args)
    invalid = tmp_path / "invalid.fits.gz"
    with gzip.open(invalid, "wb") as handle:
        handle.write(b"not a FITS file")
    with pytest.raises(OSError):
        read_cso_fits(str(invalid), cache_root=cache)
    assert not list(cache.iterdir())


def test_lazy_mismatched_channel_labels_fail_before_partial_construction(
    tmp_path, monkeypatch
):
    from solar_toolkit.radio import _lazy_spectrogram as lazy

    path = source_file(
        tmp_path / "mismatch.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    with fits.open(path, mode="update", memmap=False) as hdus:
        hdus[0].data = np.ones((3, 2, 2))
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()
    calls = []
    original = lazy.LazySpectrogram

    def construct(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lazy, "LazySpectrogram", construct)
    with pytest.raises(ValueError, match="channel count"):
        lazy.read_cso_fits(str(path), cache_root=tmp_path / "cache")
    assert not calls and not list((tmp_path / "cache").iterdir())


def test_lazy_body_baseexception_releases_views_and_cache(tmp_path, monkeypatch):
    from solar_toolkit.radio import _lazy_spectrogram as lazy

    path = source_file(
        tmp_path / "interrupt.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    with gzip.open(str(path) + ".gz", "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("synthetic interruption after spectrum construction")

    monkeypatch.setattr(lazy, "print", interrupted, raising=False)
    with pytest.raises(KeyboardInterrupt) as error:
        lazy.read_cso_fits(str(path), cache_root=tmp_path / "cache")
    # Keep the traceback alive during the cleanup assertion.
    assert error.value.__traceback__ is not None
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("compressed", [False, True])
def test_lazy_close_preserves_external_array_views(tmp_path, compressed):
    import mmap

    from solar_toolkit.radio._lazy_spectrogram import read_cso_fits

    path = source_file(
        tmp_path / "external.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    if compressed:
        with gzip.open(str(path) + ".gz", "wb") as handle:
            handle.write(path.read_bytes())
        path.unlink()
    _, hdus = read_cso_fits(str(path), cache_root=tmp_path / "cache")
    view = hdus[0].data[0]
    backing = view
    while getattr(backing, "base", None) is not None:
        backing = backing.base
    assert isinstance(backing, mmap.mmap)
    try:
        hdus.close()
    except RuntimeError as exc:
        # Windows cannot unlink the temporary mapping while an external view
        # survives; releasing it and retrying close must finish cleanup.
        assert compressed and "release external array views" in str(exc)
        assert not backing.closed
        np.testing.assert_equal(view, np.ones((2, 2)))
        del view, backing
        hdus.close()
    else:
        # Check ownership before touching memory: force-closing a live mmap
        # would make even reading the view unsafe.
        assert not backing.closed
        np.testing.assert_equal(view, np.ones((2, 2)))


def test_lazy_failed_temporary_cleanup_can_be_retried(tmp_path, monkeypatch):
    import shutil
    from contextlib import contextmanager

    from solar_toolkit.radio import _lazy_spectrogram as lazy

    path = source_file(
        tmp_path / "retry.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    temporary = tmp_path / "expanded"
    temporary.mkdir()
    readable = temporary / path.name
    shutil.copyfile(path, readable)

    @contextmanager
    def blocked_cleanup(*args, **kwargs):
        yield readable
        raise PermissionError("synthetic Windows external-view lock")

    monkeypatch.setattr(lazy, "materialize_fits", blocked_cleanup)
    _, hdus = lazy.read_cso_fits(str(path))
    view = hdus[0].data[0]
    with pytest.raises(RuntimeError, match="release external array views"):
        hdus.close()
    np.testing.assert_equal(view, np.ones((2, 2)))
    assert temporary.exists()
    del view
    hdus.close()
    hdus.close()
    assert not temporary.exists()


def test_legacy_source_change_error_releases_internal_mapping(tmp_path, monkeypatch):
    import mmap

    from solar_toolkit.radio import cso_extract

    path = source_file(
        tmp_path / "changed.fits", [0, 1], np.ones((2, 2)), np.ones((2, 2))
    )
    source = Path(str(path) + ".gz")
    with gzip.open(source, "wb") as handle:
        handle.write(path.read_bytes())
    path.unlink()
    original = cso_extract.fits.open
    opened = 0

    def changed(*args, **kwargs):
        nonlocal opened
        result = original(*args, **kwargs)
        opened += 1
        if opened == 2:
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(cso_extract.fits, "open", changed)
    with pytest.raises(RuntimeError, match="Source changed") as error:
        extract_spectrum(
            [path],
            [946684800, 946684801],
            quantized=False,
            frequency_bounds=[100, 102],
            time_bin_seconds=1,
            frequency_bin_mhz=1,
            cache_root=tmp_path / "cache",
        )
    traceback = error.value.__traceback__
    found = False
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_extract_spectrum":
            backing = traceback.tb_frame.f_locals["data"]
            while getattr(backing, "base", None) is not None:
                backing = backing.base
            assert isinstance(backing, mmap.mmap) and backing.closed
            found = True
        traceback = traceback.tb_next
    assert found and not list((tmp_path / "cache").iterdir())
