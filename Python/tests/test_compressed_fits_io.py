"""Compression is a storage boundary, preserving FITS pixels and explicit HDUs."""

import gzip
import os
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def compressed(path, data=None):
    raw = path.with_suffix("")
    fits.writeto(
        raw, np.arange(24, dtype=np.int16).reshape(4, 6) if data is None else data
    )
    path.write_bytes(gzip.compress(raw.read_bytes()))
    raw.unlink()
    return path


def test_resolver_preserves_exact_path_then_only_tries_appended_gzip(tmp_path):
    from solar_toolkit.io import resolve_fits_path

    source = compressed(tmp_path / "source.fits.gz")
    legacy = tmp_path / "source.fits"
    assert resolve_fits_path(legacy) == source
    legacy.write_bytes(b"explicit existing source")
    assert resolve_fits_path(legacy) == legacy
    with pytest.raises(FileNotFoundError):
        resolve_fits_path(tmp_path / "source.fit")
    with pytest.raises(IsADirectoryError):
        resolve_fits_path(tmp_path)


def test_gzip_context_materializes_seekable_fits_and_cleans_up(tmp_path):
    from solar_toolkit.io import materialize_fits

    source = compressed(tmp_path / "source.fits.gz")
    before = sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "private-cache"
    with materialize_fits(source, cache_root=cache) as path:
        assert isinstance(path, Path) and path != source
        with fits.open(path, memmap=True) as hdul:
            np.testing.assert_array_equal(
                hdul[0].section[1:3, 2:5], [[8, 9, 10], [14, 15, 16]]
            )
        assert path.is_file()
    assert not path.exists() and not list(cache.iterdir())
    assert sha256(source.read_bytes()).hexdigest() == before


def test_context_has_no_persistent_cache_and_never_removes_plain_inputs(tmp_path):
    from solar_toolkit.io import materialize_fits

    source = compressed(tmp_path / "source.fits.gz")
    cache = tmp_path / "cache"
    with materialize_fits(source, cache_root=cache) as first:
        with materialize_fits(source, cache_root=cache) as second:
            assert first != second and first.is_file() and second.is_file()
        assert not second.exists() and first.is_file()
    assert not first.exists()
    plain = tmp_path / "plain.fits"
    fits.writeto(plain, np.zeros((2, 2)))
    with materialize_fits(plain, cache_root=tmp_path / "unused") as actual:
        assert actual == plain
    assert plain.is_file() and not (tmp_path / "unused").exists()


def test_context_cleans_partial_output_and_propagates_body_errors(tmp_path):
    from solar_toolkit.io import materialize_fits

    source = compressed(tmp_path / "source.fits.gz")
    cache = tmp_path / "cache"
    with pytest.raises(LookupError, match="consumer failure"):
        with materialize_fits(source, cache_root=cache) as expanded:
            raise LookupError("consumer failure")
    assert not expanded.exists() and not list(cache.iterdir())
    source.write_bytes(source.read_bytes()[:-12])
    with pytest.raises((EOFError, OSError)):
        with materialize_fits(source, cache_root=cache):
            pytest.fail("A truncated gzip stream must not be yielded")
    assert not list(cache.iterdir())


def test_context_detects_source_identity_changes_and_cleans_output(tmp_path):
    from solar_toolkit.io import materialize_fits

    source = compressed(tmp_path / "source.fits.gz")
    with pytest.raises(RuntimeError, match="changed"):
        with materialize_fits(source, cache_root=tmp_path / "cache") as expanded:
            source.write_bytes(source.read_bytes() + b"changed")
    assert not expanded.exists()


def test_context_detects_replacement_even_with_matching_size_and_mtime(tmp_path):
    from solar_toolkit.io import materialize_fits

    source = compressed(tmp_path / "source.fits.gz")
    before = source.stat()
    replacement = tmp_path / "replacement.gz"
    replacement.write_bytes(source.read_bytes())
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match="changed"):
        with materialize_fits(source, cache_root=tmp_path / "cache") as expanded:
            replacement.replace(source)
    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert not expanded.exists()


def test_scaled_blank_gzip_read_is_detached_and_keeps_legacy_hdu_default(tmp_path):
    from solar_toolkit.io import read_fits_data_header

    raw = tmp_path / "scaled.fits"
    hdu = fits.PrimaryHDU(np.array([[1, -999], [3, 4]], dtype=np.int16))
    hdu.header.update(BSCALE=2, BZERO=10, BLANK=-999)
    hdu.writeto(raw)
    zipped = raw.with_name(raw.name + ".gz")
    zipped.write_bytes(gzip.compress(raw.read_bytes()))
    raw.unlink()
    data, header = read_fits_data_header(raw)
    np.testing.assert_array_equal(data, [[12, np.nan], [16, 18]])
    assert header["NAXIS"] == 2 and data.flags.owndata
    source = tmp_path / "extension.fits.gz"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=fits.Header({"DATE-OBS": "2000-01-01T00:00:00"})),
            fits.ImageHDU(np.ones((2, 3))),
        ]
    ).writeto(source)
    assert read_fits_data_header(source)[0] is None
    auto_data, auto_header = read_fits_data_header(source, hdu_index=None)
    assert (
        auto_data.shape == (2, 3) and auto_header["DATE-OBS"] == "2000-01-01T00:00:00"
    )
    assert read_fits_data_header(source, hdu_index=1)[0].shape == (2, 3)


def test_discovery_compression_suffixes_raw_preference_and_physical_size_filter(
    tmp_path,
):
    from solar_toolkit.io import scan_fits

    source = compressed(tmp_path / "frame2.FITS.gz")
    plain = tmp_path / "frame2.FITS"
    fits.writeto(plain, np.zeros((2, 2)))
    nested = tmp_path / "nested"
    nested.mkdir()
    compressed(nested / "frame10.fts.gz")
    (tmp_path / "not_fits.csv.gz").write_bytes(b"ignore")
    assert scan_fits(tmp_path) == [plain]
    assert len(scan_fits(tmp_path, recursive=True)) == 2
    plain.unlink()
    assert scan_fits(tmp_path) == [source]
    assert scan_fits(tmp_path, min_size_kb=1) == []  # Physical compressed bytes.


def test_sorted_discovery_keeps_compressed_utc_and_explicit_small_file_threshold(
    tmp_path,
):
    from solar_toolkit.io.discovery import get_sorted_fits_files

    source = compressed(tmp_path / "aia_2000-01-01T044800Z.fits.gz")
    assert get_sorted_fits_files(tmp_path, min_size_kb=0)[0][0] == source
    assert (
        get_sorted_fits_files(tmp_path, min_size_kb=0)[0][1].isoformat()
        == "2000-01-01T04:48:00"
    )


def test_auto_hdu_skips_tables_and_rejects_an_imageless_file(tmp_path):
    from solar_toolkit.io import read_fits_data_header

    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="value", format="D", array=[7.0])]
    )
    path = tmp_path / "table-and-image.fits.gz"
    fits.HDUList(
        [fits.PrimaryHDU(), table, fits.ImageHDU(np.full((2, 2), 9.0))]
    ).writeto(path)
    np.testing.assert_array_equal(
        read_fits_data_header(path, hdu_index=None)[0], np.full((2, 2), 9.0)
    )
    path = tmp_path / "table-only.fits.gz"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    with pytest.raises(ValueError, match="image HDU"):
        read_fits_data_header(path, hdu_index=None)
