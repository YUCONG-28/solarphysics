from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.io import fits


def test_scan_files_natural_sort_and_min_size(tmp_path):
    from solar_toolkit.io import scan_files, scan_fits

    tiny = tmp_path / "frame1.fits"
    tiny.write_bytes(b"x")
    fits.writeto(tmp_path / "frame10.fits", np.zeros((2, 2)), overwrite=True)
    fits.writeto(tmp_path / "frame2.fits", np.ones((2, 2)), overwrite=True)
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    assert [path.name for path in scan_files(tmp_path, suffixes=[".fits"])] == [
        "frame1.fits",
        "frame2.fits",
        "frame10.fits",
    ]
    assert [path.name for path in scan_fits(tmp_path, min_size_kb=1)] == [
        "frame2.fits",
        "frame10.fits",
    ]


def test_sorted_fits_discovery_can_recurse_into_instrument_folders(
    tmp_path,
) -> None:
    from solar_toolkit.io.discovery import get_sorted_fits_files

    nested = tmp_path / "171" / "aia_2025-01-24T044800Z.fits"
    nested.parent.mkdir()
    nested.write_bytes(b"x" * 2048)

    assert get_sorted_fits_files(tmp_path) == []
    discovered = get_sorted_fits_files(tmp_path, recursive=True)

    assert [path for path, _timestamp in discovered] == [nested]


def test_read_fits_data_header_and_manifest_roundtrip(tmp_path):
    from solar_toolkit.io import read_fits_data_header, read_manifest, write_manifest

    fits_path = tmp_path / "sample.fits"
    fits.writeto(fits_path, np.arange(4).reshape(2, 2), overwrite=True)

    data, header = read_fits_data_header(fits_path)
    assert data.shape == (2, 2)
    assert header["NAXIS"] == 2

    manifest_path = tmp_path / "manifest.csv"
    rows = [{"path": str(fits_path), "kind": "fits"}]
    written = write_manifest(rows, manifest_path)
    roundtrip = read_manifest(written)

    assert written == manifest_path
    assert isinstance(roundtrip, pd.DataFrame)
    assert roundtrip.to_dict("records") == rows
