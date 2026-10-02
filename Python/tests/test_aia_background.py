from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from solar_toolkit.aia.background import (
    find_nearest_aia,
    read_aia_background,
    scan_aia_folder,
)


def _write_aia(path, obs_time: str, wavelength: int = 171) -> None:
    image = np.arange(16, dtype=float).reshape(4, 4)
    header = fits.Header()
    header["DATE-OBS"] = obs_time
    header["WAVELNTH"] = wavelength
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    fits.writeto(path, image, header, overwrite=True)


def test_scans_aia_folder_and_finds_nearest_file(tmp_path):
    _write_aia(tmp_path / "aia_171_20000101_044837.fits", "2000-01-01T04:48:37")

    table = scan_aia_folder(tmp_path)
    matched = find_nearest_aia(
        table,
        pd.Timestamp("2000-01-01T04:48:39"),
        max_dt_seconds=5.0,
    )
    too_far = find_nearest_aia(
        table,
        pd.Timestamp("2000-01-01T04:49:30"),
        max_dt_seconds=5.0,
    )
    empty = find_nearest_aia(
        pd.DataFrame(columns=["path", "obs_time"]),
        pd.Timestamp("2000-01-01T04:48:39"),
        max_dt_seconds=5.0,
    )

    assert len(table) == 1
    assert table.loc[0, "wavelength"] == "171"
    assert matched.status == "matched"
    assert matched.delta_seconds == pytest.approx(2.0)
    assert too_far.status == "too_far"
    assert empty.status == "no_files"


def test_reads_lightweight_aia_background_grid(tmp_path):
    path = tmp_path / "aia_171_20000101_044837.fits"
    _write_aia(path, "2000-01-01T04:48:37")

    background = read_aia_background(
        path,
        max_pixels=2,
        percentile_limits=(0.0, 100.0),
        log_scale=False,
    )

    assert background.z.shape == (2, 2)
    assert background.x_arcsec.tolist() == pytest.approx([0.0, 4.0])
    assert background.y_arcsec.tolist() == pytest.approx([0.0, 4.0])
    assert background.obs_time == pd.Timestamp("2000-01-01T04:48:37")
    assert "171" in background.label
