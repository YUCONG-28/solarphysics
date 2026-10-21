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
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
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
    # Two display cells cover the original four-pixel footprint, with true
    # regular angular center axes after full-WCS display resampling.
    assert background.x_arcsec.tolist() == pytest.approx([1.0, 5.0], abs=1e-8)
    assert background.y_arcsec.tolist() == pytest.approx([1.0, 5.0], abs=1e-8)
    assert background.obs_time == pd.Timestamp("2000-01-01T04:48:37")
    assert "171" in background.label


@pytest.mark.parametrize(
    "tolerance", [np.nan, np.inf, -1, True, np.bool_(True), [1], None]
)
@pytest.mark.parametrize("empty", [False, True])
def test_nearest_aia_rejects_invalid_tolerance_before_empty_table(tolerance, empty):
    table = (
        pd.DataFrame(columns=["path", "obs_time"])
        if empty
        else pd.DataFrame(
            {
                "path": ["synthetic.fits"],
                "obs_time": [pd.Timestamp("2000-01-01T00:00:00")],
            }
        )
    )
    with pytest.raises(ValueError, match="max_dt_seconds"):
        find_nearest_aia(table, "2000-01-01T01:00:00", max_dt_seconds=tolerance)


def test_nearest_aia_zero_tolerance_requires_an_exact_instant():
    table = pd.DataFrame(
        {"path": ["synthetic.fits"], "obs_time": [pd.Timestamp("2000-01-01T00:00:00")]}
    )
    assert (
        find_nearest_aia(table, "2000-01-01T00:00:00Z", max_dt_seconds=0).status
        == "matched"
    )
    assert (
        find_nearest_aia(
            table, "2000-01-01T00:00:00.000000001Z", max_dt_seconds=0
        ).status
        == "too_far"
    )
