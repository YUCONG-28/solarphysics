"""Actual AIA/HMI readers retain science semantics across gzip storage."""

import gzip
from hashlib import sha256
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import sunpy.map
from astropy.io import fits

from solar_toolkit.aia.config import AIAConfig


def write_observation(folder, *, time="000000", hmi=False, extension=False):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{'hmi' if hmi else 'aia'}_2000-01-01T{time}Z.fts"
    path = folder / name
    data = np.random.default_rng(7).normal(100, 25, (32, 32)).astype(np.float32)
    data[5, 5] = np.nan
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 1.0,
            "CDELT2": 1.0,
            "DATE-OBS": f"2000-01-01T{time[:2]}:{time[2:4]}:{time[4:]}",
            "DSUN_OBS": 149597870700.0,
            "HGLN_OBS": 0.0,
            "HGLT_OBS": 0.0,
            "RSUN_REF": 695700000.0,
            "WAVELNTH": 6173 if hmi else 171,
            "WAVEUNIT": "Angstrom",
            "BUNIT": "G" if hmi else "DN",
            "EXPTIME": 2.0,
        }
    )
    if extension:
        fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(data)]).writeto(
            path
        )
    else:
        fits.writeto(path, data, header)
    zipped = path.with_name(path.name + ".gz")
    zipped.write_bytes(gzip.compress(path.read_bytes()))
    path.unlink()
    return zipped, data


def test_aia_file_selection_shares_gzip_fallback_and_deduplication(tmp_path):
    from solar_toolkit.aia.io import (
        parse_timestr,
        resolve_files,
        resolve_single_files,
        resolve_test_file,
    )

    path, _ = write_observation(tmp_path / "171")
    legacy = path.with_suffix("")
    assert parse_timestr(path) == parse_timestr(legacy)
    assert resolve_files(legacy, 0, None) == [path]
    assert resolve_files(tmp_path, 0, None) == [path]
    cfg = AIAConfig(data_path=str(legacy), test_file=str(legacy))
    assert resolve_single_files(cfg) == [path]
    assert resolve_test_file(cfg) == path
    legacy.write_bytes(gzip.decompress(path.read_bytes()))
    assert resolve_files(tmp_path, 0, None) == [legacy]


@pytest.mark.parametrize(
    "name",
    ["aia_20000101000000001.fts", "undated.fit", "aia.2000-01-01T000000Z.image.fits"],
)
def test_gzip_never_changes_the_legacy_aia_sort_key(name):
    from solar_toolkit.aia.io import parse_timestr

    assert parse_timestr(Path(name)) == parse_timestr(Path(name + ".gz"))


def test_background_gzip_metadata_uses_no_materialized_copy_and_sunpy_reads_once(
    tmp_path, monkeypatch
):
    from solar_toolkit.aia import background as module

    path, _ = write_observation(tmp_path, extension=True)
    before = sha256(path.read_bytes()).hexdigest()
    materialize = module.materialize_fits
    calls = []

    def tracked(*args, **kwargs):
        calls.append(args[0])
        return materialize(*args, **kwargs)

    monkeypatch.setattr(module, "materialize_fits", tracked)
    table = module.scan_aia_folder(tmp_path)
    assert calls == [] and len(table) == 1 and table.iloc[0].hdu_index == 1
    assert table.iloc[0].path == str(path)
    background = module.read_aia_background(
        path.with_suffix(""), wcs_mode="sunpy", log_scale=False
    )
    assert calls == [path] and background.path == str(path)
    assert background.obs_time.isoformat() == "2000-01-01T00:00:00"
    assert np.isnan(background.z).any()
    assert sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("entry", ["difference", "mosaic", "single"])
def test_actual_aia_entries_read_gzip_extension_and_normalize_once(
    tmp_path, monkeypatch, entry
):
    from solar_toolkit.aia import _euv_processor_impl as impl

    path, data = write_observation(tmp_path, extension=True)
    before = sha256(path.read_bytes()).hexdigest()
    cfg = AIAConfig(
        data_path=str(tmp_path),
        roi_bounds=(2, 10, 3, 12),
        save_image=False,
        show_image=False,
        show_grid=False,
    )
    plotted = []
    real_plot = sunpy.map.GenericMap.plot

    def capture(smap, *args, **kwargs):
        plotted.append(smap)
        return real_plot(smap, *args, **kwargs)

    monkeypatch.setattr(sunpy.map.GenericMap, "plot", capture)
    if entry == "difference":
        result = impl._load_exposure_normalized_map(path.with_suffix(""))
        np.testing.assert_allclose(result.data, data / 2, equal_nan=True)
    elif entry == "mosaic":
        result = impl._load_aia_cutout_panel(path, 171, cfg).cutout_map
    else:
        success, message = impl._process_single_worker(path, cfg)
        assert success, message
        result = plotted[-1]
    assert result.unit == u.DN / u.s and result.exposure_time == 2 * u.s
    assert np.isnan(result.data).any()
    assert sha256(path.read_bytes()).hexdigest() == before


def test_hmi_magnetogram_actual_workflow_reads_compressed_sources(tmp_path):
    from solar_toolkit.hmi.magnetogram import run_magnetogram_workflow

    first, _ = write_observation(tmp_path / "hmi", hmi=True, extension=True)
    second, _ = write_observation(
        tmp_path / "hmi", time="000012", hmi=True, extension=True
    )
    hashes = [sha256(path.read_bytes()).hexdigest() for path in (first, second)]
    results = run_magnetogram_workflow(
        first.parent, tmp_path / "out", roi_bounds=(2, 3, 10, 12), dpi=30
    )
    assert len(results) == 1 and results[0].is_file()
    assert [sha256(path.read_bytes()).hexdigest() for path in (first, second)] == hashes


def test_hmi_overlay_actual_workflow_discovers_gzip_aia_and_hmi(tmp_path, monkeypatch):
    from solar_toolkit.hmi.overlay import run_overlay_workflow

    paths = []
    for hmi in (False, True):
        for time in ("000000", "000012"):
            path, _ = write_observation(
                tmp_path / ("hmi" if hmi else "aia"), time=time, hmi=hmi, extension=True
            )
            paths.append(path)
    before = [sha256(path.read_bytes()).hexdigest() for path in paths]
    monkeypatch.setattr(
        "solar_toolkit.visualization.plotting.setup_chinese_font", lambda: None
    )
    results = run_overlay_workflow(
        tmp_path / "aia",
        tmp_path / "hmi",
        tmp_path / "out",
        roi_bounds=(2, 3, 10, 12),
        gaussian_sigma=0,
        show_progress=False,
        dpi=30,
    )
    assert len(results) == 2 and all(path.is_file() for path in results)
    assert [sha256(path.read_bytes()).hexdigest() for path in paths] == before


def test_actual_aia_normalizer_applies_storage_scaling_before_exposure(tmp_path):
    from solar_toolkit.aia._euv_processor_impl import _load_exposure_normalized_map

    reference, _ = write_observation(tmp_path)
    header = fits.getheader(reference)
    raw = tmp_path / "scaled.fits"
    hdu = fits.PrimaryHDU(np.array([[1, -999], [3, 4]], dtype=np.int16), header=header)
    hdu.header.update(BSCALE=2.0, BZERO=10.0, BLANK=-999)
    hdu.writeto(raw)
    zipped = raw.with_name(raw.name + ".gz")
    zipped.write_bytes(gzip.compress(raw.read_bytes()))
    raw.unlink()
    before = sha256(zipped.read_bytes()).hexdigest()
    result = _load_exposure_normalized_map(raw)
    np.testing.assert_array_equal(result.data, [[6, np.nan], [8, 9]])
    assert result.unit == u.DN / u.s
    assert sha256(zipped.read_bytes()).hexdigest() == before
