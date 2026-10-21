"""Exercise the actual AIA image/panel/difference loaders, not only the helper."""

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import sunpy.map

from solar_toolkit.aia import _euv_processor_impl as impl
from solar_toolkit.aia.config import AIAConfig
from solar_toolkit.aia.io import build_multi_band_slots


@pytest.fixture
def source_and_capture(monkeypatch):
    original_factory = sunpy.map.Map
    source = None
    plotted = []
    original_plot = sunpy.map.GenericMap.plot

    def capture_plot(smap, *args, **kwargs):
        plotted.append(smap)
        return original_plot(smap, *args, **kwargs)

    monkeypatch.setattr(sunpy.map.GenericMap, "plot", capture_plot)

    def create(unit, exposure):
        nonlocal source
        header = dict(
            ctype1="HPLN-TAN",
            ctype2="HPLT-TAN",
            cunit1="arcsec",
            cunit2="arcsec",
            crpix1=1,
            crpix2=1,
            crval1=0,
            crval2=0,
            cdelt1=1,
            cdelt2=1,
            date_obs="2000-01-01T00:00:00",
            dsun_obs=149597870700,
            hgln_obs=0,
            hglt_obs=0,
            rsun_ref=695700000,
            wavelnth=171,
            waveunit="Angstrom",
            bunit=unit,
        )
        if exposure is not None:
            header["exptime"] = exposure
        mask = np.zeros((16, 16), bool)
        mask[5, 5] = True
        source = original_factory(np.full((16, 16), 100.0), header, mask=mask)
        return source

    def read_or_construct(value, *args, **kwargs):
        return (
            source
            if isinstance(value, (Path, str))
            else original_factory(value, *args, **kwargs)
        )

    monkeypatch.setattr(impl.sunpy.map, "Map", read_or_construct)
    return create, plotted


def config():
    return AIAConfig(
        data_path="synthetic/aia",
        roi_bounds=(2, 10, 3, 12),
        save_image=False,
        show_image=False,
        show_grid=False,
    )


def exercise(entry, plotted):
    path = Path("synthetic_20000101T000000Z.fits")
    if entry == "difference-loader":
        return impl._load_exposure_normalized_map(path)
    if entry == "mosaic-panel":
        return impl._load_aia_cutout_panel(path, 171, config()).cutout_map
    success, message = impl._process_single_worker(path, config())
    assert success, message
    return plotted[-1]


@pytest.mark.parametrize(
    "entry", ["difference-loader", "mosaic-panel", "single-worker"]
)
@pytest.mark.parametrize("unit,expected", [("DN", 50.0), ("DN/s", 100.0)])
def test_actual_aia_entries_preserve_rate_units_mask_and_original_exposure(
    source_and_capture, entry, unit, expected
):
    create, plotted = source_and_capture
    source = create(unit, 2.0)
    before_data, before_mask, before_meta = (
        source.data.copy(),
        source.mask.copy(),
        source.meta.copy(),
    )
    result = exercise(entry, plotted)
    np.testing.assert_allclose(result.data, expected)
    assert result.unit == u.DN / u.s
    assert result.exposure_time == 2 * u.s
    assert result.mask is not None and result.mask.sum() == 1
    np.testing.assert_array_equal(source.data, before_data)
    np.testing.assert_array_equal(source.mask, before_mask)
    assert source.meta == before_meta


@pytest.mark.parametrize(
    "entry", ["difference-loader", "mosaic-panel", "single-worker"]
)
@pytest.mark.parametrize("exposure", [0, np.nan, np.inf, None])
def test_actual_aia_entries_reject_invalid_integrated_exposure(
    source_and_capture, entry, exposure
):
    create, _ = source_and_capture
    create("DN", exposure)
    path = Path("synthetic.fits")
    if entry == "single-worker":
        success, message = impl._process_single_worker(path, config())
        assert not success and "exposure" in message.lower()
    else:
        with pytest.raises(ValueError, match="exposure"):
            exercise(entry, [])


def test_legacy_multiband_slots_keep_index_pairing_explicit(monkeypatch):
    files = {
        171: [Path("171_t0.fits"), Path("171_t12.fits")],
        304: [Path("304_t12.fits"), Path("304_t24.fits")],
    }
    monkeypatch.setattr(
        "solar_toolkit.aia.io.sorted_fits_for_band",
        lambda path, wave, subdirs: files[wave],
    )
    assert build_multi_band_slots(config(), (171, 304)) == [
        (files[171][0], files[304][0]),
        (files[171][1], files[304][1]),
    ]
