import numpy as np
import pytest
import sunpy.map
from astropy.io import fits

from solar_toolkit.map.euvi_preprocessing import prepare_euvi_map, prepared_map
from solar_toolkit.map.jet_annotations import intensity_per_second


def example(unit="DN", **extra):
    header = dict(
        CTYPE1="HPLN-TAN",
        CTYPE2="HPLT-TAN",
        CUNIT1="arcsec",
        CUNIT2="arcsec",
        CRPIX1=2,
        CRPIX2=2,
        CRVAL1=0,
        CRVAL2=0,
        CDELT1=1.6,
        CDELT2=1.6,
        DETECTOR="EUVI",
        OBSRVTRY="STEREO_A",
        DATE_OBS="2000-01-01T12:00:00",
        EXPTIME=4,
        BUNIT=unit,
        SYNTHET=True,
        HGLN_OBS=29,
        HGLT_OBS=-7,
        DSUN_OBS=1.47e11,
        CROTA=1.5,
        WAVELNTH=304,
    )
    header.update(extra)
    return sunpy.map.Map(
        np.array([[0.0, 20, np.nan], [8, 12, 16], [4, np.inf, 24]]), fits.Header(header)
    )


def test_dn_once_masks_and_native_wcs():
    m = example(NMISSING=1)
    m.mask = np.zeros(m.data.shape, bool)
    m.mask[1, 1] = True
    original = m.meta.copy()
    out, audit = prepared_map(m)
    assert out.data[0, 1] == 5
    assert out.data[0, 0] == 0
    assert np.isnan(out.data[1, 1]) and np.isnan(out.data[2, 1])
    np.testing.assert_equal(intensity_per_second(out), out.data)
    assert m.meta == original
    assert out.wcs.to_header() == m.wcs.to_header()
    assert "telemetry_missing_blocks_not_decoded" in audit["warnings"]
    assert not audit["idl_invoked"]


@pytest.mark.parametrize(
    "unit", ["DN/s", "DN / s", "DN s-1", "photon/s", "DN/s/CCDPIX"]
)
def test_rate_units_preserved(unit):
    result = prepare_euvi_map(example(unit))
    assert result.data[0, 1] == 20
    assert result.unit == unit
    assert result.audit["exposure_action"] == "already_rate_no_division"


@pytest.mark.parametrize("unit", ["", "unknown", "DN/sr", "photon"])
def test_unknown_not_assumed_dn_or_rate(unit):
    result = prepare_euvi_map(example(unit))
    assert result.data[0, 1] == 20
    assert result.unit == unit
    assert result.audit["exposure_action"] == "unchanged_requires_review"


def test_ambiguous_processed_history_and_bad_exposure():
    assert prepare_euvi_map(example(JETNORM=True)).data[0, 1] == 20
    result = prepare_euvi_map(example(EXPTIME=0))
    assert result.display_usable and not result.quantitative_usable
    assert result.data[0, 1] == 20
    assert "invalid_exposure_for_normalization" in result.reasons


def test_unknown_difference_is_not_published(tmp_path):
    from solar_toolkit.map.jet_registration import registered_difference

    p = tmp_path / "unknown.fits"
    example("").save(p)
    result, audit = registered_difference(
        example(""), p, "source", tmp_path / "difference.fits"
    )
    assert result is None
    assert audit["status"] == "inputs_not_quantitatively_usable"
    assert "unit_or_normalization_history_ambiguous" in audit["reasons"]
    assert not (tmp_path / "difference.fits").exists()


def test_equivalent_units_and_no_detector_unit_substitution():
    from solar_toolkit.map.euvi_preprocessing import unit_factor

    assert unit_factor("DN s-1", "DN/s") == 1
    assert unit_factor("DN/min", "DN/s") == pytest.approx(1 / 60)
    for wrong in ("photon/s", "DN/s/pix", "DN/s/CCDPIX", "unknown"):
        with pytest.raises(ValueError):
            unit_factor(wrong, "DN/s")
    assert prepare_euvi_map(example("2 DN")).data[0, 1] == 10
    assert (
        "invalid_missing_block_count"
        in prepare_euvi_map(example(NMISSING="nan")).reasons
    )


def test_preparation_has_separate_display_and_quantitative_readiness():
    m = example(NMISSING=1)
    p = prepare_euvi_map(m)
    assert p.display_usable and not p.quantitative_usable
    assert "telemetry_missing_blocks_not_decoded" in p.reasons
    del m.meta["hgln_obs"]
    from solar_toolkit.map.jet_viewpoint import build_mapping

    with pytest.raises(ValueError, match="hgln_obs"):
        build_mapping(m, m)
    assert prepare_euvi_map(m).display_usable


def test_difference_updates_unit_and_preserves_fixed_mask(tmp_path):
    from solar_toolkit.map.jet_registration import registered_difference

    rng = np.random.default_rng(3)
    image = 30 + rng.normal(size=(96, 96))
    header = example().meta.copy()
    header.update(dict(crpix1=48, crpix2=48, exptime=2))
    before = sunpy.map.Map(image * 2, header)
    previous = tmp_path / "previous.fits"
    before.save(previous)
    header["date_obs"] = "2000-01-01T12:00:12"
    current = sunpy.map.Map(image * 2 + 4, header)
    destination = tmp_path / "delta.fits"
    result, audit = registered_difference(current, previous, "hash", destination)
    assert result == destination
    delta = sunpy.map.Map(result)
    assert delta.meta["bunit"] == "DN/s"
    assert np.nanmedian(delta.data) == pytest.approx(2, abs=0.05)
    current.meta["wavelnth"] = 171
    result, audit = registered_difference(
        current, previous, "hash", tmp_path / "wrong.fits"
    )
    assert result is None and "different_wavelnth" in audit["reasons"]
