"""Scientific bounds, exposure units and missing-pixel contracts."""

import astropy.units as u
import numpy as np
import pytest

from solar_toolkit.map.operations import create_aia_submap, normalize_aia_exposure


def solar_map(*, unit="DN", exposure=2.0):
    import sunpy.map

    data = np.full((12, 12), 100.0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[4, 5] = True
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
    )
    if unit is not None:
        header["bunit"] = unit
    if exposure is not None:
        header["exptime"] = exposure
    return sunpy.map.Map(data, header, mask=mask)


def test_four_roi_bounds_are_used_and_legacy_mode_is_explicit():
    source = solar_map()
    result = create_aia_submap(source, (2, 7, 3, 9))
    legacy = create_aia_submap(source, (2, 7, 3, 9), legacy_roi=True)
    assert result.data.shape == (7, 6)
    assert legacy.data.shape == (1, 6)
    np.testing.assert_array_equal(result.mask, source.mask[3:10, 2:8])


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 1, 2),
        (0, 0, 0, 1),
        (2, 1, 0, 1),
        (0, 1, 3, 2),
        (0, np.inf, 0, 1),
        (0, 1, np.nan, 2),
    ],
)
def test_roi_bounds_must_be_finite_and_increasing(bounds):
    with pytest.raises(ValueError, match="roi_bounds"):
        create_aia_submap(solar_map(), bounds)


def test_legacy_roi_must_be_boolean():
    with pytest.raises(TypeError, match="legacy_roi"):
        create_aia_submap(solar_map(), (0, 1, 0, 1), legacy_roi="false")


def test_exposure_normalization_keeps_original_exposure_mask_and_input():
    source = solar_map()
    source.meta["history"] = ["original calibration"]
    data, mask, meta = source.data.copy(), source.mask.copy(), source.meta.copy()
    result = normalize_aia_exposure(source)
    assert result.unit == u.DN / u.s
    assert result.exposure_time == 2 * u.s
    np.testing.assert_allclose(result.data, 50)
    np.testing.assert_array_equal(result.mask, mask)
    np.testing.assert_array_equal(source.data, data)
    assert source.meta == meta
    assert not np.shares_memory(result.data, source.data)
    assert not np.shares_memory(result.mask, source.mask)
    assert "original calibration" in result.meta["history"]
    assert any("exposure" in item.lower() for item in result.meta["history"])
    again = normalize_aia_exposure(result)
    np.testing.assert_array_equal(again.data, result.data)
    assert again.meta == result.meta


@pytest.mark.parametrize("unit", ["DN/s", "DN/ms", "count/s", "photon/s"])
def test_existing_rates_are_not_divided_again(unit):
    source = solar_map(unit=unit, exposure=0)
    result = normalize_aia_exposure(source)
    np.testing.assert_array_equal(result.data, source.data)
    assert result.unit == source.unit
    assert result.meta == source.meta


@pytest.mark.parametrize("exposure", [None, 0, -1, np.nan, np.inf])
def test_invalid_integrated_exposure_is_rejected(exposure):
    with pytest.raises(ValueError, match="exposure"):
        normalize_aia_exposure(solar_map(exposure=exposure))


@pytest.mark.parametrize("unit", [None, "G"])
def test_exposure_requires_explicit_detector_units(unit):
    with pytest.raises(ValueError, match="unit|BUNIT"):
        normalize_aia_exposure(solar_map(unit=unit))
