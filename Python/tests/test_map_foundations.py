"""Scientific map masks, WCS, uncertainty and detached FITS contracts."""

from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty
from astropy.time import Time
from astropy.wcs import WCS

from solar_toolkit.io.fits import read_fits_data_header
from solar_toolkit.map.coordinates import (
    calculate_fits_extent_from_header,
    hpc_pixel_coordinates,
    hpc_pixel_grid,
    hpc_world_bounds,
    hpc_world_to_pixel,
)
from solar_toolkit.map.image import crop_roi, normalize_image
from solar_toolkit.map.metadata import get_map_obs_time
from solar_toolkit.map.operations import align_maps_to_reference, normalize_aia_exposure
from solar_toolkit.radio.centers import compute_source_center
from solar_toolkit.radio.reprojection import (
    interpolate_scattered_to_grid,
    reproject_radio_array,
)
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    build_radio_roi_mask,
    measure_radio_roi,
)


def header():
    return dict(
        CTYPE1="HPLN-TAN",
        CTYPE2="HPLT-TAN",
        CUNIT1="arcsec",
        CUNIT2="arcsec",
        CRPIX1=3,
        CRPIX2=3,
        CRVAL1=0,
        CRVAL2=0,
        CDELT1=2,
        CDELT2=3,
        NAXIS1=5,
        NAXIS2=5,
        **{
            "DATE-OBS": "2000-01-01T00:00:00",
            "DSUN_OBS": 149597870700,
            "HGLN_OBS": 0,
            "HGLT_OBS": 0,
            "RSUN_REF": 695700000,
            "BUNIT": "DN",
            "EXPTIME": 2,
            "INSTRUME": "synthetic",
        },
    )


def solar_map(uncertainty=None):
    import sunpy.map

    data = np.arange(25.0).reshape(5, 5)
    data[2, 2] = 999
    mask = np.zeros(data.shape, dtype=bool)
    mask[2, 2] = True
    return sunpy.map.Map(data, header(), mask=mask, uncertainty=uncertainty)


@pytest.mark.parametrize(
    "kind,before,after",
    [
        (StdDevUncertainty, 4, 2),
        (VarianceUncertainty, 16, 4),
        (InverseVariance, 1 / 16, 1 / 4),
    ],
)
def test_exposure_propagates_uncertainty_without_aliasing(kind, before, after):
    source = solar_map(kind(np.full((5, 5), before)))
    original = deepcopy(source.meta)
    result = normalize_aia_exposure(source)
    np.testing.assert_allclose(result.uncertainty.array, after)
    np.testing.assert_allclose(source.uncertainty.array, before)
    assert isinstance(result.uncertainty, kind)
    assert not np.shares_memory(result.uncertainty.array, source.uncertainty.array)
    assert result.uncertainty.unit == source.uncertainty.unit / u.s ** (
        2 if kind is VarianceUncertainty else -2 if kind is InverseVariance else 1
    )
    assert source.meta == original
    assert result.unit == u.DN / u.s
    assert result.exposure_time == source.exposure_time
    np.testing.assert_array_equal(result.mask, source.mask)


def test_missing_map_date_does_not_use_now():
    import sunpy.map

    meta = header()
    del meta["DATE-OBS"]
    source = sunpy.map.Map(np.zeros((5, 5)), meta)
    with pytest.raises(KeyError, match="observation"):
        get_map_obs_time(source)
    with pytest.raises(KeyError, match="observation"):
        get_map_obs_time(meta)


@pytest.mark.parametrize("unit", ["DN", "DN / s"])
def test_exposure_merges_intrinsic_explicit_and_nonfinite_masks(unit):
    import sunpy.map

    data = np.ma.array(np.ones((5, 5)), mask=False)
    data.mask[1, 1] = True
    data.data[1, 1] = 999
    data.data[2, 2] = np.inf
    meta = header()
    meta["BUNIT"] = unit
    source = sunpy.map.Map(data, meta)
    source.mask[3, 3] = True
    result = normalize_aia_exposure(source)
    assert result.mask[1, 1] and result.mask[2, 2] and result.mask[3, 3]
    assert np.isinf(source.data[2, 2])
    assert not source.mask[2, 2]
    assert np.isnan(result.data[2, 2])
    assert not np.shares_memory(result.mask, source.mask)


def test_align_excludes_mask_and_preserves_acquisition_metadata():
    source = solar_map()
    original = source.data.copy(), source.mask.copy(), deepcopy(source.meta)
    target = source.wcs.deepcopy()
    result, footprint = align_maps_to_reference(source, target, return_footprint=True)
    assert np.isnan(result.data[2, 2]) and result.mask[2, 2]
    assert footprint[2, 2] == 0
    assert result.unit == u.DN and result.exposure_time == 2 * u.s
    assert result.meta["instrume"] == "synthetic"
    assert result.date == source.date
    assert result.wcs.wcs.crval == pytest.approx(target.wcs.crval)
    np.testing.assert_array_equal(source.data, original[0])
    np.testing.assert_array_equal(source.mask, original[1])
    assert source.meta == original[2]
    assert not np.shares_memory(result.data, source.data)


def test_align_uncertainty_requires_explicit_drop():
    source = solar_map(StdDevUncertainty(np.ones((5, 5))))
    with pytest.raises(ValueError, match="uncertainty"):
        align_maps_to_reference(source, source.wcs)
    result = align_maps_to_reference(source, source.wcs, uncertainty_policy="drop")
    assert result.uncertainty is None
    assert "uncertainty explicitly dropped" in str(result.meta["history"])
    assert source.uncertainty is not None


def test_align_distinguishes_source_acquisition_from_target_reference_time():
    source = solar_map()
    target = source.wcs.deepcopy()
    target.wcs.dateobs = "2000-01-02T00:00:00"
    target.wcs.dateavg = "2000-01-02T00:00:00"
    target.wcs.mjdobs = Time(target.wcs.dateobs).mjd
    target.wcs.mjdavg = target.wcs.mjdobs
    result = align_maps_to_reference(source, target)
    assert get_map_obs_time(result) == get_map_obs_time(source)
    assert Time(result.meta["date-avg"]) == Time(target.wcs.dateavg)
    assert result.meta["mjd-obs"] == Time(source.date).mjd
    assert result.meta["mjd-avg"] == Time(target.wcs.dateavg).mjd


@pytest.mark.parametrize("reverse", [False, True])
def test_full_wcs_inverse_handles_rotated_axes_units_and_nonlinear_projection(reverse):
    meta = header()
    meta.update(
        CRVAL1=18000,
        CRVAL2=3600,
        CDELT1=600,
        CDELT2=-900,
        PC1_1=0.9,
        PC1_2=0.3,
        PC2_1=-0.2,
        PC2_2=0.8,
    )
    if reverse:
        # Swap complete world-coordinate rows while retaining native pixels.
        for name in ("CTYPE", "CUNIT", "CRVAL", "CDELT"):
            meta[f"{name}1"], meta[f"{name}2"] = meta[f"{name}2"], meta[f"{name}1"]
        for column in (1, 2):
            meta[f"PC1_{column}"], meta[f"PC2_{column}"] = (
                meta[f"PC2_{column}"],
                meta[f"PC1_{column}"],
            )
    x, y = np.array([0, 2.2, 4]), np.array([4, 1.1, 0])
    wx, wy = hpc_pixel_coordinates(meta, x * u.pix, y * u.pix)
    bx, by = hpc_world_to_pixel(meta, wx / 3600 * u.deg, wy * u.arcsec)
    np.testing.assert_allclose(bx, x, atol=2e-10)
    np.testing.assert_allclose(by, y, atol=2e-10)
    wcs = WCS(meta)
    world = wcs.all_pix2world(np.column_stack((x, y)), 0)
    lon, lat = (1, 0) if reverse else (0, 1)
    np.testing.assert_allclose(wx, world[:, lon] * 3600)
    np.testing.assert_allclose(wy, world[:, lat] * 3600)


def test_display_extent_cannot_hide_rotation_but_full_grid_remains_available():
    meta = header()
    meta.update(PC1_1=0, PC1_2=-1, PC2_1=1, PC2_2=0)
    with pytest.raises(ValueError, match="WCSAxes"):
        calculate_fits_extent_from_header(meta)
    x, y = hpc_pixel_grid(meta, (5, 5))
    assert x.shape == y.shape == (5, 5)
    bounds = hpc_world_bounds(meta, (5, 5))
    assert bounds[0] < x.min() < x.max() < bounds[1]
    assert bounds[2] < y.min() < y.max() < bounds[3]
    with pytest.warns(UserWarning):
        legacy = calculate_fits_extent_from_header(meta, legacy_linear=True)
    assert len(legacy) == 4


def test_cd_only_geometry_and_explicit_angular_units():
    meta = header()
    del meta["CDELT1"], meta["CDELT2"]
    meta.update(CD1_1=2, CD2_2=3)
    x, y = hpc_pixel_grid(meta, (5, 5))
    assert x[2, 2] == pytest.approx(0, abs=1e-9)
    assert y[2, 2] == pytest.approx(0, abs=1e-9)
    del meta["CUNIT1"]
    with pytest.raises(ValueError, match="CUNIT1"):
        hpc_pixel_grid(meta, (5, 5))


@pytest.mark.parametrize("as_mapping", [False, True])
def test_wcs_accepts_multicard_commentary_without_mutating_input(as_mapping):
    cards = fits.Header(header())
    cards.add_history("calibration one")
    cards.add_history("calibration two")
    cards.add_comment("description one")
    cards.add_comment("description two")
    before = cards.tostring()
    metadata = dict(cards) if as_mapping else cards
    expected = hpc_pixel_coordinates(header(), [1, 2], [2, 3])
    actual = hpc_pixel_coordinates(metadata, [1, 2], [2, 3])
    np.testing.assert_allclose(actual, expected)
    px, py = hpc_world_to_pixel(metadata, *actual)
    np.testing.assert_allclose([px, py], [[1, 2], [2, 3]], atol=1e-9)
    with pytest.warns(UserWarning):
        assert len(calculate_fits_extent_from_header(metadata)) == 4
    assert cards.tostring() == before


def test_roi_uses_exact_curved_wcs_and_excludes_mask():
    meta = header()
    meta.update(
        CRVAL1=18000,
        CRVAL2=3600,
        CDELT1=6000,
        CDELT2=9000,
        PC1_1=1,
        PC1_2=0.3,
        PC2_1=-0.2,
        PC2_2=1,
    )
    wcs = WCS(meta)
    py, px = np.indices((5, 5))
    world = wcs.all_pix2world(np.column_stack((px.ravel(), py.ravel())), 0) * 3600
    wx = ((world[:, 0] + 648000) % 1296000 - 648000).reshape(5, 5)
    wy = world[:, 1].reshape(5, 5)
    roi = RadioRoi.from_box(10000, -2000, 30000, 15000)
    expected = (wx >= 10000) & (wx <= 30000) & (wy >= -2000) & (wy <= 15000)
    np.testing.assert_array_equal(
        build_radio_roi_mask(fits.Header(meta), (5, 5), roi), expected
    )
    values = np.ma.array(np.ones((5, 5)), mask=False)
    index = tuple(np.argwhere(expected)[0])
    values[index] = 999
    values.mask[index] = True
    measured = measure_radio_roi(values, fits.Header(meta), roi)
    assert measured["raw_sum"] == expected.sum() - 1
    assert measured["valid_pixel_count"] == expected.sum() - 1
    assert values.data[index] == 999


def test_normalization_keeps_missing_and_copies():
    data = np.ma.array([3, 999, 3, np.inf], mask=[False, True, False, False])
    np.testing.assert_allclose(
        normalize_image(data), [0, np.nan, 0, np.nan], equal_nan=True
    )
    assert np.isnan(normalize_image(np.ma.masked_all((2, 2)))).all()
    image = np.ma.array(np.ones((3, 3)), mask=False)
    image.mask[1, 1] = True
    cropped = crop_roi(image, x_range=(0, 2), y_range=(0, 2))
    assert np.isnan(cropped[1, 1])
    assert not np.shares_memory(cropped, image.data)


@pytest.mark.parametrize("method", ["linear", "cubic", "nearest"])
def test_reprojection_support_and_explicit_nearest_fallback(method):
    points = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
    values = np.array([1, 2, 3, 4])
    output, footprint, diagnostics = interpolate_scattered_to_grid(
        points, values, (5, 5), method=method, return_diagnostics=True
    )
    assert np.isnan(output[0, 0]) and footprint[0, 0] == 0
    assert np.isfinite(output[2, 2]) and footprint[2, 2] == 1
    assert diagnostics["method_used"] == method
    extrapolated, footprint, diagnostics = interpolate_scattered_to_grid(
        points,
        values,
        (5, 5),
        method=method,
        fallback="nearest",
        return_diagnostics=True,
    )
    assert np.isfinite(extrapolated[0, 0]) and footprint[0, 0] == 0
    assert diagnostics["fallback_reason"]


def test_degenerate_reprojection_is_not_silently_nearest():
    points = np.array([[2, 2]])
    data = interpolate_scattered_to_grid(points, [9], (5, 5))
    assert np.isnan(data).all()
    data, footprint, diagnostic = interpolate_scattered_to_grid(
        points, [9], (5, 5), fallback="nearest", return_diagnostics=True
    )
    np.testing.assert_array_equal(data, 9)
    assert footprint.sum() == footprint[2, 2] == 1
    assert diagnostic["method_used"] == "nearest"
    assert "degenerate_geometry" in diagnostic["fallback_reason"]


def test_radio_reprojection_excludes_masked_extreme_and_preserves_input():
    data = np.ma.array([[1, 2], [3, 999]], mask=[[False, False], [False, True]])
    result = reproject_radio_array(data, (3, 3), lambda x, y: (x, y), method="nearest")
    assert result.mapped_sample_count == 3
    assert result.amplitude == 3
    assert result.footprint[1, 1] == 0
    assert np.isnan(result.data[1, 1])
    assert data.data[1, 1] == 999 and data.mask[1, 1]


def test_source_center_excludes_masked_extreme_before_thresholding():
    data = np.ma.array(np.zeros((5, 5)), mask=False)
    data[1, 3], data[4, 0] = 10, 999
    data.mask[4, 0] = True
    center = compute_source_center(data, fits.Header(header()))
    assert center["peak_x_pix"] == 3 and center["peak_y_pix"] == 1
    assert center["center_x_pix"] == 3 and center["center_y_pix"] == 1


def test_fits_output_is_detached_and_closed_with_scaled_blank(tmp_path, monkeypatch):
    path = tmp_path / "scaled.fits"
    hdu = fits.PrimaryHDU(np.array([[1, -999], [3, 4]], dtype=np.int16))
    hdu.header.update(BSCALE=2, BZERO=10, BLANK=-999)
    hdu.writeto(path)
    opened = []
    original = fits.open

    def record(*args, **kwargs):
        hdul = original(*args, **kwargs)
        opened.append(hdul)
        return hdul

    monkeypatch.setattr(fits, "open", record)
    data, meta = read_fits_data_header(path)
    assert opened[0]._file.closed
    assert not np.shares_memory(data, opened[0][0].data)
    np.testing.assert_allclose(data, [[12, np.nan], [16, 18]], equal_nan=True)
    meta["CUSTOM"] = "independent"
    assert "CUSTOM" not in opened[0][0].header
    empty = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(empty)
    assert read_fits_data_header(empty)[0] is None
