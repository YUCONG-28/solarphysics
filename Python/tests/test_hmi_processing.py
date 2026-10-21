"""Numerical contracts for magnetic-field thresholding and missing pixels."""

import astropy.units as u
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from solar_toolkit.hmi.processing import (
    create_magnetic_contour_levels,
    process_hmi_magnetic_field,
)


def magnetic_map(data, *, unit="G"):
    import sunpy.map

    header = {
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "crpix1": 1,
        "crpix2": 1,
        "crval1": 0,
        "crval2": 0,
        "cdelt1": 1,
        "cdelt2": 1,
        "date-obs": "2000-01-01T00:00:00",
    }
    if unit is not None:
        header["bunit"] = unit
    return sunpy.map.Map(np.asarray(data, dtype=float), header)


def test_threshold_uses_absolute_field_and_equivalent_units_without_mutation():
    source = magnetic_map([[-100, -50, -49], [49, 50, 100]])
    data, metadata = source.data.copy(), source.meta.copy()
    source.data.setflags(write=False)

    result = process_hmi_magnetic_field(source, threshold=5 * u.mT, sigma=0)

    np.testing.assert_array_equal(result.data, [[-100, -50, 0], [0, 50, 100]])
    np.testing.assert_array_equal(source.data, data)
    assert source.meta == metadata
    assert result.unit == u.Gauss
    assert not np.shares_memory(source.data, result.data)


def test_finite_smoothing_matches_gaussian_filter_after_threshold():
    source = magnetic_map([[-100, -30, 80], [20, 50, 120], [-90, 0, 40]])
    expected = source.data.copy()
    expected[np.abs(expected) < 40] = 0

    result = process_hmi_magnetic_field(source, threshold=40 * u.Gauss, sigma=0.8)

    np.testing.assert_allclose(result.data, gaussian_filter(expected, sigma=0.8))


def test_tesla_map_keeps_its_unit_and_scales_the_gauss_threshold():
    source = magnetic_map([[0.001, 0.01], [-0.005, -0.0001]], unit="T")

    result = process_hmi_magnetic_field(source, threshold=50 * u.Gauss, sigma=0)

    np.testing.assert_allclose(result.data, [[0, 0.01], [-0.005, 0]])
    assert result.unit == u.T


def test_absent_unit_assumes_gauss_only_in_output_metadata():
    source = magnetic_map([[10, -20], [30, -40]], unit=None)
    metadata = source.meta.copy()

    result = process_hmi_magnetic_field(source, sigma=0)

    assert source.meta == metadata
    assert "bunit" not in source.meta
    assert result.unit == u.Gauss
    np.testing.assert_array_equal(result.data, source.data)


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_measurements_do_not_contaminate_valid_neighbors(missing):
    data = np.full((9, 9), 75.0)
    data[4, 4] = missing
    data[0, 0] = missing
    source = magnetic_map(data)

    result = process_hmi_magnetic_field(source, sigma=1)

    missing_mask = ~np.isfinite(data)
    np.testing.assert_array_equal(np.isnan(result.data), missing_mask)
    np.testing.assert_allclose(result.data[~missing_mask], 75.0)
    np.testing.assert_array_equal(source.data, data)


def test_all_missing_map_remains_missing():
    result = process_hmi_magnetic_field(magnetic_map(np.full((3, 3), np.nan)))
    assert np.isnan(result.data).all()


def test_explicit_mask_excludes_finite_bad_pixels_and_is_preserved():
    import sunpy.map

    data = np.full((9, 9), 75.0)
    data[4, 4] = 1e6
    mask = np.zeros(data.shape, dtype=bool)
    mask[4, 4] = True
    source = sunpy.map.Map(data, magnetic_map(data).meta, mask=mask)
    original_mask = source.mask.copy()
    result = process_hmi_magnetic_field(source, sigma=1)
    np.testing.assert_allclose(result.data[~mask], 75)
    assert np.isnan(result.data[4, 4])
    np.testing.assert_array_equal(result.mask, mask)
    np.testing.assert_array_equal(source.mask, original_mask)
    np.testing.assert_array_equal(source.data, data)
    assert not np.shares_memory(source.mask, result.mask)


def test_all_masked_hmi_map_remains_missing():
    import sunpy.map

    source = magnetic_map(np.ones((3, 3)))
    source = sunpy.map.Map(source.data, source.meta, mask=np.ones((3, 3), bool))
    result = process_hmi_magnetic_field(source)
    assert np.isnan(result.data).all()
    assert result.mask.all()


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf])
def test_invalid_smoothing_width_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        process_hmi_magnetic_field(magnetic_map(np.ones((2, 2))), sigma=sigma)


@pytest.mark.parametrize(
    "level", [-1 * u.Gauss, np.nan * u.Gauss, np.inf * u.Gauss, [1, 2] * u.Gauss]
)
def test_invalid_field_magnitudes_are_rejected(level):
    with pytest.raises(ValueError, match="magnitude"):
        process_hmi_magnetic_field(magnetic_map(np.ones((2, 2))), threshold=level)
    with pytest.raises(ValueError, match="magnitude"):
        create_magnetic_contour_levels(level)


def test_field_units_are_required_for_maps_thresholds_and_contours():
    with pytest.raises(u.UnitConversionError):
        process_hmi_magnetic_field(magnetic_map(np.ones((2, 2)), unit="DN"))
    with pytest.raises(u.UnitConversionError):
        process_hmi_magnetic_field(
            magnetic_map(np.ones((2, 2))), threshold=1 * u.second
        )
    with pytest.raises(u.UnitConversionError):
        create_magnetic_contour_levels(1 * u.second)
    with pytest.raises(TypeError, match="quantity"):
        create_magnetic_contour_levels(50)


def test_contours_are_symmetric_and_keep_requested_magnetic_unit():
    np.testing.assert_array_equal(create_magnetic_contour_levels().value, [-50, 50])
    levels = create_magnetic_contour_levels(0.005 * u.T)
    assert levels.unit == u.T
    np.testing.assert_allclose(levels.to_value(u.Gauss), [-50, 50])


def test_hmi_workflows_require_explicit_roi_before_creating_output(tmp_path):
    from solar_toolkit.hmi.magnetogram import run_magnetogram_workflow
    from solar_toolkit.hmi.overlay import run_overlay_workflow

    output = tmp_path / "output"
    with pytest.raises(ValueError, match="roi_bounds"):
        run_magnetogram_workflow(tmp_path / "missing", output)
    with pytest.raises(ValueError, match="roi_bounds"):
        run_overlay_workflow(tmp_path / "missing_aia", tmp_path / "missing_hmi", output)
    assert not output.exists()
