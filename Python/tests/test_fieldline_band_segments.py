"""Conditional frequency bands clip original 3D edges without inventing paths."""

import astropy.units as u
import numpy as np
import pytest

from solar_toolkit.radio.fieldline_association import fieldline_band_segments
from solar_toolkit.radio.newkirk import PLASMA_FREQ_COEFF_KHZ, newkirk_density_cm3


def plasma_frequency(radius, multiplier=1, harmonic=1):
    return (
        harmonic
        * PLASMA_FREQ_COEFF_KHZ
        / 1000
        * np.sqrt(newkirk_density_cm3(radius, multiplier))
    )


def test_plasma_band_intersects_exact_radial_shells_and_keeps_original_edge_fractions():
    points = np.array([[1.2, 0, 0], [2.3, 0, 0]])
    original = points.copy()
    result = fieldline_band_segments(
        points,
        [plasma_frequency(2), plasma_frequency(1.5)],
        mechanism="plasma",
        harmonic=1,
        rss=2.5,
        newkirk_multiplier=1,
    )
    assert result["metadata"]["status"] == "ok"
    (segment,) = result["segments"]
    np.testing.assert_allclose(segment["xyz_carrington_rsun"], [[1.5, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(segment["source_edge_fractions"], [[3 / 11, 8 / 11]])
    np.testing.assert_array_equal(segment["source_edge_indices"], [0])
    np.testing.assert_allclose(
        segment["frequency_mhz"], [plasma_frequency(1.5), plasma_frequency(2)]
    )
    np.testing.assert_array_equal(points, original)
    assert result["metadata"]["independent_geometry_valid"] is False


def test_one_chord_crosses_both_branches_without_connecting_through_the_sun():
    result = fieldline_band_segments(
        [[-3, 0, 0], [3, 0, 0]],
        [plasma_frequency(2), plasma_frequency(1.5)],
        mechanism="plasma",
        harmonic=1,
        rss=2.5,
        newkirk_multiplier=1,
    )
    assert len(result["segments"]) == 2
    np.testing.assert_allclose(
        result["segments"][0]["xyz_carrington_rsun"], [[-2, 0, 0], [-1.5, 0, 0]]
    )
    np.testing.assert_allclose(
        result["segments"][1]["xyz_carrington_rsun"], [[1.5, 0, 0], [2, 0, 0]]
    )


def test_cyclotron_interpolation_uses_local_field_not_radius_and_preserves_gaps():
    points = np.array([[1.1, 0, 0], [1.6, 0, 0], [2.1, 0, 0]])
    strength = np.array([0, 20.0, 0])
    before = strength.copy()
    result = fieldline_band_segments(
        points,
        [14, 42] * u.MHz,
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=strength,
    )
    assert len(result["segments"]) == 2
    for index, segment in enumerate(result["segments"]):
        np.testing.assert_array_equal(segment["source_edge_indices"], [index])
        np.testing.assert_allclose(segment["source_edge_fractions"], [[0.25, 0.75]])
    np.testing.assert_allclose(result["segments"][0]["frequency_mhz"], [14, 42])
    np.testing.assert_allclose(result["segments"][1]["frequency_mhz"], [42, 14])
    np.testing.assert_array_equal(strength, before)


def test_contiguous_edges_merge_with_one_source_record_per_returned_edge():
    result = fieldline_band_segments(
        [[1.1, 0, 0], [1.5, 0.2, 0], [2, 0.2, 0]],
        [10, 20],
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=np.full(3, 5) * u.G,
    )
    (segment,) = result["segments"]
    assert segment["xyz_carrington_rsun"].shape == (3, 3)
    np.testing.assert_array_equal(segment["source_edge_indices"], [0, 1])
    np.testing.assert_array_equal(segment["source_edge_fractions"], [[0, 1], [0, 1]])


@pytest.mark.parametrize("missing", [np.nan, -1])
def test_missing_or_negative_field_samples_never_bridge_adjacent_valid_points(missing):
    result = fieldline_band_segments(
        [[1.1, 0, 0], [1.5, 0, 0], [2, 0, 0]],
        [10, 20],
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=[5, missing, 5],
    )
    assert result["segments"] == []


def test_domain_clipping_and_quantity_conversion_are_explicit():
    result = fieldline_band_segments(
        np.array([[0.5, 0, 0], [3, 0, 0]]) * u.R_sun,
        [50, 60] * u.MHz,
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5 * u.R_sun,
        field_strength_gauss=[0.002, 0.002] * u.T,
    )
    np.testing.assert_allclose(
        result["segments"][0]["xyz_carrington_rsun"], [[1, 0, 0], [2.5, 0, 0]]
    )


def test_isolated_point_touches_are_omitted_but_an_in_band_plateau_is_a_segment():
    touch = fieldline_band_segments(
        [[1.1, 0, 0], [1.5, 0, 0], [2, 0, 0]],
        [14, 20],
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=[0, 5, 0],
    )
    assert not touch["segments"] and touch["metadata"]["omitted_point_touches"] > 0
    plateau = fieldline_band_segments(
        [[1.1, 0, 0], [2, 0, 0]],
        [14, 20],
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=[5, 5],
    )
    assert len(plateau["segments"]) == 1


def test_nan_geometry_and_unavailable_field_are_explicit_not_radial_fallbacks():
    result = fieldline_band_segments(
        [[1.1, 0, 0], [np.nan, 0, 0], [2, 0, 0]],
        [10, 100],
        mechanism="plasma",
        harmonic=1,
        rss=2.5,
        newkirk_multiplier=1,
    )
    assert not result["segments"]
    missing = fieldline_band_segments(
        [[1.1, 0, 0], [2, 0, 0]], [10, 100], mechanism="cyclotron", harmonic=1, rss=2.5
    )
    assert missing["metadata"]["status"] == "field_strength_unavailable"
    assert not missing["segments"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"harmonic": True},
        {"harmonic": 1.5},
        {"harmonic": 0},
        {"rss": 1},
        {"rss": np.nan},
        {"mechanism": "ecm"},
        {"frequency_bounds_mhz": [10, 10]},
        {"frequency_bounds_mhz": [20, 10]},
        {"newkirk_multiplier": None},
    ],
)
def test_invalid_configuration_is_rejected_even_for_empty_geometry(kwargs):
    config = dict(
        mechanism="plasma",
        harmonic=1,
        rss=2.5,
        newkirk_multiplier=1,
        frequency_bounds_mhz=[10, 20],
    )
    config.update(kwargs)
    bounds = config.pop("frequency_bounds_mhz")
    with pytest.raises(ValueError):
        fieldline_band_segments(np.empty((0, 3)), bounds, **config)


def test_field_shape_and_incompatible_units_raise():
    with pytest.raises(ValueError, match="shape"):
        fieldline_band_segments(
            [[1.1, 0, 0], [2, 0, 0]],
            [10, 20],
            mechanism="cyclotron",
            harmonic=1,
            rss=2.5,
            field_strength_gauss=[5],
        )
    with pytest.raises(u.UnitConversionError):
        fieldline_band_segments(
            np.ones((2, 3)) * u.s,
            [10, 20],
            mechanism="plasma",
            harmonic=1,
            rss=2.5,
            newkirk_multiplier=1,
        )
