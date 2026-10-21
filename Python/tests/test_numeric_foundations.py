"""Synthetic contracts for units, masks, and numerical precision boundaries."""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.nddata import NDData
from astropy.time import Time
from astropy.utils.masked import Masked

from solar_toolkit._utils.numeric import as_value, numeric_data
from solar_toolkit.radio.cso_processing import (
    block_mean_rebin,
    calc_polarization_ratio,
    finite_color_limits,
    safe_log10,
)
from solar_toolkit.radio.newkirk import (
    newkirk_density_cm3,
    newkirk_radius_from_density,
    newkirk_speed_from_drift_rate,
    plasma_density_from_frequency_mhz,
)
from solar_toolkit.radio.quality_autoencoder import (
    QualityAutoencoderSample,
    QualityAutoencoderValidationError,
    prepare_confirmed_good_frames,
)
from solar_toolkit.radio.quality_science import (
    compute_context_similarity,
    compute_scientific_quality_features,
)
from solar_toolkit.radio.raw_quality import compute_raw_quality_metrics
from solar_toolkit.timeseries import derivative_series, smooth_series
from solar_toolkit.xray_dem.processing import calculate_derivative, smooth_flux_data


@pytest.mark.parametrize(
    "source",
    [
        np.ma.array([1, 999, 3], mask=[False, True, False]),
        Masked(np.array([1, 999, 3]), mask=[False, True, False]),
        Masked([1, 999, 3] * u.Jy, mask=[False, True, False]),
        NDData([1, 999, 3], mask=[False, True, False], unit=u.Jy),
    ],
)
def test_numeric_data_merges_masks_without_mutating_input(source):
    result = numeric_data(source, mask=[False, False, True])
    assert result.dtype == np.float64
    np.testing.assert_equal(result, [1, np.nan, np.nan])
    result[0] = 20
    np.testing.assert_equal(numeric_data(source), [1, np.nan, 3])
    np.testing.assert_array_equal(source.mask, [False, True, False])


@pytest.mark.parametrize("source", [5, np.array(5), 5 * u.s])
def test_numeric_data_keeps_scalar_shape(source):
    result = numeric_data(source)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result == 5


def test_numeric_data_represents_nonfinite_values_as_nan_and_broadcasts_mask():
    source = np.array([[1.0, np.inf], [np.nan, -np.inf]])
    np.testing.assert_equal(numeric_data(source), [[1, np.nan], [np.nan, np.nan]])
    np.testing.assert_equal(
        numeric_data(source, mask=[[True], [False]]),
        [[np.nan, np.nan], [np.nan, np.nan]],
    )
    assert np.isnan(numeric_data(np.ma.masked_all(3))).all()
    with pytest.raises(ValueError, match="broadcastable"):
        numeric_data(source, mask=[True, False, True])


def test_numeric_data_does_not_treat_pandas_mask_method_as_missingness():
    np.testing.assert_equal(numeric_data(pd.Series([1, 2, 3])), [1, 2, 3])


@pytest.mark.parametrize(
    "source",
    [
        Masked([1000.0, 999.0] * u.ms, mask=[False, True]),
        NDData([1000.0, 999.0], unit=u.ms, mask=[False, True]),
    ],
)
def test_as_value_converts_units_and_preserves_mask(source):
    np.testing.assert_equal(as_value(source, u.s, "cadence"), [1, np.nan])


def test_as_value_bare_numbers_are_canonical_and_wrong_dimensions_are_explicit():
    assert as_value(2, u.MHz, "frequency") == 2
    with pytest.raises(u.UnitConversionError, match="frequency"):
        as_value(2 * u.s, u.MHz, "frequency")


def test_newkirk_equivalent_frequency_units_and_scalar_array_types():
    expected = plasma_density_from_frequency_mhz(200)
    assert isinstance(expected, float)
    assert plasma_density_from_frequency_mhz(0.2 * u.GHz) == pytest.approx(expected)
    assert plasma_density_from_frequency_mhz(200e6 * u.Hz) == pytest.approx(expected)
    frequencies = Masked([0.2, 900.0] * u.GHz, mask=[False, True])
    actual = plasma_density_from_frequency_mhz(frequencies)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, [expected, np.nan], equal_nan=True)
    with pytest.raises(u.UnitConversionError, match="freq_mhz"):
        plasma_density_from_frequency_mhz(200 * u.s)


def test_newkirk_radius_density_and_drift_units_use_canonical_output_units():
    density = newkirk_density_cm3(2)
    assert newkirk_density_cm3((2 * u.R_sun).to(u.km)) == pytest.approx(density)
    assert newkirk_radius_from_density(
        (density / u.cm**3).to(u.m**-3)
    ) == pytest.approx(2)
    expected = newkirk_speed_from_drift_rate(200, -2, 2)
    actual = newkirk_speed_from_drift_rate(
        0.2 * u.GHz, -2000 * u.kHz / u.s, (2 * u.R_sun).to(u.km)
    )
    assert actual == expected
    with pytest.raises(u.UnitConversionError, match="drift_rate_mhz_s"):
        newkirk_speed_from_drift_rate(200, 2 * u.km, 2)


@pytest.mark.parametrize(
    "function", [smooth_series, derivative_series, smooth_flux_data]
)
def test_series_masks_are_equivalent_to_nan_without_mutation(function):
    source = np.ma.array([1.0, 1e20, 3.0, 4.0, 5.0], mask=[0, 1, 0, 0, 0])
    expected = function([1, np.nan, 3, 4, 5])
    np.testing.assert_allclose(function(source), expected, equal_nan=True)
    assert source.data[1] == 1e20
    assert source.mask[1]


def test_savgol_smoothing_preserves_masked_missingness():
    source = np.ma.array([1.0, 999.0, 3.0, 4.0, 5.0], mask=[0, 1, 0, 0, 0])
    kwargs = {"method": "savgol", "window_length": 3, "polyorder": 1}
    np.testing.assert_allclose(
        smooth_flux_data(source, **kwargs),
        smooth_flux_data(source.filled(np.nan), **kwargs),
        equal_nan=True,
    )


@pytest.mark.parametrize("window", [3, 4, 5])
def test_savgol_missing_neighborhoods_keep_finite_edges_and_original_filter(window):
    source = np.arange(20, dtype=float) ** 2
    expected = smooth_flux_data(
        source, method="savgol", window_length=window, polyorder=2
    )
    source[10] = np.nan
    result = smooth_flux_data(
        source, method="savgol", window_length=window, polyorder=2
    )
    np.testing.assert_allclose(
        result[np.isfinite(result)], expected[np.isfinite(result)]
    )
    assert np.isfinite(result[:3]).all()
    assert np.isfinite(result[-3:]).all()
    assert np.isnan(result[10])


@pytest.mark.parametrize("function", [derivative_series, calculate_derivative])
def test_derivative_spacing_converts_quantity_and_rejects_wrong_dimension(function):
    expected = function([0, 1, 2], spacing_seconds=1)
    np.testing.assert_equal(function([0, 1, 2], spacing_seconds=1000 * u.ms), expected)
    with pytest.raises(u.UnitConversionError, match="spacing_seconds"):
        function([], spacing_seconds=1 * u.kg)
    with pytest.raises(ValueError, match="finite.*positive"):
        function([], spacing_seconds=Masked(1 * u.s, mask=True))


@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_xray_missing_singleton_does_not_become_zero(method, value):
    result = calculate_derivative(["2000-01-01T00:00:00"], [value], method=method)
    if method == "forward":
        assert result.size == 0
    else:
        np.testing.assert_equal(result, [np.nan])


@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
def test_xray_masked_singleton_and_longer_flux_are_nan_equivalent(method):
    missing = np.ma.array([999], mask=True)
    result = calculate_derivative(["2000-01-01T00:00:00"], missing, method=method)
    np.testing.assert_equal(result, [] if method == "forward" else [np.nan])
    times = [f"2000-01-01T00:00:0{i}" for i in range(5)]
    source = np.ma.array([1.0, 1e20, 3.0, 4.0, 5.0], mask=[0, 1, 0, 0, 0])
    np.testing.assert_allclose(
        calculate_derivative(times, source, method=method),
        calculate_derivative(times, source.filled(np.nan), method=method),
        equal_nan=True,
    )


@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
def test_xray_derivative_uses_si_seconds_across_utc_leap_second(method):
    times = ["2016-12-31T23:59:59", "2017-01-01T00:00:00"]
    np.testing.assert_allclose(calculate_derivative(times, [0, 2], method=method), 1)
    np.testing.assert_allclose(
        calculate_derivative(Time(times, scale="utc"), [0, 2], method=method), 1
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_centered_derivative_does_not_revive_missing_center(invalid):
    values = np.array([1.0, invalid, 3.0])
    assert np.isnan(derivative_series(values)).all()
    times = ["2000-01-01T00:00:00", "2000-01-01T00:00:01", "2000-01-01T00:00:03"]
    for method in ("gradient", "median-gradient"):
        assert np.isnan(calculate_derivative(times, values, method=method)).all()
        assert np.isnan(
            calculate_derivative(times, values, method=method, spacing_seconds=1)
        ).all()
    assert np.isnan(calculate_derivative(times, values, method="forward")).all()
    assert values[0] == 1 and values[2] == 3


@pytest.mark.parametrize(
    ("dtype", "right", "left", "expected"),
    [
        (np.uint8, 10, 20, -1 / 3),
        (np.uint8, 200, 100, 1 / 3),
        (np.uint16, 60000, 40000, 0.2),
        (np.int16, 30000, 20000, 0.2),
    ],
)
def test_cso_ratio_promotes_before_integer_addition_and_subtraction(
    dtype, right, left, expected
):
    actual = calc_polarization_ratio(
        np.array([right], dtype=dtype), np.array([left], dtype=dtype)
    )
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, [expected], rtol=1e-7)


def test_cso_ratio_invalid_samples_and_masks_stay_missing():
    right = np.ma.array([3, 999, 1, np.nan], mask=[0, 1, 0, 0])
    result = calc_polarization_ratio(right, [1, 1, -1, 1])
    np.testing.assert_equal(result, [0.5, np.nan, np.nan, np.nan])


def test_cso_ratio_converts_both_polarizations_to_a_common_unit():
    np.testing.assert_equal(calc_polarization_ratio([1] * u.Jy, [1000] * u.mJy), [0])
    np.testing.assert_equal(calc_polarization_ratio([1], [1] * u.Jy), [0])
    np.testing.assert_equal(calc_polarization_ratio([1] * u.Jy, [1]), [0])
    masked = Masked([1000, 999] * u.mJy, mask=[False, True])
    np.testing.assert_equal(calc_polarization_ratio([1, 1] * u.Jy, masked), [0, np.nan])
    with pytest.raises(u.UnitConversionError, match="left"):
        calc_polarization_ratio([1] * u.Jy, [1] * u.s)


def test_cso_rebin_preserves_float64_mean_and_explicit_precision_choice():
    source = np.array([[1e8, 1e8 + 1], [1e8 + 2, 1e8 + 3]], dtype=np.float64)
    result = block_mean_rebin(source, frequency_bin=2, time_bin=2)
    assert result.dtype == np.float64
    assert result[0, 0] == 100000001.5
    reduced = block_mean_rebin(source, frequency_bin=2, time_bin=2, dtype=np.float32)
    assert reduced.dtype == np.float32
    assert reduced[0, 0] == 100000000
    with pytest.raises(TypeError, match="floating-point"):
        block_mean_rebin(source, dtype=np.int64)


def test_cso_helpers_treat_masked_samples_like_nan():
    source = np.ma.array([[1, 1e20], [10, 100]], mask=[[0, 1], [0, 0]])
    expected = source.filled(np.nan)
    np.testing.assert_equal(block_mean_rebin(source), expected)
    assert np.isnan(block_mean_rebin(source, frequency_bin=2, time_bin=2)).all()
    np.testing.assert_equal(safe_log10(source), safe_log10(expected))
    assert finite_color_limits(source) == finite_color_limits(expected)


def test_raw_quality_excludes_masked_extreme_samples():
    data = np.arange(1, 101, dtype=float).reshape(10, 10)
    data[5, 5] = 1e12
    source = np.ma.array(data, mask=False)
    source.mask[5, 5] = True
    actual = compute_raw_quality_metrics(source)
    expected = compute_raw_quality_metrics(source.filled(np.nan))
    assert asdict(actual) == asdict(expected)
    assert actual.finite_pixel_count == 99
    assert actual.max_log10 == 2
    assert data[5, 5] == 1e12


def test_scientific_features_masks_and_nan_have_identical_statistics_and_thumbnail():
    source = np.ma.array(np.arange(100, dtype=float).reshape(10, 10), mask=False)
    source.data[3, 3] = 1e20
    source.mask[3, 3] = True
    actual, thumbnail = compute_scientific_quality_features(source)
    expected, expected_thumbnail = compute_scientific_quality_features(
        source.filled(np.nan)
    )
    for key, value in asdict(actual).items():
        if isinstance(value, float):
            # Linear algebra backends may differ by a few ulps even for
            # identical finite samples; this still rejects mask contamination.
            np.testing.assert_allclose(
                value, getattr(expected, key), rtol=1e-14, atol=1e-14
            )
        else:
            assert value == getattr(expected, key)
    assert actual.finite_pixel_count == 99
    np.testing.assert_equal(thumbnail, expected_thumbnail)
    invalid, thumbnail = compute_scientific_quality_features(np.ma.masked_all((5, 5)))
    assert invalid.invalid_reason == "no_finite_pixels"
    assert thumbnail is None


def test_context_similarity_uses_only_paired_valid_samples():
    source = np.ma.array([1, 1e20, 3, 4], mask=[0, 1, 0, 0])
    masked = compute_context_similarity(source, [[1, -1e20, 3, 4]])
    missing = compute_context_similarity(source.filled(np.nan), [[1, -1e20, 3, 4]])
    assert masked == missing
    assert masked == (1, 0)
    assert all(
        np.isnan(value)
        for value in compute_context_similarity(np.ma.masked_all(4), [[1, 2, 3, 4]])
    )


def test_autoencoder_prepare_preserves_masks_without_loading_torch():
    source = np.ma.array(np.arange(16, dtype=float).reshape(4, 4), mask=False)
    source.data[1, 1] = 1e20
    source.mask[1, 1] = True
    selection = prepare_confirmed_good_frames(
        [
            {
                "sample_id": "synthetic",
                "image": source,
                "quality_label": "good",
                "label_source": "human",
            }
        ]
    )
    np.testing.assert_equal(selection.samples[0].image, source.filled(np.nan))
    assert source.data[1, 1] == 1e20
    assert source.mask[1, 1]


@pytest.mark.parametrize("as_mapping", [False, True])
def test_autoencoder_prepare_rejects_all_masked_images_without_torch(as_mapping):
    sample = QualityAutoencoderSample(
        sample_id="missing",
        image=np.ma.masked_all((4, 4)),
        quality_label="good",
        label_source="human",
    )
    raw = (
        {
            "sample_id": sample.sample_id,
            "image": sample.image,
            "quality_label": sample.quality_label,
            "label_source": sample.label_source,
        }
        if as_mapping
        else sample
    )
    with pytest.raises(QualityAutoencoderValidationError, match="no finite"):
        prepare_confirmed_good_frames([raw])
