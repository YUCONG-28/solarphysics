"""Time-axis precision and validity for finite-difference diagnostics."""

import numpy as np
import pandas as pd
import pytest

from solar_toolkit.xray_dem.processing import calculate_derivative


@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
def test_derivative_keeps_submicrosecond_cadence_at_modern_epoch(method):
    times = np.array(
        [
            "2020-01-01T00:00:00.000000000",
            "2020-01-01T00:00:00.000000100",
            "2020-01-01T00:00:00.000000200",
        ],
        dtype="datetime64[ns]",
    )
    values = np.array([0.0, 1.0, 2.0])
    result = calculate_derivative(times, values, method=method)
    np.testing.assert_allclose(result, 1e7, rtol=1e-12)
    np.testing.assert_array_equal(values, [0, 1, 2])


@pytest.mark.parametrize(
    "method,expected",
    [
        ("gradient", [1.0, 2.0, 4.0]),
        ("forward", [1.0, 4.0]),
        ("median-gradient", [2 / 3, 3.0, 16 / 3]),
    ],
)
def test_irregular_cadence_preserves_explicit_numerical_modes(method, expected):
    times = pd.to_datetime(
        ["2000-01-01T00:00:00Z", "2000-01-01T00:00:01Z", "2000-01-01T00:00:03Z"]
    )
    np.testing.assert_allclose(
        calculate_derivative(times, [0.0, 1.0, 9.0], method=method), expected
    )


@pytest.mark.parametrize(
    "times",
    [
        ["2000-01-01T00:00:00", "2000-01-01T00:00:00"],
        ["2000-01-01T00:00:01", "2000-01-01T00:00:00"],
        ["2000-01-01T00:00:00", "NaT"],
    ],
)
@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
def test_invalid_time_axes_are_rejected_even_with_explicit_spacing(times, method):
    with pytest.raises(ValueError, match="time"):
        calculate_derivative(times, [0.0, 1.0], method=method, spacing_seconds=1)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, True])
@pytest.mark.parametrize("method", ["median-gradient", "gradient", "forward"])
def test_invalid_spacing_is_rejected_for_short_and_long_series(spacing, method):
    for values in ([], [1.0], [1.0, 2.0]):
        with pytest.raises(ValueError, match="spacing_seconds"):
            calculate_derivative(values, spacing_seconds=spacing, method=method)


@pytest.mark.parametrize("values", [[], [1.0]])
def test_unknown_method_is_rejected_before_short_series_return(values):
    with pytest.raises(ValueError, match="method"):
        calculate_derivative(
            pd.date_range("2000-01-01", periods=len(values), freq="s"),
            values,
            method="unknown",
        )


def test_derivative_requires_one_dimensional_equal_length_arrays():
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_derivative([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="same length"):
        calculate_derivative(["2000-01-01T00:00:00"], [1.0, 2.0])
