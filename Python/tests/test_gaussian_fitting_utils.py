"""Tests for shared Gaussian fitting helpers using synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest

from solar_toolkit.modeling.gaussian import (
    elliptical_gaussian_2d,
    fit_elliptical_gaussian,
    initial_guess_from_peak,
    true_indices,
    unravel_2d_index,
)


def test_elliptical_gaussian_peak_value():
    x = np.array([[0.0]])
    y = np.array([[0.0]])

    value = elliptical_gaussian_2d((x, y), 5.0, 0.0, 0.0, 2.0, 3.0, 0.0)

    assert value.shape == (1, 1)
    assert value[0, 0] == 5.0


def test_index_helpers_match_legacy_behavior():
    assert unravel_2d_index(5, (2, 3)) == (1, 2)
    assert true_indices(np.array([False, True, True, False])).tolist() == [1, 2]


def test_initial_guess_from_peak_uses_maximum_location():
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([10.0, 20.0])
    data = np.array([[0.0, 1.0, 0.0], [0.0, 4.0, 0.0]])

    guess = initial_guess_from_peak(data, x, y)

    assert guess[0] == 4.0
    assert guess[1] == 0.0
    assert guess[2] == 20.0
    assert guess[5] == 0.0


def test_fit_elliptical_gaussian_recovers_synthetic_source():
    x = np.linspace(-5.0, 5.0, 31)
    y = np.linspace(-4.0, 4.0, 25)
    X, Y = np.meshgrid(x, y)
    expected = (8.0, 0.5, -0.75, 1.2, 0.9, 0.15)
    data = elliptical_gaussian_2d((X, Y), *expected)

    popt, pcov = fit_elliptical_gaussian(data, x, y, initial_guess=expected)

    np.testing.assert_allclose(popt, expected, rtol=1e-5, atol=1e-5)
    assert pcov.shape == (6, 6)


@pytest.mark.parametrize(
    "reverse_x,reverse_y", [(False, False), (True, False), (False, True), (True, True)]
)
def test_default_gaussian_fit_supports_descending_axes(reverse_x, reverse_y):
    x = np.linspace(-5.0, 5.0, 31)[:: -1 if reverse_x else 1]
    y = np.linspace(-4.0, 4.0, 25)[:: -1 if reverse_y else 1]
    X, Y = np.meshgrid(x, y)
    expected = (8.0, 0.5, -0.75, 1.2, 0.9, 0.15)
    data = elliptical_gaussian_2d((X, Y), *expected)
    original = data.copy()
    guess = initial_guess_from_peak(data, x, y)
    assert guess[3] > 0 and guess[4] > 0
    fitted, _ = fit_elliptical_gaussian(data, x, y)
    np.testing.assert_allclose(fitted, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(data, original)


def test_descending_axes_keep_fallback_widths_positive():
    data = np.zeros((3, 3))
    data[1, 1] = 10
    guess = initial_guess_from_peak(
        data, np.array([2.0, 1.0, 0.0]), np.array([3.0, 2.0, 1.0])
    )
    assert guess[3] == pytest.approx(0.2)
    assert guess[4] == pytest.approx(0.2)
