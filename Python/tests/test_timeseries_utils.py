from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_normalize_time_column_and_crop_range():
    from solar_toolkit.timeseries import crop_time_range, normalize_time_column

    frame = pd.DataFrame(
        {
            "time_tag": ["2000-01-01T04:48:00Z", "2000-01-01T04:49:00Z"],
            "flux": [1.0, 3.0],
        }
    )
    normalized = normalize_time_column(frame, source_column="time_tag")
    cropped = crop_time_range(
        normalized, "2000-01-01T04:48:30Z", "2000-01-01T04:49:30Z"
    )

    assert "obs_time" in normalized.columns
    assert cropped["flux"].tolist() == [3.0]


def test_smooth_and_derivative_series_are_shape_stable():
    from solar_toolkit.timeseries import derivative_series, smooth_series

    values = pd.Series([0.0, 2.0, 4.0, 6.0, 8.0])

    assert np.allclose(
        smooth_series(values, window_length=3), [2 / 3, 2.0, 4.0, 6.0, 14 / 3]
    )
    assert np.allclose(
        derivative_series(values, spacing_seconds=2.0), [1.0, 1.0, 1.0, 1.0, 1.0]
    )


@pytest.mark.parametrize("values", [[], [9.0], [1.0, 2.0], [1.0, 2.0, 3.0]])
def test_smoothing_short_series_keeps_input_length_and_zero_padding(values):
    from solar_toolkit.timeseries import smooth_series

    # Direct padded-neighborhood sums are independent of np.convolve's length rule.
    expected = [
        sum(values[max(0, index - 2) : index + 3]) / 5 for index in range(len(values))
    ]
    actual = smooth_series(values, window_length=5)
    assert actual.shape == (len(values),)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("window", [0, -3, 2.5, True, float("nan"), float("inf")])
def test_smoothing_rejects_invalid_windows(window):
    from solar_toolkit.timeseries import smooth_series

    with pytest.raises(ValueError, match="positive integer"):
        smooth_series([1.0, 2.0], window_length=window)


def test_smoothing_keeps_even_window_adjustment_and_nan_propagation():
    from solar_toolkit.timeseries import smooth_series

    np.testing.assert_allclose(smooth_series([1.0, 2.0], window_length=2), [1.0, 1.0])
    values = [1.0, float("nan"), 3.0, 4.0, 5.0]
    actual = smooth_series(values, window_length=3)
    np.testing.assert_allclose(
        actual, [np.nan, np.nan, np.nan, 4.0, 3.0], equal_nan=True
    )
