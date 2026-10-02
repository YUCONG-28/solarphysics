from __future__ import annotations

import numpy as np
import pandas as pd


def test_load_sxr_data_smooth_and_derivative(tmp_path):
    from solar_toolkit.xray_dem import (
        calculate_derivative,
        load_sxr_data,
        smooth_flux_data,
    )

    csv_path = tmp_path / "goes.csv"
    pd.DataFrame(
        {
            "time": [
                "2000-01-01T04:48:00Z",
                "2000-01-01T04:49:00Z",
                "2000-01-01T04:50:00Z",
            ],
            "xrsa": [1.0, 3.0, 5.0],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_sxr_data(csv_path, "2000-01-01T04:48:30Z", "2000-01-01T04:50:30Z")
    smoothed = smooth_flux_data(pd.Series([1.0, 3.0, 5.0]), window_length=3)
    derivative = calculate_derivative(pd.Series([1.0, 3.0, 5.0]), spacing_seconds=60)

    assert loaded["xrsa"].tolist() == [3.0, 5.0]
    assert np.allclose(smoothed, [4 / 3, 3.0, 8 / 3])
    assert np.allclose(derivative, [1 / 30, 1 / 30, 1 / 30])
