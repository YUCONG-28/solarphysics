from __future__ import annotations

import math

import numpy as np
import pandas as pd

from solar_toolkit.radio.height_comparison import (
    build_gaussian_newkirk_height_summary_table,
    build_gaussian_newkirk_height_table,
    compute_gaussian_projected_height,
)


def test_gaussian_projected_height_is_zero_at_one_solar_radius():
    result = compute_gaussian_projected_height(960.0, 0.0, 960.0)

    assert result["gaussian_rho_rsun"] == pytest_approx(1.0)
    assert result["gaussian_height_rsun"] == pytest_approx(0.0)
    assert result["height_valid"] is True


def test_gaussian_projected_height_is_positive_outside_limb():
    result = compute_gaussian_projected_height(1200.0, 0.0, 960.0)

    assert result["gaussian_rho_rsun"] > 1.0
    assert result["gaussian_height_rsun"] > 0.0
    assert result["height_valid"] is True


def test_gaussian_projected_height_inside_disk_is_flagged_projected_only():
    result = compute_gaussian_projected_height(480.0, 0.0, 960.0)

    assert result["gaussian_rho_rsun"] == pytest_approx(0.5)
    assert result["gaussian_height_rsun"] == pytest_approx(-0.5)
    assert result["height_valid"] is False
    assert result["height_invalid_reason"] == "inside_disk_projected_distance_only"
    assert result["gaussian_projected_height_valid"] is False
    assert (
        result["gaussian_projected_height_reason"] == "projected_inside_limb_or_bad_fit"
    )


def test_lower_frequency_gives_larger_newkirk_height():
    df = pd.DataFrame(
        [
            _gaussian_row(freq=220.0),
            _gaussian_row(freq=120.0),
        ]
    )

    out = build_gaussian_newkirk_height_table(
        df,
        {
            "solar_radius_arcsec": 960.0,
            "selected_models": [{"multiplier": 1.0, "harmonic": 1}],
        },
    ).sort_values("frequency_mhz")

    assert out.iloc[0]["frequency_mhz"] == 120.0
    assert out.iloc[0]["newkirk_height_rsun"] > out.iloc[1]["newkirk_height_rsun"]


def test_height_residual_equals_gaussian_minus_newkirk():
    out = build_gaussian_newkirk_height_table(
        pd.DataFrame([_gaussian_row(freq=150.0, x=1200.0, y=0.0)]),
        {
            "solar_radius_arcsec": 960.0,
            "selected_models": [{"multiplier": 1.0, "harmonic": 1}],
        },
    )
    row = out.iloc[0]

    assert row["height_residual_rsun"] == pytest_approx(
        row["gaussian_height_rsun"] - row["newkirk_height_rsun"]
    )
    assert row["height_residual_arcsec"] == pytest_approx(
        row["height_residual_rsun"] * row["solar_radius_arcsec"]
    )


def test_nan_gaussian_centers_do_not_crash():
    out = build_gaussian_newkirk_height_table(
        pd.DataFrame([_gaussian_row(freq=150.0, x=np.nan, y=0.0)]),
        {
            "solar_radius_arcsec": 960.0,
            "selected_models": [{"multiplier": 1.0, "harmonic": 1}],
        },
    )

    assert len(out) == 1
    assert not out.iloc[0]["height_valid"]
    assert out.iloc[0]["height_invalid_reason"] == "nonfinite_projected_coordinate"
    assert not out.iloc[0]["gaussian_projected_height_valid"]
    assert (
        out.iloc[0]["gaussian_projected_height_reason"]
        == "gaussian_fit_or_coordinate_invalid"
    )


def test_height_summary_excludes_invalid_projected_heights_from_statistics():
    raw = build_gaussian_newkirk_height_table(
        pd.DataFrame(
            [
                _gaussian_row(freq=149.0, x=480.0, y=0.0),
                _gaussian_row(freq=149.0, x=1200.0, y=0.0),
            ]
        ),
        {
            "solar_radius_arcsec": 960.0,
            "selected_models": [
                {"multiplier": 1.0, "harmonic": 1},
                {"multiplier": 1.0, "harmonic": 2},
                {"multiplier": 2.0, "harmonic": 1},
                {"multiplier": 2.0, "harmonic": 2},
                {"multiplier": 4.0, "harmonic": 1},
                {"multiplier": 4.0, "harmonic": 2},
            ],
        },
    )

    summary = build_gaussian_newkirk_height_summary_table(
        raw,
        {
            "comparison_frequency_mhz": [149],
            "reference_newkirk_assumption": "2xH2",
        },
    )
    row = summary.iloc[0]

    assert row["gaussian_valid_count"] == 1
    assert row["gaussian_invalid_count"] == 1
    assert row["gaussian_projected_height_median_rsun"] == pytest_approx(0.25)
    assert row["newkirk_height_rsun_1xH2"] == pytest_approx(
        row["newkirk_height_rsun_4xH1"]
    )
    assert "abs_delta_reference_rsun" in summary.columns


def test_drift_selection_match_propagates_label_and_source_type():
    out = build_gaussian_newkirk_height_table(
        pd.DataFrame([_gaussian_row(freq=140.0, x=1200.0, y=0.0)]),
        {
            "solar_radius_arcsec": 960.0,
            "selected_models": [{"multiplier": 1.0, "harmonic": 1}],
            "drift_time_tolerance_s": 0.5,
            "drift_frequency_tolerance_mhz": 2.0,
            "drift_selections": [
                {
                    "label": "drift_001",
                    "source_type": "typeIII",
                    "t_start": "2000-01-01T04:48:38",
                    "t_end": "2000-01-01T04:48:42",
                    "f_start_mhz": 180.0,
                    "f_end_mhz": 100.0,
                }
            ],
        },
    )

    assert out.iloc[0]["drift_label"] == "drift_001"
    assert out.iloc[0]["source_type"] == "typeIII"


def _gaussian_row(freq=150.0, x=960.0, y=0.0):
    return {
        "time": "2000-01-01T04:48:40",
        "freq": freq,
        "center_x_arcsec": x,
        "center_y_arcsec": y,
        "quality_flag": "ok",
        "overlay_valid": True,
        "trajectory_valid": True,
    }


def pytest_approx(value):
    try:
        import pytest
    except ModuleNotFoundError:

        class Approx:
            def __eq__(self, other):
                return math.isclose(other, value, rel_tol=1e-12, abs_tol=1e-12)

        return Approx()

    return pytest.approx(value, rel=1e-12, abs=1e-12)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
