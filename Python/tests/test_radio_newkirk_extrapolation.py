from __future__ import annotations

import math

import pandas as pd

from solar_toolkit.radio.newkirk import (
    attach_newkirk_height_to_gaussian,
    extrapolate_drift_line_with_newkirk,
    newkirk_density_cm3,
    newkirk_height_from_frequency_mhz,
    newkirk_radius_from_density,
    newkirk_radius_from_frequency_mhz,
    newkirk_speed_from_drift_rate,
    plasma_density_from_frequency_mhz,
)


def test_newkirk_density_and_inverse_radius_round_trip():
    radius = 1.45
    density = newkirk_density_cm3(radius, multiplier=2.0)

    recovered = newkirk_radius_from_density(density, multiplier=2.0)

    assert recovered == pytest_approx(radius)


def test_newkirk_definition_is_pinned_to_published_constants_and_height_convention():
    radius = 1.5
    multiplier = 2.0
    expected_density = 4.2e4 * multiplier * 10.0 ** (4.32 / radius)

    assert newkirk_density_cm3(radius, multiplier=multiplier) == pytest_approx(
        expected_density
    )
    recovered = newkirk_radius_from_density(expected_density, multiplier=multiplier)
    assert recovered == pytest_approx(radius)
    frequency_mhz = 8.98e-3 * math.sqrt(expected_density)
    assert newkirk_height_from_frequency_mhz(
        frequency_mhz, multiplier=multiplier, harmonic=1
    ) == pytest_approx(radius - 1.0)


def test_frequency_height_uses_harmonic_density_relation():
    fundamental = newkirk_radius_from_frequency_mhz(150.0, multiplier=1.0, harmonic=1)
    harmonic = newkirk_radius_from_frequency_mhz(150.0, multiplier=1.0, harmonic=2)

    assert plasma_density_from_frequency_mhz(
        150.0, harmonic=2
    ) < plasma_density_from_frequency_mhz(150.0, harmonic=1)
    assert harmonic > fundamental
    assert newkirk_height_from_frequency_mhz(
        150.0, multiplier=1.0, harmonic=1
    ) == pytest_approx(fundamental - 1.0)


def test_newkirk_speed_from_negative_drift_is_positive_outward():
    speed = newkirk_speed_from_drift_rate(150.0, -10.0, 1.5)

    assert speed["dr_dt_rsun_s"] > 0
    assert speed["speed_km_s"] > 0


def test_attach_newkirk_height_filters_invalid_gaussian_rows():
    df = pd.DataFrame(
        [
            {
                "time": "2025-01-24T04:48:50",
                "freq": 150.0,
                "center_x_arcsec": 100.0,
                "center_y_arcsec": 200.0,
                "quality_flag": "ok",
                "overlay_valid": True,
                "trajectory_valid": True,
            },
            {
                "time": "2025-01-24T04:48:51",
                "freq": 150.0,
                "center_x_arcsec": 100.0,
                "center_y_arcsec": 200.0,
                "quality_flag": "low_snr",
                "overlay_valid": True,
                "trajectory_valid": True,
            },
        ]
    )

    out = attach_newkirk_height_to_gaussian(
        df,
        multiplier=1.0,
        harmonic=1,
    )

    assert len(out) == 1
    assert out.iloc[0]["newkirk_r_rsun"] > 1.0
    assert math.isfinite(out.iloc[0]["newkirk_height_rsun"])
    assert "newkirk_z_arcsec" not in out.columns
    assert "newkirk_geometry_valid" not in out.columns


def test_extrapolate_drift_line_with_newkirk_maps_midpoint_speed():
    row = {
        "label": "drift_001",
        "f_start_mhz": 180.0,
        "f_end_mhz": 120.0,
        "drift_rate_mhz_s": -10.0,
    }

    out = extrapolate_drift_line_with_newkirk(row, multiplier=1.0, harmonic=1)

    assert out["label"] == "drift_001"
    assert out["mid_frequency_mhz"] == 150.0
    assert out["speed_km_s"] > 0


def test_newkirk_harmonic_density_degeneracy_and_speed_aliases():
    row = {
        "label": "drift_001",
        "f_start_mhz": 180.0,
        "f_end_mhz": 120.0,
        "drift_rate_mhz_s": -10.0,
    }

    harmonic = extrapolate_drift_line_with_newkirk(row, multiplier=1.0, harmonic=2)
    dense_fundamental = extrapolate_drift_line_with_newkirk(
        row, multiplier=4.0, harmonic=1
    )

    assert harmonic["density_multiplier"] == pytest_approx(1.0)
    assert harmonic["emission_harmonic"] == pytest_approx(2.0)
    assert harmonic["effective_density_factor"] == pytest_approx(4.0)
    assert dense_fundamental["effective_density_factor"] == pytest_approx(4.0)
    assert harmonic["newkirk_height_rsun"] == pytest_approx(
        dense_fundamental["newkirk_height_rsun"]
    )
    assert harmonic["newkirk_speed_km_s"] == pytest_approx(
        dense_fundamental["newkirk_speed_km_s"]
    )
    assert harmonic["newkirk_speed_c"] == pytest_approx(
        harmonic["newkirk_speed_km_s"] / 299792.458
    )
    assert harmonic["newkirk_assumption_label"] == "1x Newkirk, H=2, N*s^2=4"


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
