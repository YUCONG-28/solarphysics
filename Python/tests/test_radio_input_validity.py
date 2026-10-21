"""Scientific input boundaries must not manufacture valid observations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.time import Time

from solar_toolkit.radio.drift_rate import calculate_drift_rate_from_line
from solar_toolkit.radio.io import truthy
from solar_toolkit.radio.newkirk import (
    attach_newkirk_height_to_gaussian,
    effective_density_factor,
    extrapolate_drift_line_with_newkirk,
    newkirk_density_cm3,
    newkirk_radius_from_frequency_mhz,
    plasma_density_from_frequency_mhz,
)
from solar_toolkit.radio.physical_diagnostics import build_drift_newkirk_table
from solar_toolkit.radio.reprojection import nearest_time_index
from solar_toolkit.radio.trajectory import (
    filter_time_range,
    normalize_centers_dataframe,
)


@pytest.mark.parametrize(
    "value", [None, np.nan, pd.NA, np.inf, -1, 2, "unknown", "", "false", [], [True]]
)
def test_unknown_quality_values_are_not_valid(value):
    assert truthy(value) is False


@pytest.mark.parametrize("value", [True, np.bool_(True), 1, 1.0, "true", " YES ", "ok"])
def test_explicit_quality_values_remain_valid(value):
    assert truthy(value) is True


def _center(**updates):
    return {
        "obs_time": "2000-01-01T00:00:00Z",
        "freq_mhz": 150.0,
        "center_x_arcsec": 10.0,
        "center_y_arcsec": 20.0,
        **updates,
    }


def test_missing_validity_column_is_legacy_but_missing_value_is_not_valid():
    assert len(normalize_centers_dataframe(pd.DataFrame([_center()]))) == 1
    for flag in (np.nan, pd.NA, "unknown"):
        assert normalize_centers_dataframe(
            pd.DataFrame([_center(overlay_valid=flag)])
        ).empty
        assert normalize_centers_dataframe(
            pd.DataFrame([_center(quality_flag=flag)])
        ).empty


@pytest.mark.parametrize(
    "updates", [{"freq_mhz": -1.0}, {"freq_mhz": np.inf}, {"center_x_arcsec": np.inf}]
)
def test_invalid_physical_centers_are_excluded(updates):
    assert normalize_centers_dataframe(pd.DataFrame([_center(**updates)])).empty


@pytest.mark.parametrize(
    "when",
    [
        "2000-01-01T08:00:00+08:00",
        pd.Timestamp("2000-01-01T08:00:00+08:00"),
        Time("2000-01-01T00:00:00", scale="utc").tai,
    ],
)
def test_equivalent_instants_survive_a_utc_window(when):
    frame = normalize_centers_dataframe(pd.DataFrame([_center(obs_time=when)]))
    selected = filter_time_range(
        frame, start="2000-01-01T00:00:00Z", end="2000-01-01T00:00:00Z"
    )
    assert len(selected) == 1
    assert selected.iloc[0].obs_time == pd.Timestamp("2000-01-01T00:00:00")


@pytest.mark.parametrize("freq", [-150.0, 0.0, np.nan, np.inf])
def test_invalid_observation_frequency_has_no_density_or_height(freq):
    assert np.isnan(plasma_density_from_frequency_mhz(freq))
    assert np.isnan(newkirk_radius_from_frequency_mhz(freq))


@pytest.mark.parametrize("bad", [-1.0, 0.0, np.nan, np.inf])
def test_model_configuration_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError, match="harmonic"):
        plasma_density_from_frequency_mhz(150.0, harmonic=bad)
    with pytest.raises(ValueError, match="multiplier"):
        newkirk_density_cm3(1.5, multiplier=bad)
    with pytest.raises(ValueError):
        effective_density_factor(bad, 1)


def test_invalid_sample_is_retained_with_reason_and_input_is_unchanged():
    data = pd.DataFrame(
        {
            "freq": [150.0, -150.0, np.nan],
            "quality_flag": ["ok"] * 3,
            "overlay_valid": [True] * 3,
            "trajectory_valid": [True] * 3,
        }
    )
    before = data.copy(deep=True)
    out = attach_newkirk_height_to_gaussian(data, multiplier=1, harmonic=1)
    assert len(out) == 3
    assert out.loc[0, "newkirk_quality_flag"] == "ok"
    assert out.loc[1:, "newkirk_r_rsun"].isna().all()
    assert set(out.loc[1:, "newkirk_invalid_reason"]) == {"invalid_frequency"}
    pd.testing.assert_frame_equal(data, before)


def test_unknown_fit_validity_is_excluded_from_model():
    data = pd.DataFrame(
        {
            "freq": [150.0],
            "quality_flag": ["ok"],
            "overlay_valid": [np.nan],
            "trajectory_valid": [np.nan],
        }
    )
    assert attach_newkirk_height_to_gaussian(data, 1, 1).empty


@pytest.mark.parametrize("freq", [-150.0, 0.0, np.nan, np.inf, None, pd.NA, ""])
def test_invalid_drift_frequency_is_flagged(freq):
    result = calculate_drift_rate_from_line(
        {
            "t_start": "2000-01-01T00:00:00Z",
            "t_end": "2000-01-01T00:00:01Z",
            "f_start_mhz": freq,
            "f_end_mhz": 120.0,
        }
    )
    assert result.quality_flag == "invalid_frequency"
    assert np.isnan(result.drift_rate_mhz_s)


@pytest.mark.parametrize(
    "updates",
    [
        {"f_start_mhz": -180.0},
        {"f_start_mhz": np.nan},
        {"duration_s": 0.0},
        {"drift_rate_mhz_s": np.inf},
    ],
)
def test_invalid_drift_line_cannot_produce_a_model_speed(updates):
    row = {
        "f_start_mhz": 180.0,
        "f_end_mhz": 120.0,
        "drift_rate_mhz_s": -10.0,
        **updates,
    }
    out = extrapolate_drift_line_with_newkirk(row, 1, 1)
    assert out["newkirk_quality_flag"] != "ok"
    assert out["newkirk_invalid_reason"]
    assert np.isnan(out["speed_km_s"])


@pytest.mark.parametrize("value", [None, pd.NA, ""])
@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("f_start_mhz", "invalid_frequency"),
        ("drift_rate_mhz_s", "invalid_drift_rate"),
        ("duration_s", "invalid_duration"),
    ],
)
def test_missing_drift_samples_retain_a_reason(field, reason, value):
    row = {"f_start_mhz": 180.0, "f_end_mhz": 120.0, "drift_rate_mhz_s": -10.0}
    row[field] = value
    out = extrapolate_drift_line_with_newkirk(row, 1, 1)
    assert out["newkirk_invalid_reason"] == reason
    assert np.isnan(out["speed_km_s"])
    assert row[field] is value


def test_model_quality_normalization_matches_table_filter():
    row = {
        "f_start_mhz": 180.0,
        "f_end_mhz": 120.0,
        "drift_rate_mhz_s": -10.0,
        "quality_flag": " OK ",
    }
    out = extrapolate_drift_line_with_newkirk(row, 1, 1)
    assert out["newkirk_quality_flag"] == "ok"
    assert np.isfinite(out["speed_km_s"])


@pytest.mark.parametrize("key", ["multipliers", "harmonics"])
@pytest.mark.parametrize("value", [None, 0.0, -1.0, np.nan, np.inf, "unknown"])
def test_invalid_model_configuration_is_rejected_without_observed_rows(key, value):
    with pytest.raises(ValueError, match=key):
        build_drift_newkirk_table([], {key: [value]})


@pytest.mark.parametrize(
    "candidate",
    [
        pd.Timestamp("2000-01-01T08:00:00+08:00"),
        Time("2000-01-01T00:00:00", scale="utc").tai,
    ],
)
def test_nearest_lookup_normalizes_equivalent_time_representations(candidate):
    target = pd.Timestamp("2000-01-01T00:00:00").to_pydatetime()
    assert nearest_time_index(target, [candidate], max_delta_seconds=0.0) == 0


def test_nearest_lookup_preserves_submicrosecond_offsets():
    target = Time("2000-01-01T00:00:00.000000300", scale="utc")
    candidates = [
        Time("2000-01-01T00:00:00.000000100", scale="utc"),
        Time("2000-01-01T00:00:00.000000310", scale="utc"),
    ]
    assert nearest_time_index(target, candidates, max_delta_seconds=10e-9) == 1
    assert nearest_time_index(target, candidates, max_delta_seconds=9e-9) is None
