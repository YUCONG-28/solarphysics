"""Focused contracts for the public radio computation modules."""

from __future__ import annotations

import ast
import importlib
import inspect
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from solar_toolkit import radio
from solar_toolkit.radio.cso_processing import (
    block_mean_rebin,
    calc_polarization_ratio,
    finite_color_limits,
    safe_log10,
)
from solar_toolkit.radio.physical_diagnostics import build_drift_newkirk_table
from solar_toolkit.radio.reprojection import (
    nearest_time_index,
    reproject_radio_array,
)


def test_public_modules_have_explicit_pure_exports():
    expected = {
        "cso_processing": {
            "block_mean_rebin",
            "calc_polarization_ratio",
            "finite_color_limits",
            "safe_log10",
        },
        "physical_diagnostics": {
            "build_drift_newkirk_table",
            "filter_accepted_drift_rows",
        },
        "reprojection": {
            "RadioReprojectionResult",
            "interpolate_scattered_to_grid",
            "nearest_time_index",
            "reproject_radio_array",
        },
    }
    for module_name, exports in expected.items():
        module = importlib.import_module(f"solar_toolkit.radio.{module_name}")
        assert set(module.__all__) == exports
        assert getattr(radio, module_name) is module
        tree = ast.parse(inspect.getsource(module))
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any("cli" in name or "workflow" in name for name in imports)


def test_cso_block_rebin_and_numerical_helpers():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    rebinned = block_mean_rebin(data, frequency_bin=2, time_bin=2)
    np.testing.assert_allclose(rebinned, [[2.5, 4.5], [10.5, 12.5]])

    ratio = calc_polarization_ratio(
        np.array([3.0, 1.0, np.nan]),
        np.array([1.0, -1.0, 1.0]),
    )
    np.testing.assert_allclose(ratio[:1], [0.5])
    assert np.isnan(ratio[1:]).all()
    np.testing.assert_allclose(safe_log10(np.array([1.0, 10.0])), [0.0, 1.0])
    assert np.isnan(safe_log10(np.array([0.0, -1.0]))).all()
    assert finite_color_limits(
        np.array([-2.0, 1.0, np.nan]),
        lower_percentile=0,
        upper_percentile=100,
        symmetric=True,
    ) == (-2.0, 2.0)


def test_reproject_radio_array_with_explicit_pixel_mapper():
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = reproject_radio_array(
        source,
        (2, 2),
        lambda x, y: (x, y),
        method="nearest",
    )

    assert result is not None
    np.testing.assert_allclose(result.data, source)
    assert result.peak_pixel == (1.0, 1.0)
    assert result.amplitude == 4.0
    assert result.mapped_sample_count == 4


def test_nearest_time_index_honors_tolerance_and_first_tie():
    target = datetime(2000, 1, 1, 4, 48, 40)
    candidates = [target - timedelta(seconds=2), target + timedelta(seconds=2)]

    assert nearest_time_index(target, candidates) == 0
    assert nearest_time_index(target, candidates, max_delta_seconds=1.5) is None
    with pytest.raises(ValueError, match="non-negative"):
        nearest_time_index(target, candidates, max_delta_seconds=-1)


def test_physical_diagnostics_filters_quality_and_expands_model_grid():
    accepted = {
        "drift_label": "accepted",
        "f_start_mhz": 160.0,
        "f_end_mhz": 140.0,
        "drift_rate_mhz_s": -10.0,
        "quality_flag": "ok",
    }
    table = build_drift_newkirk_table(
        pd.DataFrame(
            [accepted, {**accepted, "drift_label": "rejected", "quality_flag": "bad"}]
        ),
        {"multipliers": [1, 2], "harmonics": [1, 2]},
    )

    assert table["drift_label"].tolist() == ["accepted"] * 4
    assert set(table["newkirk_multiplier"]) == {1.0, 2.0}
    assert set(table["newkirk_harmonic"]) == {1, 2}
    assert np.isfinite(table["speed_km_s"]).all()
