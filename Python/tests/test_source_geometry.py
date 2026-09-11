import datetime as dt

import numpy as np
import pandas as pd
import pytest

from solar_toolkit.radio.height_comparison import build_gaussian_newkirk_height_table
from solar_toolkit.radio.io import parse_datetime_value
from solar_toolkit.radio.source_geometry import (
    compare_geometry_newkirk,
    hpc_ray,
    intersect_polyline,
    intersect_radial_direction,
    match_burst_crossings,
    normalize_source_observations,
    position_on_inclination_cone,
    project_hpc,
    ray_height_bound,
    triangulate_rays,
    triangulate_tie_point,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("20250124044829509", dt.datetime(2025, 1, 24, 4, 48, 29, 509000)),
        ("20250124044830 11", dt.datetime(2025, 1, 24, 4, 48, 30, 11000)),
        ("20250124044842  2", dt.datetime(2025, 1, 24, 4, 48, 42, 2000)),
        ("2025-01-24T12:48:30.011+08:00", dt.datetime(2025, 1, 24, 4, 48, 30, 11000)),
        ("20250199044829509", None),
        ("20250124044829.0", None),
    ],
)
def test_timestamp_formats(text, expected):
    assert parse_datetime_value(text) == expected


def sources():
    return normalize_source_observations(
        pd.DataFrame(
            [
                dict(
                    time=f"2025-01-24T04:48:{second:02d}Z",
                    freq=f,
                    center_x_arcsec=1200.0,
                    center_y_arcsec=-300.0,
                    quality_flag="ok",
                    trajectory_valid=True,
                )
                for f in [150.0, 200.0]
                for second in [30, 31, 32, 33]
            ]
        )
    )


def test_observation_identity_keeps_valid_fit_without_float_coordinate_identity():
    original = sources().iloc[:1].drop(columns=["source_id"])
    bad = original.copy()
    bad["center_x_arcsec"] = np.nan
    result = normalize_source_observations(pd.concat([bad, original]))
    assert len(result) == 1 and result.iloc[0].source_valid
    assert result.iloc[0].duplicate_count == 1


def test_actual_crossing_no_endpoint_extension_and_offset():
    drift = pd.DataFrame(
        [
            dict(
                label="a",
                t_start="2025-01-24T04:48:30Z",
                t_end="2025-01-24T04:48:32Z",
                f_start_mhz=190,
                f_end_mhz=140,
            )
        ]
    )
    matches = match_burst_crossings(sources(), drift)
    assert (
        matches.loc[matches.frequency_mhz == 200, "match_reason"].iloc[0]
        == "no_frequency_crossing"
    )
    row = matches.loc[matches.frequency_mhz == 150].iloc[0]
    assert row.match_valid and row.delta_time_s == pytest.approx(0.4)
    shifted = match_burst_crossings(sources(), drift, clock_offset_s=0.8)
    assert shifted.loc[shifted.frequency_mhz == 150, "delta_time_s"].iloc[
        0
    ] == pytest.approx(-0.4)


def test_ambiguous_bursts_are_not_primary():
    drift = pd.DataFrame(
        [
            dict(
                label=label,
                t_start="2025-01-24T04:48:30Z",
                t_end="2025-01-24T04:48:32Z",
                f_start_mhz=200,
                f_end_mhz=150,
            )
            for label in ["a", "b"]
        ]
    )
    result = match_burst_crossings(sources(), drift)
    assert not result.match_valid.any()
    assert set(result.match_reason) == {"ambiguous_burst"}


def test_exact_two_view_geometry_and_minimum_height():
    point = np.array([0.6, 1.25, -0.3])
    a = np.array([215.0, 0.0, 0.0])
    b = np.array([186.0, 107.0, 0.0])
    oa, ra = hpc_ray(*project_hpc(point, a), a)
    ob, rb = hpc_ray(*project_hpc(point, b), b)
    result = triangulate_rays(oa, ra, ob, rb)
    assert result["valid"]
    np.testing.assert_allclose(result["point_rsun"], point, atol=1e-10)
    low, high = ray_height_bound(oa, ra)
    assert low <= np.linalg.norm(point) - 1 and np.isinf(high)
    assert triangulate_rays(oa, ra, oa, ra)["reason"] == "weak_parallax"


def test_cone_recovers_independent_point_and_reprojects_exactly():
    point = np.array([0.6, 1.25, -0.3])
    observer = np.array([215.0, 0.0, 0.0])
    theta = np.degrees(
        np.arccos(point @ observer / np.linalg.norm(point) / np.linalg.norm(observer))
    )
    origin, ray = hpc_ray(*project_hpc(point, observer), observer)
    result = position_on_inclination_cone(origin, ray, theta)
    np.testing.assert_allclose(result, point, atol=1e-10)
    with pytest.raises(ValueError):
        position_on_inclination_cone(origin, ray, 0)


def test_tie_point_requires_evidence_and_localization_errors():
    point = np.array([0.6, 1.25, -0.3])
    a = np.array([215.0, 0.0, 0.0])
    b = np.array([186.0, 107.0, 0.0])
    x, y = project_hpc(point, a), project_hpc(point, b)
    assert not triangulate_tie_point(x, y, a, b)["geometry_valid"]
    assert not triangulate_tie_point(x, y, a, b, sigma_a_arcsec=1, sigma_b_arcsec=1)[
        "geometry_valid"
    ]
    result = triangulate_tie_point(
        x,
        y,
        a,
        b,
        sigma_a_arcsec=1,
        sigma_b_arcsec=1,
        correspondence_verified=True,
        samples=200,
        seed=42,
    )
    low, high = result["height_interval_rsun"]
    assert result["geometry_valid"] and low < np.linalg.norm(point) - 1 < high
    bad = triangulate_tie_point(
        x,
        y + [0, 100],
        a,
        b,
        sigma_a_arcsec=1,
        sigma_b_arcsec=1,
        correspondence_verified=True,
    )
    assert not bad["geometry_valid"] and bad["reason"] == "reprojection_exceeds_3sigma"


def test_polyline_multiple_depth_solutions_and_no_extension():
    points = np.array([[0, -1, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0]])
    result = intersect_polyline([10, 0, 0], [-1, 0, 0], points, max_miss_rsun=0.001)
    assert len(result) == 2
    assert intersect_polyline([10, 3, 0], [-1, 0, 0], points, max_miss_rsun=0.001) == []


def test_radial_skew_lines_keep_distinct_closest_points_and_require_a_gate():
    args = ([10, 0.4, 2], [-1, 0, 0], [0, 0, 1])
    diagnostic = intersect_radial_direction(*args)
    assert diagnostic["numerical_valid"]
    assert not diagnostic["valid"] and not diagnostic["geometry_valid"]
    assert diagnostic["reason"] == "compatibility_threshold_unavailable"
    np.testing.assert_allclose(diagnostic["axis_point_rsun"], [0, 0, 2])
    np.testing.assert_allclose(diagnostic["los_point_rsun"], [0, 0.4, 2], atol=1e-12)
    assert diagnostic["miss_rsun"] == pytest.approx(0.4)
    assert np.isnan(diagnostic["point_rsun"]).all()
    assert np.isnan(diagnostic["height_rsun"])

    rejected = intersect_radial_direction(*args, max_miss_rsun=0.1)
    assert rejected["numerical_valid"] and not rejected["geometry_valid"]
    assert rejected["reason"] == "miss_exceeds_threshold"
    accepted = intersect_radial_direction(*args, max_miss_rsun=0.5)
    assert accepted["geometry_valid"] and not accepted["independent_geometry_valid"]
    np.testing.assert_allclose(accepted["point_rsun"], [0, 0, 2])
    assert accepted["height_rsun"] == pytest.approx(1)


@pytest.mark.parametrize("height", [0.5, -2.0])
def test_radial_axis_rejects_interior_and_opposite_radial_extension(height):
    result = intersect_radial_direction(
        [10, 0, height], [-1, 0, 0], [0, 0, 1], max_miss_rsun=0.01
    )
    assert result["numerical_valid"] and not result["geometry_valid"]
    assert result["reason"] == "below_photosphere"
    assert np.isnan(result["height_rsun"])


def test_radial_axis_rejects_a_point_behind_the_observer():
    result = intersect_radial_direction(
        [10, 0, 2], [1, 0, 0], [0, 0, 1], max_miss_rsun=0.01
    )
    assert result["reason"] == "behind_observer"
    assert not result["geometry_valid"]


def test_radial_photospheric_boundary_survives_numerical_roundoff():
    point = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    origin = np.array([20.0, -1.0, 0.0])
    result = intersect_radial_direction(
        origin, point - origin, point, max_miss_rsun=1e-10
    )
    assert result["geometry_valid"]
    assert result["height_rsun"] == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(result["point_rsun"], point, atol=1e-12)


def test_axis_independence_requires_both_measured_axis_and_radio_association():
    args = ([10, 0, 2], [-1, 0, 0], [0, 0, 1])
    for association, independent in [(False, False), (True, False), (False, True)]:
        result = intersect_radial_direction(
            *args,
            max_miss_rsun=0.001,
            association_verified=association,
            axis_independent=independent,
        )
        assert result["geometry_valid"] and not result["independent_geometry_valid"]
    result = intersect_radial_direction(
        *args, max_miss_rsun=0.001, association_verified=True, axis_independent=True
    )
    assert result["independent_geometry_valid"]


def test_nearly_parallel_axes_expose_weak_angle_without_an_invented_position():
    direction = [np.cos(np.radians(0.5)), np.sin(np.radians(0.5)), 0]
    radial = intersect_radial_direction(
        [10, 0, 2], [-1, 0, 0], direction, max_miss_rsun=10
    )
    assert radial["reason"] == "weak_crossing_angle"
    assert radial["angle_deg"] == pytest.approx(0.5)
    assert not radial["numerical_valid"] and np.isnan(radial["point_rsun"]).all()

    points = [[0, 0, 2], [2, 0.01, 2]]
    assert intersect_polyline([10, 0, 2], [-1, 0, 0], points, max_miss_rsun=10) == []
    diagnostics = intersect_polyline(
        [10, 0, 2], [-1, 0, 0], points, max_miss_rsun=10, include_rejected=True
    )
    assert len(diagnostics) == 1 and diagnostics[0]["reason"] == "weak_crossing_angle"
    assert diagnostics[0]["angle_deg"] < 1
    assert not diagnostics[0]["geometry_valid"]


def test_polyline_shared_vertex_is_one_solution_but_separate_depths_remain():
    result = intersect_polyline(
        [10, 0, 2],
        [-1, 0, 0],
        [[0, -1, 2], [0, 0, 2], [0, 1, 2], [1, 1, 2], [1, -1, 2]],
        max_miss_rsun=0.001,
    )
    assert len(result) == 2
    assert result[0]["segment_ids"] == [0, 1]
    assert result[1]["segment_ids"] == [3]
    assert result[0]["height_rsun"] == pytest.approx(1)
    assert result[1]["height_rsun"] == pytest.approx(np.sqrt(5) - 1)


def test_polyline_large_gate_cannot_extend_a_measured_segment():
    args = ([10, 2, 2], [-1, 0, 0], [[0, -1, 2], [0, 1, 2]])
    assert intersect_polyline(*args, max_miss_rsun=10) == []
    result = intersect_polyline(*args, max_miss_rsun=10, include_rejected=True)[0]
    assert result["reason"] == "outside_measured_segment"
    assert result["segment_fraction"] == pytest.approx(1.5)
    assert not result["geometry_valid"] and np.isnan(result["height_rsun"])


def test_polyline_skew_miss_and_missing_threshold_cannot_be_valid():
    args = ([10, 0.4, 2], [-1, 0, 0], [[0, 0, 1], [0, 0, 3]])
    assert intersect_polyline(*args) == []
    result = intersect_polyline(*args, include_rejected=True)[0]
    assert result["numerical_valid"] and not result["geometry_valid"]
    assert result["reason"] == "compatibility_threshold_unavailable"
    assert intersect_polyline(*args, max_miss_rsun=0.1) == []
    result = intersect_polyline(*args, max_miss_rsun=0.5)[0]
    np.testing.assert_allclose(result["los_point_rsun"], [0, 0.4, 2], atol=1e-12)
    np.testing.assert_allclose(result["axis_point_rsun"], [0, 0, 2])
    assert not result["independent_geometry_valid"]


@pytest.mark.parametrize("threshold", [-1, np.inf, np.nan])
def test_invalid_axis_compatibility_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="Miss threshold"):
        intersect_radial_direction(
            [10, 0, 2], [-1, 0, 0], [0, 0, 1], max_miss_rsun=threshold
        )


def test_newkirk_does_not_set_depth_and_degenerate_models_equal():
    src = sources().iloc[:1]
    geom = pd.DataFrame(
        [
            dict(
                source_id=src.iloc[0].source_id,
                height_rsun=0.5,
                geometry_valid=True,
                method="independent_test",
                evidence_id="synthetic",
            )
        ]
    )
    result = compare_geometry_newkirk(src, geom, [(1, 2), (4, 1)])
    assert result.height_rsun.nunique() == 1
    assert result.newkirk_height_rsun.nunique() == 1
    assert result.comparison_valid.all()
    geom["height_rsun"] = np.nan
    geom["geometry_valid"] = False
    result = compare_geometry_newkirk(src, geom)
    assert result.height_residual_3d_rsun.isna().all()
    assert not result.comparison_valid.any()


def test_legacy_height_interface_independent_geometry_is_optional():
    src = sources().iloc[:1]
    geom = pd.DataFrame(
        [
            dict(
                source_id=src.iloc[0].source_id,
                height_rsun=0.5,
                geometry_valid=True,
                method="independent_test",
                evidence_id="synthetic",
            )
        ]
    )
    old = build_gaussian_newkirk_height_table(src, {"solar_radius_arcsec": 960})
    new = build_gaussian_newkirk_height_table(src, {"solar_radius_arcsec": 960}, geom)
    np.testing.assert_array_equal(old.gaussian_height_rsun, new.gaussian_height_rsun)
    assert new.height_3d_valid.all()
    assert new.source_height_3d_rsun.eq(0.5).all()
