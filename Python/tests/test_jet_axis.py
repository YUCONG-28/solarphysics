"""Nisticò-inspired jet PCA is descriptive, native and endpoint-aware."""

import json
from copy import deepcopy

import numpy as np
import pytest

from solar_toolkit.map.jet_axis import fit_jet_axis


def rows(positions):
    return [
        dict(
            number=i + 1,
            role="jet",
            admissible_candidate=True,
            xyz_Rsun=list(p),
            order=i + 1,
            endpoint=(
                "inner" if i == 0 else "outer" if i == len(positions) - 1 else None
            ),
        )
        for i, p in enumerate(positions)
    ]


def test_straight_inclined_axis_and_radial_angle_use_centroid():
    direction = np.array([1, 2, -1]) / np.sqrt(6)
    centre = np.array([1.1, 0.2, 0.1])
    points = rows(centre + np.linspace(-0.2, 0.2, 5)[:, None] * direction)
    before = deepcopy(points)
    result = fit_jet_axis(
        points,
        reference_frame="HeliographicStonyhurst",
        reference_time_utc="2000-01-01T12:00:31.000",
    )
    assert result["valid"] and result["directed"]
    assert not result["geometry_valid"]
    np.testing.assert_allclose(result["direction_xyz"], direction, atol=1e-12)
    np.testing.assert_allclose(result["centroid_xyz_Rsun"], centre, atol=1e-12)
    expected = np.rad2deg(np.arccos(direction @ centre / np.linalg.norm(centre)))
    assert result["radial_angle_deg"] == pytest.approx(expected)
    assert result["rms_perpendicular_Rsun"] < 1e-8
    assert result["reference_frame"] == "HeliographicStonyhurst"
    assert result["reference_time_utc"] == "2000-01-01T12:00:31.000"
    assert points == before
    json.dumps(result, allow_nan=False)


def test_reversing_explicit_endpoints_changes_only_direction_and_directed_angle():
    points = rows([[1.1, 0, 0], [1.2, 0.01, 0], [1.3, 0, 0]])
    before = fit_jet_axis(points)
    points[0]["endpoint"], points[-1]["endpoint"] = "outer", "inner"
    after = fit_jet_axis(points)
    np.testing.assert_allclose(
        after["direction_xyz"], -np.asarray(before["direction_xyz"])
    )
    assert after["radial_angle_deg"] == pytest.approx(180 - before["radial_angle_deg"])
    for key in (
        "eigenvalues_Rsun2",
        "centroid_xyz_Rsun",
        "rms_perpendicular_Mm",
        "point_residuals",
        "unsigned_radial_angle_deg",
    ):
        assert after[key] == before[key]


def test_bent_axis_scatter_is_actual_perpendicular_rms_and_not_uncertainty():
    points = rows(
        [
            [1.05, -0.20, 0],
            [1.09, -0.10, 0.02],
            [1.14, 0, 0.03],
            [1.17, 0.1, 0.02],
            [1.18, 0.2, 0],
        ]
    )
    result = fit_jet_axis(points)
    distances = np.asarray([p["distance_Rsun"] for p in result["point_residuals"]])
    assert result["rms_perpendicular_Rsun"] == pytest.approx(
        np.sqrt(np.mean(distances**2)), abs=1e-12
    )
    assert result["rms_perpendicular_Mm"] > 0
    assert "confidence_interval" not in result and "sigma" not in result
    offset = np.array([1, 2, -3])
    moved = rows(np.asarray([p["xyz_Rsun"] for p in points]) + offset)
    shifted = fit_jet_axis(moved)
    np.testing.assert_allclose(
        shifted["eigenvalues_Rsun2"], result["eigenvalues_Rsun2"], atol=1e-14
    )
    np.testing.assert_allclose(
        shifted["direction_xyz"], result["direction_xyz"], atol=1e-14
    )


def test_reference_bad_and_duplicate_positions_do_not_change_fit_weight():
    points = rows([[1.1, 0, 0], [1.2, 0.02, 0], [1.3, 0, 0]])
    expected = fit_jet_axis(points)
    points += [
        dict(number=4, role="reference", admissible_candidate=True, xyz_Rsun=[4, 8, 2]),
        dict(
            number=5,
            role="jet",
            admissible_candidate=False,
            xyz_Rsun=[8, 2, 3],
            issues=["below_photosphere"],
        ),
        dict(
            number=6,
            role="jet",
            admissible_candidate=True,
            xyz_Rsun=points[1]["xyz_Rsun"],
        ),
        dict(number=7, role="jet", admissible_candidate=True, xyz_Rsun=[np.nan, 0, 0]),
    ]
    result = fit_jet_axis(points)
    assert result["used_point_numbers"] == [1, 2, 3]
    assert [p["number"] for p in result["excluded_points"]] == [4, 5, 6, 7]
    assert result["eigenvalues_Rsun2"] == expected["eigenvalues_Rsun2"]
    assert result["direction_xyz"] == expected["direction_xyz"]
    assert result["rms_perpendicular_Mm"] == expected["rms_perpendicular_Mm"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "positions",
    [[], [[1, 0, 0]], [[1, 0, 0], [2, 0, 0]], [[1, 0, 0], [1, 0, 0], [2, 0, 0]]],
)
def test_three_distinct_points_are_required(positions):
    result = fit_jet_axis(rows(positions))
    assert not result["valid"] and result["direction_xyz"] is None
    assert result["status"] == "insufficient_distinct_points"


def test_equal_largest_eigenvalues_do_not_define_unique_axis():
    points = rows([[1.2, 0.1, 0], [1.2, 0, 0.1], [1.2, -0.1, 0], [1.2, 0, -0.1]])
    result = fit_jet_axis(points)
    assert result["status"] == "degenerate_point_distribution"
    assert not result["valid"] and result["direction_xyz"] is None
    assert "principal_eigenline_not_unique" in result["issues"]


def test_missing_or_ambiguous_endpoints_return_unsigned_axis_only():
    points = rows([[1.1, 0, 0], [1.2, 0.01, 0], [1.3, 0, 0]])
    points[0]["endpoint"] = None
    result = fit_jet_axis(points)
    assert result["valid"] and not result["directed"]
    assert result["radial_angle_deg"] is None
    assert 0 <= result["unsigned_radial_angle_deg"] <= 90
    assert result["sign_method"] == "unoriented"
    points[0]["endpoint"] = "inner"
    points[1]["endpoint"] = "inner"
    assert not fit_jet_axis(points)["directed"]


def test_duplicate_number_and_excluded_endpoint_are_diagnostic():
    points = rows([[1.1, 0, 0], [1.2, 0.01, 0], [1.3, 0, 0], [1.4, 0, 0]])
    points[0]["admissible_candidate"] = False
    result = fit_jet_axis(points)
    assert result["valid"] and not result["directed"]
    assert "endpoint_excluded_from_fit" in result["issues"]
    points[0]["admissible_candidate"] = True
    points[1]["number"] = 1
    result = fit_jet_axis(points)
    assert not result["valid"]
    assert all(
        "duplicate_point_number" in p["reasons"] for p in result["excluded_points"]
    )


def test_public_map_function_is_same_implementation():
    from solar_toolkit.map import fit_jet_axis as public

    assert public is fit_jet_axis
