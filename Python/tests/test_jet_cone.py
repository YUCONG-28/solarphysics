import numpy as np
import pytest

from solar_toolkit.radio.jet_cone import (
    JetCone,
    height_interval_distance,
    intersect_jet_cone,
)


def test_cylinder_crossing_and_interior_height_minimum():
    cone = JetCone((2, 0, 0), (0, 0, 1), 0.5, 0)
    row = intersect_jet_cone((10, 0, 2), (-1, 0, 0), cone)[0]
    assert row["distance_min_rsun"] == pytest.approx(7.5)
    assert row["distance_max_rsun"] == pytest.approx(8.5)
    cone = JetCone((0, 0, 2), (0, 0, 1), 3, 0)
    row = intersect_jet_cone((10, 0, 2), (-1, 0, 0), cone)[0]
    assert row["height_min_rsun"] == pytest.approx(1)
    assert row["height_max_rsun"] == pytest.approx(np.sqrt(13) - 1)


def test_tangent_no_intersection_and_forward_only():
    cone = JetCone((2, 0, 0), (0, 0, 1), 0.5, 0)
    row = intersect_jet_cone((10, 0.5, 2), (-1, 0, 0), cone)[0]
    assert row["distance_min_rsun"] == pytest.approx(row["distance_max_rsun"])
    assert intersect_jet_cone((10, 0.6, 2), (-1, 0, 0), cone) == []
    assert intersect_jet_cone((10, 0, 2), (1, 0, 0), cone) == []


def test_axis_parallel_unbounded_and_occultation():
    cone = JetCone((2, 0, 0), (1, 0, 0), 0.2, 10)
    row = intersect_jet_cone((3, 0, 0), (1, 0, 0), cone)[0]
    assert row["height_min_rsun"] == pytest.approx(2)
    assert np.isinf(row["height_max_rsun"])
    # Far-side cone cannot be observed through the photosphere.
    far = JetCone((-2, 0, 0), (-1, 0, 0), 0.2, 10)
    assert intersect_jet_cone((10, 0, 0), (-1, 0, 0), far) == []


def test_quadratic_two_roots_discard_back_cone():
    cone = JetCone((2, 0, 0), (0, 0, 1), 0.1, 10)
    rows = intersect_jet_cone((2, 0, 10), (0, 0, -1), cone)
    assert len(rows) == 1
    assert rows[0]["distance_max_rsun"] == pytest.approx(10)


def test_rotation_invariance_and_direct_membership():
    cone = JetCone((2, 0.3, 0), (0, 0, 1), 0.1, 20)
    o = np.array([10, 0.4, 2])
    u = np.array([-1, 0, 0])
    rows = intersect_jet_cone(o, u, cone)
    q, _ = np.linalg.qr(np.random.default_rng(42).normal(size=(3, 3)))
    rotated = intersect_jet_cone(
        q @ o,
        q @ u,
        JetCone(
            tuple(q @ np.array(cone.origin_rsun)),
            tuple(q @ np.array(cone.axis)),
            0.1,
            20,
        ),
    )
    assert rows[0]["height_min_rsun"] == pytest.approx(rotated[0]["height_min_rsun"])
    r = rows[0]
    for d in np.linspace(r["distance_min_rsun"], r["distance_max_rsun"], 101):
        v = o + d * u - np.array(cone.origin_rsun)
        assert np.linalg.norm(v[:2]) <= 0.1 + v[2] * np.tan(np.deg2rad(20)) + 1e-8


def test_density_independent_and_degeneracy():
    from solar_toolkit.radio.newkirk import newkirk_height_from_frequency_mhz

    cone = JetCone((2, 0, 0), (0, 0, 1), 0.5, 10)
    first = intersect_jet_cone((10, 0, 2), (-1, 0, 0), cone)
    a = newkirk_height_from_frequency_mhz(180, 1, 2)
    b = newkirk_height_from_frequency_mhz(180, 4, 1)
    assert a == b
    assert first == intersect_jet_cone((10, 0, 2), (-1, 0, 0), cone)
    assert height_interval_distance(0.3, 0.2, 0.5) == 0
    assert height_interval_distance(0.1, 0.2, np.inf) == pytest.approx(0.1)


@pytest.mark.parametrize("angle", [-1, 90, np.nan])
def test_bad_angle(angle):
    with pytest.raises(ValueError):
        JetCone((2, 0, 0), (1, 0, 0), 0.1, angle)


def test_known_curved_stereo_points_and_common_registration_bias():
    from solar_toolkit.radio.source_geometry import project_hpc, triangulate_tie_point

    oa = np.array([215.0, 0, 0])
    ob = 215 * np.array([np.cos(0.51), np.sin(0.51), 0])
    points = np.array(
        [[1.02, 0.5, -0.15], [1.07, 0.54, -0.17], [1.12, 0.6, -0.19], [1.2, 0.7, -0.2]]
    )
    recovered = []
    biased = []
    for p in points:
        a = project_hpc(p, oa)
        b = project_hpc(p, ob)
        fit = triangulate_tie_point(
            a,
            b,
            oa,
            ob,
            sigma_a_arcsec=0.5,
            sigma_b_arcsec=1.6,
            correspondence_verified=True,
            samples=100,
            seed=42,
        )
        assert fit["geometry_valid"]
        recovered.append(fit["point_rsun"])
        bias = triangulate_tie_point(
            a,
            b + [2.0, 0],
            oa,
            ob,
            sigma_a_arcsec=0.5,
            sigma_b_arcsec=1.6,
            correspondence_verified=False,
            samples=100,
        )
        assert not bias["geometry_valid"]
        biased.append(bias["point_rsun"])
    np.testing.assert_allclose(recovered, points, atol=1e-9)
    assert np.max(abs(np.asarray(biased) - points)) > 1e-4


def test_random_intervals_against_direct_frustum_predicate():
    rng = np.random.default_rng(7)
    for _ in range(30):
        cone = JetCone(
            tuple([2, rng.uniform(-0.4, 0.4), 0]),
            tuple([0, 0, 1]),
            rng.uniform(0.1, 0.8),
            rng.uniform(0, 30),
        )
        o = np.array([10.0, rng.uniform(-1, 1), rng.uniform(1.1, 3)])
        ray = np.array([-1.0, 0, 0])
        intervals = intersect_jet_cone(o, ray, cone)
        for d in np.linspace(0, 20, 200):
            p = o + d * ray
            v = p - np.array(cone.origin_rsun)
            direct = v[2] >= 0 and np.linalg.norm(v[:2]) <= cone.width_rsun + v[
                2
            ] * np.tan(np.deg2rad(cone.half_angle_deg))
            returned = any(
                x["distance_min_rsun"] <= d <= x["distance_max_rsun"] for x in intervals
            )
            assert direct == returned
