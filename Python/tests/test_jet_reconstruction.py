"""Native stereo candidates: exact rays, explicit identities and honest lengths."""

import json
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import frames

from solar_toolkit.map.jet_annotations import JetDocument
from solar_toolkit.map.jet_reconstruction import (
    RECONSTRUCTION_VERSION,
    epipolar_native_pixels,
    reconstruct_jet,
)
from solar_toolkit.radio.source_geometry import project_hpc, triangulate_rays

POINTS = np.array([[0.65, 0.82, -0.20], [0.68, 0.85, -0.215], [0.73, 0.89, -0.23]])
RADIUS = 695700000.0


def observations(
    tmp_path,
    *,
    longitudes=(0.0, 30.0),
    seconds=(0, 0),
    wcs_offsets=(0, 0),
    points=POINTS,
):
    reference = frames.HeliographicStonyhurst(obstime="2000-01-01T12:00:31")
    docs = []
    for i, (lon, offset) in enumerate(zip(longitudes, seconds, strict=True)):
        date = Time("2000-01-01T12:00:30") + offset * u.s
        native = frames.HeliographicStonyhurst(obstime=date + wcs_offsets[i] * u.s)
        phi, b = np.deg2rad(lon), np.deg2rad(-5)
        obs = 212 * np.array(
            [np.cos(b) * np.cos(phi), np.cos(b) * np.sin(phi), np.sin(b)]
        )
        positions = (
            SkyCoord(CartesianRepresentation(points.T * RADIUS * u.m), frame=reference)
            .transform_to(native)
            .cartesian.xyz.to_value(u.m)
            .T
            / RADIUS
        )
        hpc = np.asarray([project_hpc(p, obs) for p in positions])
        header = fits.Header(
            dict(
                CTYPE1="HPLN-TAN",
                CTYPE2="HPLT-TAN",
                CUNIT1="arcsec",
                CUNIT2="arcsec",
                CRPIX1=128,
                CRPIX2=128,
                CRVAL1=hpc[0, 0],
                CRVAL2=hpc[0, 1],
                CDELT1=1,
                CDELT2=1,
                EXPTIME=2,
                DATE_OBS=date.isot,
                DSUN_OBS=212 * RADIUS,
                HGLN_OBS=lon,
                HGLT_OBS=-5,
                RSUN_REF=RADIUS,
                TELESCOP="SDO/AIA",
                INSTRUME="AIA",
                WAVELNTH=304,
                WAVEUNIT="angstrom",
                BUNIT="DN",
                SYNTHET=True,
            )
        )
        header["DATE-AVG"] = native.obstime.isot
        path = tmp_path / f"view{i}.fits"
        fits.writeto(path, np.ones((256, 256)) * (i + 1), header)
        doc = JetDocument(path)
        for n, (tx, ty) in enumerate(hpc, 1):
            px = doc.map.world_to_pixel(
                SkyCoord(tx * u.arcsec, ty * u.arcsec, frame=doc.map.coordinate_frame)
            )
            doc.add_tiepoint([px.x.value, px.y.value], n, "possible", f"candidate {n}")
            doc.state["tiepoints"][-1].update(
                role="jet",
                order=n,
                endpoint="inner" if n == 1 else "outer" if n == len(points) else None,
            )
        docs.append(doc)
    return docs


def reconstruct(docs):
    return reconstruct_jet(docs, pairing={"status": "matched", "tolerance_s": 15})


def test_exact_known_points_without_roi_or_axis(tmp_path):
    docs = observations(tmp_path)
    assert all(not d.state["axis"] and d.state["roi"] is None for d in docs)
    before = [deepcopy(d.state) for d in docs]
    result = reconstruct(docs)
    assert result["numerical_valid"] and not result["geometry_valid"]
    assert not result["display_geometry_used"]
    assert [d.state for d in docs] == before
    positions = np.asarray([p["xyz_Rsun"] for p in result["points"]])
    assert np.max(abs(positions - POINTS)) < 1e-9
    for p in result["points"]:
        assert p["gap_Mm"] < 1e-6
        assert max(p["residual_arcsec"]) < 1e-7
        assert p["within_3sigma"] is None
        assert len(p["closest_points_Rsun"]) == 2
        assert min(p["ray_distances_Rsun"]) > 0
        assert p["visible"] == [True, True]
    assert result["summary"]["chord_Rsun"] == pytest.approx(
        np.linalg.norm(POINTS[-1] - POINTS[0]), abs=1e-9
    )
    assert result["summary"]["polyline_length_Rsun"] == pytest.approx(
        np.linalg.norm(np.diff(POINTS, axis=0), axis=1).sum(), abs=1e-9
    )
    assert not result["summary"]["incomplete"]
    assert result["algorithm_version"] == RECONSTRUCTION_VERSION
    assert result["summary"]["axis"]["valid"]
    assert result["summary"]["axis"]["reference_frame"] == "HeliographicStonyhurst"
    json.dumps(result, allow_nan=False)


def test_cross_epoch_observers_reproject_to_each_native_frame(tmp_path):
    docs = observations(tmp_path, seconds=(0, 10), wcs_offsets=(1, 2))
    result = reconstruct(docs)
    assert (
        np.max(abs(np.asarray([p["xyz_Rsun"] for p in result["points"]]) - POINTS))
        < 1e-9
    )
    assert abs(result["provenance"]["center_delta_emission_s"] + 10) < 1e-4
    assert (
        result["provenance"]["frames"][1]["wcs_time_utc"] == "2000-01-01T12:00:42.000"
    )
    for p in result["points"]:
        assert np.max(abs(np.asarray(p["projected_pixel_xy"]) - p["pixel_xy"])) < 0.1


def test_references_and_old_unknown_roles_never_enter_jet_length(tmp_path):
    docs = observations(tmp_path)
    for d in docs:
        d.state["tiepoints"][1]["role"] = "reference"
    result = reconstruct(docs)
    assert result["summary"]["point_numbers"] == [1, 3]
    assert result["summary"]["polyline_length_Rsun"] == pytest.approx(
        result["summary"]["chord_Rsun"]
    )
    for d in docs:
        for point in d.state["tiepoints"]:
            point.pop("role")
    result = reconstruct(docs)
    assert result["numerical_valid"]
    assert result["summary"]["point_numbers"] == []
    assert result["summary"]["chord_Rsun"] is None
    assert "unknown_roles_excluded" in result["summary"]["issues"]


def test_partial_endpoints_not_inferred_and_order_conflicts(tmp_path):
    docs = observations(tmp_path)
    docs[1].state["tiepoints"].pop()
    result = reconstruct(docs)
    assert result["summary"]["chord_Rsun"] is None
    assert result["points"][-1]["reason"] == "missing_counterpart"
    assert result["summary"]["incomplete"]
    docs[0].state["tiepoints"].pop()
    docs[1].state["tiepoints"][1]["order"] = 99
    result = reconstruct(docs)
    assert "order_conflict" in result["points"][1]["issues"]
    assert result["summary"]["polyline_length_Rsun"] is None


def test_same_image_duplicate_numbers_and_repeated_coordinates(tmp_path):
    docs = observations(tmp_path)
    assert reconstruct([docs[0], docs[0]])["issues"] == [
        "same_native_image_in_both_views"
    ]
    docs[0].state["tiepoints"].append(deepcopy(docs[0].state["tiepoints"][0]))
    result = reconstruct(docs)
    assert result["points"][0]["reason"] == "duplicate_number"
    docs[0].state["tiepoints"].pop()
    docs[0].state["tiepoints"][1]["pixel_xy"] = (
        docs[0].state["tiepoints"][0]["pixel_xy"].copy()
    )
    result = reconstruct(docs)
    assert "duplicate_native_position" in result["points"][0]["issues"]


def test_weak_parallax_is_diagnostic(tmp_path):
    docs = observations(tmp_path, longitudes=(0.0, 0.00001))
    result = reconstruct(docs)
    assert not result["numerical_valid"]
    assert all(p["reason"] == "weak_parallax" for p in result["points"])
    json.dumps(result, allow_nan=False)


def test_mismatch_exceeds_sigma_without_manufactured_confidence(tmp_path):
    docs = observations(tmp_path)
    for d in docs:
        for p in d.state["tiepoints"]:
            p["sigma_arcsec"] = 0.5
    docs[1].state["tiepoints"][1]["pixel_xy"][1] += 15
    result = reconstruct(docs)
    row = result["points"][1]
    assert row["numerical_valid"] and row["within_3sigma"] is False
    assert row["reason"] == "reprojection_exceeds_3sigma"
    assert row["gap_Mm"] > 1
    assert not result["geometry_valid"]
    assert "confidence_interval" not in json.dumps(result)


def test_time_and_wavelength_evidence_do_not_get_silently_accepted(tmp_path):
    docs = observations(tmp_path)
    result = reconstruct_jet(
        docs, pairing={"status": "outside_tolerance", "tolerance_s": 1}
    )
    assert result["numerical_valid"] and result["summary"]["incomplete"]
    assert all(not p["admissible_candidate"] for p in result["points"])
    result = reconstruct_jet(docs)
    assert "time_pairing_unverified" in result["issues"]
    assert result["summary"]["incomplete"]
    result = reconstruct_jet(
        docs,
        pairing={
            "status": "matched",
            "tolerance_s": 15,
            "AIA": {"image_sha256": "an_old_frame"},
            "EUVI": {"image_sha256": docs[1].sha256},
        },
    )
    assert "pairing_image_identity_mismatch" in result["issues"]
    assert all(not p["admissible_candidate"] for p in result["points"])
    from sunpy.map import Map

    meta = docs[1].map.meta.copy()
    meta["wavelnth"] = 171
    docs[1].map = Map(docs[1].map.data, meta)
    result = reconstruct(docs)
    assert result["numerical_valid"]
    assert "different_wavelengths" in result["issues"]
    assert result["summary"]["polyline_length_Rsun"] is None


def test_below_surface_and_occluded_positions_are_not_clipped(tmp_path):
    points = np.array([[0.5, 0.6, -0.1]])
    docs = observations(tmp_path, points=points)
    result = reconstruct(docs)
    row = result["points"][0]
    assert row["height_Rsun"] < 0 and row["below_photosphere"]
    assert row["visible"] == [False, False]
    assert not row["admissible_candidate"]


def test_ray_api_retains_actual_closest_points_and_rejects_invalid_origin():
    r = triangulate_rays([3, 0, 0], [-1, 0, 0], [0, 3, 0.1], [0, -1, 0])
    assert r["valid"]
    assert np.allclose(r["closest_points_rsun"], [[0, 0, 0], [0, 0, 0.1]])
    assert np.allclose(r["point_rsun"], [0, 0, 0.05])
    assert np.allclose(r["ray_distances_rsun"], [3, 3])
    with pytest.raises(ValueError):
        triangulate_rays([np.nan, 0, 0], [-1, 0, 0], [0, 3, 0], [0, -1, 0])


def _distance_to_locus(point, line):
    a, b = line[:-1], line[1:]
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[ok], b[ok]
    vector = b - a
    along = np.clip(
        np.sum((point - a) * vector, axis=1) / np.sum(vector**2, axis=1), 0, 1
    )
    return np.linalg.norm(point - (a + along[:, None] * vector), axis=1).min()


def test_native_epipolar_cross_epoch_and_vertical_wcs(tmp_path):
    from sunpy.map import Map

    docs = observations(tmp_path, seconds=(0, 10), wcs_offsets=(1, 2))
    for rotation in (0, 90):
        meta = docs[1].map.meta.copy()
        meta["crota2"] = rotation
        docs[1].map = Map(docs[1].map.data, meta)
        for source in docs[0].state["tiepoints"]:
            line = epipolar_native_pixels(docs[0], docs[1], source["pixel_xy"])
            # Expected target pixels independently recomputed from known 3-D.
            position = SkyCoord(
                CartesianRepresentation(POINTS[source["number"] - 1] * RADIUS * u.m),
                frame=frames.HeliographicStonyhurst(obstime="2000-01-01T12:00:31"),
            )
            expected = docs[1].map.world_to_pixel(
                position.transform_to(docs[1].map.coordinate_frame)
            )
            assert (
                _distance_to_locus(np.array([expected.x.value, expected.y.value]), line)
                < 0.1
            )
    assert epipolar_native_pixels(docs[0], docs[0], [100, 100]).shape == (0, 2)


def test_independent_sunpy_projected_points_close_full_reconstruction(tmp_path):
    """Generate tiepoints through SunPy, not the reconstruction projection API."""
    docs = observations(tmp_path, seconds=(0, 9), wcs_offsets=(1, 2))
    known = SkyCoord(
        CartesianRepresentation(POINTS.T * RADIUS * u.m),
        frame=frames.HeliographicStonyhurst(obstime="2000-01-01T12:00:31"),
    )
    for doc in docs:
        expected_pixels = doc.map.world_to_pixel(
            known.transform_to(doc.map.coordinate_frame)
        )
        for row, x, y in zip(
            doc.state["tiepoints"],
            expected_pixels.x.value,
            expected_pixels.y.value,
            strict=True,
        ):
            row["pixel_xy"] = [float(x), float(y)]
    result = reconstruct(docs)
    np.testing.assert_allclose(
        [p["xyz_Rsun"] for p in result["points"]], POINTS, atol=1e-9
    )
    assert result["summary"]["axis"]["used_point_numbers"] == [1, 2, 3]
    assert result["summary"]["axis"]["reference_time_utc"] == "2000-01-01T12:00:31.000"
    assert max(max(p["residual_arcsec"]) for p in result["points"]) < 1e-7


@pytest.mark.parametrize(
    "failure",
    [
        "missing_counterpart",
        "missing_native_pixel",
        "duplicate_number",
        "role_conflict",
    ],
)
def test_failed_middle_jet_node_is_not_bridged(tmp_path, failure):
    docs = observations(tmp_path)
    if failure == "missing_counterpart":
        docs[1].state["tiepoints"].pop(1)
    elif failure == "duplicate_number":
        docs[1].state["tiepoints"].append(deepcopy(docs[1].state["tiepoints"][1]))
    elif failure == "role_conflict":
        docs[1].state["tiepoints"][1]["role"] = "reference"
    else:
        x, y = docs[1].state["tiepoints"][1]["pixel_xy"]
        docs[1].map.data[int(round(y)), int(round(x))] = np.nan
    result = reconstruct(docs)
    summary = result["summary"]
    assert summary["incomplete"] and summary["polyline_has_gaps"]
    assert summary["polyline_length_Rsun"] is None
    assert summary["polyline_xyz_Rsun"] == []
    assert [s["point_numbers"] for s in summary["polyline_segments"]] == [[1], [3]]
    assert summary["chord_Rsun"] == pytest.approx(
        np.linalg.norm(POINTS[-1] - POINTS[0]), abs=1e-9
    )


def test_partial_polyline_sums_supported_segments_only(tmp_path):
    points = POINTS[0] + np.arange(5)[:, None] * (POINTS[-1] - POINTS[0]) / 4
    docs = observations(tmp_path, points=points)
    docs[1].state["tiepoints"].pop(2)
    result = reconstruct(docs)
    summary = result["summary"]
    assert [s["point_numbers"] for s in summary["polyline_segments"]] == [
        [1, 2],
        [4, 5],
    ]
    expected = np.linalg.norm(points[1] - points[0]) + np.linalg.norm(
        points[4] - points[3]
    )
    assert summary["polyline_length_Rsun"] == pytest.approx(expected, abs=1e-9)
    assert summary["status"] == "partial_candidate_polyline"
    assert summary["incomplete"] and summary["polyline_xyz_Rsun"] == []


def test_epipolar_consistent_wrong_depth_is_not_identity_validation(tmp_path):
    """Even a perfect locus match can select a different depth on the first ray."""
    docs = observations(tmp_path)
    known = SkyCoord(
        CartesianRepresentation(POINTS[0] * RADIUS * u.m),
        frame=frames.HeliographicStonyhurst(obstime="2000-01-01T12:00:31"),
    )
    origin = (
        docs[0]
        .map.observer_coordinate.transform_to(known.frame)
        .cartesian.xyz.to_value(u.m)
        / RADIUS
    )
    # A nearby but wrong depth has exactly the same AIA pixel. It is a valid
    # epipolar candidate in EUVI, not evidence that the clicked structure agrees.
    wrong_position = POINTS[0] + 0.015 * (POINTS[0] - origin) / np.linalg.norm(
        POINTS[0] - origin
    )
    wrong = SkyCoord(
        CartesianRepresentation(wrong_position * RADIUS * u.m), frame=known.frame
    )
    pixels = docs[1].map.world_to_pixel(
        wrong.transform_to(docs[1].map.coordinate_frame)
    )
    docs[1].state["tiepoints"][0]["pixel_xy"] = [pixels.x.value, pixels.y.value]
    before = deepcopy(docs[1].state["tiepoints"][0]["pixel_xy"])
    result = reconstruct(docs)
    row = result["points"][0]
    np.testing.assert_allclose(row["xyz_Rsun"], wrong_position, atol=1e-9)
    assert max(row["residual_arcsec"]) < 1e-7
    assert not row["geometry_valid"] and row["within_3sigma"] is None
    assert docs[1].state["tiepoints"][0]["pixel_xy"] == before
