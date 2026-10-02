"""Joint fitting uses native WCS observations, not display warps or test depths."""

import json
from copy import deepcopy

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import CartesianRepresentation, SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map
from test_jet_reconstruction import observations

from solar_toolkit.map.jet_geometry_fit import (
    GEOMETRY_FIT_VERSION,
    _NativeProjectionContext,
    fit_jet_geometry,
    freeze_jet_documents,
)
from solar_toolkit.map.jet_reconstruction import reconstruct_jet

PAIR = {"status": "matched", "tolerance_s": 15}
START, END = np.array([0.65, 0.82, -0.2]), np.array([0.73, 0.89, -0.23])


def independent_observations(tmp_path, n=8, bend=0.0, noise=0.0, scales=(0.6, 1.6)):
    t = np.linspace(0, 1, n) ** 1.1
    transverse = np.array([-0.07, 0.08, 0.0])
    transverse *= bend / np.linalg.norm(transverse)
    xyz = START + t[:, None] * (END - START) + 4 * (t * (1 - t))[:, None] * transverse
    docs = observations(tmp_path, points=xyz, seconds=(0, 8), wcs_offsets=(1, 2))
    known = SkyCoord(
        CartesianRepresentation(xyz.T * 695700000 * u.m),
        frame=frames.HeliographicStonyhurst(obstime="2000-01-01T12:00:31"),
    )
    rng = np.random.default_rng(741)
    for i, doc in enumerate(docs):
        meta = doc.map.meta.copy()
        meta.update(dict(cdelt1=scales[i], cdelt2=scales[i], crota2=15 * i))
        doc.map = Map(doc.map.data, meta)
        pixels = doc.map.world_to_pixel(known.transform_to(doc.map.coordinate_frame))
        for row, x, y in zip(
            doc.state["tiepoints"], pixels.x.value, pixels.y.value, strict=True
        ):
            row["pixel_xy"] = (np.array([x, y]) + rng.normal(0, noise, 2)).tolist()
    return docs, xyz


def test_joint_native_line_closure_and_interior_prediction(tmp_path):
    docs, xyz = independent_observations(tmp_path, n=6)
    before = [deepcopy(d.state) for d in docs]
    result = fit_jet_geometry(docs, pairing=PAIR)
    fit = result["line"]
    assert fit["valid"], fit["issues"]
    assert result["method_version"] == GEOMETRY_FIT_VERSION
    np.testing.assert_allclose(
        [p["xyz_Rsun"] for p in fit["fitted_points"]], xyz, atol=2e-8
    )
    assert fit["pixel_rmse"] < 1e-6
    assert fit["length_Rsun"] == pytest.approx(np.linalg.norm(END - START), abs=2e-8)
    assert fit["validation"]["complete"] and fit["validation"]["pixel_rmse"] < 1e-5
    assert fit["optimizer"]["jacobian_rank"] == fit["optimizer"]["parameter_count"]
    assert result["weighting"]["mode"] == "equal_native_pixels"
    assert result["selected_model"] == "line" and not result["geometry_valid"]
    assert not result["display_geometry_used"]
    assert [d.state for d in docs] == before
    json.dumps(result, allow_nan=False)


def test_single_bend_curve_fits_native_coordinates_without_automatic_selection(
    tmp_path,
):
    docs, xyz = independent_observations(tmp_path, n=8, bend=0.012)
    result = fit_jet_geometry(docs, pairing=PAIR, include_curve=True)
    curve, line = result["curve"], result["line"]
    assert curve["valid"], curve["issues"]
    assert curve["pixel_rmse"] < 1e-5
    np.testing.assert_allclose(
        [p["xyz_Rsun"] for p in curve["fitted_points"]], xyz, atol=2e-7
    )
    assert curve["length_Rsun"] > np.linalg.norm(END - START)
    assert curve["validation"]["complete"] and curve["validation"]["pixel_rmse"] < 1e-4
    assert line["pixel_rmse"] > 0.5
    assert result["selected_model"] == "line"
    assert not result["comparison"]["automatic_model_selection"]
    params = curve["parameters"]
    chord = np.asarray(params["endpoint_xyz_Rsun"])[1] - params["endpoint_xyz_Rsun"][0]
    assert abs(np.dot(chord, params["bend_xyz_Rsun"])) < 1e-12
    assert (
        params["t"][0] == 0
        and params["t"][-1] == 1
        and np.all(np.diff(params["t"]) > 0)
    )


def test_curve_gate_and_three_point_line(tmp_path):
    docs, _ = independent_observations(tmp_path, n=3)
    result = fit_jet_geometry(docs, pairing=PAIR, include_curve=True)
    assert result["line"]["valid"]
    assert not result["curve"]["valid"]
    assert "at_least_six_ordered_features_required" in result["curve"]["issues"]
    assert result["line"]["validation"]["expected_fold_count"] == 1


def test_snapshot_and_stale_reconstruction_protection(tmp_path):
    docs, _ = independent_observations(tmp_path, n=3)
    frozen = freeze_jet_documents(docs)
    result = reconstruct_jet(frozen, pairing=PAIR)
    original = frozen[0].state["tiepoints"][0]["pixel_xy"].copy()
    docs[0].state["tiepoints"][0]["pixel_xy"][0] += 1
    docs[0].map.data[0, 0] = -500
    assert frozen[0].state["tiepoints"][0]["pixel_xy"] == original
    assert frozen[0].map.data[0, 0] != -500 and not frozen[0].map.data.flags.writeable
    fit = fit_jet_geometry(
        frozen, reconstruction=result, pairing=PAIR, cross_validate=False
    )
    assert fit["line"]["valid"]
    stale = fit_jet_geometry(docs, reconstruction=result, pairing=PAIR)
    assert "reconstruction_input_signature_stale" in stale["issues"]


def test_partial_and_invalid_sigmas_remain_uniform_engineering_weights(tmp_path):
    docs, _ = independent_observations(tmp_path, n=3, noise=0.1)
    unweighted = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)
    docs[0].state["tiepoints"][0]["sigma_arcsec"] = 0.5
    docs[1].state["tiepoints"][0]["sigma_arcsec"] = "unmeasured"
    partial = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)
    assert partial["weighting"]["mode"] == "equal_native_pixels"
    assert partial["weighting"]["supplied_sigma_count"] == 1
    assert partial["line"]["pixel_rmse"] == pytest.approx(
        unweighted["line"]["pixel_rmse"], abs=1e-8
    )


def test_complete_sigmas_use_local_native_wcs_weights(tmp_path):
    docs, _ = independent_observations(tmp_path, n=5, noise=0.1)
    for doc in docs:
        for point in doc.state["tiepoints"]:
            point["sigma_arcsec"] = 1.0
    result = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)
    assert result["line"]["valid"], result["line"]["issues"]
    assert result["weighting"]["complete_sigma"]
    assert result["weighting"]["mode"] == "supplied_angular_sigma"
    assert not result["geometry_valid"]


def test_robust_loss_keeps_same_points_and_does_not_upgrade_identity(tmp_path):
    docs, _ = independent_observations(tmp_path, n=8)
    docs[1].state["tiepoints"][3]["pixel_xy"][1] += 8
    linear = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)
    robust = fit_jet_geometry(docs, pairing=PAIR, loss="soft_l1", cross_validate=False)
    assert linear["line"]["converged"] and robust["line"]["converged"]
    assert (
        linear["input_point_numbers"]
        == robust["input_point_numbers"]
        == list(range(1, 9))
    )
    assert len(robust["line"]["fitted_points"]) == 8
    assert (
        min(np.ravel(robust["line"]["fitted_points"][3]["robust_score_weights"])) < 0.5
    )
    assert robust["settings"]["robust_scale"] == 1
    assert not robust["geometry_valid"]


@pytest.mark.parametrize("failure", ["missing", "order", "endpoint"])
def test_incomplete_or_ambiguous_input_allows_descriptive_line_but_not_curve(
    tmp_path, failure
):
    docs, _ = independent_observations(tmp_path, n=6)
    if failure == "missing":
        docs[1].state["tiepoints"].pop(2)
    elif failure == "order":
        docs[1].state["tiepoints"][2]["order"] = 1
    else:
        docs[1].state["tiepoints"][-1]["endpoint"] = None
    result = fit_jet_geometry(
        docs, pairing=PAIR, include_curve=True, cross_validate=False
    )
    assert result["line"]["valid"], result["line"]["issues"]
    assert not result["curve"]["valid"] and result["curve"]["issues"]
    if failure == "endpoint":
        assert (
            not result["line"]["directed"]
            and result["line"]["radial_angle_deg"] is None
        )
    if failure == "missing":
        assert len(result["input_point_numbers"]) == 5
        assert "not_confirmed_complete" in result["line"]["length_interpretation"]


def test_order_metadata_conflict_does_not_erase_numerical_point_or_pca(tmp_path):
    docs, _ = independent_observations(tmp_path, n=3)
    docs[1].state["tiepoints"][1]["order"] = 77
    report = reconstruct_jet(docs, pairing=PAIR)
    middle = report["points"][1]
    assert middle["numerical_valid"] and not middle["admissible_candidate"]
    assert middle["eligibility"]["axis_candidate"]
    assert not middle["eligibility"]["ordered_jet_candidate"]
    assert report["summary"]["axis"]["used_point_numbers"] == [1, 2, 3]
    assert report["summary"]["polyline_length_Rsun"] is None


def test_no_convergence_preserves_diagnostic_fitted_positions(tmp_path):
    docs, _ = independent_observations(tmp_path, n=8, noise=0.2)
    result = fit_jet_geometry(docs, pairing=PAIR, max_nfev=1, cross_validate=False)
    assert not result["line"]["valid"]
    assert "optimizer_not_converged" in result["line"]["issues"]
    assert len(result["line"]["fitted_points"]) == 8


def test_vectorized_context_matches_independent_sunpy_projection(tmp_path):
    docs, xyz = independent_observations(tmp_path, n=8, bend=0.01)
    context = _NativeProjectionContext(docs)
    pixels, _ = context.project(xyz)
    for side, doc in enumerate(docs):
        np.testing.assert_allclose(
            pixels[:, side], [p["pixel_xy"] for p in doc.state["tiepoints"]], atol=1e-7
        )
    assert reconstruct_jet(docs, pairing=PAIR)["processing"]["native_wcs_batches"] == 2


def test_reversed_endpoints_preserve_lengths_and_reverse_direction(tmp_path):
    docs, _ = independent_observations(tmp_path, n=5, bend=0)
    first = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)["line"]
    for doc in docs:
        for point in doc.state["tiepoints"]:
            point["order"] = 6 - point["order"]
            point["endpoint"] = {"inner": "outer", "outer": "inner", None: None}[
                point["endpoint"]
            ]
    second = fit_jet_geometry(docs, pairing=PAIR, cross_validate=False)["line"]
    assert second["valid"]
    assert second["length_Rsun"] == pytest.approx(first["length_Rsun"], abs=1e-8)
    np.testing.assert_allclose(
        second["direction_xyz"], -np.asarray(first["direction_xyz"]), atol=1e-7
    )


def test_invalid_options_and_public_api(tmp_path):
    from solar_toolkit.map import fit_jet_geometry as public

    assert public is fit_jet_geometry
    with pytest.raises(ValueError):
        fit_jet_geometry([], loss="invented")
    with pytest.raises(ValueError):
        fit_jet_geometry([], max_nfev=False)
    assert fit_jet_geometry([])["issues"] == ["missing_view"]


def test_no_endpoint_or_order_metadata_still_fits_unsigned_line(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6)
    for doc in docs:
        for point in doc.state["tiepoints"]:
            point.pop("order", None)
            point.pop("endpoint", None)
    result = fit_jet_geometry(
        docs, pairing=PAIR, include_curve=True, cross_validate=False
    )
    line = result["line"]
    assert line["valid"] and not line["directed"]
    assert (
        line["radial_angle_deg"] is None
        and line["unsigned_radial_angle_deg"] is not None
    )
    assert line["ordering_source"] == "PCA_projection_for_nuisance_parameters_only"
    assert not result["curve"]["valid"]
    assert "unique_annotated_jet_order_required_for_curve" in result["curve"]["issues"]
    assert all("order" not in p for d in docs for p in d.state["tiepoints"])


def test_true_cancellation_propagates_from_optimizer_objective(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6, noise=0.1)
    calls = []

    def cancelled():
        calls.append(1)
        return len(calls) >= 4

    with pytest.raises(InterruptedError, match="cancelled"):
        fit_jet_geometry(docs, pairing=PAIR, cancelled=cancelled)
    assert len(calls) == 4


def test_exact_radial_minimum_does_not_use_only_model_endpoints():
    from solar_toolkit.map.jet_geometry_fit import _path_height_range

    endpoints = np.array([[1.1, -0.3, 0], [1.1, 0.3, 0]])
    result = _path_height_range(endpoints, np.array([-0.2, 0, 0]))
    assert result["min_height_Rsun"] == pytest.approx(-0.1)
    assert result["max_height_Rsun"] > 0
    assert np.any(np.isclose(result["extremum_parameters"], 0.5, rtol=0, atol=1e-12))


def test_masks_are_preserved_and_excluded_from_quantitative_fits(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6)
    mask = np.zeros(docs[1].map.data.shape, dtype=bool)
    x, y = docs[1].state["tiepoints"][2]["pixel_xy"]
    mask[int(round(y)), int(round(x))] = True
    docs[1].map = Map(docs[1].map.data, docs[1].map.meta, mask=mask)
    frozen = freeze_jet_documents(docs)
    assert np.array_equal(frozen[1].map.mask, mask)
    report = reconstruct_jet(frozen, pairing=PAIR)
    assert not report["points"][2]["eligibility"]["data_usable"]
    fit = fit_jet_geometry(
        frozen,
        reconstruction=report,
        pairing=PAIR,
        include_curve=True,
        cross_validate=False,
    )
    assert 3 not in fit["input_point_numbers"]
    assert fit["line"]["valid"] and not fit["curve"]["valid"]


def test_weak_parallax_cannot_be_rescued_by_joint_fitting(tmp_path):
    xyz = START + np.linspace(0, 1, 6)[:, None] * (END - START)
    docs = observations(tmp_path, points=xyz, longitudes=(0, 0.00001))
    result = fit_jet_geometry(docs, pairing=PAIR, include_curve=True)
    assert not result["line"]["valid"] and not result["curve"]["valid"]
    assert not result["input_point_numbers"]
    assert all("weak_parallax" in p["reasons"] for p in result["excluded_points"])


def test_frame_time_mismatch_is_preserved_instead_of_fitted_away(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6)
    result = fit_jet_geometry(
        docs, pairing={"status": "outside_tolerance", "tolerance_s": 1}
    )
    assert not result["line"]["valid"] and not result["input_point_numbers"]
    assert all(
        "time_pairing_incompatible" in p["reasons"] for p in result["excluded_points"]
    )


def test_same_images_with_changed_wcs_invalidate_reconstruction(tmp_path):
    docs, _ = independent_observations(tmp_path, n=3)
    report = reconstruct_jet(docs, pairing=PAIR)
    meta = docs[1].map.meta.copy()
    meta["crpix1"] += 0.5
    docs[1].map = Map(docs[1].map.data, meta)
    result = fit_jet_geometry(docs, reconstruction=report, pairing=PAIR)
    assert "reconstruction_input_signature_stale" in result["issues"]


def test_curve_zero_bend_has_no_control_point_gauge_singularity(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6)
    result = fit_jet_geometry(
        docs, pairing=PAIR, include_curve=True, cross_validate=False
    )
    assert result["curve"]["valid"], result["curve"]["issues"]
    assert np.linalg.norm(result["curve"]["parameters"]["bend_xyz_Rsun"]) < 1e-8
    optimizer = result["curve"]["optimizer"]
    assert optimizer["jacobian_rank"] == optimizer["parameter_count"] == 12


def test_nonstraight_inconsistent_stereo_annotations_leave_model_residuals(tmp_path):
    docs, _ = independent_observations(tmp_path, n=6, bend=0.014)
    docs[1].state["tiepoints"][2]["pixel_xy"][1] += 7
    result = fit_jet_geometry(
        docs, pairing=PAIR, include_curve=True, cross_validate=False
    )
    assert result["curve"]["pixel_rmse"] > 0.1
    assert not result["geometry_valid"]
    assert result["selected_model"] == "line"
