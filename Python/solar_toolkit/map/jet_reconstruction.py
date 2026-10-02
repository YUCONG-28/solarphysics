"""Native-image stereoscopy for identified jet features, without display shells.

The closest-ray midpoint is a *candidate estimate*, never an exact intersection
or independent identity confirmation. Both actual closest points are retained.
Positions use no density model, image reprojection, segmentation or fitted
axis. A descriptive jet axis is fitted only after the native-ray reconstruction.
"""

from collections import Counter

import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from sunpy.coordinates import frames

from solar_toolkit.radio.source_geometry import hpc_ray, triangulate_rays

from .euvi_preprocessing import geometry_issues
from .jet_annotations import json_safe, pixel_hpc
from .jet_axis import fit_jet_axis
from .jet_viewpoint import observer

__all__ = ["RECONSTRUCTION_VERSION", "reconstruct_jet", "epipolar_native_pixels"]

RECONSTRUCTION_VERSION = "native-rays-layered-v3"
RADIUS_M = 695700000.0
LIGHT_SPEED_M_S = 299792458.0


def _transform(point, source, target):
    """Transform positions, including epoch-dependent solar-origin translation."""
    return (
        SkyCoord(
            CartesianRepresentation(np.asarray(point) * RADIUS_M * u.m), frame=source
        )
        .transform_to(target)
        .cartesian.xyz.to_value(u.m)
        / RADIUS_M
    )


def _visible(point, origin):
    direction = point - origin
    t = np.clip(-origin @ direction / (direction @ direction), 0, 1)
    closest = origin + t * direction
    return bool(np.linalg.norm(closest) >= 1 - 1e-10)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _eligibility(row, global_issues):
    """Separate geometry, pairing and annotation policy without losing diagnostics."""
    issues = set(row.get("issues", []))
    data_failures = {
        "invalid_native_coordinates",
        "native_pixel_missing",
        "duplicate_number",
        "duplicate_native_position",
        "missing_counterpart",
    }
    data_usable = row.get("hpc_arcsec") is not None and not bool(issues & data_failures)
    physical = bool(
        row.get("numerical_valid")
        and row.get("below_photosphere") is False
        and row.get("visible") == [True, True]
    )
    pair_codes = {
        "time_pairing_incompatible",
        "source_time_outside_tolerance",
        "different_wavelengths",
        "wavelength_unavailable",
    }
    pairing_compatible = not bool(issues & pair_codes)
    pairing_evidenced = pairing_compatible and not any(
        "time" in code or code == "pairing_image_identity_mismatch"
        for code in global_issues
    )
    candidate = bool(data_usable and physical and pairing_compatible)
    sigmas = [
        p.get("sigma_arcsec") if p else None
        for p in row.get("input_correspondences", [])
    ]
    try:
        values = np.asarray(sigmas, float)
        errors_available = (
            values.shape == (2,) and np.isfinite(values).all() and (values > 0).all()
        )
    except (TypeError, ValueError):
        errors_available = False
    return dict(
        data_usable=bool(data_usable),
        numerical_position=bool(row.get("numerical_valid")),
        physical_candidate=physical,
        pairing_compatible=pairing_compatible,
        pairing_evidence_available=pairing_evidenced,
        identity_confirmed=row.get("identity_status")
        == ["manual_confirmed", "manual_confirmed"],
        localization_scales_available=bool(errors_available),
        axis_candidate=candidate and row.get("role") == "jet",
        ordered_jet_candidate=candidate
        and row.get("role") == "jet"
        and _integer(row.get("order")),
        endpoint_metadata_compatible="endpoint_conflict" not in issues,
        interpretation="operation_specific_gates; identity_and_evolution_not_upgraded",
    )


def epipolar_native_pixels(source_doc, target_doc, xy):
    """Native target-pixel epipolar locus, including the two WCS epochs.

    This is only a geometric suggestion. No depth, identity, visibility or
    correspondence is inferred. The great-circle parameterization also handles
    vertical loci and rotated WCS; it does not assume y is a function of x.
    """
    source = frames.HeliographicStonyhurst(
        obstime=source_doc.map.coordinate_frame.obstime
    )
    target = frames.HeliographicStonyhurst(
        obstime=target_doc.map.coordinate_frame.obstime
    )
    origin = observer(source_doc.map, source)
    target_origin = observer(target_doc.map, target)
    tx, ty = pixel_hpc(source_doc.map, [xy])[0]
    _, direction = hpc_ray(tx, ty, origin)
    transformed_origin = _transform(origin, source, target)
    transformed_ray = (
        _transform(origin + direction, source, target) - transformed_origin
    )
    baseline = target_origin - transformed_origin
    normal = np.cross(transformed_ray, baseline)
    if np.linalg.norm(baseline) < 1e-10 or np.linalg.norm(
        normal
    ) < 1e-10 * np.linalg.norm(baseline):
        return np.empty((0, 2))
    normal /= np.linalg.norm(normal)
    h, w = target_doc.map.data.shape
    positions = np.array(
        [[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1], [(w - 1) / 2, (h - 1) / 2]]
    )
    hpc = pixel_hpc(target_doc.map, positions)
    rays = np.array([hpc_ray(*p, target_origin)[1] for p in hpc])
    centre = rays[-1] - (rays[-1] @ normal) * normal
    if np.linalg.norm(centre) < 1e-10:
        return np.empty((0, 2))
    centre /= np.linalg.norm(centre)
    across = np.cross(normal, centre)
    angles = np.arctan2(rays[:-1] @ across, rays[:-1] @ centre)
    theta = np.linspace(angles.min(), angles.max(), 800)
    directions = np.cos(theta)[:, None] * centre + np.sin(theta)[:, None] * across
    ez = target_origin / np.linalg.norm(target_origin)
    ex = np.cross([0, 0, 1], ez)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    tx = np.arctan2(directions @ ex, -(directions @ ez))
    ty = np.arcsin(np.clip(directions @ ey, -1, 1))
    pixels = target_doc.map.world_to_pixel(
        SkyCoord(tx * u.rad, ty * u.rad, frame=target_doc.map.coordinate_frame)
    )
    result = np.c_[pixels.x.value, pixels.y.value]
    valid = (
        np.isfinite(result).all(axis=1)
        & (result[:, 0] >= 0)
        & (result[:, 0] <= w - 1)
        & (result[:, 1] >= 0)
        & (result[:, 1] <= h - 1)
    )
    # NaNs prevent plotting through unsupported branches, while clipping avoids
    # a hint expanding the native image's display range.
    result[~valid] = np.nan
    return result if valid.any() else np.empty((0, 2))


def _summary(points, global_issues, provenance=None):
    issues = []
    selected = [
        p
        for p in points
        if p["role"] == "jet"
        or (
            p["role"] == "unknown"
            and any(
                t and t.get("role") == "jet" for t in p.get("input_correspondences", [])
            )
        )
    ]
    usable = [
        p
        for p in selected
        if p.get("eligibility", {}).get(
            "axis_candidate", p.get("admissible_candidate") and p.get("role") == "jet"
        )
    ]
    # Sort all selected nodes, including failed reconstructions. Filtering first
    # would fabricate a segment across an unmeasured middle position.
    all_ordered = [p for p in selected if p["order"] is not None]
    if len(all_ordered) != len(selected):
        issues.append("jet_order_missing")
    if len(usable) != len(selected):
        issues.append("jet_points_unusable")
    if len(set(p["order"] for p in all_ordered)) != len(all_ordered):
        issues.append("duplicate_jet_order")
        all_ordered = []
    all_ordered.sort(key=lambda p: p["order"])
    usable_numbers = {p["number"] for p in usable}
    ordered = [p for p in all_ordered if p["number"] in usable_numbers]
    inner = [p for p in selected if p["endpoint"] == "inner"]
    outer = [p for p in selected if p["endpoint"] == "outer"]
    result = dict(
        status=(
            "candidate_polyline" if len(ordered) >= 2 else "insufficient_ordered_points"
        ),
        point_numbers=[p["number"] for p in ordered],
        polyline_xyz_Rsun=[p["xyz_Rsun"] for p in ordered],
        polyline_length_Rsun=None,
        polyline_length_Mm=None,
        polyline_segments=[],
        polyline_has_gaps=len(usable) != len(selected),
        chord_Rsun=None,
        chord_Mm=None,
        direction_xyz=None,
        inner_number=inner[0]["number"] if len(inner) == 1 else None,
        outer_number=outer[0]["number"] if len(outer) == 1 else None,
        incomplete=True,
        issues=issues,
        length_interpretation="piecewise_chords; unobserved_curvature_not_reconstructed",
        axis=fit_jet_axis(
            points,
            reference_frame=(provenance or {}).get("reference_frame"),
            reference_time_utc=(provenance or {}).get("reference_time_utc"),
        ),
    )
    if any(p["role"] == "unknown" for p in points):
        issues.append("unknown_roles_excluded")
    if len(inner) != 1 or len(outer) != 1:
        issues.append("explicit_unique_endpoints_required")
    elif not all(p["number"] in usable_numbers for p in inner + outer):
        issues.append("endpoint_unusable")
    elif inner[0]["order"] is None or outer[0]["order"] is None:
        issues.append("endpoint_order_missing")
    elif inner[0]["order"] >= outer[0]["order"]:
        issues.append("endpoint_order_conflict")
    else:
        vector = np.asarray(outer[0]["xyz_Rsun"]) - inner[0]["xyz_Rsun"]
        length = float(np.linalg.norm(vector))
        if length <= 1e-10:
            issues.append("coincident_endpoints")
        else:
            result.update(
                chord_Rsun=length,
                chord_Mm=length * RADIUS_M / 1e6,
                direction_xyz=vector / length,
            )
        if any(
            p["order"] < inner[0]["order"] or p["order"] > outer[0]["order"]
            for p in ordered
        ):
            issues.append("points_outside_endpoint_order")
    if len(ordered) >= 2 and not any("order" in code for code in issues):
        runs, run = [], []
        for point in all_ordered:
            if point["number"] in usable_numbers:
                run.append(point)
            else:
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        for segment in runs:
            xyz = [p["xyz_Rsun"] for p in segment]
            length = (
                float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                if len(xyz) >= 2
                else 0.0
            )
            result["polyline_segments"].append(
                dict(
                    point_numbers=[p["number"] for p in segment],
                    xyz_Rsun=xyz,
                    length_Rsun=length,
                    length_Mm=length * RADIUS_M / 1e6,
                )
            )
        if any(len(segment) >= 2 for segment in runs):
            length = sum(
                segment["length_Rsun"] for segment in result["polyline_segments"]
            )
            result.update(
                polyline_length_Rsun=length, polyline_length_Mm=length * RADIUS_M / 1e6
            )
        if result["polyline_has_gaps"]:
            result["status"] = "partial_candidate_polyline"
            # Old consumers must not connect the flattened points across gaps.
            result["polyline_xyz_Rsun"] = []
    if not selected:
        issues.append("no_explicit_jet_points")
    # Candidate geometry remains qualified even when all requested nodes exist.
    if global_issues:
        issues.append("pair_or_input_conditions_unresolved")
    if result["polyline_has_gaps"] or any("order" in code for code in issues):
        result["polyline_xyz_Rsun"] = []
    result["incomplete"] = bool(issues)
    return result


def reconstruct_jet(documents, *, pairing=None):
    """Return JSON-safe native-ray candidates, segmented lengths and jet PCA.

    ``documents`` are two JetDocument-compatible objects. Tiepoint metadata:
    ``number``, ``pixel_xy``, ``role`` (jet/reference/unknown), integer ``order``,
    optional ``endpoint`` (inner/outer), and optional positive ``sigma_arcsec``.
    Old records with no role remain unknown and never enter the jet length.
    ``pairing`` may contain status, tolerance_s and actual-frame pairing evidence.
    A sigma permits a residual diagnostic only; identity/evolution are not
    established by this routine and geometry_valid therefore remains False.
    ``summary.axis`` is descriptive and cannot change the reconstructed points.
    ``summary.polyline_segments`` retains gaps; a partial reported length sums
    only the supported piecewise chords, not links across failed nodes.
    """
    report = dict(
        schema_version=1,
        method="native_stereo_rays",
        algorithm_version=RECONSTRUCTION_VERSION,
        numerical_valid=False,
        geometry_valid=False,
        issues=[],
        points=[],
        summary={},
        provenance={},
        display_geometry_used=False,
    )
    issues = report["issues"]
    if len(documents) != 2 or any(d is None for d in documents):
        issues.append("missing_view")
        report["summary"] = _summary([], issues)
        return report
    report["provenance"] = dict(
        physical_radius_m=RADIUS_M,
        frames=[
            dict(
                path=str(d.path),
                sha256=d.sha256,
                midpoint_utc=d.info.get("midpoint_utc"),
            )
            for d in documents
        ],
    )
    if documents[0].sha256 == documents[1].sha256:
        issues.append("same_native_image_in_both_views")
    for i, doc in enumerate(documents):
        issues.extend(f"view_{i}:{reason}" for reason in geometry_issues(doc.map))
    if issues:
        report["summary"] = _summary([], issues)
        return report
    from .jet_geometry_fit import _input_signature, _NativeProjectionContext

    try:
        context = _NativeProjectionContext(documents)
    except (ValueError, np.linalg.LinAlgError) as error:
        issues.append("projection_context_unavailable:" + str(error))
        report["summary"] = _summary([], issues)
        return report
    common = context.common
    native_obs, origins, midpoints = (
        context.native_origins,
        context.origins,
        context.midpoints,
    )
    emission = [
        t - np.linalg.norm(o) * RADIUS_M / LIGHT_SPEED_M_S
        for t, o in zip(midpoints, native_obs, strict=True)
    ]
    delta = float(emission[0] - emission[1])
    report["provenance"] = dict(
        physical_radius_m=RADIUS_M,
        reference_frame="HeliographicStonyhurst",
        reference_time_utc=common.obstime.utc.isot,
        assumption="fixed_position_between_exposures; no_differential_rotation",
        input_signature=_input_signature(documents, pairing),
        center_delta_emission_s=delta,
        pairing=pairing,
        frames=[
            dict(
                path=str(d.path),
                sha256=d.sha256,
                midpoint_utc=d.info["midpoint_utc"],
                wcs_time_utc=d.map.coordinate_frame.obstime.utc.isot,
                observer_Rsun=o,
                detector=str(d.map.detector),
                wavelength=str(d.map.wavelength),
                parent_sha256=d.info.get("parent_sha256"),
                parent_pixel_offset_xy=d.info.get("parent_pixel_offset_xy"),
                pixel_origin="zero_based",
                native_wcs=d.map.wcs.to_header_string(),
            )
            for d, o in zip(documents, origins, strict=True)
        ],
    )
    wavelength_issue = None
    try:
        bands = np.array([d.map.wavelength.to_value(u.angstrom) for d in documents])
        if not np.isfinite(bands).all() or (bands <= 0).any():
            wavelength_issue = "wavelength_unavailable"
        elif not np.isclose(*bands, rtol=0, atol=1e-6):
            wavelength_issue = "different_wavelengths"
    except (AttributeError, ValueError, TypeError):
        wavelength_issue = "wavelength_unavailable"
    if wavelength_issue:
        issues.append(wavelength_issue)
    time_mismatch = False
    if pairing is None:
        issues.append("time_pairing_unverified")
    else:
        for label, document in zip(("AIA", "EUVI"), documents, strict=True):
            if label in pairing:
                record = pairing[label]
                if record is None or record.get("image_sha256") != document.sha256:
                    issues.append("pairing_image_identity_mismatch")
                    time_mismatch = True
                    break
        if pairing.get("status") != "matched":
            issues.append("time_pairing_" + str(pairing.get("status", "unverified")))
            time_mismatch = True
        tolerance = pairing.get("tolerance_s")
        if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0:
            issues.append("time_tolerance_unavailable")
        elif abs(delta) > tolerance + 1e-6:
            issues.append("center_time_outside_tolerance")
            time_mismatch = True
    ties = [d.state.get("tiepoints", []) for d in documents]
    counts = [
        Counter(t.get("number") for t in side if _integer(t.get("number")))
        for side in ties
    ]
    if any(not _integer(t.get("number")) for side in ties for t in side):
        issues.append("invalid_point_number")
    duplicates = []
    for side in ties:
        positions = [tuple(t.get("pixel_xy", [])) for t in side]
        repeated = {xy for xy, n in Counter(positions).items() if n > 1}
        duplicates.append(
            {t.get("number") for t in side if tuple(t.get("pixel_xy", [])) in repeated}
        )
    numbers = sorted(set(counts[0]) | set(counts[1]))
    # One native WCS batch per image; invalid rows still retain their own reasons.
    ray_cache = []
    for view, side in enumerate(ties):
        usable_pixels = []
        usable_numbers = []
        for item in side:
            try:
                xy = np.asarray(item.get("pixel_xy"), float)
                if (
                    xy.shape == (2,)
                    and np.isfinite(xy).all()
                    and _integer(item.get("number"))
                ):
                    usable_pixels.append(xy)
                    usable_numbers.append(item["number"])
            except (ValueError, TypeError):
                pass
        rays, hpc = context.rays(view, usable_pixels) if usable_pixels else ([], [])
        ray_cache.append(
            {
                number: (ray, angular)
                for number, ray, angular in zip(usable_numbers, rays, hpc, strict=True)
            }
        )
    for number in numbers:
        pair = [
            next((t for t in side if t.get("number") == number), None) for side in ties
        ]
        row = dict(
            number=number,
            role="unknown",
            order=None,
            endpoint=None,
            numerical_valid=False,
            geometry_valid=False,
            admissible_candidate=False,
            reason="missing_counterpart",
            issues=[],
            xyz_Rsun=None,
            height_Rsun=None,
            closest_points_Rsun=None,
            ray_distances_Rsun=None,
            gap_Mm=None,
            ray_angle_deg=None,
            reprojection_arcsec=None,
            residual_arcsec=None,
            projected_pixel_xy=None,
            below_photosphere=None,
            visible=None,
            source_delta_emission_s=None,
            within_3sigma=None,
            pixel_xy=[p.get("pixel_xy") if p else None for p in pair],
            hpc_arcsec=None,
            input_correspondences=pair,
            image_sha256=[d.sha256 for d in documents],
            feature=[p.get("feature", "") if p else None for p in pair],
            identity_status=[
                p.get("identity_status", "uncertain") if p else None for p in pair
            ],
        )
        report["points"].append(row)
        # A one-sided jet annotation is still a known missing node in the jet
        # order. Preserve that diagnostic metadata, without inventing a mate.
        present = [p for p in pair if p is not None]
        roles = [p.get("role", "unknown") for p in present]
        if len(set(roles)) == 1 and roles[0] in {"jet", "reference", "unknown"}:
            row["role"] = roles[0]
        else:
            row["issues"].append("role_conflict")
        orders = [p.get("order") for p in present]
        if len(set(orders)) == 1 and _integer(orders[0]):
            row["order"] = int(orders[0])
        elif any(x is not None for x in orders):
            row["issues"].append("order_conflict")
        endpoints = [p.get("endpoint") for p in present]
        if len(set(endpoints)) == 1 and endpoints[0] in {"inner", "outer", None}:
            row["endpoint"] = endpoints[0]
        else:
            row["issues"].append("endpoint_conflict")
        if any(c[number] > 1 for c in counts):
            row["reason"] = "duplicate_number"
            row["issues"].append(row["reason"])
            continue
        if any(p is None for p in pair):
            row["issues"].append(row["reason"])
            continue
        if any(number in group for group in duplicates):
            row["issues"].append("duplicate_native_position")
        try:
            pixels = np.asarray(row["pixel_xy"], float)
            if pixels.shape != (2, 2) or not np.isfinite(pixels).all():
                raise ValueError("invalid pixel coordinates")
            for d, xy in zip(documents, pixels, strict=True):
                h, w = d.map.data.shape
                if not (-0.5 <= xy[0] < w - 0.5 and -0.5 <= xy[1] < h - 0.5):
                    raise ValueError("outside native image")
                if not np.isfinite(d.map.data[int(round(xy[1])), int(round(xy[0]))]):
                    row["issues"].append("native_pixel_missing")
                elif (
                    d.map.mask is not None
                    and d.map.mask[int(round(xy[1])), int(round(xy[0]))]
                ):
                    row["issues"].append("native_pixel_missing")
            rays = [cache[number][0] for cache in ray_cache]
            hpc = np.asarray([cache[number][1] for cache in ray_cache])
            geometry = triangulate_rays(origins[0], rays[0], origins[1], rays[1])
        except (ValueError, IndexError, TypeError) as error:
            row.update(reason="invalid_native_coordinates", detail=str(error))
            row["issues"].append(row["reason"])
            continue
        row.update(
            hpc_arcsec=hpc,
            ray_angle_deg=geometry["angle_deg"],
            reason=geometry["reason"],
            closest_points_Rsun=geometry["closest_points_rsun"],
            ray_distances_Rsun=geometry["ray_distances_rsun"],
            gap_Mm=geometry["separation_rsun"] * RADIUS_M / 1e6,
        )
        if not geometry["valid"]:
            row["issues"].append(geometry["reason"])
            continue
        position = geometry["point_rsun"]
        native_positions = [p[0] for p in context.native_positions([position])]
        pixels_batch, hpc_batch = context.project([position])
        projected_pixels, predicted = pixels_batch[0], hpc_batch[0]
        visible = [
            _visible(p, o) for p, o in zip(native_positions, native_obs, strict=True)
        ]
        source_times = [
            t - np.linalg.norm(p - o) * RADIUS_M / LIGHT_SPEED_M_S
            for t, p, o in zip(midpoints, native_positions, native_obs, strict=True)
        ]
        row.update(
            numerical_valid=True,
            xyz_Rsun=position,
            height_Rsun=np.linalg.norm(position) - 1,
            reprojection_arcsec=predicted - hpc,
            residual_arcsec=np.linalg.norm(predicted - hpc, axis=1),
            projected_pixel_xy=projected_pixels,
            below_photosphere=bool(np.linalg.norm(position) < 1 - 1e-10),
            visible=visible,
            source_delta_emission_s=float(source_times[0] - source_times[1]),
            position_estimator="closest_ray_midpoint_candidate",
        )
        if row["below_photosphere"]:
            row["issues"].append("below_photosphere")
        if not all(visible):
            row["issues"].append("solar_occultation")
        if time_mismatch:
            row["issues"].append("time_pairing_incompatible")
        if wavelength_issue:
            row["issues"].append(wavelength_issue)
        if pairing and pairing.get("tolerance_s") is not None:
            if abs(row["source_delta_emission_s"]) > pairing["tolerance_s"] + 1e-6:
                row["issues"].append("source_time_outside_tolerance")
        sigmas = [p.get("sigma_arcsec") for p in pair]
        if any(s is None for s in sigmas):
            uncertainty = "localization_sigma_and_evolution_not_available"
        else:
            try:
                sigma = np.asarray(sigmas, float)
                if (
                    sigma.shape != (2,)
                    or not np.isfinite(sigma).all()
                    or np.any(sigma <= 0)
                ):
                    raise ValueError("positive scalar sigma required")
                row["within_3sigma"] = bool(
                    np.max(abs(predicted - hpc) / sigma[:, None]) <= 3
                )
                row["max_reprojection_sigma"] = float(
                    np.max(abs(predicted - hpc) / sigma[:, None])
                )
                uncertainty = "identity_and_evolution_unverified"
                if not row["within_3sigma"]:
                    row["issues"].append("reprojection_exceeds_3sigma")
            except (ValueError, TypeError):
                uncertainty = "invalid_localization_sigma"
                row["issues"].append(uncertainty)
        row["admissible_candidate"] = not row["issues"]
        row["reason"] = row["issues"][0] if row["issues"] else uncertainty
    report["numerical_valid"] = any(p["numerical_valid"] for p in report["points"])
    if not numbers:
        issues.append("no_correspondences")
    if any(p["issues"] for p in report["points"]):
        issues.append("point_diagnostics_require_review")
    for row in report["points"]:
        row["eligibility"] = _eligibility(row, issues)
    report["processing"] = dict(
        native_wcs_batches=2,
        frame_context="per_pair_frozen_affine_transform",
        legacy_admissible_candidate="preserved; use eligibility for operation-specific gates",
    )
    report["summary"] = _summary(report["points"], issues, report["provenance"])
    return json_safe(report)
