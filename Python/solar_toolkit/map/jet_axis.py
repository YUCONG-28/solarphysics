"""Descriptive jet-axis fits to native-stereo candidate positions.

Independent implementation informed by Nisticò (2023), Sections 3 and 4.3,
doi:10.1007/s11207-023-02122-9. Unlike a second moment about a selected reference
point, this fit uses the point centroid and 1/N equal weights. Its transverse
scatter is not a localization uncertainty or a correspondence-validation test.
"""

from collections import Counter

import numpy as np

__all__ = ["AXIS_METHOD_VERSION", "fit_jet_axis"]

AXIS_METHOD_VERSION = 1
_RADIUS_MM = 695.7
_POSITION_ATOL = 1e-10
_EIGENGAP_RTOL = 1e-10


def fit_jet_axis(points, *, reference_frame=None, reference_time_utc=None):
    """Fit an equal-weight PCA line to admissible, explicit jet candidates.

    ``points`` contains reconstruction rows with ``number``, ``role``,
    ``admissible_candidate``, ``xyz_Rsun`` and optional ``endpoint``. Reference
    and invalid points are excluded with reasons; duplicate positions have no
    extra weight. Three distinct positions are required. A tied largest
    eigenvalue does not define an axis. Coordinates are neither changed nor
    relabelled: the supplied frame and epoch accompany the result.

    ``valid`` means that a numerical line exists, never that correspondence or
    dynamics have been verified. Unique usable inner/outer endpoints orient the
    line. Otherwise ``direction_xyz`` is a deterministic representative of an
    *unoriented* eigenline and only the unsigned acute radial angle is reported.
    Eigenvalues use 1/N normalization so that the transverse RMS is exactly the
    root mean square of the saved point-to-line distances.
    """
    result = dict(
        method="mean_centered_equal_weight_pca",
        method_version=AXIS_METHOD_VERSION,
        citation_doi="10.1007/s11207-023-02122-9",
        valid=False,
        directed=False,
        geometry_valid=False,
        status="insufficient_distinct_points",
        issues=[],
        reference_frame=reference_frame,
        reference_time_utc=reference_time_utc,
        used_point_numbers=[],
        excluded_points=[],
        centroid_xyz_Rsun=None,
        direction_xyz=None,
        radial_unit_xyz=None,
        radial_angle_deg=None,
        unsigned_radial_angle_deg=None,
        eigenvalues_Rsun2=None,
        eigenvalue_ratios_to_largest=None,
        rms_perpendicular_Rsun=None,
        rms_perpendicular_Mm=None,
        point_residuals=[],
        inner_number=None,
        outer_number=None,
        sign_method="unoriented",
        sample_count=0,
        interpretation="descriptive_candidate_axis; scatter_is_not_localization_uncertainty",
        parameters=dict(
            centering="centroid",
            weighting="equal",
            normalization="1/N",
            position_atol_Rsun=_POSITION_ATOL,
            eigengap_rtol=_EIGENGAP_RTOL,
        ),
    )
    points = list(points)
    counts = Counter(p.get("number") for p in points)
    rows, positions = [], []
    for row in points:
        reasons = []
        if row.get("role") != "jet":
            reasons.append("not_explicit_jet_point")
        if not row.get("eligibility", {}).get(
            "axis_candidate", row.get("admissible_candidate")
        ):
            reasons.append("inadmissible_candidate")
            reasons.extend(row.get("issues") or [])
        if counts[row.get("number")] > 1:
            reasons.append("duplicate_point_number")
        try:
            position = np.asarray(row.get("xyz_Rsun"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("a finite three-vector is required")
        except (TypeError, ValueError):
            reasons.append("invalid_position")
            position = None
        if not reasons and any(
            np.linalg.norm(position - p) <= _POSITION_ATOL for p in positions
        ):
            reasons.append("duplicate_position")
        if reasons:
            result["excluded_points"].append(
                dict(number=row.get("number"), reasons=list(dict.fromkeys(reasons)))
            )
        else:
            rows.append(row)
            positions.append(position)
    result["used_point_numbers"] = [p["number"] for p in rows]
    result["sample_count"] = len(rows)
    if len(rows) < 3:
        result["issues"].append(
            "at_least_three_distinct_admissible_jet_points_required"
        )
        return result

    positions = np.asarray(positions)
    centre = positions.mean(axis=0)
    offsets = positions - centre
    covariance = offsets.T @ offsets / len(rows)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    # Numerical roundoff may make a mathematically zero eigenvalue negative.
    eigenvalues = np.maximum(eigenvalues, 0)
    result.update(
        centroid_xyz_Rsun=centre.tolist(), eigenvalues_Rsun2=eigenvalues.tolist()
    )
    largest = eigenvalues[-1]
    if (
        largest <= _POSITION_ATOL**2
        or largest - eigenvalues[-2] <= _EIGENGAP_RTOL * largest
    ):
        result.update(status="degenerate_point_distribution")
        result["issues"].append("principal_eigenline_not_unique")
        return result
    direction = vectors[:, -1]
    # A representative sign keeps exports reproducible; it does not imply flow.
    if direction[np.argmax(abs(direction))] < 0:
        direction = -direction
    inner = [
        p for p in points if p.get("role") == "jet" and p.get("endpoint") == "inner"
    ]
    outer = [
        p for p in points if p.get("role") == "jet" and p.get("endpoint") == "outer"
    ]
    result["inner_number"] = inner[0].get("number") if len(inner) == 1 else None
    result["outer_number"] = outer[0].get("number") if len(outer) == 1 else None
    used = set(result["used_point_numbers"])
    if len(inner) != 1 or len(outer) != 1:
        result["issues"].append("explicit_unique_endpoints_required_for_direction")
    elif inner[0].get("number") not in used or outer[0].get("number") not in used:
        result["issues"].append("endpoint_excluded_from_fit")
    else:
        chord = np.asarray(outer[0]["xyz_Rsun"]) - inner[0]["xyz_Rsun"]
        length = np.linalg.norm(chord)
        projection = float(chord @ direction)
        if length <= _POSITION_ATOL or abs(projection) <= _EIGENGAP_RTOL * length:
            result["issues"].append("endpoints_do_not_orient_principal_axis")
        else:
            if projection < 0:
                direction = -direction
            result.update(
                directed=True, sign_method="explicit_inner_to_outer_endpoints"
            )
    distances = np.linalg.norm(
        offsets - np.outer(offsets @ direction, direction), axis=1
    )
    rms = float(np.sqrt(eigenvalues[0] + eigenvalues[1]))
    result.update(
        valid=True,
        status="fitted",
        direction_xyz=direction.tolist(),
        eigenvalue_ratios_to_largest=(eigenvalues / largest).tolist(),
        rms_perpendicular_Rsun=rms,
        rms_perpendicular_Mm=rms * _RADIUS_MM,
        point_residuals=[
            dict(
                number=p["number"],
                distance_Rsun=float(d),
                distance_Mm=float(d) * _RADIUS_MM,
            )
            for p, d in zip(rows, distances, strict=True)
        ],
    )
    radius = np.linalg.norm(centre)
    if radius <= _POSITION_ATOL:
        result["issues"].append("centroid_radial_direction_undefined")
    else:
        radial = centre / radius
        dot = float(np.clip(direction @ radial, -1, 1))
        result.update(
            radial_unit_xyz=radial.tolist(),
            unsigned_radial_angle_deg=float(np.rad2deg(np.arccos(abs(dot)))),
        )
        if result["directed"]:
            result["radial_angle_deg"] = float(np.rad2deg(np.arccos(dot)))
    return result
