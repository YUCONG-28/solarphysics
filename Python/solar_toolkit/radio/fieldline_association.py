"""Forward magnetic-model comparisons, never independent height measurements."""

import numpy as np
import pandas as pd

from .newkirk import newkirk_radius_from_frequency_mhz
from .source_geometry import hpc_ray, intersect_polyline


def sphere_intersections(points, radius):
    """All distinct intersections of a finite 3D polyline with a radial shell."""
    points = np.asarray(points, float)
    if not np.isfinite(radius) or radius < 1:
        return []
    hits = []
    for index, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
        d = b - a
        aa = d @ d
        if aa <= 1e-24:
            continue
        bb, cc = 2 * a @ d, a @ a - radius**2
        disc = bb * bb - 4 * aa * cc
        if disc < -1e-12:
            continue
        for t in [
            (-bb - np.sqrt(max(0, disc))) / (2 * aa),
            (-bb + np.sqrt(max(0, disc))) / (2 * aa),
        ]:
            if -1e-10 <= t <= 1 + 1e-10:
                p = a + np.clip(t, 0, 1) * d
                if not any(
                    np.linalg.norm(p - old["point_rsun"]) < 1e-8 for old in hits
                ):
                    hits.append(dict(segment_id=index, point_rsun=p))
    return hits


def projected_distances(sources_xy, curve):
    """Distances to every valid finite projected segment, preserving gaps."""
    sources_xy, curve = np.asarray(sources_xy, float), np.asarray(curve, float)
    a, d = curve[:-1], np.diff(curve, axis=0)
    denom = np.sum(d * d, axis=1)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(d).all(axis=1) & (denom > 0)
    if not len(d):
        return np.empty((len(sources_xy), 0)), np.empty((len(sources_xy), 0))
    safe = np.where(valid, denom, 1)
    fraction = np.clip(np.sum((sources_xy[:, None, :] - a) * d, axis=2) / safe, 0, 1)
    separation = np.linalg.norm(
        sources_xy[:, None, :] - (a + fraction[:, :, None] * d), axis=2
    )
    separation[:, ~valid] = np.inf
    return separation, fraction


def associate_sources(
    sources, records, projections, *, observer_carrington_rsun, max_residual_arcsec=None
):
    """Return branch diagnostics on every preselected line, with explicit evidence.

    Inputs must share one observation time/observer. The optional residual cutoff
    is an exploratory angular gate, not a statistical confidence interval.
    Caller groups records by their actual UTC before invoking this function.
    """
    source_frame = pd.DataFrame(sources)
    if max_residual_arcsec is not None and (
        not np.isfinite(max_residual_arcsec) or max_residual_arcsec < 0
    ):
        raise ValueError("Angular gate must be finite and nonnegative")
    rows = []
    xy = source_frame[["center_x_arcsec", "center_y_arcsec"]].to_numpy(float)
    for record, projected in zip(records, projections, strict=True):
        if not record["trace_valid"]:
            continue
        points = np.asarray(record["xyz_carrington_rsun"])
        distances, fractions = projected_distances(xy, projected["hpc_arcsec"])
        if not distances.shape[1]:
            continue
        for source_index, (_, source) in enumerate(source_frame.iterrows()):
            dist = distances[source_index]
            # All local minima retain ambiguous branches along the same line.
            local = (
                np.isfinite(dist)
                & (dist <= np.r_[np.inf, dist[:-1]])
                & (dist <= np.r_[dist[1:], np.inf])
            )
            for branch, segment in enumerate(np.flatnonzero(local)):
                origin, ray = hpc_ray(*xy[source_index], observer_carrington_rsun)
                checks = intersect_polyline(
                    origin, ray, points[segment : segment + 2], include_rejected=True
                )
                check = checks[0] if checks else {}
                point = points[segment] + fractions[source_index, segment] * (
                    points[segment + 1] - points[segment]
                )
                # A finite diagnostic endpoint is not promoted to a valid LOS hit.
                in_segment = (
                    0 <= check.get("segment_fraction", np.nan) <= 1
                    and check.get("ray_distance_rsun", -1) >= 0
                )
                numerical = bool(check.get("numerical_valid", False)) and in_segment
                if (
                    numerical
                    and np.isfinite(check.get("axis_point_rsun", np.nan)).all()
                ):
                    point = check["axis_point_rsun"]
                accepted = (
                    numerical
                    and max_residual_arcsec is not None
                    and dist[segment] <= max_residual_arcsec
                )
                rows.append(
                    dict(
                        source_id=source.source_id,
                        time_utc=str(source.time_utc),
                        frequency_mhz=float(source.frequency_mhz),
                        fieldline_id=record["fieldline_id"],
                        scenario_id=record["fieldline_id"],
                        solution_id=f"branch-{branch}",
                        segment_id=int(segment),
                        method="conditional_pfss_field_line",
                        evidence_id=f"fieldline:{record['fieldline_id']};source:{source.source_id}",
                        geometry_valid=bool(accepted),
                        independent_geometry_valid=False,
                        jet_association_verified=False,
                        projected_residual_arcsec=float(dist[segment]),
                        conditional_candidate_height_rsun=float(
                            np.linalg.norm(point) - 1
                        ),
                        height_rsun=(
                            float(np.linalg.norm(point) - 1) if accepted else np.nan
                        ),
                        axis_x_rsun=float(point[0]),
                        axis_y_rsun=float(point[1]),
                        axis_z_rsun=float(point[2]),
                        los_x_rsun=float(check.get("los_point_rsun", [np.nan] * 3)[0]),
                        los_y_rsun=float(check.get("los_point_rsun", [np.nan] * 3)[1]),
                        los_z_rsun=float(check.get("los_point_rsun", [np.nan] * 3)[2]),
                        separation_rsun=float(check.get("miss_rsun", np.nan)),
                        crossing_angle_deg=float(check.get("angle_deg", np.nan)),
                        reason=(
                            "conditional_angular_gate"
                            if accepted
                            else check.get("reason", "no_intersection")
                        ),
                    )
                )
    return pd.DataFrame(rows)


def model_shell_candidates(records, frequencies, *, rss, models=None):
    """Newkirk shells on fixed magnetic lines; no line selection or tuning."""
    models = models or [(m, s) for m in (1, 2, 4) for s in (1, 2)]
    candidates = []
    for record in records:
        if not record["trace_valid"]:
            continue
        for frequency in sorted(set(frequencies)):
            for multiplier, harmonic in models:
                try:
                    radius = float(
                        newkirk_radius_from_frequency_mhz(
                            frequency, multiplier, harmonic
                        )
                    )
                except (ValueError, ZeroDivisionError):
                    radius = np.nan
                valid_radius = np.isfinite(radius) and 1 <= radius <= rss
                intersections = (
                    sphere_intersections(record["xyz_carrington_rsun"], radius)
                    if valid_radius
                    else []
                )
                if intersections:
                    reason = "ok"
                elif not np.isfinite(radius) or radius <= 0:
                    reason = "invalid_frequency_density_inversion"
                elif radius < 1:
                    reason = "below_photosphere"
                elif radius > rss:
                    reason = "outside_pfss_domain"
                else:
                    reason = "no_shell_intersection"
                for index, hit in enumerate(
                    intersections
                    or [{"point_rsun": np.full(3, np.nan), "segment_id": -1}]
                ):
                    candidates.append(
                        dict(
                            fieldline_id=record["fieldline_id"],
                            solution_id=f"shell-{index}",
                            frequency_mhz=float(frequency),
                            multiplier=multiplier,
                            harmonic=harmonic,
                            newkirk_height_rsun=radius - 1,
                            xyz_carrington_rsun=hit["point_rsun"],
                            segment_id=hit["segment_id"],
                            model_valid=bool(intersections),
                            reason=reason,
                            method="pfss_and_newkirk_conditional_forward_model",
                            independent_geometry_valid=False,
                        )
                    )
    return candidates


__all__ = [
    "sphere_intersections",
    "projected_distances",
    "associate_sources",
    "model_shell_candidates",
]
