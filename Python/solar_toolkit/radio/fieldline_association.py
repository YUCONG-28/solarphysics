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


def _band_values(value, unit, name):
    from solar_toolkit._utils.numeric import as_value

    if np.asanyarray(value).dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain real numerical values")
    return as_value(value, unit, name)


def _band_scalar(value, unit, name):
    result = _band_values(value, unit, name)
    if (
        isinstance(value, (bool, np.bool_))
        or result.ndim
        or not np.isfinite(result)
        or result <= 0
    ):
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(result)


def _edge_sphere_fractions(start, direction, length, radius):
    """Both chord/sphere roots; tangent roots are retained for diagnostics."""
    along = float(start @ direction)
    perpendicular = start - along * direction
    square = radius * radius - float(perpendicular @ perpendicular)
    tolerance = (
        64 * np.finfo(float).eps * max(1.0, radius * radius, float(start @ start))
    )
    if square < -tolerance:
        return []
    half = np.sqrt(max(0.0, square))
    roots = [(-along - half) / length, (-along + half) / length]
    return [float(np.clip(t, 0, 1)) for t in roots if -1e-12 <= t <= 1 + 1e-12]


def fieldline_band_segments(
    xyz_carrington_rsun,
    frequency_bounds_mhz,
    *,
    mechanism,
    harmonic,
    rss,
    newkirk_multiplier=None,
    field_strength_gauss=None,
):
    """Clip original 3D edges to a conditionally predicted emission band.

    Coordinates are ``(N, 3)`` Carrington Cartesian solar radii. Frequency
    bounds are an increasing positive MHz pair, ``rss > 1`` is a solar radius,
    and the emission harmonic is a positive integer. Compatible Quantities
    are converted explicitly. Inputs are copied; masked/nonfinite coordinates
    break paths rather than being joined or filled.

    ``plasma`` requires an explicit positive Newkirk multiplier and intersects
    each straight edge with exact radial model shells. ``cyclotron`` requires
    an aligned ``(N,)`` field-magnitude array in Gauss, using
    ``f_MHz = 2.8 * harmonic * B_G`` and linear interpolation of sampled B on
    each edge. Negative/nonfinite B samples break paths. The rounded gyro
    coefficient follows the NRL Plasma Formulary. Unused model inputs are not
    consumed. No density, magnetic field, calibration or mechanism is inferred.

    Return ``{'segments': [...], 'metadata': {...}}``. Each segment contains
    ``xyz_carrington_rsun`` (M,3), ``frequency_mhz`` (M,), original
    ``source_edge_indices`` (M-1,) and ``source_edge_fractions`` (M-1,2).
    Adjacent supported edges stay joined; gaps remain separate. Both models
    are clipped to ``1 <= r <= rss``. Isolated boundary touches and paths
    shorter than 1e-12 R_sun are omitted: they have no finite path support.
    ``omitted_point_touches`` counts contacts per original edge. Constant-B
    edges lying on a band boundary are finite supported segments and remain.

    This is a forward-model candidate, never an independent measured height or
    evidence that the selected emission mechanism operates. Missing B gives
    ``field_strength_unavailable`` without a radial approximation. Invalid
    configuration or misaligned arrays raises ValueError, including for empty
    input. Old shell/source-association interfaces are unchanged.
    """
    from astropy import units as u

    from .newkirk import PLASMA_FREQ_COEFF_KHZ, newkirk_density_cm3

    if mechanism not in {"plasma", "cyclotron"}:
        raise ValueError("mechanism must be 'plasma' or 'cyclotron'")
    harmonic = _band_scalar(harmonic, u.dimensionless_unscaled, "harmonic")
    if not harmonic.is_integer():
        raise ValueError("harmonic must be a positive integer")
    harmonic = int(harmonic)
    rss = _band_scalar(rss, u.R_sun, "rss")
    if rss <= 1:
        raise ValueError("rss must be greater than one solar radius")
    bounds = _band_values(frequency_bounds_mhz, u.MHz, "frequency_bounds_mhz")
    if (
        bounds.shape != (2,)
        or not np.isfinite(bounds).all()
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError(
            "frequency_bounds_mhz must be two increasing positive MHz values"
        )
    multiplier = None
    if mechanism == "plasma":
        if newkirk_multiplier is None:
            raise ValueError("plasma requires explicit newkirk_multiplier")
        multiplier = _band_scalar(
            newkirk_multiplier, u.dimensionless_unscaled, "newkirk_multiplier"
        )
    points = _band_values(xyz_carrington_rsun, u.R_sun, "xyz_carrington_rsun")
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("xyz_carrington_rsun must have shape (N, 3)")
    metadata = dict(
        mechanism=mechanism,
        harmonic=harmonic,
        rss=rss,
        frequency_bounds_mhz=bounds.tolist(),
        newkirk_multiplier=multiplier,
        geometry_method="conditional_model_candidate",
        independent_geometry_valid=False,
        status="no_in_band_segment",
        input_point_count=len(points),
        invalid_coordinate_count=int(
            np.count_nonzero(~np.isfinite(points).all(axis=1))
        ),
        omitted_point_touches=0,
        zero_length_tolerance_rsun=1e-12,
        interpolation=(
            "exact Newkirk radial shells on piecewise-linear 3D edges"
            if mechanism == "plasma"
            else "linear sampled field magnitude on piecewise-linear 3D edges"
        ),
    )
    result = {"segments": [], "metadata": metadata}
    strength = None
    shell_inner, shell_outer = 1.0, rss
    if mechanism == "cyclotron":
        if field_strength_gauss is None:
            metadata["status"] = "field_strength_unavailable"
            return result
        strength = _band_values(field_strength_gauss, u.G, "field_strength_gauss")
        if strength.shape != (len(points),):
            raise ValueError(
                "field_strength_gauss must have shape (N,) aligned with xyz"
            )
        strength[strength < 0] = np.nan
        metadata["invalid_field_count"] = int(np.count_nonzero(~np.isfinite(strength)))
        metadata["gyrofrequency_mhz_per_gauss"] = 2.8
        if not np.isfinite(strength).any():
            metadata["status"] = "field_strength_unavailable"
            return result
    else:
        domain_frequency = (
            harmonic
            * PLASMA_FREQ_COEFF_KHZ
            / 1000
            * np.sqrt(newkirk_density_cm3([rss, 1.0], multiplier))
        )
        if not np.isfinite(domain_frequency).all():
            raise ValueError("Model frequencies overflow within the PFSS domain")
        low, high = max(bounds[0], domain_frequency[0]), min(
            bounds[1], domain_frequency[1]
        )
        if low > high:
            return result
        shell_inner, shell_outer = np.clip(
            newkirk_radius_from_frequency_mhz([high, low], multiplier, harmonic),
            1.0,
            rss,
        )

    def frequency_at(index, fraction, point):
        if mechanism == "plasma":
            return float(
                harmonic
                * PLASMA_FREQ_COEFF_KHZ
                / 1000
                * np.sqrt(newkirk_density_cm3(np.linalg.norm(point), multiplier))
            )
        return float(
            2.8
            * harmonic
            * (strength[index] + fraction * (strength[index + 1] - strength[index]))
        )

    def inside(index, fraction, point):
        radius = np.linalg.norm(point)
        tolerance = 64 * np.finfo(float).eps * max(1.0, rss)
        if not shell_inner - tolerance <= radius <= shell_outer + tolerance:
            return False
        if mechanism == "plasma":
            return True
        freq = frequency_at(index, fraction, point)
        tolerance = 64 * np.finfo(float).eps * max(1.0, bounds[-1])
        return (
            np.isfinite(freq) and bounds[0] - tolerance <= freq <= bounds[1] + tolerance
        )

    current = None
    for index in range(max(0, len(points) - 1)):
        start, end = points[index : index + 2]
        if not np.isfinite([start, end]).all() or (
            strength is not None and not np.isfinite(strength[index : index + 2]).all()
        ):
            current = None
            continue
        delta = end - start
        length = np.linalg.norm(delta)
        if not np.isfinite(length) or length <= 1e-12:
            current = None
            continue
        knots = [0.0, 1.0]
        for radius in {1.0, rss, float(shell_inner), float(shell_outer)}:
            knots.extend(_edge_sphere_fractions(start, delta / length, length, radius))
        if strength is not None and strength[index] != strength[index + 1]:
            frequencies = 2.8 * harmonic * strength[index : index + 2]
            for bound in bounds:
                fraction = (bound - frequencies[0]) / (frequencies[1] - frequencies[0])
                if 0 <= fraction <= 1:
                    knots.append(float(fraction))
        knots = sorted(set(knots))
        intervals = [
            (left, right)
            for left, right in zip(knots[:-1], knots[1:], strict=True)
            if (right - left) * length > 1e-12
            and inside(index, (left + right) / 2, start + (left + right) / 2 * delta)
        ]
        for fraction in knots:
            if inside(index, fraction, start + fraction * delta) and not any(
                left - 1e-12 <= fraction <= right + 1e-12 for left, right in intervals
            ):
                metadata["omitted_point_touches"] += 1
        for left, right in intervals:
            a, b = start + left * delta, start + right * delta
            fa, fb = frequency_at(index, left, a), frequency_at(index, right, b)
            join = (
                current is not None
                and np.linalg.norm(current["xyz_carrington_rsun"][-1] - a) <= 1e-12
            )
            if join:
                previous = current["source_edge_indices"][-1]
                join = previous == index or (
                    previous == index - 1
                    and current["source_edge_fractions"][-1][1] >= 1 - 1e-12
                    and left <= 1e-12
                )
            if not join:
                current = dict(
                    xyz_carrington_rsun=[a],
                    frequency_mhz=[fa],
                    source_edge_indices=[],
                    source_edge_fractions=[],
                )
                result["segments"].append(current)
            current["xyz_carrington_rsun"].append(b)
            current["frequency_mhz"].append(fb)
            current["source_edge_indices"].append(index)
            current["source_edge_fractions"].append([left, right])
        if not intervals:
            current = None
    for segment in result["segments"]:
        for key in segment:
            segment[key] = np.asarray(
                segment[key], dtype=np.int64 if key == "source_edge_indices" else float
            )
    if result["segments"]:
        metadata["status"] = "ok"
    return result


__all__ = [
    "sphere_intersections",
    "projected_distances",
    "associate_sources",
    "model_shell_candidates",
    "fieldline_band_segments",
]
