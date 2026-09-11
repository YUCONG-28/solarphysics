"""Independent line-of-sight geometry, observation identity and burst sampling.

Cartesian vectors share a heliocentric frame and are measured in solar radii.
No density model is used to determine a source position. Conditional geometry
must never be promoted to an independently measured radio source position.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .io import parse_datetime_value, truthy

__all__ = [
    "normalize_source_observations",
    "match_burst_crossings",
    "hpc_ray",
    "project_hpc",
    "triangulate_rays",
    "ray_height_bound",
    "intersect_radial_direction",
    "intersect_polyline",
    "compare_geometry_newkirk",
    "position_on_inclination_cone",
    "triangulate_tie_point",
]

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    if (
        vector.shape != (3,)
        or not np.isfinite(vector).all()
        or np.linalg.norm(vector) == 0
    ):
        raise ValueError("Expected a finite, nonzero three-vector")
    return vector / np.linalg.norm(vector)


def normalize_source_observations(frame):
    """Return one deterministically selected fit per observation/component.

    Output timestamps are timezone-aware UTC; input strings are retained. A
    missing component denotes the single fitted component, not all components.
    """
    df = pd.DataFrame(frame).copy()
    df["time_original"] = df["time"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time"].map(parse_datetime_value), utc=True)
    df["frequency_mhz"] = pd.to_numeric(
        df.get("frequency_mhz", df.get("freq")), errors="coerce"
    )
    for col, default in [
        ("polarization", "unknown"),
        ("component_id", "single"),
        ("instrument", "radio"),
    ]:
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default).astype(str)
    valid = df.time_utc.notna() & np.isfinite(df.frequency_mhz)
    for col in ["center_x_arcsec", "center_y_arcsec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        valid &= np.isfinite(df[col])
    for col in ["trajectory_valid", "overlay_valid", "fwhm_valid"]:
        if col in df:
            valid &= df[col].fillna(False).map(truthy)
    if "quality_flag" in df:
        valid &= df.quality_flag.fillna("").isin(["", "ok"])
    df["source_valid"] = valid
    keys = ["time_utc", "frequency_mhz", "polarization", "component_id", "instrument"]
    # Invalid timestamps have no reliable shared identity: retain them separately.
    identities = []
    for n, (_, row) in enumerate(df.iterrows()):
        identity = "|".join(str(row[k]) for k in keys)
        if pd.isna(row.time_utc):
            identity += f"|invalid-row-{n}"
        identities.append(hashlib.sha256(identity.encode()).hexdigest()[:20])
    df["source_id"] = identities
    df["duplicate_count"] = df.groupby("source_id")["source_id"].transform("size") - 1
    df["_residual"] = pd.to_numeric(
        df.get("residual_rms", pd.Series(np.inf, index=df.index)), errors="coerce"
    ).fillna(np.inf)
    df = df.sort_values(
        ["source_valid", "_residual"], ascending=[False, True], kind="stable"
    )
    return (
        df.drop_duplicates("source_id")
        .drop(columns="_residual")
        .sort_values(["time_utc", "frequency_mhz"])
        .reset_index(drop=True)
    )


def match_burst_crossings(sources, drifts, *, clock_offset_s=0.0, clock_sigma_s=0.0):
    """Match real segment crossings only; positive offset adds to drift times.

    No endpoint/frequency extrapolation. Cadence is estimated from all frames,
    before quality filtering. All failures remain as auditable table rows.
    """
    if clock_sigma_s < 0 or not np.isfinite([clock_offset_s, clock_sigma_s]).all():
        raise ValueError("Clock offset must be finite and uncertainty nonnegative")
    rows = []
    for _, burst in pd.DataFrame(drifts).iterrows():
        t0, t1 = [parse_datetime_value(burst.get(k)) for k in ["t_start", "t_end"]]
        f0, f1 = [float(burst.get(k, np.nan)) for k in ["f_start_mhz", "f_end_mhz"]]
        for freq, group in sources.groupby("frequency_mhz"):
            result = dict(
                burst_id=str(burst.get("label", "")),
                frequency_mhz=float(freq),
                source_id="",
                match_valid=False,
                match_reason="no_frequency_crossing",
                crossing_time_utc="",
                delta_time_s=np.nan,
                tolerance_s=np.nan,
                clock_offset_s=clock_offset_s,
                clock_sigma_s=clock_sigma_s,
            )
            if (
                t0 is None
                or t1 is None
                or not np.isfinite([f0, f1]).all()
                or f0 == f1
                or t0 == t1
            ):
                result["match_reason"] = "invalid_drift_segment"
            elif min(f0, f1) <= freq <= max(f0, f1):
                crossing = (
                    pd.Timestamp(t0, tz="UTC")
                    + (pd.Timestamp(t1) - pd.Timestamp(t0)) * ((freq - f0) / (f1 - f0))
                    + pd.Timedelta(seconds=clock_offset_s)
                )
                result["crossing_time_utc"] = crossing.isoformat()
                times = group.time_utc.dropna().drop_duplicates().sort_values()
                dt = times.diff().dt.total_seconds().dropna()
                cadence = float(dt[dt > 0].median()) if len(dt) else np.nan
                result["tolerance_s"] = cadence / 2 + clock_sigma_s
                eligible = group[group.source_valid & group.time_utc.notna()]
                if eligible.empty:
                    result["match_reason"] = "no_valid_frame"
                elif not np.isfinite(cadence):
                    result["match_reason"] = "unknown_cadence"
                else:
                    delta = (eligible.time_utc - crossing).dt.total_seconds()
                    smallest = delta.abs().min()
                    candidates = eligible.loc[
                        np.isclose(delta.abs(), smallest, atol=1e-9, rtol=0)
                    ]
                    result["delta_time_s"] = float(delta.loc[candidates.index[0]])
                    if smallest > result["tolerance_s"] + 1e-9:
                        result["match_reason"] = "no_frame_within_tolerance"
                    elif len(candidates) != 1:
                        result["match_reason"] = "ambiguous_nearest_frame"
                    else:
                        result.update(
                            source_id=candidates.iloc[0].source_id,
                            match_valid=True,
                            match_reason="matched",
                        )
            rows.append(result)
    out = pd.DataFrame(rows)
    if len(out):
        matched = out.match_valid
        conflicts = out.loc[matched].groupby("source_id").burst_id.nunique()
        out.loc[
            out.source_id.isin(conflicts[conflicts > 1].index),
            ["match_valid", "match_reason"],
        ] = [False, "ambiguous_burst"]
    return out


def hpc_ray(tx_arcsec, ty_arcsec, observer_rsun, north=(0, 0, 1)):
    """Exact Thompson HPC ray in an arbitrary heliocentric Cartesian frame."""
    observer = np.asarray(observer_rsun, dtype=float)
    ez = _unit(observer)
    ex = _unit(np.cross(_unit(north), ez))
    ey = _unit(np.cross(ez, ex))
    tx, ty = np.asarray([tx_arcsec, ty_arcsec], dtype=float) / ARCSEC_PER_RAD
    ray = np.cos(ty) * np.sin(tx) * ex + np.sin(ty) * ey - np.cos(ty) * np.cos(tx) * ez
    return observer, _unit(ray)


def project_hpc(point_rsun, observer_rsun, north=(0, 0, 1)):
    ez = _unit(observer_rsun)
    ex = _unit(np.cross(_unit(north), ez))
    ey = _unit(np.cross(ez, ex))
    direction = _unit(np.asarray(point_rsun) - np.asarray(observer_rsun))
    return (
        np.array(
            [
                np.arctan2(direction @ ex, -direction @ ez),
                np.arcsin(np.clip(direction @ ey, -1, 1)),
            ]
        )
        * ARCSEC_PER_RAD
    )


def triangulate_rays(origin_a, ray_a, origin_b, ray_b, *, min_angle_deg=1.0):
    """Closest positive ray points; separation is not a positional uncertainty."""
    a, b = np.asarray(origin_a, float), np.asarray(origin_b, float)
    u, v = _unit(ray_a), _unit(ray_b)
    angle = np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1)))
    if angle < min_angle_deg:
        return dict(
            valid=False,
            reason="weak_parallax",
            angle_deg=angle,
            point_rsun=np.full(3, np.nan),
            separation_rsun=np.nan,
        )
    distances = np.linalg.lstsq(np.column_stack((u, -v)), b - a, rcond=None)[0]
    pa, pb = a + distances[0] * u, b + distances[1] * v
    valid = bool((distances >= 0).all())
    return dict(
        valid=valid,
        reason="ok" if valid else "behind_observer",
        angle_deg=angle,
        point_rsun=(pa + pb) / 2,
        separation_rsun=float(np.linalg.norm(pa - pb)),
    )


def triangulate_tie_point(
    hpc_a,
    hpc_b,
    observer_a,
    observer_b,
    *,
    sigma_a_arcsec=None,
    sigma_b_arcsec=None,
    correspondence_verified=False,
    samples=500,
    seed=0,
):
    """3-sigma reprojection gate and conditional 95% localization interval.

    Supplied sigmas must include registration/feature-location uncertainty.
    Sampling assumes independent Gaussian angular errors and fixed observers;
    it does not estimate shared calibration, evolution or radio propagation.
    Geometric consistency alone does not verify correspondence identity.
    """
    xy = np.asarray([hpc_a, hpc_b], float)
    oa, ra = hpc_ray(*xy[0], observer_a)
    ob, rb = hpc_ray(*xy[1], observer_b)
    result = triangulate_rays(oa, ra, ob, rb)
    result.update(
        geometry_valid=False,
        correspondence_verified=correspondence_verified,
        height_interval_rsun=(np.nan, np.nan),
        covariance_rsun2=np.full((3, 3), np.nan),
    )
    if not result["valid"]:
        return result
    predicted = np.array(
        [
            project_hpc(result["point_rsun"], observer_a),
            project_hpc(result["point_rsun"], observer_b),
        ]
    )
    result["reprojection_residual_arcsec"] = predicted - xy
    if sigma_a_arcsec is None or sigma_b_arcsec is None:
        result["reason"] = "localization_uncertainty_unavailable"
        return result
    sigmas = np.array([sigma_a_arcsec, sigma_b_arcsec], float)[:, None]
    if not np.isfinite(sigmas).all() or (sigmas <= 0).any():
        raise ValueError("Localization sigmas must be finite and positive")
    result["max_reprojection_sigma"] = float(np.max(abs((predicted - xy) / sigmas)))
    if result["max_reprojection_sigma"] > 3:
        result["reason"] = "reprojection_exceeds_3sigma"
        return result
    if not correspondence_verified:
        result["reason"] = "correspondence_unverified"
        return result
    if samples < 100:
        raise ValueError("Use at least 100 uncertainty samples")
    rng = np.random.default_rng(seed)
    positions = []
    for drawn in rng.normal(xy, sigmas, size=(samples, 2, 2)):
        a, u = hpc_ray(*drawn[0], observer_a)
        b, v = hpc_ray(*drawn[1], observer_b)
        sample = triangulate_rays(a, u, b, v)
        if sample["valid"]:
            positions.append(sample["point_rsun"])
    if len(positions) < 0.9 * samples:
        result["reason"] = "unstable_geometry_under_localization_errors"
        return result
    positions = np.asarray(positions)
    result.update(
        geometry_valid=True,
        reason="ok",
        covariance_rsun2=np.cov(positions.T),
        height_interval_rsun=tuple(
            np.quantile(np.linalg.norm(positions, axis=1) - 1, [0.025, 0.975])
        ),
    )
    return result


def ray_height_bound(origin, ray):
    """Geometric apparent-source lower bound, with no finite upper bound."""
    origin, ray = np.asarray(origin, float), _unit(ray)
    closest = origin + max(0.0, -origin @ ray) * ray
    return max(0.0, float(np.linalg.norm(closest) - 1)), np.inf


def position_on_inclination_cone(origin, ray, theta_deg):
    """Exact apparent-source position for an ASSUMED heliocentric viewing angle.

    theta is the angle between the solar-center-to-source and observer vectors.
    Different image position angles lie on different generators of this cone;
    this is not a single flux tube or a measured radio depth.
    """
    origin, ray = np.asarray(origin, float), _unit(ray)
    gamma = np.arccos(np.clip(-ray @ _unit(origin), -1, 1))
    theta = np.radians(float(theta_deg))
    if not 0 < theta < np.pi - gamma:
        raise ValueError("Inclination does not admit a finite forward source")
    distance = np.linalg.norm(origin) * np.sin(theta) / np.sin(theta + gamma)
    return origin + distance * ray


def _axis_closest_points(origin, ray, anchor, direction, min_angle_deg):
    """Numerical line solution only; neither domain nor evidence is accepted."""
    origin, anchor = np.asarray(origin, float), np.asarray(anchor, float)
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (origin, anchor)):
        raise ValueError("Expected finite three-vector origins")
    ray, direction = _unit(ray), _unit(direction)
    angle = float(np.degrees(np.arccos(np.clip(abs(ray @ direction), 0, 1))))
    result = dict(
        numerical_valid=False,
        valid=False,
        geometry_valid=False,
        independent_geometry_valid=False,
        reason="weak_crossing_angle",
        angle_deg=angle,
        point_rsun=np.full(3, np.nan),
        los_point_rsun=np.full(3, np.nan),
        axis_point_rsun=np.full(3, np.nan),
        height_rsun=np.nan,
        ray_distance_rsun=np.nan,
        axis_distance_rsun=np.nan,
        separation_rsun=np.nan,
        miss_rsun=np.nan,
    )
    if angle < min_angle_deg:
        return result
    distance, along = np.linalg.lstsq(
        np.column_stack((ray, -direction)), anchor - origin, rcond=None
    )[0]
    los_point, axis_point = origin + distance * ray, anchor + along * direction
    miss = float(np.linalg.norm(los_point - axis_point))
    result.update(
        numerical_valid=True,
        reason="compatibility_threshold_unavailable",
        los_point_rsun=los_point,
        axis_point_rsun=axis_point,
        ray_distance_rsun=float(distance),
        axis_distance_rsun=float(along),
        separation_rsun=miss,
        miss_rsun=miss,
    )
    return result


def _validate_axis_gate(max_miss_rsun, min_angle_deg):
    if not np.isfinite(min_angle_deg) or not 0 < min_angle_deg <= 90:
        raise ValueError("Minimum crossing angle must be in (0, 90] degrees")
    if max_miss_rsun is not None and (
        not np.isfinite(max_miss_rsun) or max_miss_rsun < 0
    ):
        raise ValueError("Miss threshold must be finite and nonnegative")


def _accept_axis_solution(
    result, max_miss_rsun, association_verified, axis_independent
):
    if max_miss_rsun is None:
        return
    if result["miss_rsun"] > max_miss_rsun:
        result["reason"] = "miss_exceeds_threshold"
        return
    result.update(
        valid=True,
        geometry_valid=True,
        independent_geometry_valid=bool(association_verified and axis_independent),
        reason="ok",
        point_rsun=result["axis_point_rsun"].copy(),
        height_rsun=float(np.linalg.norm(result["axis_point_rsun"]) - 1),
    )


def intersect_radial_direction(
    origin,
    ray,
    radial_direction,
    *,
    max_miss_rsun=None,
    min_angle_deg=1.0,
    association_verified=False,
    axis_independent=False,
):
    """Closest LOS/radial-axis points, restricted to radial distance >= 1.

    A supplied miss threshold gates conditional geometry; the caller must derive
    it from the positional/axis uncertainties appropriate to the observation.
    No uncertainty, sigma interpretation or source association is invented.
    ``point_rsun`` is the accepted axis point, never the midpoint of skew lines.
    Without a threshold, only diagnostic closest points are returned. Independent
    validity additionally requires an independently measured axis AND verified
    radio association. An assumed photospheric radial axis is conditional.
    """
    _validate_axis_gate(max_miss_rsun, min_angle_deg)
    result = _axis_closest_points(
        origin, ray, np.zeros(3), radial_direction, min_angle_deg
    )
    result.update(
        association_verified=bool(association_verified),
        axis_independent=bool(axis_independent),
        max_miss_rsun=max_miss_rsun,
    )
    if not result["numerical_valid"]:
        return result
    if result["ray_distance_rsun"] < 0:
        result["reason"] = "behind_observer"
    elif result["axis_distance_rsun"] < 1 - 1e-12:
        result["reason"] = "below_photosphere"
    else:
        if result["axis_distance_rsun"] < 1:
            result["axis_distance_rsun"] = 1.0
            result["axis_point_rsun"] = _unit(radial_direction)
            result["miss_rsun"] = result["separation_rsun"] = float(
                np.linalg.norm(result["los_point_rsun"] - result["axis_point_rsun"])
            )
        _accept_axis_solution(
            result, max_miss_rsun, association_verified, axis_independent
        )
    return result


def intersect_polyline(
    origin,
    ray,
    points,
    *,
    max_miss_rsun=None,
    min_angle_deg=1.0,
    association_verified=False,
    axis_independent=False,
    include_rejected=False,
):
    """Return compatible finite-segment solutions, retaining distinct depths.

    ``include_rejected`` exposes weak-angle/domain/miss diagnostics; its invalid
    rows must not enter height statistics. Closest diagnostic points may lie on
    the segment's infinite supporting line, but are never accepted outside the
    measured segment. Conditional extensions must be supplied separately.
    Shared-vertex hits on adjacent segments are one solution with ``segment_ids``.
    Threshold and evidence semantics match ``intersect_radial_direction``.
    """
    _validate_axis_gate(max_miss_rsun, min_angle_deg)
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Polyline points must be a finite N by 3 array")
    if len(points) < 2:
        raise ValueError("Polyline requires at least two points")
    solutions = []
    for i, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
        length = float(np.linalg.norm(b - a))
        if length == 0:
            continue
        result = _axis_closest_points(origin, ray, a, b - a, min_angle_deg)
        result.update(
            segment_id=i,
            segment_ids=[i],
            segment_fraction=np.nan,
            association_verified=bool(association_verified),
            axis_independent=bool(axis_independent),
            max_miss_rsun=max_miss_rsun,
        )
        if result["numerical_valid"]:
            fraction = result["axis_distance_rsun"] / length
            result["segment_fraction"] = float(fraction)
            if result["ray_distance_rsun"] < 0:
                result["reason"] = "behind_observer"
            elif not -1e-12 <= fraction <= 1 + 1e-12:
                result["reason"] = "outside_measured_segment"
            else:
                # Snap machine-roundoff at endpoints, never extend the structure.
                result["segment_fraction"] = float(np.clip(fraction, 0, 1))
                result["axis_point_rsun"] = a + result["segment_fraction"] * (b - a)
                result["miss_rsun"] = result["separation_rsun"] = float(
                    np.linalg.norm(result["los_point_rsun"] - result["axis_point_rsun"])
                )
                _accept_axis_solution(
                    result, max_miss_rsun, association_verified, axis_independent
                )
        if result["geometry_valid"]:
            previous = next(
                (
                    s
                    for s in solutions
                    if s["geometry_valid"]
                    and i - 1 in s["segment_ids"]
                    and np.allclose(s["axis_point_rsun"], a, atol=1e-10, rtol=0)
                    and np.allclose(result["axis_point_rsun"], a, atol=1e-10, rtol=0)
                    and np.allclose(
                        s["los_point_rsun"],
                        result["los_point_rsun"],
                        atol=1e-10,
                        rtol=0,
                    )
                ),
                None,
            )
            if previous is not None:
                previous["segment_ids"].append(i)
                continue
        if result["geometry_valid"] or include_rejected:
            solutions.append(result)
    return solutions


def compare_geometry_newkirk(sources, geometry, models=None):
    """Join independent geometry 1:1 by source_id; keep intervals and invalidity.

    Geometry requires method/evidence_id/geometry_valid and height_rsun columns.
    An unbounded geometric constraint is not a measured height or a model rank.
    """
    from .newkirk import newkirk_radius_from_frequency_mhz

    models = models or [(n, s) for n in (1.0, 2.0, 4.0) for s in (1, 2)]
    geometry = pd.DataFrame(geometry)
    if geometry.source_id.duplicated().any():
        raise ValueError(
            "Geometry must contain one solution per source_id; compare scenarios separately"
        )
    df = sources.merge(geometry, on="source_id", how="left", validate="one_to_one")
    rows = []
    for _, row in df.iterrows():
        for multiplier, harmonic in models:
            height = float(
                newkirk_radius_from_frequency_mhz(
                    row.frequency_mhz, multiplier=multiplier, harmonic=harmonic
                )
                - 1
            )
            result = row.to_dict()
            valid = (
                bool(row.source_valid)
                and pd.notna(row.get("geometry_valid"))
                and truthy(row.get("geometry_valid"))
                and np.isfinite(row.get("height_rsun", np.nan))
            )
            result.update(
                newkirk_multiplier=multiplier,
                harmonic=harmonic,
                effective_density_factor=multiplier * harmonic**2,
                newkirk_height_rsun=height,
                comparison_valid=valid,
                height_residual_3d_rsun=(
                    float(row.height_rsun - height) if valid else np.nan
                ),
            )
            rows.append(result)
    return pd.DataFrame(rows)
