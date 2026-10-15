"""Actual-frame pairing and geometry prechecks, independent of display shells."""

from collections import Counter
from decimal import Decimal

import numpy as np
from sunpy.coordinates import frames

from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
)

from .jet_annotations import pixel_hpc
from .jet_viewpoint import observer


def pair_frames(records, master="auto", band=None):
    """Cover all real frames, using UTC order and explicit centre-light-time cost.

    ``master`` is retained as a legacy UI argument. The smallest nonempty stream
    supplies the root and the largest last stream supplies the complete row axis.
    Tolerance is a quality flag, and never removes a selected counterpart.
    """
    records = [r for r in records if band is None or float(r["band"]) == float(band)]
    if master not in ("auto", "AIA", "EUVI", None):
        raise ValueError("master must be auto, AIA or EUVI")
    if any(r["instrument"] not in ("AIA", "EUVI") for r in records):
        raise ValueError("Jet timeline supports AIA and EUVI observations")
    if len({float(r["band"]) for r in records}) > 1:
        raise ValueError("Select one band before matching the Jet timeline")
    if not records:
        return []
    frames = {"AIA": [], "EUVI": []}
    for record in records:
        stamp = normalize_time(record["midpoint_utc"], source="FITS:exposure_midpoint")
        distance = Decimal(str(record["dsun_m"]))
        if not distance.is_finite() or distance <= 0:
            raise ValueError("A positive finite observer distance is required")
        emission = stamp.shift(
            -distance / Decimal(299792458), source="solar_center_emission"
        )
        frames[record["instrument"]].append(
            TimeFrame(
                record["image_sha256"],
                stamp,
                payload=record,
                match_time=emission,
            )
        )
    tolerances = {}
    for instrument, values in frames.items():
        times = sorted({frame.match_time.tai_ns for frame in values})
        intervals = sorted(b - a for a, b in zip(times[:-1], times[1:], strict=True))
        if intervals:
            middle = len(intervals) // 2
            median = (
                Decimal(intervals[middle])
                if len(intervals) % 2
                else (Decimal(intervals[middle - 1]) + Decimal(intervals[middle])) / 2
            )
            tolerances[instrument] = median / Decimal(2_000_000_000)
        else:
            tolerances[instrument] = None
    plan = build_sequence_match_plan(
        [
            TimeStream(name, values, tolerance_seconds=tolerances[name] or 0)
            for name, values in frames.items()
        ]
    )
    result = []
    other = "EUVI" if plan.output_source_id == "AIA" else "AIA"
    tolerance = tolerances[other]
    for row in plan.rows:
        selected = row.selections[other]
        paired = selected.frame is not None
        delta = (
            row.selections["AIA"].match_time.tai_ns
            - row.selections["EUVI"].match_time.tai_ns
            if paired
            else None
        )
        within = selected.within_tolerance if tolerance is not None and paired else None
        status = (
            "no_other_frame"
            if not paired
            else "cadence_unavailable"
            if tolerance is None
            else "matched"
            if within
            else "outside_tolerance"
        )
        out = {
            "master_id": row.master_frame.frame_id,
            "master_instrument": plan.master_source_id,
            "output_id": row.output_frame.frame_id,
            "output_instrument": plan.output_source_id,
            "output_utc": row.time.utc,
            "reference_instrument": other,
            "AIA": row.selections["AIA"].frame.payload
            if row.selections["AIA"].frame
            else None,
            "EUVI": row.selections["EUVI"].frame.payload
            if row.selections["EUVI"].frame
            else None,
            "tolerance_s": float(tolerance) if tolerance is not None else None,
            "status": status,
            "within_tolerance": within,
            "delta_emission_s": delta / 1e9 if delta is not None else None,
            "delta_emission_ns": delta,
            "repeated_other": selected.reused,
            "reference_use_count": selected.use_count,
            "time_match_policy": "hierarchical_monotone_full_coverage_solar_center_emission_cost",
            "algorithm_version": 3,
            "source_order": list(plan.source_order),
            "parent_matches": row.to_dict()["parent_matches"],
            "time_match_fingerprint": plan.fingerprint,
            "selected_indices": {
                name: {
                    "original_index": item.original_index,
                    "sorted_index": item.sorted_index,
                }
                for name, item in row.selections.items()
            },
        }
        result.append(out)
    return result


def geometry_precheck(documents):
    """Return residual evidence, never infer correspondence/3-sigma validity."""
    from solar_toolkit.radio.source_geometry import (
        hpc_ray,
        project_hpc,
        triangulate_rays,
    )

    if len(documents) != 2 or any(d is None for d in documents):
        return {"geometry_valid": False, "issues": ["missing_view"], "points": []}
    if documents[0].sha256 == documents[1].sha256:
        return {
            "geometry_valid": False,
            "issues": ["same_native_image_in_both_views"],
            "points": [],
            "display_geometry_used": False,
        }
    from .euvi_preprocessing import geometry_issues

    issues = [
        f"view_{i}:{reason}"
        for i, d in enumerate(documents)
        for reason in geometry_issues(d.map)
    ]
    if issues:
        return {
            "geometry_valid": False,
            "issues": issues,
            "points": [],
            "display_geometry_used": False,
        }
    frame = frames.HeliographicStonyhurst(obstime=documents[0].map.date)
    observers = [observer(d.map, frame) for d in documents]
    issues = []
    ties = []
    for v, d in enumerate(documents):
        counts = Counter(t["number"] for t in d.state["tiepoints"])
        if any(n > 1 for n in counts.values()):
            issues.append(f"view_{v}_duplicate_numbers")
        ties.append(
            {t["number"]: t for t in d.state["tiepoints"] if counts[t["number"]] == 1}
        )
        if d.state["axis_status"] == "pending":
            issues.append(f"view_{v}_axis_pending")
        if not d.state["axis"]:
            issues.append(f"view_{v}_axis_missing")
    rows = []
    for n in sorted(set(ties[0]) | set(ties[1])):
        row = {"number": n, "geometry_valid": False, "reason": "missing_counterpart"}
        if n in ties[0] and n in ties[1]:
            xy = [
                pixel_hpc(d.map, [t[n]["pixel_xy"]])[0]
                for d, t in zip(documents, ties, strict=True)
            ]
            rays = [hpc_ray(*p, o)[1] for p, o in zip(xy, observers, strict=True)]
            g = triangulate_rays(observers[0], rays[0], observers[1], rays[1])
            row.update(reason=g["reason"], ray_angle_deg=g["angle_deg"])
            if g["valid"]:
                row.update(
                    reason="localization_sigma_and_evolution_not_available",
                    separation_Mm=g["separation_rsun"] * 695.7,
                    residual_arcsec=[
                        float(np.linalg.norm(project_hpc(g["point_rsun"], o) - p))
                        for o, p in zip(observers, xy, strict=True)
                    ],
                )
                for v, d in enumerate(documents):
                    axis = pixel_hpc(d.map, d.state["axis"])
                    row[f"view_{v}_nearest_axis_index"] = (
                        int(np.linalg.norm(axis - xy[v], axis=1).argmin())
                        if len(axis)
                        else None
                    )
        rows.append(row)
    # Diagnose order on both axes without manufacturing equal-arc-length matches.
    both = [
        r
        for r in rows
        if r.get("view_0_nearest_axis_index") is not None
        and r.get("view_1_nearest_axis_index") is not None
    ]
    ordered = sorted(both, key=lambda r: r["view_0_nearest_axis_index"])
    if len(ordered) > 1 and any(
        np.diff([r["view_1_nearest_axis_index"] for r in ordered]) < 0
    ):
        issues.append("axis_order_disagreement")
    return {
        "geometry_valid": False,
        "issues": issues,
        "points": rows,
        "display_geometry_used": False,
    }


__all__ = ["pair_frames", "geometry_precheck"]
