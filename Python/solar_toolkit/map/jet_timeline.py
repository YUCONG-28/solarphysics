"""Actual-frame pairing and geometry prechecks, independent of display shells."""

from collections import Counter

import numpy as np
from astropy.time import Time
from sunpy.coordinates import frames

from .jet_annotations import pixel_hpc
from .jet_viewpoint import observer


def pair_frames(records, master="EUVI", band=None):
    """Pair header emission times; missing and tied nearest frames remain explicit."""
    records = [r for r in records if band is None or float(r["band"]) == float(band)]

    def emission(r):
        return Time(r["midpoint_utc"]).unix - float(r["dsun_m"]) / 299792458.0

    aa = sorted([r for r in records if r["instrument"] == master], key=emission)
    bb = sorted([r for r in records if r["instrument"] != master], key=emission)
    dt = np.diff(sorted(set(emission(r) for r in bb)))
    tolerance = float(np.median(dt) / 2) if len(dt) else None
    result = []
    previous = None
    for a in aa:
        out = {
            "master_id": a["image_sha256"],
            "AIA": a if master == "AIA" else None,
            "EUVI": a if master == "EUVI" else None,
            "tolerance_s": tolerance,
            "status": "no_other_frame",
            "delta_emission_s": None,
            "repeated_other": False,
        }
        if bb and tolerance is not None:
            distance = np.array([abs(emission(b) - emission(a)) for b in bb])
            ids = np.where(np.isclose(distance, distance.min(), atol=1e-6, rtol=0))[0]
            if len(ids) > 1:
                out["status"] = "ambiguous_nearest"
            elif distance.min() > tolerance + 1e-6:
                out["status"] = "outside_tolerance"
            else:
                b = bb[ids[0]]
                other = "AIA" if master == "EUVI" else "EUVI"
                out[other] = b
                out["status"] = "matched"
                out["delta_emission_s"] = emission(out["AIA"]) - emission(out["EUVI"])
                out["repeated_other"] = b["image_sha256"] == previous
                previous = b["image_sha256"]
        elif bb:
            out["status"] = "cadence_unavailable"
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
