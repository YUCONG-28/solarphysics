"""Descriptive comparisons of jet annotations on the same image.

These are repeatability diagnostics, not accuracy, identity or 3-D validation.
The saved automatic_axis is the automatic path of the selected branch, not
necessarily the originally displayed longest path.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .jet_annotations import pixel_hpc
from .jet_extraction import axis_metrics


def angle_difference_deg(first, second):
    """Signed second-minus-first direction difference, without reversing axes."""
    return float((second - first + 180) % 360 - 180)


def compare_axes(first_hpc, second_hpc):
    a, b = np.asarray(first_hpc, float), np.asarray(second_hpc, float)
    for points in (a, b):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Both axes require at least two HPC positions")
        if not np.isfinite(points).all():
            raise ValueError("Non-finite axis coordinates")
        if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == 0:
            raise ValueError("Zero-length axis has no direction")
    ma, mb = axis_metrics(a), axis_metrics(b)
    same = np.linalg.norm(a[[0, -1]] - b[[0, -1]], axis=1)
    opposite = np.linalg.norm(a[[0, -1]] - b[[-1, 0]], axis=1)
    directions = []
    reversed_orientation = bool(opposite.sum() + 1e-8 < same.sum())
    for da, db in zip(
        ma["terminal_directions"], mb["terminal_directions"], strict=True
    ):
        delta = angle_difference_deg(da["direction_deg"], db["direction_deg"])
        full_length = not (da["short_segment"] or db["short_segment"])
        directions.append(
            {
                "requested_arcsec": da["requested_arcsec"],
                "first_available_arcsec": da["available_arcsec"],
                "second_available_arcsec": db["available_arcsec"],
                "first_direction_deg": da["direction_deg"],
                "second_direction_deg": db["direction_deg"],
                "difference_deg": delta,
                "same_requested_length_available": full_length,
                "within_5deg_engineering_target": (
                    bool(abs(delta) <= 5)
                    if full_length and not reversed_orientation
                    else None
                ),
            }
        )
    return {
        "inner_endpoint_difference_arcsec": float(same[0]),
        "outer_endpoint_difference_arcsec": float(same[1]),
        "orientation_maybe_reversed": reversed_orientation,
        "first_length_arcsec": ma["length_arcsec"],
        "second_length_arcsec": mb["length_arcsec"],
        "length_difference_arcsec": mb["length_arcsec"] - ma["length_arcsec"],
        "directions": directions,
        "projected_only": True,
    }


def width_summary(measurements):
    rows = measurements.get("widths", [])
    gaussian = [
        r["gaussian"]["fwhm_arcsec"]
        for r in rows
        if r["gaussian"]["status"] == "valid"
        and r["gaussian"].get("fwhm_arcsec") is not None
    ]
    mask = [
        r["mask_width_arcsec"]
        for r in rows
        if r.get("mask_status") == "valid" and r.get("mask_width_arcsec") is not None
    ]
    return {
        "section_count": len(rows),
        "valid_gaussian_count": len(gaussian),
        "valid_mask_count": len(mask),
        "median_fwhm_arcsec": float(np.median(gaussian)) if gaussian else None,
        "median_mask_width_arcsec": float(np.median(mask)) if mask else None,
        "gaussian_status_counts": dict(Counter(r["gaussian"]["status"] for r in rows)),
        "interpretation": "unpaired_descriptive_widths_not_centroid_uncertainty",
    }


def operation_summary(document):
    counts = Counter(row["action"] for row in document.history)
    return {
        "recorded_action_count": len(document.history),
        "actions": dict(counts),
        "branch_selection_actions": counts["choose_branch"],
        "warning": "actions_include_undo; branch_selection_count_is_not_verified_branch_jumps",
    }


def compare_documents(first, second, *, same_structure=False):
    """Require identical image bytes; do not infer identity from overlap."""
    result = {"status": "unavailable", "independent_3d_valid": False}
    if first is None or second is None:
        return {**result, "reason": "missing_view"}
    result.update(
        image_sha256=first.sha256,
        first_axis_status=first.state["axis_status"],
        second_axis_status=second.state["axis_status"],
    )
    if first.sha256 != second.sha256:
        return {**result, "reason": "different_image_or_observation"}
    if not first.state["axis"] or not second.state["axis"]:
        return {**result, "reason": "missing_axis"}
    result.update(
        compare_axes(
            pixel_hpc(first.map, first.state["axis"]),
            pixel_hpc(second.map, second.state["axis"]),
        )
    )
    if first.selected is not None and second.selected is not None:
        union = np.count_nonzero(first.selected | second.selected)
        result["mask_iou"] = (
            float(np.count_nonzero(first.selected & second.selected) / union)
            if union
            else None
        )
    confirmed = {"automatic_confirmed", "manual_corrected"}
    eligible = (
        same_structure
        and first.state["axis_status"] in confirmed
        and second.state["axis_status"] in confirmed
        and not result["orientation_maybe_reversed"]
    )
    result.update(
        status="repeatability_comparison" if eligible else "descriptive_only",
        same_structure_asserted=bool(same_structure),
        eligible_for_repeatability=eligible,
        first_widths=width_summary(first.measurements()),
        second_widths=width_summary(second.measurements()),
        first_operations=operation_summary(first),
        second_operations=operation_summary(second),
    )
    if not eligible:
        result["reason"] = "identity_orientation_or_axis_confirmation_required"
    # A small angular difference alone cannot pass an unconfirmed comparison.
    for direction in result["directions"]:
        direction["eligible_engineering_pass"] = (
            direction["within_5deg_engineering_target"] if eligible else None
        )
    return result


def compare_saved_automatic(document):
    if (
        document is None
        or not document.state["automatic_axis"]
        or not document.state["axis"]
    ):
        return {"status": "unavailable", "reason": "missing_automatic_or_edited_axis"}
    return {
        "status": "descriptive_only",
        "comparison": "selected_branch_automatic_vs_saved_axis",
        "image_sha256": document.sha256,
        "saved_axis_status": document.state["axis_status"],
        "independent_3d_valid": False,
        **compare_axes(
            pixel_hpc(document.map, document.state["automatic_axis"]),
            pixel_hpc(document.map, document.state["axis"]),
        ),
    }


__all__ = [
    "angle_difference_deg",
    "compare_axes",
    "width_summary",
    "operation_summary",
    "compare_documents",
    "compare_saved_automatic",
]
