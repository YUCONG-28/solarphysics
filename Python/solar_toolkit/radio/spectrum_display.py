"""Display-only relative intensities and optional missing-column compaction.

Relative backgrounds preserve original UTC. Compaction is an independent,
explicitly requested display transform with a mapping back to original UTC.
"""

import hashlib
from datetime import datetime, timezone

import numpy as np

__all__ = ["relative_background_db", "compact_spectrum_time"]


def relative_background_db(data, time, max_gap_seconds=0.0, background_percentile=20.0):
    """Return 10 log10(I / per-channel P20) and provenance of estimated cells.

    Only bounded interior runs whose bracketing times are within the configured
    tolerance are estimated linearly in linear intensity. Never compress time.
    """
    values = np.asarray(data, dtype=float).copy()
    times = np.asarray(time, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != len(times)
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("Expected frequency x increasing time")
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    baseline = np.nanpercentile(values, background_percentile, axis=1)
    estimated = np.zeros_like(values, dtype=bool)
    for row, mask in zip(values, estimated, strict=True):
        good = np.flatnonzero(np.isfinite(row))
        for a, b in zip(good[:-1], good[1:], strict=True):
            if b > a + 1 and times[b] - times[a] <= max_gap_seconds + 1e-6:
                row[a + 1 : b] = np.interp(times[a + 1 : b], times[[a, b]], row[[a, b]])
                mask[a + 1 : b] = True
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10 * np.log10(values / baseline[:, None])
    return out, {
        "unit": "dB above per-frequency background",
        "definition": "10 log10(I / B(f)); B(f)=20th percentile of positive original samples in displayed window",
        "background_percentile": background_percentile,
        "background_linear": baseline.tolist(),
        "max_bracketing_gap_seconds": max_gap_seconds,
        "estimated_cells": int(estimated.sum()),
        "estimated_time_indices": np.flatnonzero(estimated.any(axis=0)).tolist(),
        "remaining_missing_cells": int(np.isnan(out).sum()),
        "interpolation": "linear intensity, interior short gaps only; display-only; original UTC retained",
    }


def _utc(epoch):
    return (
        datetime.fromtimestamp(float(epoch), timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def compact_spectrum_time(
    arrays, target, *, allow_outside_window=False, allow_missing_time=False
):
    """Omit all-empty time columns for display, preserving their UTC mapping."""
    values = arrays["intensity"]
    real_edges = np.asarray(arrays["time_edges_unix"], dtype=float)
    if real_edges.shape != (values.shape[1] + 1,) or np.any(np.diff(real_edges) <= 0):
        raise ValueError("Invalid spectrum time edges")
    valid = np.any(np.isfinite(values), axis=0)
    retained = np.flatnonzero(valid)
    if not retained.size:
        raise ValueError("No observed spectrum time columns")
    field = values[:, retained]
    widths = np.diff(real_edges)[retained]
    display_edges = np.r_[0.0, np.cumsum(widths)]
    original_column = int(np.searchsorted(real_edges, target, side="right") - 1)
    if target == real_edges[-1]:
        original_column = values.shape[1] - 1
    position = int(np.searchsorted(retained, original_column))
    outside = target < real_edges[0] or target > real_edges[-1]
    marker, error = None, None
    if outside and allow_outside_window:
        cursor_status = "image_time_outside_spectrum_window"
    else:
        missing_target = not outside and (
            position >= retained.size or retained[position] != original_column
        )
        if missing_target and allow_missing_time:
            cursor_status = "image_time_no_spectrum_data"
        elif outside or missing_target:
            raise ValueError("Preview cursor has no spectrum data")
        else:
            fraction = (target - real_edges[original_column]) / widths[position]
            marker = float(display_edges[position] + fraction * widths[position])
            recovered = real_edges[original_column] + (marker - display_edges[position])
            error = float(abs(recovered - target))
            if error > 1e-6:
                raise ValueError("Compacted cursor UTC roundtrip failed")
            cursor_status = "shown"
    runs = np.r_[0, np.flatnonzero(np.diff(retained) != 1) + 1, retained.size]
    segments = []
    for start, stop in zip(runs[:-1], runs[1:], strict=True):
        a, b = int(retained[start]), int(retained[stop - 1] + 1)
        segments.append(
            {
                "original_bin_start": a,
                "original_bin_stop_exclusive": b,
                "utc_start": _utc(real_edges[a]),
                "utc_end": _utc(real_edges[b]),
                "display_start_seconds": float(display_edges[start]),
                "display_end_seconds": float(display_edges[stop]),
            }
        )
    missing = ~valid
    leading, trailing = int(retained[0]), int(values.shape[1] - retained[-1] - 1)
    info = {
        "time_axis_policy": "all-empty time columns omitted; retained columns adjacent; tick labels and cursor retain original UTC",
        "time_axis_is_continuous_utc": False,
        "original_time_columns": int(values.shape[1]),
        "displayed_time_columns": int(retained.size),
        "removed_no_data_time_columns": int(missing.sum()),
        "trimmed_leading_time_columns": leading,
        "trimmed_trailing_time_columns": trailing,
        "removed_internal_time_columns": int(missing.sum()) - leading - trailing,
        "display_start_utc": _utc(real_edges[retained[0]]),
        "display_end_utc": _utc(real_edges[retained[-1] + 1]),
        "removed_time_seconds": float(np.diff(real_edges)[missing].sum()),
        "displayed_elapsed_seconds": float(display_edges[-1]),
        "retained_time_segments": segments,
        "retained_original_bin_indices_sha256": hashlib.sha256(
            retained.astype("<i8").tobytes()
        ).hexdigest(),
        "retained_values_sha256": hashlib.sha256(field.tobytes()).hexdigest(),
        "displayed_empty_time_columns": int(
            np.sum(~np.any(np.isfinite(field), axis=0))
        ),
        "cursor_original_utc": _utc(target),
        "cursor_display_seconds": marker,
        "cursor_visible": marker is not None,
        "cursor_display_status": cursor_status,
        "cursor_utc_roundtrip_error_seconds": error,
        "source_values_modified": False,
        "interpolation_applied": False,
    }
    return field, display_edges, retained, marker, info
