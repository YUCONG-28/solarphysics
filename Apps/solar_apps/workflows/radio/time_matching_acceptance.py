"""Validate original FITS and videos for explicitly supplied image sequences.

Inputs and output destinations are explicit CLI arguments. No observations are
changed. This runner exercises the public time plan and exact radio-plane reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from functools import lru_cache
from collections import Counter

import numpy as np
from astropy.wcs import WCS

from solar_toolkit.radio.time_matching import (
    index_radio_frames,
    pair_radio_frames,
    materialize_radio_frame,
    compatible_radio_payloads,
)
from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
    seconds_to_ns,
    filter_time_frames,
)


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _paths(folder):
    return sorted(path for path in Path(folder).glob("*.fits") if path.is_file())


def _index(paths, errors, *, frequency=None, polarization=None):
    frames = []
    for path in paths:
        try:
            frames.extend(
                index_radio_frames(
                    [path], frequency_mhz=frequency, polarization=polarization
                )
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    return frames


def _locator(payload):
    return (
        payload["path"],
        payload["hdu_index"],
        tuple(payload.get("plane_indices", ())),
    )


def _stokes_i(payload):
    header = payload["header"]
    axes = range(int(header.get("NAXIS", 2)), 2, -1)
    indices = dict(zip(axes, payload.get("plane_indices", ())))
    codes = [
        float(header.get(f"CRVAL{axis}", 0))
        + (indices[axis] + 1 - float(header.get(f"CRPIX{axis}", 1)))
        * float(header.get(f"CDELT{axis}", 1))
        for axis in indices
        if str(header.get(f"CTYPE{axis}", "")).upper() == "STOKES"
    ]
    return len(codes) == 1 and codes[0] == 1


def audit_event(
    aia_directory,
    frequencies,
    muser_directory,
    output,
    *,
    dart_directory,
    dart_mode="RR+LL",
    aia_waves=(171, 304),
):
    """Read original headers once, retaining exact planes and rejection reasons."""
    errors, streams, scanned = [], {}, {}
    root = Path(aia_directory)
    for wave in aia_waves:
        paths = _paths(root / str(wave))
        scanned[f"aia:{wave}"] = len(paths)
        frames = _index(paths, errors)
        accepted = []
        for frame in frames:
            header = frame.payload["header"]
            exposure = float(header.get("EXPTIME", 0))
            if (
                int(header.get("QUALITY", -1)) != 0
                or not np.isfinite(exposure)
                or exposure <= 0
                or float(header.get("WAVELNTH", -1)) != wave
                or not str(header.get("CTYPE1", "")).startswith("HPLN")
                or not str(header.get("CTYPE2", "")).startswith("HPLT")
                or any(frame.payload.get("plane_indices", ()))
            ):
                errors.append(
                    {
                        "path": frame.payload["path"],
                        "reason": "AIA requires the requested wavelength, helioprojective image, QUALITY=0 and finite positive EXPTIME",
                    }
                )
            else:
                payload = dict(
                    frame.payload,
                    frequency_mhz=None,
                    polarization="not_applicable",
                    wavelength_A=wave,
                )
                accepted.append(TimeFrame(frame.frame_id, frame.time, payload))
        streams[f"aia:{wave}"] = accepted
    for frequency in frequencies:
        folder = Path(dart_directory) / f"{frequency:g}MHz"
        for pol in ("I",) if dart_mode == "I" else ("RR", "LL"):
            paths = _paths(folder / pol)
            source = (
                f"dart:{frequency:g}"
                if dart_mode == "I"
                else f"dart:{frequency:g}:{pol}"
            )
            scanned[source] = len(paths)
            streams[source] = _index(
                paths, errors, frequency=frequency, polarization=pol
            )
    muser = {frequency: [] for frequency in frequencies}
    paths = _paths(muser_directory) if muser_directory is not None else []
    scanned["muser:combined"] = len(paths)
    for path in paths:
        try:
            frames = index_radio_frames([path])
            frames = [frame for frame in frames if _stokes_i(frame.payload)]
            if not frames:
                raise ValueError("MUSER requires explicit Stokes I")
            for frequency in frequencies:
                chosen = min(
                    frames,
                    key=lambda frame: abs(frame.payload["frequency_mhz"] - frequency),
                )
                if abs(chosen.payload["frequency_mhz"] - frequency) > 5:
                    errors.append(
                        {
                            "path": str(path),
                            "reason": f"No MUSER plane within 5 MHz of {frequency:g}",
                        }
                    )
                else:
                    muser[frequency].append(chosen)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})
    for frequency, frames in muser.items():
        streams[f"muser:{frequency:g}"] = frames
    _write_json(
        output / "header_audit.json",
        {
            "scanned_file_counts": scanned,
            "valid_indexed_file_counts": {
                source: len({frame.payload["path"] for frame in frames})
                for source, frames in streams.items()
            },
            "excluded": errors,
        },
    )
    return streams, errors


def _window(frames, start, end):
    return [frame for frame in frames if start.tai_ns <= frame.time.tai_ns < end.tai_ns]


def build_event_plan(
    indexed,
    *,
    frequencies,
    start,
    end,
    include_muser,
    dart_mode="RR+LL",
    aia_waves=(171, 304),
    aia_tolerance_seconds=12,
    radio_tolerance_seconds=0.1,
    pair_tolerance_seconds=0.1,
):
    streams, pairs_audit = [], {}
    for wave in aia_waves:
        streams.append(
            TimeStream(f"aia:{wave}", indexed[f"aia:{wave}"], aia_tolerance_seconds)
        )
    for frequency in frequencies:
        if dart_mode == "I":
            frames = indexed[f"dart:{frequency:g}"]
            selected = filter_time_frames(frames, time_start=start, time_end=end)
            pairs_audit[f"{frequency:g}"] = {
                "mode": "direct_stokes_i",
                "pairing_performed": False,
                "input_frames": len(frames),
                "in_window_frames": len(selected),
                "outside_window_frames": len(frames) - len(selected),
            }
            streams.append(
                TimeStream(f"dart:{frequency:g}", frames, radio_tolerance_seconds)
            )
            if include_muser:
                streams.append(
                    TimeStream(
                        f"muser:{frequency:g}",
                        indexed[f"muser:{frequency:g}"],
                        radio_tolerance_seconds,
                    )
                )
            continue
        rr_all = indexed[f"dart:{frequency:g}:RR"]
        ll_all = indexed[f"dart:{frequency:g}:LL"]
        rr = _window(rr_all, start, end)
        ll = _window(ll_all, start, end)
        pairs = pair_radio_frames(
            rr_all,
            ll_all,
            tolerance_seconds=pair_tolerance_seconds,
            time_start=start,
            time_end=end,
        )
        used = [
            _locator(member)
            for pair in pairs
            for member in pair.payload["pair_members"]
        ]
        if len(used) != len(set(used)):
            raise AssertionError(
                "An original polarization plane entered multiple pairs"
            )
        pairs_audit[f"{frequency:g}"] = {
            "rr_frames": len(rr),
            "ll_frames": len(ll),
            "rr_original_archive_frames": len(rr_all),
            "ll_original_archive_frames": len(ll_all),
            "rr_outside_window": len(rr_all) - len(rr),
            "ll_outside_window": len(ll_all) - len(ll),
            "pairs": len(pairs),
            "unpaired_rr": sum(_locator(frame.payload) not in used for frame in rr),
            "unpaired_ll": sum(_locator(frame.payload) not in used for frame in ll),
            "member_times_outside_window": sum(
                not start.tai_ns <= time < end.tai_ns
                for pair in pairs
                for time in pair.payload["actual_times_tai_ns"]
            ),
        }
        streams.append(
            TimeStream(f"dart:{frequency:g}", pairs, radio_tolerance_seconds)
        )
        if include_muser:
            streams.append(
                TimeStream(
                    f"muser:{frequency:g}",
                    indexed[f"muser:{frequency:g}"],
                    radio_tolerance_seconds,
                )
            )
    plan = build_sequence_match_plan(streams, time_start=start, time_end=end)
    return plan, streams, pairs_audit


def _selected_streams(plan, streams):
    window = plan.to_dict()["window"]
    return [
        TimeStream(
            stream.source_id,
            filter_time_frames(
                stream.frames,
                time_start=window["start_utc"],
                time_end=window["end_utc"],
            ),
            stream.tolerance_seconds,
            stream.offset_seconds,
        )
        for stream in streams
    ]


def validate_plan(plan, streams, *, pair_tolerance_seconds=0.1):
    """Independently verify identities, coverage and every derived quality field."""
    original_streams = streams
    streams = _selected_streams(plan, original_streams)
    counts = {stream.source_id: len(stream.frames) for stream in streams}
    if not plan.rows or len(plan.rows) != max(counts.values()):
        raise AssertionError("The longest timeline is not complete")
    previous_master = None
    used = {source: set() for source in counts}
    previous = {}
    eligible_ids = {
        stream.source_id: {frame.frame_id for frame in stream.frames}
        for stream in streams
    }
    ordered = {
        stream.source_id: sorted(
            (
                (index, frame)
                for index, frame in enumerate(stream.frames)
                if frame.frame_id in eligible_ids[stream.source_id]
            ),
            key=lambda item: (item[1].time.tai_ns, item[1].frame_id, item[0]),
        )
        for stream in original_streams
    }
    by_source = {stream.source_id: stream for stream in streams}
    source_order = tuple(
        sorted(
            (source for source, count in counts.items() if count),
            key=lambda source: (counts[source], source),
        )
    )
    expected_master, expected_output = source_order[0], source_order[-1]
    if (
        plan.master_source_id != expected_master
        or plan.output_source_id != expected_output
        or tuple(plan.source_order) != source_order
    ):
        raise AssertionError(
            "Root/output sources differ from the ascending-count hierarchy"
        )
    edge_report = {}
    if set(plan.parent_maps) != set(source_order[1:]):
        raise AssertionError("Hierarchy parent maps are incomplete")
    for parent_source, child_source in zip(source_order, source_order[1:]):
        edges = plan.parent_maps[child_source]
        if len(edges) != counts[child_source]:
            raise AssertionError("Hierarchy edge does not cover every child frame")
        parent_clocks = [
            (frame.match_time or frame.time).tai_ns
            + seconds_to_ns(by_source[parent_source].offset_seconds)
            for _, frame in ordered[parent_source]
        ]
        parent_uses, previous_parent, forced = set(), -1, 0
        for child_index, edge in enumerate(edges):
            parent_index = edge["parent_sorted_index"]
            if (
                not isinstance(parent_index, int)
                or not previous_parent <= parent_index < counts[parent_source]
            ):
                raise AssertionError(
                    "Hierarchy parent indices decrease or leave the stream"
                )
            previous_parent = parent_index
            parent_original, parent = ordered[parent_source][parent_index]
            child_original, child = ordered[child_source][child_index]
            child_clock = (child.match_time or child.time).tai_ns + seconds_to_ns(
                by_source[child_source].offset_seconds
            )
            delta = parent_clocks[parent_index] - child_clock
            nearest = min(abs(clock - child_clock) for clock in parent_clocks)
            if (
                edge["parent_source_id"] != parent_source
                or edge["child_source_id"] != child_source
                or edge["child_sorted_index"] != child_index
                or edge["parent_original_index"] != parent_original
                or edge["child_original_index"] != child_original
                or edge["parent_frame_id"] != parent.frame_id
                or edge["child_frame_id"] != child.frame_id
                or edge["parent_time"]["tai_ns"] != parent.time.tai_ns
                or edge["child_time"]["tai_ns"] != child.time.tai_ns
                or edge["parent_match_time"]["tai_ns"] != parent_clocks[parent_index]
                or edge["child_match_time"]["tai_ns"] != child_clock
                or edge["delta_ns"] != delta
                or edge["nearest_distance_ns"] != nearest
                or edge["is_nearest"] != (abs(delta) == nearest)
                or edge["coverage_forced"] != (abs(delta) != nearest)
            ):
                raise AssertionError(
                    "Hierarchy edge identity, nearest-distance or forced-coverage metadata differs"
                )
            parent_uses.add(parent_index)
            forced += int(edge["coverage_forced"])
        if parent_uses != set(range(counts[parent_source])):
            raise AssertionError("Hierarchy edge omits parent frames")
        edge_report[child_source] = {
            "parent_source_id": parent_source,
            "child_frames": len(edges),
            "coverage_forced": forced,
        }
    uses = Counter(
        (source, item.sorted_index)
        for row in plan.rows
        for source, item in row.selections.items()
        if item.frame is not None
    )
    for index, row in enumerate(plan.rows):
        output = ordered[expected_output][index][1]
        master_clock = (output.match_time or output.time).tai_ns + seconds_to_ns(
            by_source[expected_output].offset_seconds
        )
        if (
            row.master_frame != row.selections[expected_master].frame
            or row.output_frame != output
            or row.time != output.time
            or row.match_time.tai_ns != master_clock
        ):
            raise AssertionError(
                "Root/output frame identity, observed UTC or effective time changed"
            )
        if set(row.selections) != set(counts):
            raise AssertionError("Plan sources differ from input streams")
        if previous_master is not None and row.time.tai_ns < previous_master:
            raise AssertionError("Master UTC decreases")
        previous_master = row.time.tai_ns
        for source, selection in row.selections.items():
            if selection.frame is None:
                if counts[source]:
                    raise AssertionError(f"Nonempty source missing: {source}")
                if (
                    selection.delta_ns is not None
                    or selection.within_tolerance is not None
                    or selection.use_count
                    or selection.reused
                    or selection.sorted_index is not None
                ):
                    raise AssertionError("Empty-source metadata is inconsistent")
                continue
            stream = by_source[source]
            if (
                not isinstance(selection.sorted_index, int)
                or not 0 <= selection.sorted_index < counts[source]
            ):
                raise AssertionError("Selected sorted index is outside the input range")
            original, expected = ordered[source][selection.sorted_index]
            if selection.frame != expected or selection.original_index != original:
                raise AssertionError(f"Exact input frame locator differs: {source}")
            delta = (
                (expected.match_time or expected.time).tai_ns
                + seconds_to_ns(stream.offset_seconds)
                - master_clock
            )
            if (
                selection.delta_ns != delta
                or selection.within_tolerance
                != (abs(delta) <= seconds_to_ns(stream.tolerance_seconds))
                or selection.use_count != uses[source, selection.sorted_index]
                or selection.reused != (uses[source, selection.sorted_index] > 1)
                or selection.match_time.tai_ns != master_clock + delta
            ):
                raise AssertionError(
                    f"Derived delta, tolerance or reuse metadata differs: {source}"
                )
            stamp = selection.frame.time.tai_ns
            if source in previous and (
                stamp < previous[source][0]
                or selection.sorted_index < previous[source][1]
            ):
                raise AssertionError(f"Selected source decreases: {source}")
            previous[source] = stamp, selection.sorted_index
            used[source].add(selection.frame.frame_id)
        if set(row.parent_matches) != set(source_order[1:]):
            raise AssertionError("Output row omits hierarchy parents")
        for source in source_order[1:]:
            edge = plan.parent_maps[source][row.selections[source].sorted_index]
            if (
                row.parent_matches[source] != edge
                or row.selections[edge["parent_source_id"]].sorted_index
                != edge["parent_sorted_index"]
            ):
                raise AssertionError(
                    "A child frame changed its fixed parent in an output row"
                )
    for source, count in counts.items():
        if len(used[source]) != count:
            raise AssertionError(f"Incomplete independent-frame coverage: {source}")
        if source.startswith("dart:"):
            originals = []
            for frame in by_source[source].frames:
                payload = frame.payload
                if "pair_members" not in payload:
                    if payload.get("polarization") != "L+R":
                        raise AssertionError(
                            "Direct DART images require explicit Stokes I"
                        )
                    continue
                a, b = payload["pair_members"]
                times = payload["actual_times_tai_ns"]
                if (
                    abs(times[1] - times[0]) > seconds_to_ns(pair_tolerance_seconds)
                    or payload["pair_delta_ns"] != times[1] - times[0]
                    or frame.time.tai_ns != (times[0] + times[1]) // 2
                    or not compatible_radio_payloads(a, b)
                ):
                    raise AssertionError("Strict RR/LL pairing metadata differs")
                originals.extend((_locator(a), _locator(b)))
            if len(originals) != len(set(originals)):
                raise AssertionError(
                    "Original RR/LL plane participates in multiple pairs"
                )
    return {
        "passed": True,
        "output_rows": len(plan.rows),
        "master_source_id": plan.master_source_id,
        "output_source_id": plan.output_source_id,
        "source_order": list(plan.source_order),
        "parent_edges": edge_report,
        "used_independent_frames": {
            source: len(values) for source, values in used.items()
        },
    }


def selection_statistics(rows, streams):
    """Comparable independent-frame, reuse, reversal and real-delta statistics."""
    result = {}
    for stream in streams:
        selected = [
            row[stream.source_id]
            for row in rows
            if row.get(stream.source_id) is not None
        ]
        identities = [frame.frame_id for frame, delta in selected]
        times = [frame.time.tai_ns for frame, delta in selected]
        deltas = [delta for frame, delta in selected]
        result[stream.source_id] = {
            "input_frames": len(stream.frames),
            "used_frames": len(set(identities)),
            "unused_frames": len(stream.frames) - len(set(identities)),
            "repeat_occurrences": len(identities) - len(set(identities)),
            "time_reversals": sum(b < a for a, b in zip(times, times[1:])),
            "maximum_abs_delta_ns": max(map(abs, deltas), default=None),
            "total_abs_delta_ns": sum(map(abs, deltas)),
        }
    return result


def compare_previous_selection_policy(plan, streams):
    """Reproduce first-radio/strict-nearest selection on the same audited times.

    This isolates selection behavior; it is not a replay of historical images
    or of timestamp parsing bugs in older code.
    """
    by_source = {stream.source_id: stream for stream in streams}
    primary = by_source[
        sorted(
            (source for source in by_source if source.startswith("dart:")),
            key=lambda source: float(source.split(":")[1]),
        )[0]
    ]
    baseline = []
    for frame in sorted(
        primary.frames, key=lambda item: (item.time.tai_ns, item.frame_id)
    ):
        selections = {}
        for stream in streams:
            if not stream.frames:
                break
            candidate = min(
                stream.frames,
                key=lambda item: (
                    abs(item.time.tai_ns - frame.time.tai_ns),
                    item.time.tai_ns,
                    item.frame_id,
                ),
            )
            delta = candidate.time.tai_ns - frame.time.tai_ns
            if abs(delta) > seconds_to_ns(stream.tolerance_seconds):
                break
            selections[stream.source_id] = (candidate, delta)
        if len(selections) == len(streams):
            baseline.append(selections)
    current = [
        {
            source: (item.frame, item.delta_ns) if item.frame else None
            for source, item in row.selections.items()
        }
        for row in plan.rows
    ]
    return {
        "scope": "selection policies on the same newly audited timestamps",
        "previous_policy": "first DART frequency; strict independent nearest; incomplete rows dropped",
        "before": {
            "output_rows": len(baseline),
            "sources": selection_statistics(baseline, streams),
        },
        "after": {
            "output_rows": len(current),
            "sources": selection_statistics(current, streams),
        },
    }


def _frame_record(frame):
    payload = frame.payload
    records = []
    for member in payload.get("pair_members", (payload,)):
        path = Path(member["path"])
        records.append(
            {
                "path": str(path),
                "hdu_index": member["hdu_index"],
                "plane_indices": member.get("plane_indices"),
                "size_bytes": member.get("size_bytes"),
                "mtime_ns": member.get("mtime_ns"),
                "frequency_mhz": member.get("frequency_mhz"),
                "unit": member["header"].get("BUNIT"),
                "time_scale": member["header"].get("TIMESYS", "UTC"),
                "time_source": member.get("time_source"),
            }
        )
    return {
        **frame.to_dict(),
        "inputs": records,
        "pair_delta_ns": payload.get("pair_delta_ns"),
    }


def verify_exact_reference(frame):
    """Reread original headers and independently resolve each selected locator."""
    actual = []
    payload = frame.payload
    members = payload.get("pair_members", (payload,))
    for member in members:
        path = Path(member["path"])
        stat = path.stat()
        if (
            stat.st_size != member["size_bytes"]
            or stat.st_mtime_ns != member["mtime_ns"]
        ):
            raise AssertionError(f"Original FITS changed before encoding: {path}")
        found = [
            candidate
            for candidate in index_radio_frames([path])
            if _locator(candidate.payload) == _locator(member)
        ]
        if len(found) != 1:
            raise AssertionError(
                f"Actual HDU/plane locator is ambiguous: {_locator(member)}"
            )
        candidate = found[0]
        observed = candidate.payload
        if tuple(observed["shape"]) != tuple(member["shape"]):
            raise AssertionError("Actual image-plane shape differs from plan")
        if "wavelength_A" in member:
            if float(observed["header"].get("WAVELNTH", -1)) != member["wavelength_A"]:
                raise AssertionError("Actual AIA wavelength differs from plan")
        elif observed["polarization"] != member["polarization"] or not np.isclose(
            observed["frequency_mhz"], member["frequency_mhz"], rtol=1e-5, atol=1e-6
        ):
            raise AssertionError(
                "Actual radio frequency/polarization plane differs from plan"
            )
        actual.append(
            {
                "locator": list(_locator(member)),
                "time": candidate.time.to_dict(),
                "frequency_mhz": (
                    None if "wavelength_A" in member else observed["frequency_mhz"]
                ),
                "polarization": member["polarization"],
                "wavelength_A": member.get("wavelength_A"),
                "unit": observed["header"].get("BUNIT"),
                "shape": observed["shape"],
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    observed_times = tuple(item["time"]["tai_ns"] for item in actual)
    expected_times = tuple(payload.get("actual_times_tai_ns", (frame.time.tai_ns,)))
    if observed_times != expected_times or frame.time.tai_ns != sum(
        observed_times
    ) // len(observed_times):
        raise AssertionError("Actual original-FITS observation times differ from plan")
    return {
        "frame_id": frame.frame_id,
        "actual_inputs": actual,
        "observed_tai_ns": frame.time.tai_ns,
    }


def validate_encoded_references(path, plan):
    """Check all written encoder references, without sampling or trusting counts."""
    with Path(path).open(encoding="utf-8") as file:
        count = 0
        for count, line in enumerate(file, 1):
            if count > len(plan.rows):
                raise AssertionError("Encoder emitted extra metadata rows")
            record, row = json.loads(line), plan.rows[count - 1]
            if (
                record["row"] != count
                or record["master_frame_id"] != row.master_frame.frame_id
                or record.get("output_frame_id") != row.output_frame.frame_id
                or record.get("parent_matches") != row.to_dict()["parent_matches"]
            ):
                raise AssertionError(
                    "Encoder row/master reference differs from fixed plan"
                )
            expected = {
                source: item.frame.frame_id
                for source, item in row.selections.items()
                if item.frame is not None
            }
            actual = {
                source: item["frame_id"]
                for source, item in record["selections"].items()
            }
            if actual != expected:
                raise AssertionError(
                    "Encoder selected references differ from fixed plan"
                )
            for source, item in record["selections"].items():
                if item["observed_tai_ns"] != row.selections[source].frame.time.tai_ns:
                    raise AssertionError(
                        "Encoder observation UTC differs from fixed plan"
                    )
        if count != len(plan.rows):
            raise AssertionError("Encoder reference row count differs from fixed plan")
    return count


def render_video(
    plan, output, frequencies, *, include_muser, fov, fps=10, aia_waves=(171, 304)
):
    """Render every planned original image, then independently decode the MP4."""
    import av
    from fractions import Fraction
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sunpy.visualization.colormaps import cm as solar_cmaps

    lookup = {
        selection.frame.frame_id: selection.frame
        for row in plan.rows
        for selection in row.selections.values()
        if selection.frame is not None
    }

    @lru_cache(maxsize=40)
    def pixels(identity):
        evidence = verify_exact_reference(lookup[identity])
        data, header = materialize_radio_frame(lookup[identity])
        if data.ndim != 2 or not np.isfinite(data).any():
            raise ValueError(f"Unreadable original pixels: {identity}")
        wcs = WCS(header, naxis=2)
        corner_x, corner_y = wcs.all_world2pix(
            np.array([fov[0], fov[1], fov[0], fov[1]]) / 3600,
            np.array([fov[2], fov[2], fov[3], fov[3]]) / 3600,
            0,
        )
        if np.isfinite(corner_x).all() and np.isfinite(corner_y).all():
            x0, x1 = (
                max(0, int(np.floor(corner_x.min())) - 2),
                min(data.shape[1], int(np.ceil(corner_x.max())) + 3),
            )
            y0, y1 = (
                max(0, int(np.floor(corner_y.min())) - 2),
                min(data.shape[0], int(np.ceil(corner_y.max())) + 3),
            )
            evidence["fov_intersects_image"] = x1 > x0 and y1 > y0
            if x1 <= x0 or y1 <= y0:
                x0, x1, y0, y1 = 0, data.shape[1], 0, data.shape[0]
        else:
            raise ValueError(
                f"Original image WCS cannot resolve requested field of view: {identity}"
            )
        step = max(1, int(np.ceil(max(y1 - y0, x1 - x0) / 600)))
        ys, xs = slice(y0, y1, step), slice(x0, x1, step)
        crop = data[ys, xs].copy()
        x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
        lon, lat = wcs.all_pix2world(x, y, 0)
        xx, yy = ((lon + 180) % 360 - 180) * 3600, lat * 3600
        evidence["display_pixels_sha256"] = hashlib.sha256(
            np.ascontiguousarray(crop).tobytes()
        ).hexdigest()
        return crop, xx, yy, float(np.nanmax(data)), evidence

    def checked_pixels(frame):
        # Cache reuse must still reject changes to an original file in any row.
        for member in frame.payload.get("pair_members", (frame.payload,)):
            stat = Path(member["path"]).stat()
            if (
                stat.st_size != member["size_bytes"]
                or stat.st_mtime_ns != member["mtime_ns"]
            ):
                raise AssertionError(
                    "Original FITS changed while reusing a cached display frame"
                )
        return pixels(frame.frame_id)

    figure = Figure(figsize=(11, 10), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.85, wspace=0.08)
    colors = ("#ff595e", "#ffca3a", "#8ac926", "#00bbf9", "#b388ff", "#ff69b4")
    path = output / "comparison.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = 1100, 1000
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "20", "preset": "fast"}
    samples = {0, len(plan.rows) // 2, len(plan.rows) - 1}
    transitions = [
        i
        for i in range(1, len(plan.rows))
        if any(
            row.frame is not None
            and row.frame.frame_id != plan.rows[i - 1].selections[source].frame.frame_id
            for source, row in plan.rows[i].selections.items()
            if plan.rows[i - 1].selections[source].frame is not None
        )
    ]
    if transitions:
        samples.add(transitions[0])
    artists = []
    references_path = output / "encoded_references.jsonl"
    references = references_path.open("w", encoding="utf-8")
    try:
        for index, row in enumerate(plan.rows):
            selected_pixels = {
                source: checked_pixels(item.frame)
                for source, item in row.selections.items()
                if item.frame is not None
            }
            for artist in artists:
                artist.remove()
            artists = []
            for wave, axis in zip(aia_waves, axes):
                axis.clear()
                axis.set_facecolor("black")
                chosen = row.selections[f"aia:{wave}"]
                if chosen.frame is not None:
                    data, x, y, _, _ = selected_pixels[f"aia:{wave}"]
                    image = np.log10(np.maximum(data, 0) + 1)
                    finite = image[np.isfinite(image)]
                    lo, hi = np.percentile(finite, [1, 99.7]) if len(finite) else (0, 1)
                    axis.pcolormesh(
                        x,
                        y,
                        image,
                        cmap=solar_cmaps.cmlist[f"sdoaia{wave}"],
                        vmin=lo,
                        vmax=max(hi, lo + 1e-6),
                        shading="auto",
                        rasterized=True,
                    )
                    label = (
                        f"AIA {wave} | {chosen.frame.time.utc[11:]}\nΔt={chosen.delta_ns / 1e9:+.6f}s"
                        + (" | OVER LIMIT" if not chosen.within_tolerance else "")
                    )
                else:
                    label = f"AIA {wave} | NO DATA"
                axis.set_title(label, fontsize=9)
                for instrument, style in (("dart", "-"), ("muser", "--")):
                    if instrument == "muser" and not include_muser:
                        continue
                    for frequency, color in zip(frequencies, colors):
                        selection = row.selections[f"{instrument}:{frequency:g}"]
                        if selection.frame is None:
                            continue
                        data, x, y, peak, _ = selected_pixels[
                            f"{instrument}:{frequency:g}"
                        ]
                        if (
                            np.isfinite(peak)
                            and peak > 0
                            and np.nanmin(data) <= 0.9 * peak <= np.nanmax(data)
                        ):
                            axis.contour(
                                x,
                                y,
                                data,
                                levels=[0.9 * peak],
                                colors=[color],
                                linewidths=1.2,
                                linestyles=style,
                            )
                axis.set(
                    xlim=fov[:2], ylim=fov[2:], aspect="equal", xlabel="HPLN [arcsec]"
                )
            axes[0].set_ylabel("HPLT [arcsec]")
            figure.suptitle(
                f"{'AIA + DART + MUSER' if include_muser else 'AIA + DART'} | {row.time.utc} UTC\n"
                f"root={plan.master_source_id} | output={plan.output_source_id} | "
                f"{index + 1}/{len(plan.rows)} | full coverage | native 90% peak contours",
                fontsize=11,
            )
            for k, (frequency, color) in enumerate(zip(frequencies, colors)):
                notes = []
                for instrument in ("dart", "muser") if include_muser else ("dart",):
                    selected = row.selections[f"{instrument}:{frequency:g}"]
                    if selected.frame is None:
                        notes.append(f"{instrument.upper()}: NO DATA")
                    else:
                        notes.append(
                            f"{instrument.upper()} {selected.frame.time.utc[11:]} Δ={selected.delta_ns / 1e9:+.3f}s"
                            + (" OVER" if not selected.within_tolerance else "")
                            + (" reused" if selected.reused else "")
                        )
                artists.append(
                    figure.text(
                        0.07,
                        0.11 - k * 0.019,
                        f"{frequency:g} MHz  " + " | ".join(notes),
                        color=color,
                        fontsize=7,
                    )
                )
            figure.canvas.draw()
            if index in samples:
                figure.savefig(output / f"preview_{index + 1:05d}.png", dpi=100)
            rgba = np.asarray(figure.canvas.buffer_rgba())
            encoded = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(rgba[:, :, :3]), format="rgb24"
            )
            encoded.pts = index
            encoded.time_base = Fraction(1, fps)
            for packet in stream.encode(encoded):
                container.mux(packet)
            references.write(
                json.dumps(
                    {
                        "row": index + 1,
                        "master_frame_id": row.master_frame.frame_id,
                        "master_utc": row.master_frame.time.utc,
                        "output_frame_id": row.output_frame.frame_id,
                        "output_utc": row.time.utc,
                        "parent_matches": row.to_dict()["parent_matches"],
                        "selections": {
                            source: value[4]
                            for source, value in selected_pixels.items()
                        },
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            if index % 50 == 0:
                _write_json(
                    output / "progress.json",
                    {
                        "phase": "rendering",
                        "frames": index + 1,
                        "total": len(plan.rows),
                    },
                )
                print(f"{output.name}: {index + 1}/{len(plan.rows)}", flush=True)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
        references.close()
        figure.clear()
        pixels.cache_clear()
    for frame in lookup.values():
        for member in frame.payload.get("pair_members", (frame.payload,)):
            stat = Path(member["path"]).stat()
            if (
                stat.st_size != member["size_bytes"]
                or stat.st_mtime_ns != member["mtime_ns"]
            ):
                raise AssertionError(
                    "Original FITS changed before final video verification"
                )
    reference_rows = validate_encoded_references(references_path, plan)
    decoded = 0
    with av.open(str(path)) as check:
        for frame in check.decode(video=0):
            if Fraction(frame.pts) * frame.time_base != Fraction(decoded, fps):
                raise AssertionError(
                    "Decoded video presentation times differ from plan order"
                )
            decoded += 1
    if decoded != len(plan.rows):
        raise AssertionError("Decoded video frame count differs from match plan")
    return {
        "decoded_frame_count": decoded,
        "verified_encoder_reference_rows": reference_rows,
        "verified_video_presentation_times": True,
        "fps": fps,
        "preview_rows": sorted(index + 1 for index in samples),
        "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _acceptance_windows(time_start=None, time_end=None):
    """Require explicit observed UTC bounds before opening inputs or outputs."""
    if time_start is None or time_end is None:
        raise ValueError("--time-start and --time-end are both required")
    start, end = normalize_time(time_start), normalize_time(time_end)
    if start.tai_ns >= end.tai_ns:
        raise ValueError("Time window must satisfy time_start < time_end")
    return {"burst": (start.utc + "Z", end.utc + "Z")}


def run_acceptance(args):
    windows = _acceptance_windows(args.time_start, args.time_end)
    requested_window = {
        "time_start": windows["burst"][0],
        "time_end": windows["burst"][1],
        "boundary": "[start,end)",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frequencies, aia_waves = tuple(args.frequencies), tuple(args.aia_waves)
    indexed, errors = audit_event(
        args.aia_directory,
        frequencies,
        args.muser_directory if "aia-dart-muser" in args.scenarios else None,
        output,
        dart_directory=args.dart_directory,
        dart_mode=args.dart_mode,
        aia_waves=aia_waves,
    )
    overview = []
    for window, (start_text, end_text) in windows.items():
        start, end = normalize_time(start_text), normalize_time(end_text)
        for scenario in dict.fromkeys(args.scenarios):
            include_muser = scenario == "aia-dart-muser"
            name = f"{window}_{'aia_dart_muser' if include_muser else 'aia_dart'}"
            destination = output / name
            destination.mkdir()
            plan, original_streams, pairs = build_event_plan(
                indexed,
                frequencies=frequencies,
                start=start,
                end=end,
                include_muser=include_muser,
                dart_mode=args.dart_mode,
                aia_waves=aia_waves,
                aia_tolerance_seconds=args.aia_tolerance_seconds,
                radio_tolerance_seconds=args.radio_tolerance_seconds,
                pair_tolerance_seconds=args.pair_tolerance_seconds,
            )
            streams = _selected_streams(plan, original_streams)
            missing = [stream.source_id for stream in streams if not stream.frames]
            if missing:
                raise ValueError(
                    f"Required real-event image streams have no valid observations in {name}: {missing}"
                )
            verification = validate_plan(
                plan,
                original_streams,
                pair_tolerance_seconds=args.pair_tolerance_seconds,
            )
            comparison = compare_previous_selection_policy(plan, streams)
            _write_json(
                destination / "resolved_config.json",
                {
                    "event": args.event_id,
                    "start_utc": start.utc,
                    "end_utc": end.utc,
                    "window_name": window,
                    "window_source": "explicit_utc",
                    "requested_window": requested_window,
                    "frequencies_mhz": frequencies,
                    "aia_waves": aia_waves,
                    "fov_arcsec": args.fov,
                    "fps": args.fps,
                    "policy": "hierarchical_monotone_full_coverage_minimum_absolute_delta",
                    "algorithm_version": 3,
                    "master_source_id": plan.master_source_id,
                    "output_source_id": plan.output_source_id,
                    "source_order": list(plan.source_order),
                    "outside_tolerance": "keep_flagged",
                    "dart_mode": args.dart_mode,
                    "rrll": (
                        "strict_one_to_one"
                        if args.dart_mode == "RR+LL"
                        else "not_applicable"
                    ),
                    "aia_tolerance_seconds": args.aia_tolerance_seconds,
                    "radio_tolerance_seconds": args.radio_tolerance_seconds,
                    "pair_tolerance_seconds": args.pair_tolerance_seconds,
                    "pairing_algorithm_version": 2,
                    "include_muser": include_muser,
                    "matching_fingerprint": plan.fingerprint,
                },
            )
            _write_json(
                destination / "input_manifest.json",
                {
                    stream.source_id: [_frame_record(frame) for frame in stream.frames]
                    for stream in streams
                },
            )
            _write_json(destination / "matching_plan.json", plan.to_dict())
            _write_json(destination / "before_after.json", comparison)
            with (destination / "matching.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "row",
                        "master_utc",
                        "output_utc",
                        "output_frame_id",
                        "root_source_id",
                        "output_source_id",
                        "source_id",
                        "frame_id",
                        "actual_utc",
                        "delta_ns",
                        "within_tolerance",
                        "use_count",
                        "reused",
                        "sorted_index",
                        "parent_source_id",
                        "parent_frame_id",
                        "parent_delta_ns",
                        "is_nearest",
                        "coverage_forced",
                    ],
                )
                writer.writeheader()
                for i, row in enumerate(plan.rows):
                    for source, item in row.selections.items():
                        writer.writerow(
                            {
                                "row": i + 1,
                                "master_utc": row.master_frame.time.utc,
                                "output_utc": row.time.utc,
                                "output_frame_id": row.output_frame.frame_id,
                                "root_source_id": plan.master_source_id,
                                "output_source_id": plan.output_source_id,
                                "source_id": source,
                                "frame_id": item.frame.frame_id if item.frame else "",
                                "actual_utc": item.frame.time.utc if item.frame else "",
                                "delta_ns": item.delta_ns,
                                "within_tolerance": item.within_tolerance,
                                "use_count": item.use_count,
                                "reused": item.reused,
                                "sorted_index": item.sorted_index,
                                "parent_source_id": row.parent_matches.get(
                                    source, {}
                                ).get("parent_source_id"),
                                "parent_frame_id": row.parent_matches.get(
                                    source, {}
                                ).get("parent_frame_id"),
                                "parent_delta_ns": row.parent_matches.get(
                                    source, {}
                                ).get("delta_ns"),
                                "is_nearest": row.parent_matches.get(source, {}).get(
                                    "is_nearest"
                                ),
                                "coverage_forced": row.parent_matches.get(
                                    source, {}
                                ).get("coverage_forced"),
                            }
                        )
            video = render_video(
                plan,
                destination,
                frequencies,
                include_muser=include_muser,
                fov=args.fov,
                fps=args.fps,
                aia_waves=aia_waves,
            )
            summary = {
                "status": "passed",
                "window_name": window,
                "start_utc": start.utc + "Z",
                "end_utc": end.utc + "Z",
                "requested_window": requested_window,
                "verification": verification,
                "sources": plan.to_dict()["summary"]["sources"],
                "rrll": pairs,
                "video": video,
                "before_after": comparison,
            }
            _write_json(destination / "matching_summary.json", summary)
            overview.append({"scenario": name, **summary})
            print(
                f"PASSED {name}: master={plan.master_source_id}, frames={len(plan.rows)}",
                flush=True,
            )
    _write_json(
        output / "acceptance.json",
        {
            "status": "passed",
            "scenarios": overview,
            "excluded_header_inputs": errors,
            "requested_window": requested_window,
            "windows": {
                name: {
                    "start_utc": bounds[0],
                    "end_utc": bounds[1],
                    "boundary": "[start,end)",
                }
                for name, bounds in windows.items()
            },
        },
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aia-directory", required=True, help="Parent of wavelength directories."
    )
    parser.add_argument(
        "--dart-directory",
        required=True,
        help="Parent of <frequency>MHz/I or RR, LL directories.",
    )
    parser.add_argument(
        "--muser-directory", help="Combined frequency/Stokes image FITS directory."
    )
    parser.add_argument(
        "--event-id", help="Optional user-supplied label, without inferred settings."
    )
    parser.add_argument("--output", required=True, help="New private output directory.")
    parser.add_argument(
        "--time-start", required=True, help="Inclusive original observed UTC."
    )
    parser.add_argument(
        "--time-end", required=True, help="Exclusive original observed UTC."
    )
    parser.add_argument("--frequencies", nargs="+", type=float, required=True)
    parser.add_argument("--aia-waves", nargs=2, type=int, required=True)
    parser.add_argument(
        "--fov",
        nargs=4,
        type=float,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    parser.add_argument("--dart-mode", choices=("I", "RR+LL"), default="RR+LL")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--aia-tolerance-seconds", type=float, default=12)
    parser.add_argument("--radio-tolerance-seconds", type=float, default=0.1)
    parser.add_argument("--pair-tolerance-seconds", type=float, default=0.1)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("aia-dart", "aia-dart-muser"),
        default=["aia-dart", "aia-dart-muser"],
    )
    args = parser.parse_args(argv)
    try:
        _acceptance_windows(args.time_start, args.time_end)
        if not 1 <= len(args.frequencies) <= 6 or len(set(args.frequencies)) != len(
            args.frequencies
        ):
            raise ValueError("Supply one to six distinct frequencies")
        if any(not math.isfinite(value) or value <= 0 for value in args.frequencies):
            raise ValueError("Frequencies must be finite and positive")
        if len(set(args.aia_waves)) != 2 or min(args.aia_waves) <= 0:
            raise ValueError("Supply two distinct positive AIA wavelengths")
        if any(not math.isfinite(value) for value in args.fov) or not (
            args.fov[0] < args.fov[1] and args.fov[2] < args.fov[3]
        ):
            raise ValueError("FOV requires finite increasing X and Y bounds")
        if args.fps <= 0:
            raise ValueError("FPS must be positive")
        if any(
            not math.isfinite(value) or value < 0
            for value in (
                args.aia_tolerance_seconds,
                args.radio_tolerance_seconds,
                args.pair_tolerance_seconds,
            )
        ):
            raise ValueError("Time tolerances must be finite and nonnegative")
        if "aia-dart-muser" in args.scenarios and not args.muser_directory:
            raise ValueError("--muser-directory is required for aia-dart-muser")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    output_existed = Path(args.output).exists()
    try:
        run_acceptance(args)
    except Exception as exc:
        output = Path(args.output)
        if not output_existed and output.is_dir():
            _write_json(
                output / "acceptance_failed.json",
                {"status": "failed", "reason": str(exc)},
            )
        raise


if __name__ == "__main__":
    main()
