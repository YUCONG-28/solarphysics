"""Deterministic full-coverage sequence plans and strict one-to-one pairing."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .precision import TimeStamp, normalize_time, seconds_to_ns

ALGORITHM_VERSION = 3
_ALGORITHM = "hierarchical_monotone_full_coverage_minimum_absolute_delta"


@dataclass(frozen=True)
class TimeFrame:
    frame_id: str
    time: TimeStamp
    payload: Any = None
    match_time: TimeStamp | None = None
    member_times: tuple[TimeStamp, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "member_times", tuple(self.member_times))

    def to_dict(self):
        # Payloads remain exact in-process locators. They need not be JSON data
        # and must not destabilize the fingerprint through object reprs.
        return {
            "frame_id": self.frame_id,
            "time": self.time.to_dict(),
            "match_time": self.match_time.to_dict() if self.match_time else None,
            "member_times": [stamp.to_dict() for stamp in self.member_times],
        }


@dataclass(frozen=True)
class TimeStream:
    source_id: str
    frames: Sequence[TimeFrame]
    tolerance_seconds: Any = 0
    offset_seconds: Any = 0


@dataclass(frozen=True)
class TimeSelection:
    frame: TimeFrame | None
    delta_ns: int | None
    within_tolerance: bool | None
    use_count: int
    reused: bool
    original_index: int | None
    sorted_index: int | None
    match_time: TimeStamp | None = None

    def to_dict(self):
        return {
            "frame": self.frame.to_dict() if self.frame else None,
            "delta_ns": self.delta_ns,
            "within_tolerance": self.within_tolerance,
            "use_count": self.use_count,
            "reused": self.reused,
            "original_index": self.original_index,
            "sorted_index": self.sorted_index,
            "match_time": self.match_time.to_dict() if self.match_time else None,
        }


@dataclass(frozen=True)
class SequenceMatchRow:
    time: TimeStamp
    master_frame: TimeFrame
    selections: Mapping[str, TimeSelection]
    match_time: TimeStamp | None = None
    output_frame: TimeFrame | None = None
    parent_matches: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "selections", MappingProxyType(dict(self.selections)))
        object.__setattr__(self, "parent_matches", _freeze(self.parent_matches))

    def to_dict(self):
        return {
            "time": self.time.to_dict(),
            "master_frame": self.master_frame.to_dict(),
            "selections": {
                key: item.to_dict() for key, item in self.selections.items()
            },
            "match_time": self.match_time.to_dict() if self.match_time else None,
            "output_frame": self.output_frame.to_dict() if self.output_frame else None,
            "parent_matches": _plain(self.parent_matches),
        }


@dataclass(frozen=True)
class SequenceMatchPlan:
    master_source_id: str | None
    rows: tuple[SequenceMatchRow, ...]
    fingerprint: str
    summary: Mapping[str, Any]
    output_source_id: str | None = None
    source_order: tuple[str, ...] = ()
    parent_maps: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "summary", _freeze(self.summary))
        object.__setattr__(self, "source_order", tuple(self.source_order))
        object.__setattr__(self, "parent_maps", _freeze(self.parent_maps))

    def to_dict(self):
        return {
            "schema": "solarphysics.time.sequence_match_plan",
            "version": 1,
            "algorithm": _ALGORITHM,
            "algorithm_version": ALGORITHM_VERSION,
            "master_source_id": self.master_source_id,
            "fingerprint": self.fingerprint,
            "output_source_id": self.output_source_id,
            "source_order": list(self.source_order),
            "parent_maps": _plain(self.parent_maps),
            "window": _plain(self.summary.get("window", _window_bounds(None, None)[2])),
            "summary": _plain(self.summary),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class OneToOneMatch:
    left: TimeFrame
    right: TimeFrame
    delta_ns: int
    left_original_index: int
    right_original_index: int
    left_sorted_index: int
    right_sorted_index: int

    def to_dict(self):
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "delta_ns": self.delta_ns,
            "left_original_index": self.left_original_index,
            "right_original_index": self.right_original_index,
            "left_sorted_index": self.left_sorted_index,
            "right_sorted_index": self.right_sorted_index,
        }


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _ordered(frames):
    indexed = list(enumerate(frames))
    for _, frame in indexed:
        if not isinstance(frame, TimeFrame) or not isinstance(frame.time, TimeStamp):
            raise ValueError("Every frame requires a normalized observation timestamp")
        if not isinstance(frame.frame_id, str) or not frame.frame_id:
            raise ValueError("frame_id must be a nonempty string")
        if frame.match_time is not None and not isinstance(frame.match_time, TimeStamp):
            raise ValueError("match_time must be a normalized timestamp")
        if any(not isinstance(stamp, TimeStamp) for stamp in frame.member_times):
            raise ValueError(
                "member_times must contain normalized observation timestamps"
            )
    return sorted(
        indexed, key=lambda item: (item[1].time.tai_ns, item[1].frame_id, item[0])
    )


def _clock(frame, offset=0):
    return int((frame.match_time or frame.time).tai_ns) + offset


def _window_bounds(time_start, time_end):
    start = normalize_time(time_start) if time_start is not None else None
    end = normalize_time(time_end) if time_end is not None else None
    if start is not None and end is not None and start.tai_ns >= end.tai_ns:
        raise ValueError("Time window must satisfy time_start < time_end")
    window = {
        "start_utc": start.utc if start else None,
        "end_utc": end.utc if end else None,
        "boundary": "[start,end)",
        "clock": "observed_utc",
    }
    return start.tai_ns if start else None, end.tai_ns if end else None, window


def _inside_window(frame, start, end):
    if start is None and end is None:
        return True
    times = [frame.time.tai_ns, *(stamp.tai_ns for stamp in frame.member_times)]
    payload = frame.payload
    logical_frames = getattr(payload, "logical_frames", ())
    times.extend(member.time.tai_ns for member in logical_frames)
    payload = getattr(payload, "payload", payload)
    if isinstance(payload, Mapping):
        members = payload.get("actual_times_tai_ns")
        if members is not None:
            if payload.get("pair_members") and len(members) != len(
                payload["pair_members"]
            ):
                raise ValueError(
                    "Paired observations require a time for every original member"
                )
            times.extend(members)
        elif payload.get("actual_times_utc") is not None:
            if payload.get("pair_members") and len(payload["actual_times_utc"]) != len(
                payload["pair_members"]
            ):
                raise ValueError(
                    "Paired observations require a time for every original member"
                )
            times.extend(
                normalize_time(value).tai_ns for value in payload["actual_times_utc"]
            )
        elif payload.get("pair_members") and not frame.member_times:
            raise ValueError(
                "Paired observations require original member times for window filtering"
            )
        if payload.get("pair_members") and len(times) == 1:
            raise ValueError(
                "Paired observations require original member times for window filtering"
            )
    return all(
        (start is None or value >= start) and (end is None or value < end)
        for value in times
    )


def filter_time_frames(frames, *, time_start=None, time_end=None):
    """Return original frames inside the observed-UTC half-open window.

    Scientific matching epochs and offsets never change eligibility. Every
    original member of an already synthesized pair must also be in the window.
    Input order and frame objects are preserved; sorting belongs to the planner.
    """
    start, end, _ = _window_bounds(time_start, time_end)
    frames = list(frames)
    _ordered(frames)  # Validate identities/timestamps even for an empty window.
    return [frame for frame in frames if _inside_window(frame, start, end)]


def _tolerance_ns(seconds):
    return seconds_to_ns(seconds, nonnegative=True)


def _full_coverage_indices(master, other):
    """Stay/advance DP from each child to a parent; cover every parent once."""
    n, m = len(master), len(other)
    if not m:
        return [None] * n
    if n < m:
        raise ValueError(
            "The master must have at least as many frames as every reference"
        )
    if n == m:
        return list(range(n))
    if m == 1:
        return [0] * n
    previous = [None] * m
    directions = [bytearray(m) for _ in range(n)]
    for i in range(n):
        current = [None] * m
        lower, upper = max(0, m - n + i), min(i, m - 1)
        for j in range(lower, upper + 1):
            cost = abs(master[i] - other[j])
            if i == 0:
                current[j] = cost
                continue
            stay = previous[j]
            advance = previous[j - 1] if j else None
            # Equal total costs deliberately advance, making backtracking
            # deterministic without depending on container iteration order.
            if advance is not None and (stay is None or advance <= stay):
                current[j] = advance + cost
                directions[i][j] = 1
            elif stay is not None:
                current[j] = stay + cost
        previous = current
    indices = [0] * n
    j = m - 1
    for i in range(n - 1, -1, -1):
        indices[i] = j
        if directions[i][j]:
            j -= 1
    return indices


def build_sequence_match_plan(
    streams: Sequence[TimeStream], *, time_start=None, time_end=None
) -> SequenceMatchPlan:
    """Match adjacent count-ordered layers and expand the complete output layer.

    Filter every stream to [time_start,time_end) before counting or matching.
    The smallest nonempty stream is the root; the largest last stream owns rows.
    Observed UTC determines eligibility and ordering, including when an explicit scientific
    ``match_time`` or clock offset determines cost. Tolerance is a quality flag,
    never a deletion gate. All costs and deltas use Python integer nanoseconds.
    """
    start, end, window = _window_bounds(time_start, time_end)
    streams = list(streams)
    if any(not isinstance(stream, TimeStream) for stream in streams):
        raise ValueError("streams must contain TimeStream instances")
    ids = [stream.source_id for stream in streams]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(
        ids
    ):
        raise ValueError("source_id values must be nonempty and unique")
    streams.sort(key=lambda stream: stream.source_id)
    ordered, tolerances, offsets, effective = {}, {}, {}, {}
    inputs, input_counts = [], {}
    for stream in streams:
        key = stream.source_id
        all_ordered = _ordered(stream.frames)
        input_counts[key] = len(all_ordered)
        ordered[key] = [
            item for item in all_ordered if _inside_window(item[1], start, end)
        ]
        tolerances[key] = _tolerance_ns(stream.tolerance_seconds)
        offsets[key] = seconds_to_ns(stream.offset_seconds)
        inputs.append(
            {
                "source_id": key,
                "tolerance_ns": tolerances[key],
                "offset_ns": offsets[key],
                "frames": [
                    {"original_index": original, **frame.to_dict()}
                    for original, frame in ordered[key]
                ],
            }
        )
        effective[key] = [
            (
                (frame.match_time or frame.time).shift(stream.offset_seconds)
                if offsets[key]
                else (frame.match_time or frame.time)
            )
            for _, frame in ordered[key]
        ]
    source_order = tuple(
        sorted(
            (key for key in ids if ordered[key]),
            key=lambda key: (len(ordered[key]), key),
        )
    )
    policy = {
        "version": ALGORITHM_VERSION,
        "window": window,
        "source_order": source_order,
        "ordering": "observed_tai_ns_frame_id_original_index",
        "policy": _ALGORITHM,
        "equal_cost": "advance",
        "streams": inputs,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            policy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    master_id = source_order[0] if source_order else None
    output_id = source_order[-1] if source_order else None
    output = ordered[output_id] if output_id is not None else []
    n = len(output)
    clocks = {key: [stamp.tai_ns for stamp in effective[key]] for key in ids}
    parent_indices, parent_maps = {}, {}
    for parent_id, child_id in zip(source_order[:-1], source_order[1:], strict=True):
        indices = _full_coverage_indices(clocks[child_id], clocks[parent_id])
        parent_indices[child_id] = indices
        nearest_clocks = sorted(set(clocks[parent_id]))
        edges = []
        for child_index, parent_index in enumerate(indices):
            parent_original, parent = ordered[parent_id][parent_index]
            child_original, child = ordered[child_id][child_index]
            child_clock = clocks[child_id][child_index]
            insertion = bisect_left(nearest_clocks, child_clock)
            nearest = min(
                abs(nearest_clocks[index] - child_clock)
                for index in (insertion - 1, insertion)
                if 0 <= index < len(nearest_clocks)
            )
            delta = clocks[parent_id][parent_index] - child_clock
            edges.append(
                {
                    "parent_source_id": parent_id,
                    "child_source_id": child_id,
                    "parent_frame_id": parent.frame_id,
                    "child_frame_id": child.frame_id,
                    "parent_sorted_index": parent_index,
                    "child_sorted_index": child_index,
                    "parent_original_index": parent_original,
                    "child_original_index": child_original,
                    "parent_time": parent.time.to_dict(),
                    "child_time": child.time.to_dict(),
                    "parent_match_time": effective[parent_id][parent_index].to_dict(),
                    "child_match_time": effective[child_id][child_index].to_dict(),
                    "delta_ns": delta,
                    "nearest_distance_ns": nearest,
                    "is_nearest": abs(delta) == nearest,
                    "coverage_forced": abs(delta) > nearest,
                }
            )
        parent_maps[child_id] = tuple(edges)
    chosen = {key: [None] * n for key in ids}
    if output_id is not None:
        chosen[output_id] = list(range(n))
        for parent_id, child_id in reversed(
            list(zip(source_order[:-1], source_order[1:], strict=True))
        ):
            chosen[parent_id] = [
                parent_indices[child_id][index] for index in chosen[child_id]
            ]
    counts = {
        key: Counter(index for index in chosen[key] if index is not None) for key in ids
    }
    rows = []
    for i, (_, frame) in enumerate(output):
        selections = {}
        for stream in streams:
            key, j = stream.source_id, chosen[stream.source_id][i]
            if j is None:
                selections[key] = TimeSelection(None, None, None, 0, False, None, None)
                continue
            original, selected = ordered[key][j]
            delta = clocks[key][j] - clocks[output_id][i]
            count = counts[key][j]
            selections[key] = TimeSelection(
                selected,
                delta,
                abs(delta) <= tolerances[key],
                count,
                count > 1,
                original,
                j,
                effective[key][j],
            )
        root_frame = ordered[master_id][chosen[master_id][i]][1]
        edges = {
            child_id: parent_maps[child_id][chosen[child_id][i]]
            for child_id in source_order[1:]
        }
        rows.append(
            SequenceMatchRow(
                frame.time,
                root_frame,
                selections,
                effective[output_id][i],
                frame,
                edges,
            )
        )
    source_summary = {}
    for stream in streams:
        key = stream.source_id
        selections = [row.selections[key] for row in rows]
        source_summary[key] = {
            "input_frame_count": input_counts[key],
            "in_window_frame_count": len(ordered[key]),
            "outside_window_frame_count": input_counts[key] - len(ordered[key]),
            "frame_count": len(ordered[key]),
            "used_frame_count": len(counts[key]),
            "coverage_fraction": (
                len(counts[key]) / len(ordered[key]) if ordered[key] else None
            ),
            "missing_count": sum(item.frame is None for item in selections),
            "outside_tolerance_count": sum(
                item.within_tolerance is False for item in selections
            ),
            "reused_row_count": sum(item.reused for item in selections),
            "extra_uses": sum(counts[key].values()) - len(counts[key]),
            "non_nearest_parent_frame_count": sum(
                edge["coverage_forced"] for edge in parent_maps.get(key, ())
            ),
            "total_abs_delta_ns": sum(
                abs(item.delta_ns) for item in selections if item.delta_ns is not None
            ),
        }
    summary = {
        "master_source_id": master_id,
        "output_source_id": output_id,
        "source_order": source_order,
        "row_count": n,
        "window": window,
        "parent_map_summary": {
            child: {
                "parent_source_id": edges[0]["parent_source_id"],
                "child_frame_count": len(edges),
                "total_abs_delta_ns": sum(abs(edge["delta_ns"]) for edge in edges),
                "non_nearest_count": sum(edge["coverage_forced"] for edge in edges),
            }
            for child, edges in parent_maps.items()
        },
        "sources": source_summary,
    }
    return SequenceMatchPlan(
        master_id,
        tuple(rows),
        fingerprint,
        summary,
        output_id,
        source_order,
        parent_maps,
    )


def monotone_one_to_one(
    left: Sequence[TimeFrame],
    right: Sequence[TimeFrame],
    *,
    tolerance_seconds,
    compatibility: Callable[[TimeFrame, TimeFrame], bool] | None = None,
) -> list[OneToOneMatch]:
    """Strict pairs: maximize pair count, then minimize total absolute delta.

    Each original frame is used at most once. Incompatible/outside-tolerance
    pairs are forbidden. Exact score ties prefer a pair, then skipping a right
    frame, then skipping a left frame. Unpaired originals remain caller-owned.
    """
    tolerance = _tolerance_ns(tolerance_seconds)
    aa, bb = _ordered(left), _ordered(right)
    n, m = len(aa), len(bb)
    if n == m and all(
        abs(_clock(b) - _clock(a)) <= tolerance
        and (compatibility is None or compatibility(a, b))
        for (_, a), (_, b) in zip(aa, bb, strict=True)
    ):
        # A monotone N-to-N full matching has exactly one possible path.
        return [
            OneToOneMatch(a, b, _clock(b) - _clock(a), li, ri, j, j)
            for j, ((li, a), (ri, b)) in enumerate(zip(aa, bb, strict=True))
        ]
    previous = [(0, 0)] * (m + 1)
    directions = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        current = [(0, 0)] * (m + 1)
        for j in range(1, m + 1):
            # Score is (count, negative cost), so ordinary tuple comparison
            # implements the scientific priority without fixed-width overflow.
            best, action = current[j - 1], 2
            if previous[j] > best:
                best, action = previous[j], 1
            delta = _clock(bb[j - 1][1]) - _clock(aa[i - 1][1])
            if abs(delta) <= tolerance and (
                compatibility is None or compatibility(aa[i - 1][1], bb[j - 1][1])
            ):
                prior = previous[j - 1]
                candidate = (prior[0] + 1, prior[1] - abs(delta))
                if candidate >= best:
                    best, action = candidate, 3
            current[j], directions[i][j] = best, action
        previous = current
    pairs, i, j = [], n, m
    while i and j:
        action = directions[i][j]
        if action == 3:
            li, a = aa[i - 1]
            ri, b = bb[j - 1]
            pairs.append(
                OneToOneMatch(a, b, _clock(b) - _clock(a), li, ri, i - 1, j - 1)
            )
            i, j = i - 1, j - 1
        elif action == 2:
            j -= 1
        else:
            i -= 1
    return list(reversed(pairs))


__all__ = [
    "TimeFrame",
    "TimeStream",
    "TimeSelection",
    "SequenceMatchRow",
    "SequenceMatchPlan",
    "OneToOneMatch",
    "build_sequence_match_plan",
    "monotone_one_to_one",
    "filter_time_frames",
    "ALGORITHM_VERSION",
]
