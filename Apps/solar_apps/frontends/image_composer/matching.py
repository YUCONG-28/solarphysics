# SPDX-License-Identifier: MIT
"""Frame matching for time-based and relative-index compositions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Iterable
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
    seconds_to_ns,
    filter_time_frames,
)

from .models import ComposerProject, FolderSource, ImageRecord, MATCH_MODES
from .catalog import discover_images


class MatchPlanError(ValueError):
    """Raised when a project cannot produce an unambiguous match plan."""


@dataclass(slots=True, frozen=True)
class FolderMatch:
    folder_id: str
    record: ImageRecord
    corrected_timestamp: datetime
    delta_seconds: float
    over_tolerance: bool
    delta_ns: int = 0
    reused: bool = False
    use_count: int = 1
    corrected_timestamp_text: str = ""


@dataclass(slots=True, frozen=True)
class FrameMatch:
    attempt_index: int
    output_frame_index: int | None
    master_record: ImageRecord
    master_corrected_timestamp: datetime
    matches: Mapping[str, FolderMatch]
    emitted: bool
    skip_reason: str
    master_timestamp_text: str = ""
    master_tai_ns: int = 0
    output_record: ImageRecord | None = None
    output_timestamp_text: str = ""
    output_tai_ns: int = 0
    output_corrected_timestamp_text: str = ""
    parent_matches: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_record is None:
            object.__setattr__(self, "output_record", self.master_record)
        if not self.output_timestamp_text:
            stamp = normalize_time(
                self.output_record.timestamp_text or self.output_record.timestamp
            )
            object.__setattr__(self, "output_timestamp_text", stamp.utc + "Z")
            object.__setattr__(self, "output_tai_ns", stamp.tai_ns)
        if not self.output_corrected_timestamp_text:
            object.__setattr__(
                self,
                "output_corrected_timestamp_text",
                self.master_timestamp_text
                or normalize_time(self.master_corrected_timestamp).utc + "Z",
            )
        object.__setattr__(self, "parent_matches", _freeze_json(self.parent_matches))


def validate_project(project: ComposerProject, *, require_output: bool = False) -> None:
    if project.schema_version != 2:
        raise MatchPlanError(f"Unsupported project schema: {project.schema_version}")
    if project.canvas.width < 2 or project.canvas.height < 2:
        raise MatchPlanError("Canvas width and height must be at least 2 pixels.")
    if project.canvas.width % 2 or project.canvas.height % 2:
        raise MatchPlanError("Canvas width and height must both be even.")
    if not project.slots:
        raise MatchPlanError("Add at least one image slot to the canvas.")
    if project.matching.mode not in MATCH_MODES:
        raise MatchPlanError(f"Unsupported match mode: {project.matching.mode}")
    matching_window(project)
    if (
        not math.isfinite(project.matching.tolerance_seconds)
        or project.matching.tolerance_seconds < 0
    ):
        raise MatchPlanError("Time tolerance cannot be negative.")

    if (
        project.matching.master_policy
        not in {"auto_min_frames_hierarchical", "auto_max_frames"}
        or project.matching.overflow_policy != "keep_flagged"
    ):
        raise MatchPlanError(
            "Composer requires auto_min_frames_hierarchical and keep_flagged policies."
        )
    folders = project.folder_map()

    for slot in project.slots:
        folder = folders.get(slot.folder_id)
        if folder is None:
            raise MatchPlanError(f"Slot {slot.id} references a missing folder.")
        _validate_folder_range(folder)
        if not math.isfinite(folder.offset_seconds):
            raise MatchPlanError("Clock offsets must be finite.")
        if slot.width <= 0 or slot.height <= 0:
            raise MatchPlanError(f"Slot {slot.id} has an invalid size.")
        if slot.fit not in {"contain", "cover", "stretch"}:
            raise MatchPlanError(f"Slot {slot.id} has an invalid fit mode.")
        if not 0 <= slot.opacity <= 1:
            raise MatchPlanError(f"Slot {slot.id} has invalid opacity.")

    if require_output:
        if not project.export.output_path:
            raise MatchPlanError("Select an output video path.")
        if project.export.output_format not in {"mp4", "avi"}:
            raise MatchPlanError("Output format must be MP4 or AVI.")
        if project.export.fps <= 0 or project.export.fps > 60:
            raise MatchPlanError("FPS must be greater than 0 and no more than 60.")


class StaleMatchPlanError(MatchPlanError):
    """Inputs changed after preview; rescan before exporting a new plan."""


def matching_window(project: ComposerProject) -> dict[str, object]:
    """Validate the observed-UTC interval independently of clock corrections."""

    try:
        start = (
            normalize_time(project.matching.time_start)
            if project.matching.time_start is not None
            else None
        )
        end = (
            normalize_time(project.matching.time_end)
            if project.matching.time_end is not None
            else None
        )
        if start is not None and end is not None and start.tai_ns >= end.tai_ns:
            raise ValueError("Time window must satisfy time_start < time_end")
    except (TypeError, ValueError) as exc:
        raise MatchPlanError(str(exc)) from exc
    return {
        "start_utc": start.utc if start else None,
        "end_utc": end.utc if end else None,
        "boundary": "[start,end)",
        "clock": "observed_utc",
    }


def in_window_records(
    project: ComposerProject, folder: FolderSource
) -> list[ImageRecord]:
    """Count valid original records before choosing the fullest source."""

    frames = tuple(
        TimeFrame(
            str(record.path.resolve()),
            normalize_time(record.timestamp_text or record.timestamp),
            record,
        )
        for record in folder.selected_records()
        if record.readable
    )
    return [
        frame.payload
        for frame in filter_time_frames(
            frames,
            time_start=project.matching.time_start,
            time_end=project.matching.time_end,
        )
    ]


def _freeze_json(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ComposerMatchPlan(Sequence[FrameMatch]):
    rows: tuple[FrameMatch, ...]
    fingerprint: str
    master_folder_id: str | None
    summary: Mapping
    core_fingerprint: str
    output_folder_id: str | None = None
    source_order: tuple[str, ...] = ()
    parent_maps: Mapping = field(default_factory=dict)
    settings_fingerprint: str = ""
    source_fingerprint: str = ""

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "summary", _freeze_json(self.summary))
        if self.output_folder_id is None:
            object.__setattr__(self, "output_folder_id", self.master_folder_id)
        object.__setattr__(self, "source_order", tuple(self.source_order))
        object.__setattr__(self, "parent_maps", _freeze_json(self.parent_maps))

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)

    def __deepcopy__(self, memo):
        # Frozen records and timestamps are independent of scaled canvas geometry.
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm_version": 3,
            "window": _thaw_json(self.summary["window"]),
            "fingerprint": self.fingerprint,
            "settings_fingerprint": self.settings_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "master_folder_id": self.master_folder_id,
            "output_folder_id": self.output_folder_id,
            "source_order": list(self.source_order),
            "parent_maps": _thaw_json(self.parent_maps),
            "summary": _thaw_json(self.summary),
            "rows": [
                {
                    "master_locator": str(row.master_record.path.resolve()),
                    "output_locator": str(row.output_record.path.resolve()),
                    "time_utc": row.output_timestamp_text,
                    "output_corrected_time_utc": row.output_corrected_timestamp_text,
                    "master_corrected_time_utc": row.master_timestamp_text,
                    "parent_matches": _thaw_json(row.parent_matches),
                    "matches": {
                        key: str(value.record.path.resolve())
                        for key, value in sorted(row.matches.items())
                    },
                }
                for row in self
            ],
        }


def _utc_datetime(text: str) -> datetime:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _input_signature(project: ComposerProject, core_fingerprint: str) -> str:
    used = {slot.folder_id for slot in project.slots}
    inputs = []
    for folder in sorted(project.folders, key=lambda item: item.id):
        if folder.id not in used:
            continue
        records = []
        for path in discover_images(folder.path):
            try:
                stat = path.stat()
            except OSError as exc:
                raise StaleMatchPlanError(
                    f"Image input is unavailable: {path}"
                ) from exc
            records.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        inputs.append(
            (
                folder.id,
                folder.start_index,
                folder.end_index,
                float(folder.offset_seconds),
                tuple(
                    (
                        str(record.path.resolve()),
                        record.timestamp_text or record.timestamp.isoformat(),
                        record.readable,
                    )
                    for record in folder.selected_records()
                ),
                records,
            )
        )
    payload = json.dumps(
        {
            "core": core_fingerprint,
            "inputs": inputs,
            "mode": project.matching.mode,
            "master_policy": project.matching.master_policy,
            "tolerance_seconds": float(project.matching.tolerance_seconds),
            "window": matching_window(project),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _settings_signature(project: ComposerProject) -> str:
    used = {slot.folder_id for slot in project.slots}
    values = {
        "folders": [
            (folder.id, str(folder.path.resolve()), folder.start_index,
             folder.end_index, float(folder.offset_seconds))
            for folder in sorted(project.folders, key=lambda item: item.id)
            if folder.id in used
        ],
        "mode": project.matching.mode,
        "master_policy": project.matching.master_policy,
        "tolerance_seconds": float(project.matching.tolerance_seconds),
        "window": matching_window(project),
    }
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def _source_signature(project: ComposerProject) -> str:
    """Check original observations separately from deliberate range edits."""
    used = {slot.folder_id for slot in project.slots}
    values = []
    for folder in sorted(project.folders, key=lambda item: item.id):
        if folder.id not in used:
            continue
        files = []
        for path in discover_images(folder.path):
            try:
                stat = path.stat()
            except OSError as exc:
                raise StaleMatchPlanError(f"Image input is unavailable: {path}") from exc
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        values.append((
            folder.id,
            files,
            [(str(record.path.resolve()), record.timestamp_text or record.timestamp.isoformat(),
              record.readable) for record in folder.records],
        ))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()


def build_match_plan(project: ComposerProject) -> ComposerMatchPlan:
    """Expand the sparsest root through successive full-coverage parent maps."""

    if project.matching.master_policy == "auto_max_frames":
        project.matching.master_policy = "auto_min_frames_hierarchical"
        project.sequence_plan = None
        project._match_plan_cache = None
        project.migration_note = (
            "Migrated legacy maximum-frame master policy to strategy 3: "
            "the fewest-frame source is the root, and successive parent maps expand to the output source."
        )
    validate_project(project)
    window = matching_window(project)
    settings_fingerprint = _settings_signature(project)
    source_fingerprint = _source_signature(project)
    if project.sequence_plan is not None:
        saved = project.sequence_plan
        if int(saved.get("algorithm_version", 1)) in {1, 2}:
            project.sequence_plan = None
            project._match_plan_cache = None
            project.migration_note = (
                "Rebuilt legacy strategy 1/2 matching with strategy 3: "
                "the fewest-frame source is the root, with successive full-coverage parent maps."
            )
        elif int(saved.get("algorithm_version", 1)) != 3:
            raise MatchPlanError("Unsupported saved matching strategy version.")
        elif (
            saved.get("settings_fingerprint")
            and saved["settings_fingerprint"] != settings_fingerprint
        ):
            if saved.get("source_fingerprint") != source_fingerprint:
                raise StaleMatchPlanError(
                    "The preview matching plan is stale; rescan the folders and rebuild it before export."
                )
            # A new working selection must not invalidate an already queued snapshot.
            project.sequence_plan = None
            project._match_plan_cache = None
        elif saved.get("window") != window:
            project.sequence_plan = None
            project._match_plan_cache = None
    cached = project._match_plan_cache
    if (
        isinstance(cached, ComposerMatchPlan)
        and _input_signature(project, cached.core_fingerprint) == cached.fingerprint
    ):
        if (
            project.sequence_plan is not None
            and project.sequence_plan != cached.to_dict()
        ):
            raise StaleMatchPlanError(
                "The preview matching plan is stale; rescan the folders and rebuild it before export."
            )
        return cached
    folders = project.folder_map()
    used_ids = sorted({slot.folder_id for slot in project.slots})
    streams = []
    for folder_id in used_ids:
        folder = folders[folder_id]
        frames = tuple(
            TimeFrame(
                frame_id=str(record.path.resolve()),
                time=normalize_time(
                    record.timestamp_text or record.timestamp, source=folder_id
                ),
                payload=record,
            )
            for record in folder.selected_records()
            if record.readable
        )
        streams.append(
            TimeStream(
                folder_id,
                frames,
                tolerance_seconds=project.matching.tolerance_seconds,
                offset_seconds=folder.offset_seconds,
            )
        )
    core = build_sequence_match_plan(
        streams,
        time_start=project.matching.time_start,
        time_end=project.matching.time_end,
    )
    fingerprint = _input_signature(project, core.fingerprint)
    rows = []
    for index, row in enumerate(core.rows, start=1):
        root_selection = row.selections[core.master_source_id]
        master_key = root_selection.match_time or row.master_frame.time
        master_time = _utc_datetime(master_key.utc)
        matches = {}
        for folder_id, selected in row.selections.items():
            if selected.frame is None:
                continue
            record = selected.frame.payload
            corrected = selected.match_time or selected.frame.time
            matches[folder_id] = FolderMatch(
                folder_id,
                record,
                _utc_datetime(corrected.utc),
                selected.delta_ns / 1_000_000_000,
                selected.within_tolerance is False,
                delta_ns=selected.delta_ns,
                reused=selected.reused,
                use_count=selected.use_count,
                corrected_timestamp_text=corrected.utc + "Z",
            )
        rows.append(
            FrameMatch(
                index,
                index,
                row.master_frame.payload,
                master_time,
                MappingProxyType(matches),
                True,
                "",
                master_key.utc + "Z",
                master_key.tai_ns,
                output_record=row.output_frame.payload,
                output_timestamp_text=row.time.utc + "Z",
                output_tai_ns=row.time.tai_ns,
                output_corrected_timestamp_text=(row.match_time or row.time).utc + "Z",
                parent_matches=row.parent_matches,
            )
        )
    plan = ComposerMatchPlan(
        rows,
        fingerprint=fingerprint,
        master_folder_id=core.master_source_id,
        summary=core.to_dict()["summary"],
        core_fingerprint=core.fingerprint,
        output_folder_id=core.output_source_id,
        source_order=core.source_order,
        parent_maps=core.parent_maps,
        settings_fingerprint=settings_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    if project.sequence_plan is not None and project.sequence_plan != plan.to_dict():
        raise StaleMatchPlanError(
            "The preview matching plan is stale; rescan the folders and rebuild it before export."
        )
    project._match_plan_cache = plan
    return plan


def freeze_match_plan(project: ComposerProject) -> ComposerMatchPlan:
    plan = build_match_plan(project)
    project.sequence_plan = plan.to_dict()
    project.matching.master_folder_id = plan.master_folder_id
    return plan


def select_plan_frame(
    plan: ComposerMatchPlan, current_time_utc: datetime | str
) -> FrameMatch:
    if not plan:
        raise MatchPlanError("No valid images in the selected UTC time window.")
    target = normalize_time(current_time_utc)
    return min(
        plan,
        key=lambda row: (
            abs(row.output_tai_ns - target.tai_ns),
            row.output_tai_ns,
            row.attempt_index,
        ),
    )


def nearest_record(
    folder: FolderSource,
    candidates: Iterable[ImageRecord],
    master_time: datetime,
) -> ImageRecord:
    """Choose nearest time, then earlier time, then lower natural ordinal."""

    if master_time.tzinfo is None:
        master_time = master_time.replace(tzinfo=UTC)
    items = list(candidates)
    if not items:
        raise MatchPlanError(f"Folder has no selected images: {folder.path}")
    target_ns = normalize_time(master_time).tai_ns
    offset_ns = seconds_to_ns(folder.offset_seconds)
    timed = [
        (
            record,
            normalize_time(record.timestamp_text or record.timestamp).tai_ns
            + offset_ns,
        )
        for record in items
    ]
    return min(
        timed, key=lambda item: (abs(item[1] - target_ns), item[1], item[0].ordinal)
    )[0]


def iter_match_rows(
    project: ComposerProject, plan: Iterable[FrameMatch]
) -> Iterable[dict[str, Any]]:
    """Yield long-form rows for every attempted master frame and canvas slot."""

    if not plan:
        return
    folders = project.folder_map()
    master_id = getattr(plan, "master_folder_id", project.matching.master_folder_id)
    master = folders[master_id]
    output = folders[getattr(plan, "output_folder_id", None) or master_id]
    for frame in plan:
        for slot in sorted(project.slots, key=lambda item: item.z_index):
            folder = folders[slot.folder_id]
            match = frame.matches.get(slot.folder_id)
            parent = frame.parent_matches.get(slot.folder_id, {})
            yield {
                "attempt_index": frame.attempt_index,
                "output_frame_index": frame.output_frame_index or "",
                "emitted": frame.emitted,
                "skip_reason": frame.skip_reason,
                "match_mode": project.matching.mode,
                "strict": project.matching.strict,
                "tolerance_seconds": project.matching.tolerance_seconds,
                "master_folder_id": master.id,
                "master_folder": master.name,
                "master_file": str(frame.master_record.path),
                "master_ordinal": frame.master_record.ordinal,
                "master_timestamp": frame.master_record.timestamp_text
                or frame.master_record.timestamp.isoformat(timespec="microseconds"),
                "master_time_source": frame.master_record.time_source,
                "master_offset_seconds": master.offset_seconds,
                "master_corrected_timestamp": frame.master_timestamp_text
                or frame.master_corrected_timestamp.isoformat(timespec="microseconds"),
                "output_folder_id": output.id,
                "output_folder": output.name,
                "output_file": str(frame.output_record.path),
                "output_ordinal": frame.output_record.ordinal,
                "output_timestamp": frame.output_timestamp_text,
                "output_corrected_timestamp": frame.output_corrected_timestamp_text,
                "parent_source_id": parent.get("parent_source_id", ""),
                "parent_frame_id": parent.get("parent_frame_id", ""),
                "parent_delta_ns": parent.get("delta_ns", ""),
                "parent_is_nearest": parent.get("is_nearest", ""),
                "parent_coverage_forced": parent.get("coverage_forced", ""),
                "slot_id": slot.id,
                "slot_z_index": slot.z_index,
                "source_folder_id": folder.id,
                "source_folder": folder.name,
                "source_file": str(match.record.path) if match else "",
                "source_ordinal": match.record.ordinal if match else "",
                "source_timestamp": (
                    match.record.timestamp_text
                    or match.record.timestamp.isoformat(timespec="microseconds")
                )
                if match
                else "",
                "source_time_source": match.record.time_source if match else "",
                "source_offset_seconds": folder.offset_seconds,
                "source_corrected_timestamp": (
                    match.corrected_timestamp_text
                    or match.corrected_timestamp.isoformat(timespec="microseconds")
                )
                if match
                else "",
                "delta_seconds": match.delta_seconds if match else "",
                "abs_delta_seconds": abs(match.delta_seconds) if match else "",
                "over_tolerance": match.over_tolerance if match else "",
                "delta_ns": match.delta_ns if match else "",
                "reused": match.reused if match else False,
                "use_count": match.use_count if match else 0,
                "missing_data": match is None,
                "time_start_utc": matching_window(project)["start_utc"] or "",
                "time_end_utc": matching_window(project)["end_utc"] or "",
                "overflow_policy": project.matching.overflow_policy,
                "plan_fingerprint": getattr(plan, "fingerprint", ""),
            }


def _validate_folder_range(folder: FolderSource) -> None:
    if not folder.resolved or not folder.path.is_dir():
        raise MatchPlanError(f"Folder is unresolved: {folder.path}")
    count = len(folder.records)
    if count == 0:
        raise MatchPlanError(f"Folder contains no supported images: {folder.path}")
    if folder.start_index < 1:
        raise MatchPlanError(f"Folder start index must be at least 1: {folder.name}")
    if folder.end_index < folder.start_index:
        raise MatchPlanError(f"Folder range is reversed: {folder.name}")
    if folder.end_index > count:
        raise MatchPlanError(
            f"Folder end index {folder.end_index} exceeds {count}: {folder.name}"
        )
