# SPDX-License-Identifier: MIT
"""Frame matching for time-based and relative-index compositions."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from .models import ComposerProject, FolderSource, ImageRecord, MATCH_MODES


class MatchPlanError(ValueError):
    """Raised when a project cannot produce an unambiguous match plan."""


@dataclass(slots=True, frozen=True)
class FolderMatch:
    folder_id: str
    record: ImageRecord
    corrected_timestamp: datetime
    delta_seconds: float
    over_tolerance: bool


@dataclass(slots=True, frozen=True)
class FrameMatch:
    attempt_index: int
    output_frame_index: int | None
    master_record: ImageRecord
    master_corrected_timestamp: datetime
    matches: dict[str, FolderMatch]
    emitted: bool
    skip_reason: str


def validate_project(project: ComposerProject, *, require_output: bool = False) -> None:
    if project.schema_version != 1:
        raise MatchPlanError(f"Unsupported project schema: {project.schema_version}")
    if project.canvas.width < 2 or project.canvas.height < 2:
        raise MatchPlanError("Canvas width and height must be at least 2 pixels.")
    if project.canvas.width % 2 or project.canvas.height % 2:
        raise MatchPlanError("Canvas width and height must both be even.")
    if not project.slots:
        raise MatchPlanError("Add at least one image slot to the canvas.")
    if project.matching.mode not in MATCH_MODES:
        raise MatchPlanError(f"Unsupported match mode: {project.matching.mode}")
    if project.matching.tolerance_seconds < 0:
        raise MatchPlanError("Time tolerance cannot be negative.")

    folders = project.folder_map()
    master = folders.get(project.matching.master_folder_id)
    if master is None:
        raise MatchPlanError("Select a master timeline folder.")
    _validate_folder_range(master)

    for slot in project.slots:
        folder = folders.get(slot.folder_id)
        if folder is None:
            raise MatchPlanError(f"Slot {slot.id} references a missing folder.")
        _validate_folder_range(folder)
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


def build_match_plan(project: ComposerProject) -> list[FrameMatch]:
    """Build the complete auditable plan before rendering starts."""

    validate_project(project)
    folders = project.folder_map()
    master = folders[project.matching.master_folder_id]
    master_records = master.selected_records()
    used_ids = list(dict.fromkeys(slot.folder_id for slot in project.slots))
    candidate_sets = {
        folder_id: folders[folder_id].selected_records() for folder_id in used_ids
    }
    time_indexes = {
        folder_id: _time_index(folders[folder_id], records)
        for folder_id, records in candidate_sets.items()
        if folder_id != master.id
    }

    output_index = 0
    plan: list[FrameMatch] = []
    for position, master_record in enumerate(master_records):
        master_time = master.corrected_timestamp(master_record)
        matches: dict[str, FolderMatch] = {}
        over_folders: list[str] = []
        for folder_id in used_ids:
            folder = folders[folder_id]
            if folder_id == master.id:
                selected = master_record
            elif project.matching.mode == "relative":
                selected = relative_record(
                    candidate_sets[folder_id], position, len(master_records)
                )
            else:
                selected = _nearest_from_index(
                    folder, time_indexes[folder_id], master_time
                )
            corrected = folder.corrected_timestamp(selected)
            delta = (corrected - master_time).total_seconds()
            over = (
                project.matching.mode == "time"
                and abs(delta) > project.matching.tolerance_seconds
            )
            if over:
                over_folders.append(folder.name)
            matches[folder_id] = FolderMatch(
                folder_id=folder_id,
                record=selected,
                corrected_timestamp=corrected,
                delta_seconds=delta,
                over_tolerance=over,
            )

        emitted = not (
            project.matching.mode == "time" and project.matching.strict and over_folders
        )
        if emitted:
            output_index += 1
        reason = ""
        if not emitted:
            reason = "time tolerance exceeded: " + ", ".join(over_folders)
        plan.append(
            FrameMatch(
                attempt_index=position + 1,
                output_frame_index=output_index if emitted else None,
                master_record=master_record,
                master_corrected_timestamp=master_time,
                matches=matches,
                emitted=emitted,
                skip_reason=reason,
            )
        )
    return plan


def nearest_record(
    folder: FolderSource,
    candidates: Iterable[ImageRecord],
    master_time: datetime,
) -> ImageRecord:
    """Choose nearest time, then earlier time, then lower natural ordinal."""

    items = list(candidates)
    if not items:
        raise MatchPlanError(f"Folder has no selected images: {folder.path}")
    return min(
        items,
        key=lambda record: (
            abs((folder.corrected_timestamp(record) - master_time).total_seconds()),
            folder.corrected_timestamp(record),
            record.ordinal,
        ),
    )


def _time_index(
    folder: FolderSource, candidates: Iterable[ImageRecord]
) -> tuple[list[datetime], list[ImageRecord]]:
    ordered = sorted(
        candidates,
        key=lambda record: (folder.corrected_timestamp(record), record.ordinal),
    )
    if not ordered:
        raise MatchPlanError(f"Folder has no selected images: {folder.path}")
    return [folder.corrected_timestamp(record) for record in ordered], ordered


def _nearest_from_index(
    folder: FolderSource,
    index: tuple[list[datetime], list[ImageRecord]],
    master_time: datetime,
) -> ImageRecord:
    times, records = index
    position = bisect_left(times, master_time)
    candidate_positions: list[int] = []
    if position < len(times):
        candidate_positions.append(position)
    if position > 0:
        earlier_time = times[position - 1]
        candidate_positions.append(bisect_left(times, earlier_time))
    candidates = [records[item] for item in dict.fromkeys(candidate_positions)]
    return min(
        candidates,
        key=lambda record: (
            abs((folder.corrected_timestamp(record) - master_time).total_seconds()),
            folder.corrected_timestamp(record),
            record.ordinal,
        ),
    )


def relative_record(
    candidates: Iterable[ImageRecord], master_position: int, master_count: int
) -> ImageRecord:
    """Map both selected-range endpoints with deterministic half-up rounding."""

    items = list(candidates)
    if not items:
        raise MatchPlanError("A used folder has no selected images.")
    if master_count <= 1 or len(items) == 1:
        return items[0]
    scaled = master_position * (len(items) - 1) / (master_count - 1)
    index = min(len(items) - 1, int(math.floor(scaled + 0.5)))
    return items[index]


def iter_match_rows(
    project: ComposerProject, plan: Iterable[FrameMatch]
) -> Iterable[dict[str, Any]]:
    """Yield long-form rows for every attempted master frame and canvas slot."""

    folders = project.folder_map()
    master = folders[project.matching.master_folder_id]
    for frame in plan:
        for slot in sorted(project.slots, key=lambda item: item.z_index):
            folder = folders[slot.folder_id]
            match = frame.matches[slot.folder_id]
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
                "master_timestamp": frame.master_record.timestamp.isoformat(
                    timespec="microseconds"
                ),
                "master_time_source": frame.master_record.time_source,
                "master_offset_seconds": master.offset_seconds,
                "master_corrected_timestamp": frame.master_corrected_timestamp.isoformat(
                    timespec="microseconds"
                ),
                "slot_id": slot.id,
                "slot_z_index": slot.z_index,
                "source_folder_id": folder.id,
                "source_folder": folder.name,
                "source_file": str(match.record.path),
                "source_ordinal": match.record.ordinal,
                "source_timestamp": match.record.timestamp.isoformat(
                    timespec="microseconds"
                ),
                "source_time_source": match.record.time_source,
                "source_offset_seconds": folder.offset_seconds,
                "source_corrected_timestamp": match.corrected_timestamp.isoformat(
                    timespec="microseconds"
                ),
                "delta_seconds": match.delta_seconds,
                "abs_delta_seconds": abs(match.delta_seconds),
                "over_tolerance": match.over_tolerance,
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
