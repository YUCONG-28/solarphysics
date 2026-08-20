# SPDX-License-Identifier: MIT
"""Data models for the local image composer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

FIT_MODES = frozenset({"contain", "cover", "stretch"})
MATCH_MODES = frozenset({"time", "relative"})
OUTPUT_FORMATS = frozenset({"mp4", "avi"})


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


@dataclass(slots=True, frozen=True)
class ImageRecord:
    """One naturally ordered image and its uncorrected local timestamp."""

    ordinal: int
    path: Path
    timestamp: datetime
    time_source: str


@dataclass(slots=True)
class FolderSource:
    """A folder, its selected inclusive range, and clock correction."""

    id: str
    path: Path
    name: str
    records: list[ImageRecord] = field(default_factory=list)
    start_index: int = 1
    end_index: int = 1
    offset_seconds: float = 0.0
    resolved: bool = True

    @classmethod
    def create(cls, path: str | Path, records: list[ImageRecord]) -> FolderSource:
        resolved = Path(path).expanduser().resolve()
        return cls(
            id=new_id("folder"),
            path=resolved,
            name=resolved.name or str(resolved),
            records=list(records),
            start_index=1,
            end_index=max(1, len(records)),
            resolved=resolved.is_dir(),
        )

    def selected_records(self) -> list[ImageRecord]:
        if not self.records:
            return []
        start = max(1, int(self.start_index)) - 1
        stop = max(start, int(self.end_index))
        return self.records[start:stop]

    def corrected_timestamp(self, record: ImageRecord) -> datetime:
        return record.timestamp + timedelta(seconds=float(self.offset_seconds))

    def record_by_ordinal(self, ordinal: int) -> ImageRecord | None:
        if 1 <= ordinal <= len(self.records):
            record = self.records[ordinal - 1]
            if record.ordinal == ordinal:
                return record
        return next((item for item in self.records if item.ordinal == ordinal), None)


@dataclass(slots=True)
class LayoutSlot:
    """One canvas slot bound to a folder rather than a fixed image."""

    id: str
    folder_id: str
    preview_ordinal: int
    preview_relative_path: str
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    opacity: float = 1.0
    fit: str = "contain"
    z_index: int = 0

    @classmethod
    def create(
        cls,
        folder_id: str,
        preview_ordinal: int,
        *,
        x: float,
        y: float,
        width: float = 420.0,
        height: float = 280.0,
        z_index: int = 0,
    ) -> LayoutSlot:
        return cls(
            id=new_id("slot"),
            folder_id=folder_id,
            preview_ordinal=int(preview_ordinal),
            preview_relative_path="",
            x=float(x),
            y=float(y),
            width=float(width),
            height=float(height),
            z_index=int(z_index),
        )


@dataclass(slots=True)
class CanvasSettings:
    width: int = 1280
    height: int = 720
    background: str = "#101318"


@dataclass(slots=True)
class MatchSettings:
    master_folder_id: str = ""
    mode: str = "time"
    tolerance_seconds: float = 1.0
    strict: bool = True


@dataclass(slots=True)
class ExportSettings:
    output_path: str = ""
    output_format: str = "mp4"
    fps: float = 5.0
    save_png_frames: bool = False


@dataclass(slots=True)
class ComposerProject:
    schema_version: int = 1
    canvas: CanvasSettings = field(default_factory=CanvasSettings)
    folders: list[FolderSource] = field(default_factory=list)
    slots: list[LayoutSlot] = field(default_factory=list)
    matching: MatchSettings = field(default_factory=MatchSettings)
    export: ExportSettings = field(default_factory=ExportSettings)

    def folder_map(self) -> dict[str, FolderSource]:
        return {folder.id: folder for folder in self.folders}

    def slot_by_id(self, slot_id: str) -> LayoutSlot | None:
        return next((slot for slot in self.slots if slot.id == slot_id), None)

    def normalize_z_indexes(self) -> None:
        for index, slot in enumerate(sorted(self.slots, key=lambda item: item.z_index)):
            slot.z_index = index
