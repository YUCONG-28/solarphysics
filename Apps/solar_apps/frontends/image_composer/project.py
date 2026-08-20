# SPDX-License-Identifier: MIT
"""Versioned JSON persistence for image-composer projects."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    CanvasSettings,
    ComposerProject,
    ExportSettings,
    FolderSource,
    LayoutSlot,
    MatchSettings,
)

SCHEMA_VERSION = 1


class ProjectFormatError(ValueError):
    """Raised when a project file is malformed or from an unknown schema."""


def project_to_dict(project: ComposerProject) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "canvas": {
            "width": project.canvas.width,
            "height": project.canvas.height,
            "background": project.canvas.background,
        },
        "folders": [
            {
                "id": folder.id,
                "path": str(folder.path),
                "name": folder.name,
                "start_index": folder.start_index,
                "end_index": folder.end_index,
                "offset_seconds": folder.offset_seconds,
            }
            for folder in project.folders
        ],
        "slots": [
            {
                "id": slot.id,
                "folder_id": slot.folder_id,
                "preview_ordinal": slot.preview_ordinal,
                "preview_relative_path": slot.preview_relative_path,
                "x": slot.x,
                "y": slot.y,
                "width": slot.width,
                "height": slot.height,
                "rotation": slot.rotation,
                "opacity": slot.opacity,
                "fit": slot.fit,
                "z_index": slot.z_index,
            }
            for slot in project.slots
        ],
        "matching": {
            "master_folder_id": project.matching.master_folder_id,
            "mode": project.matching.mode,
            "tolerance_seconds": project.matching.tolerance_seconds,
            "strict": project.matching.strict,
        },
        "export": {
            "output_path": project.export.output_path,
            "output_format": project.export.output_format,
            "fps": project.export.fps,
            "save_png_frames": project.export.save_png_frames,
        },
    }


def project_from_dict(payload: dict[str, Any]) -> ComposerProject:
    try:
        schema_version = int(payload["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectFormatError(
            "Project schema_version is missing or invalid."
        ) from exc
    if schema_version != SCHEMA_VERSION:
        raise ProjectFormatError(
            f"Unsupported project schema_version {schema_version}; expected {SCHEMA_VERSION}."
        )

    try:
        canvas_data = dict(payload.get("canvas") or {})
        matching_data = dict(payload.get("matching") or {})
        export_data = dict(payload.get("export") or {})
        folders = [_folder_from_dict(item) for item in payload.get("folders", [])]
        slots = [_slot_from_dict(item) for item in payload.get("slots", [])]
        project = ComposerProject(
            schema_version=SCHEMA_VERSION,
            canvas=CanvasSettings(
                width=int(canvas_data.get("width", 1280)),
                height=int(canvas_data.get("height", 720)),
                background=str(canvas_data.get("background", "#101318")),
            ),
            folders=folders,
            slots=slots,
            matching=MatchSettings(
                master_folder_id=str(matching_data.get("master_folder_id", "")),
                mode=str(matching_data.get("mode", "time")),
                tolerance_seconds=float(matching_data.get("tolerance_seconds", 1.0)),
                strict=bool(matching_data.get("strict", True)),
            ),
            export=ExportSettings(
                output_path=str(export_data.get("output_path", "")),
                output_format=str(export_data.get("output_format", "mp4")),
                fps=float(export_data.get("fps", 5.0)),
                save_png_frames=bool(export_data.get("save_png_frames", False)),
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ProjectFormatError(f"Invalid project value: {exc}") from exc
    return project


def save_project(path: str | Path, project: ComposerProject) -> Path:
    """Write UTF-8 JSON through a same-directory temporary file and replace."""

    target = _project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        project_to_dict(project), ensure_ascii=False, indent=2, sort_keys=False
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_project(path: str | Path) -> ComposerProject:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectFormatError(f"Could not read project: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProjectFormatError("Project root must be a JSON object.")
    return project_from_dict(payload)


def _project_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    name = target.name.casefold()
    if not name.endswith(".fic.json"):
        target = target.with_name(target.name + ".fic.json")
    return target


def _folder_from_dict(value: Any) -> FolderSource:
    data = dict(value)
    path = Path(str(data["path"])).expanduser().resolve()
    return FolderSource(
        id=str(data["id"]),
        path=path,
        name=str(data.get("name") or path.name or path),
        records=[],
        start_index=int(data.get("start_index", 1)),
        end_index=int(data.get("end_index", 1)),
        offset_seconds=float(data.get("offset_seconds", 0.0)),
        resolved=path.is_dir(),
    )


def _slot_from_dict(value: Any) -> LayoutSlot:
    data = dict(value)
    return LayoutSlot(
        id=str(data["id"]),
        folder_id=str(data["folder_id"]),
        preview_ordinal=int(data.get("preview_ordinal", 1)),
        preview_relative_path=str(data.get("preview_relative_path", "")),
        x=float(data.get("x", 0.0)),
        y=float(data.get("y", 0.0)),
        width=float(data.get("width", 420.0)),
        height=float(data.get("height", 280.0)),
        rotation=float(data.get("rotation", 0.0)),
        opacity=float(data.get("opacity", 1.0)),
        fit=str(data.get("fit", "contain")),
        z_index=int(data.get("z_index", 0)),
    )
