# SPDX-License-Identifier: GPL-3.0-only
"""Subprocess-only renderer for the native PyQt6 Image Composer page."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from pathlib import Path

from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.project import load_project
from solar_apps.frontends.image_composer.rendering import (
    compose_frame,
    export_project,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an App v1 composer project.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--mode", choices=("static", "sequence"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--save-png-frames", action="store_true")
    parser.add_argument("--allowed-roots", required=True)
    return parser


def _roots(raw: str) -> tuple[Path, ...]:
    roots = tuple(
        Path(item).expanduser().resolve(strict=False)
        for item in str(raw).split(os.pathsep)
        if item.strip()
    )
    if not roots:
        raise PermissionError("No allowed roots were supplied")
    return roots


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _hydrate(project, roots: tuple[Path, ...]) -> None:  # type: ignore[no-untyped-def]
    for folder in project.folders:
        resolved = folder.path.expanduser().resolve(strict=False)
        if not _inside(resolved, roots):
            raise PermissionError(
                f"Composer folder is outside allowed roots: {resolved}"
            )
        folder.records = scan_folder(resolved)
        if not folder.records:
            raise ValueError(
                f"Composer folder contains no supported images: {resolved}"
            )
        folder.resolved = True
        folder.start_index = max(1, min(folder.start_index, len(folder.records)))
        folder.end_index = max(
            folder.start_index,
            min(folder.end_index, len(folder.records)),
        )


def _scaled(project, scale: int):  # type: ignore[no-untyped-def]
    factor = int(scale)
    if factor < 1 or factor > 8:
        raise ValueError("Scale must be between 1 and 8")
    result = copy.deepcopy(project)
    result.canvas.width *= factor
    result.canvas.height *= factor
    for slot in result.slots:
        slot.x *= factor
        slot.y *= factor
        slot.width *= factor
        slot.height *= factor
    return result


def _static(project, output: Path) -> dict[str, object]:  # type: ignore[no-untyped-def]
    folders = project.folder_map()
    matched = {}
    for slot in project.slots:
        folder = folders.get(slot.folder_id)
        if folder is None:
            raise ValueError(f"Slot references missing folder: {slot.folder_id}")
        record = folder.record_by_ordinal(slot.preview_ordinal)
        if record is None:
            raise ValueError(
                f"Preview ordinal {slot.preview_ordinal} is unavailable in {folder.name}"
            )
        matched[folder.id] = record
    image = compose_frame(project, matched)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.png")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "saved",
        "image_path": str(output),
        "width": image.width,
        "height": image.height,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = _roots(args.allowed_roots)
    project_path = Path(args.project).expanduser().resolve(strict=False)
    output = Path(args.output).expanduser().resolve(strict=False)
    if not _inside(project_path, roots):
        raise PermissionError(f"Project is outside allowed roots: {project_path}")
    if not _inside(output, roots):
        raise PermissionError(f"Output is outside allowed roots: {output}")
    project = load_project(project_path)
    _hydrate(project, roots)
    project = _scaled(project, args.scale)
    print("PROGRESS 10", flush=True)
    if args.mode == "static":
        result = _static(project, output)
        print("PROGRESS 100", flush=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        project.export.output_path = str(output)
        project.export.output_format = output.suffix.casefold().lstrip(".")
        project.export.fps = float(args.fps)
        project.export.save_png_frames = bool(args.save_png_frames)

        def progress(current: int, total: int, _message: str) -> None:
            percentage = 10 + round(85 * current / max(1, total))
            print(f"PROGRESS {min(95, percentage)}", flush=True)

        exported = export_project(project, progress=progress)
        result = {
            "status": exported.status,
            "video_path": (
                str(exported.video_path) if exported.video_path is not None else None
            ),
            "csv_path": str(exported.csv_path),
            "frames_path": (
                str(exported.frames_path) if exported.frames_path is not None else None
            ),
            "attempted_frames": exported.attempted_frames,
            "emitted_frames": exported.emitted_frames,
        }
        print("PROGRESS 100", flush=True)
    print("LOG " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
