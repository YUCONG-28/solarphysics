# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 2A adapters over existing AIA/HMI/viewer implementations."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from solar_apps.frontends.image_viewer.server import scan_images
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .runtime import AppV1RuntimePaths

_IMAGE_MODULE = "solar_apps.frontends.image_viewer.cli"
_AIA_MODULE = "solar_apps.workflows.aia.application"
_HMI_MODULE = "solar_apps.workflows.hmi.overlay_cli"


@dataclass(frozen=True, slots=True)
class TaskLaunch:
    """Confirmed subprocess launch request prepared by a Phase 2 adapter."""

    title: str
    module_id: str
    python_module: str
    arguments: tuple[str, ...]
    output_dir: Path
    summary: str


@dataclass(frozen=True, slots=True)
class ImageSequenceSelection:
    folder: Path
    images: tuple[Path, ...]
    recursive: bool
    summary: str


@dataclass(frozen=True, slots=True)
class ImageGroupSelection:
    """Index-synchronized image folders for native comparison and export."""

    groups: tuple[ImageSequenceSelection, ...]
    recursive: bool
    summary: str


class Phase2AAdapter:
    """Validate inputs and build commands without duplicating science code."""

    def __init__(
        self,
        layout: RuntimeLayout,
        *,
        allowed_roots: tuple[Path, ...] | None = None,
    ) -> None:
        self.layout = layout
        self.runtime = AppV1RuntimePaths.from_layout(layout)
        self.allowed_roots = (
            tuple(
                Path(item).expanduser().resolve(strict=False) for item in allowed_roots
            )
            if allowed_roots is not None
            else configured_allowed_roots(
                environ=os.environ,
                workspace_root=layout.repo_root,
            )
        )

    def select_images(
        self,
        folder: str | Path,
        *,
        recursive: bool = False,
    ) -> ImageSequenceSelection:
        selected = self.validate_input_directory(folder)
        _root, images = scan_images(
            selected,
            recursive=recursive,
            allowed_roots=list(self.allowed_roots),
        )
        summary = "\n".join(
            (
                "Module: Image Viewer",
                f"Input: {selected}",
                f"Parameters: recursive={str(bool(recursive)).lower()}",
                "Output: none (read-only preview)",
                f"Workload: {len(images)} image(s)",
            )
        )
        return ImageSequenceSelection(
            selected,
            tuple(images),
            bool(recursive),
            summary,
        )

    def select_image_groups(
        self,
        folders: tuple[str | Path, ...],
        *,
        recursive: bool = False,
    ) -> ImageGroupSelection:
        groups = tuple(
            self.select_images(folder, recursive=recursive)
            for folder in dict.fromkeys(folders)
        )
        if not groups:
            raise ValueError("Select at least one image folder")
        summary = "\n".join(
            (
                "Module: Image Viewer",
                "Input: " + "; ".join(str(group.folder) for group in groups),
                (
                    f"Parameters: groups={len(groups)}; "
                    f"recursive={str(bool(recursive)).lower()}"
                ),
                "Output: none (read-only synchronized preview)",
                ("Workload: " f"{sum(len(group.images) for group in groups)} image(s)"),
            )
        )
        return ImageGroupSelection(groups, bool(recursive), summary)

    def build_image_export(
        self,
        selection: ImageGroupSelection,
        *,
        output_format: str = "mp4",
        composite: bool = True,
        fps: float = 5.0,
        workers: int = 1,
    ) -> TaskLaunch:
        if output_format not in {"mp4", "gif", "webm"}:
            raise ValueError("Image export format must be MP4, GIF, or WebM")
        if not 0.2 <= float(fps) <= 120:
            raise ValueError("Image export FPS must be between 0.2 and 120")
        if not 1 <= int(workers) <= 16:
            raise ValueError("Image export workers must be between 1 and 16")
        output = self._new_output_dir("image-viewer-export")
        arguments: list[str] = []
        for group in selection.groups:
            arguments.extend(["--folder", str(group.folder)])
        arguments.extend(
            [
                "--output-dir",
                str(output),
                "--format",
                output_format,
                "--mode",
                "composite" if composite else "separate",
                "--fps",
                str(float(fps)),
                "--workers",
                str(int(workers)),
            ]
        )
        if selection.recursive:
            arguments.append("--recursive")
        summary = "\n".join(
            (
                "Module: Image Viewer Export",
                "Input: " + "; ".join(str(item.folder) for item in selection.groups),
                (
                    f"Parameters: format={output_format}; "
                    f"mode={'composite' if composite else 'separate'}; "
                    f"fps={fps:g}; workers={workers}"
                ),
                f"Output: {output}",
                f"Workload: {sum(len(item.images) for item in selection.groups)} frame(s)",
            )
        )
        return TaskLaunch(
            "Image Viewer media export",
            "image-viewer",
            "solar_apps.frontends.app_v1.image_viewer_worker",
            tuple(arguments),
            output,
            summary,
        )

    def build_aia(
        self,
        input_dir: str | Path,
        *,
        mode: str = "test",
        waves: tuple[int, ...] = (171,),
        start: int | None = None,
        end: int | None = None,
        test_index: int = 0,
        workers: int = 1,
    ) -> TaskLaunch:
        selected = self.validate_input_directory(input_dir)
        if mode not in {"single", "mosaic", "test"}:
            raise ValueError("AIA mode must be single, mosaic, or test")
        clean_waves = tuple(dict.fromkeys(int(item) for item in waves))
        if not clean_waves or any(item <= 0 for item in clean_waves):
            raise ValueError("At least one positive AIA wavelength is required")
        if start is not None and start < 0:
            raise ValueError("AIA start index cannot be negative")
        if end is not None and start is not None and end < start:
            raise ValueError("AIA end index cannot precede start")
        if test_index < 0:
            raise ValueError("AIA test index cannot be negative")
        if workers < 1:
            raise ValueError("AIA worker count must be positive")
        output = self._new_output_dir("aia-processing")
        arguments = [
            "--data-path",
            str(selected),
            "--output-dir",
            str(output),
            "--mode",
            mode,
            "--waves",
            *map(str, clean_waves),
        ]
        if start is not None:
            arguments.extend(["--start", str(start)])
        if end is not None:
            arguments.extend(["--end", str(end)])
        arguments.extend(["--workers", str(workers)])
        if mode == "test":
            arguments.extend(
                [
                    "--test-wave",
                    str(clean_waves[0]),
                    "--test-index",
                    str(test_index),
                ]
            )
        workload = self._count_files(selected, {".fits", ".fit", ".fts"})
        summary = "\n".join(
            (
                "Module: AIA Processing",
                f"Input: {selected}",
                (
                    f"Parameters: mode={mode}; waves={','.join(map(str, clean_waves))}; "
                    f"start={start if start is not None else 0}; "
                    f"end={end if end is not None else 'all'}; "
                    f"test_index={test_index}; workers={workers}"
                ),
                f"Output: {output}",
                f"Workload: {workload} FITS candidate(s)",
            )
        )
        return TaskLaunch(
            "AIA processing",
            "image-viewer",
            _AIA_MODULE,
            tuple(arguments),
            output,
            summary,
        )

    def build_hmi_overlay(
        self,
        aia_dir: str | Path,
        hmi_dir: str | Path,
        *,
        dpi: int = 300,
        max_time_diff_seconds: float = 24.0,
    ) -> TaskLaunch:
        aia = self.validate_input_directory(aia_dir)
        hmi = self.validate_input_directory(hmi_dir)
        if dpi < 72 or dpi > 1200:
            raise ValueError("DPI must be between 72 and 1200")
        if max_time_diff_seconds < 0:
            raise ValueError("Time tolerance cannot be negative")
        output = self._new_output_dir("hmi-overlay")
        arguments = (
            "--input-dir-aia",
            str(aia),
            "--input-dir-hmi",
            str(hmi),
            "--output-dir",
            str(output),
            "--dpi",
            str(int(dpi)),
            "--max-time-diff-seconds",
            str(float(max_time_diff_seconds)),
            "--no-show-plot",
        )
        aia_count = self._count_files(aia, {".fits", ".fit", ".fts"})
        hmi_count = self._count_files(hmi, {".fits", ".fit", ".fts"})
        summary = "\n".join(
            (
                "Module: AIA/HMI Overlay",
                f"Input: {aia_count} AIA and {hmi_count} HMI FITS candidate(s)",
                f"Parameters: dpi={dpi}; tolerance={max_time_diff_seconds:g} s",
                f"Output: {output}",
                f"Workload: match up to {aia_count} AIA frame(s)",
            )
        )
        return TaskLaunch(
            "AIA/HMI overlay",
            "image-viewer",
            _HMI_MODULE,
            arguments,
            output,
            summary,
        )

    def validate_input_directory(self, value: str | Path) -> Path:
        if not self.allowed_roots:
            raise PermissionError(
                "No allowed roots are configured for App 1.0 data access."
            )
        path = Path(value).expanduser().resolve(strict=False)
        if not path.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {path}")
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise PermissionError(
                f"Input is outside the configured allowed roots: {path}"
            )
        return path

    def _new_output_dir(self, module_id: str) -> Path:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        return self.runtime.run_output_dir("preview", run_id, module_id)

    @staticmethod
    def _count_files(directory: Path, suffixes: set[str]) -> int:
        return sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in suffixes
        )


__all__ = [
    "ImageGroupSelection",
    "ImageSequenceSelection",
    "Phase2AAdapter",
    "TaskLaunch",
]
