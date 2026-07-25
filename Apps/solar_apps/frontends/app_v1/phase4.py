# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 4 Image Composer task adapter."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from solar_apps.frontends.image_composer.models import ComposerProject
from solar_apps.frontends.image_composer.project import save_project
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .phase2a import TaskLaunch
from .runtime import AppV1RuntimePaths

_WORKER = "solar_apps.frontends.app_v1.composer_worker"


class Phase4ComposerAdapter:
    """Persist schema-1 projects and prepare isolated render workers."""

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

    def validate_project_inputs(self, project: ComposerProject) -> None:
        if not project.folders:
            raise ValueError("Add at least one image folder")
        if not project.slots:
            raise ValueError("Drop at least one image onto the canvas")
        for folder in project.folders:
            path = folder.path.expanduser().resolve(strict=False)
            if not path.is_dir():
                raise FileNotFoundError(f"Image folder does not exist: {path}")
            if not self._inside(path):
                raise PermissionError(
                    f"Image folder is outside configured allowed roots: {path}"
                )

    def save_workspace_project(
        self,
        project: ComposerProject,
        *,
        name: str = "preview-composer",
    ) -> Path:
        self.validate_project_inputs(project)
        return save_project(self.runtime.workspaces_dir / name, project)

    def build_static_export(
        self,
        project: ComposerProject,
        *,
        scale: int = 1,
    ) -> TaskLaunch:
        if scale < 1 or scale > 8:
            raise ValueError("Export scale must be between 1 and 8")
        project_path = self.save_workspace_project(project)
        output_dir = self._new_output_dir()
        output = output_dir / "images" / "composition.png"
        return self._launch(
            title="Image Composer PNG",
            project=project,
            project_path=project_path,
            output_dir=output_dir,
            output=output,
            mode="static",
            scale=scale,
            fps=project.export.fps,
        )

    def build_sequence_export(
        self,
        project: ComposerProject,
        *,
        scale: int = 1,
        fps: float = 5.0,
        save_png_frames: bool = False,
    ) -> TaskLaunch:
        if fps <= 0 or fps > 60:
            raise ValueError("FPS must be greater than 0 and no more than 60")
        project_path = self.save_workspace_project(project)
        output_dir = self._new_output_dir()
        output = output_dir / "media" / "composition.mp4"
        launch = self._launch(
            title="Image Composer Sequence",
            project=project,
            project_path=project_path,
            output_dir=output_dir,
            output=output,
            mode="sequence",
            scale=scale,
            fps=fps,
        )
        if save_png_frames:
            launch = TaskLaunch(
                launch.title,
                launch.module_id,
                launch.python_module,
                (*launch.arguments, "--save-png-frames"),
                launch.output_dir,
                launch.summary,
            )
        return launch

    def _launch(
        self,
        *,
        title: str,
        project: ComposerProject,
        project_path: Path,
        output_dir: Path,
        output: Path,
        mode: str,
        scale: int,
        fps: float,
    ) -> TaskLaunch:
        width = project.canvas.width * int(scale)
        height = project.canvas.height * int(scale)
        return TaskLaunch(
            title,
            "image-composer",
            _WORKER,
            (
                "--project",
                str(project_path),
                "--mode",
                mode,
                "--output",
                str(output),
                "--scale",
                str(int(scale)),
                "--fps",
                str(float(fps)),
                "--allowed-roots",
                self._root_list(output_dir),
            ),
            output_dir,
            "\n".join(
                (
                    f"Module: {title}",
                    (
                        f"Input: {len(project.folders)} folder(s), "
                        f"{len(project.slots)} layer(s)"
                    ),
                    (
                        f"Parameters: mode={mode}; scale={int(scale)}; "
                        f"canvas={width}x{height}; fps={float(fps):g}"
                    ),
                    f"Output: {output}",
                    (
                        "Workload: "
                        f"{sum(len(folder.selected_records()) for folder in project.folders)} "
                        "selected image reference(s)"
                    ),
                )
            ),
        )

    def _new_output_dir(self) -> Path:
        return self.runtime.run_output_dir(
            "preview",
            f"run-{uuid.uuid4().hex[:12]}",
            "image-composer",
        )

    def _inside(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.allowed_roots)

    def _root_list(self, output: Path) -> str:
        roots = (*self.allowed_roots, self.runtime.workspaces_dir, output)
        return os.pathsep.join(map(str, roots))


__all__ = ["Phase4ComposerAdapter"]
