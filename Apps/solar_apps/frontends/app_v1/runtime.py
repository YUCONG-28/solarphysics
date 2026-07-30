# SPDX-License-Identifier: GPL-3.0-only
"""App 1.0 paths derived exclusively from the existing private runtime layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout

from .contracts import validate_identifier


@dataclass(frozen=True, slots=True)
class AppV1RuntimePaths:
    """Private App 1.0 roots; no second repository runtime is introduced."""

    state_dir: Path
    workspaces_dir: Path
    outputs_dir: Path
    logs_dir: Path
    tmp_dir: Path
    observations_dir: Path | None = None

    @classmethod
    def from_layout(cls, layout: RuntimeLayout) -> "AppV1RuntimePaths":
        return cls(
            state_dir=layout.state_dir / "app_v1",
            workspaces_dir=layout.workspaces_dir / "app_v1",
            outputs_dir=layout.outputs_dir / "app_v1",
            logs_dir=layout.logs_dir / "app_v1",
            tmp_dir=layout.tmp_dir / "app_v1",
            observations_dir=layout.observations_dir,
        )

    @property
    def time_index_path(self) -> Path:
        return self.state_dir / "time_index.sqlite3"

    def project_file(self, project_id: str) -> Path:
        project = validate_identifier(project_id, label="project_id")
        return self.workspaces_dir / f"{project}.spapp.json"

    def flow_file(self, flow_id: str) -> Path:
        flow = validate_identifier(flow_id, label="flow_id")
        return self.workspaces_dir / "flows" / f"{flow}.spflow.json"

    def run_output_dir(self, project_id: str, run_id: str, module_id: str) -> Path:
        project = validate_identifier(project_id, label="project_id")
        run = validate_identifier(run_id, label="run_id")
        module = validate_identifier(module_id, label="module_id")
        return self.outputs_dir / project / run / module

    def ensure(self) -> "AppV1RuntimePaths":
        for directory in (
            self.state_dir,
            self.workspaces_dir,
            self.outputs_dir,
            self.logs_dir,
            self.tmp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if self.observations_dir is not None:
            self.observations_dir.mkdir(parents=True, exist_ok=True)
        return self


__all__ = ["AppV1RuntimePaths"]
