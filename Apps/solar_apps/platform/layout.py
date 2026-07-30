"""Portable repository and private-runtime path discovery."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOCAL_ROOT_ENV = "SOLAR_APPS_LOCAL_ROOT"
REPO_ROOT_ENV = "SOLAR_APPS_REPO_ROOT"


def _discover_repo_root(start: Path) -> Path:
    """Find the workspace without relying on a fixed package depth."""

    resolved = start.expanduser().resolve(strict=False)
    candidates = (resolved, *resolved.parents)
    for candidate in candidates:
        if (candidate / "Apps").is_dir() and (candidate / "Python").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the solarphysics workspace containing Apps/ and Python/. "
        "Pass repo_root explicitly."
    )


@dataclass(frozen=True, slots=True)
class RuntimeLayout:
    """Resolved public source roots and the ignored local runtime tree."""

    repo_root: Path
    apps_root: Path
    python_root: Path
    local_root: Path
    config_dir: Path
    state_dir: Path
    workspaces_dir: Path
    outputs_dir: Path
    observations_dir: Path
    logs_dir: Path
    tmp_dir: Path

    @classmethod
    def discover(
        cls,
        repo_root: str | os.PathLike[str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "RuntimeLayout":
        """Resolve the workspace and optional ``SOLAR_APPS_LOCAL_ROOT`` override."""

        env = os.environ if environ is None else environ
        repository_value = repo_root or env.get(REPO_ROOT_ENV)
        repository = (
            Path(repository_value).expanduser().resolve(strict=False)
            if repository_value is not None
            else _discover_repo_root(Path(__file__))
        )
        local_value = env.get(LOCAL_ROOT_ENV)
        local = (
            Path(local_value).expanduser().resolve(strict=False)
            if local_value
            else (repository / "Local").resolve(strict=False)
        )
        return cls(
            repo_root=repository,
            apps_root=repository / "Apps",
            python_root=repository / "Python",
            local_root=local,
            config_dir=local / "configs",
            state_dir=local / "state",
            workspaces_dir=local / "workspaces",
            outputs_dir=local / "outputs",
            observations_dir=local / "observations",
            logs_dir=local / "logs",
            tmp_dir=local / "tmp",
        )

    @property
    def config_path(self) -> Path:
        """Default machine-local path configuration."""

        return self.config_dir / "paths.local.yaml"

    def ensure(self) -> "RuntimeLayout":
        """Create the complete ignored runtime directory contract."""

        for directory in (
            self.local_root,
            self.config_dir,
            self.state_dir,
            self.workspaces_dir,
            self.outputs_dir,
            self.observations_dir,
            self.logs_dir,
            self.tmp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


__all__ = ["LOCAL_ROOT_ENV", "REPO_ROOT_ENV", "RuntimeLayout"]
