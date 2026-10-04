"""Portable repository and private-runtime path discovery."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LOCAL_ROOT_ENV = "SOLAR_APPS_LOCAL_ROOT"
REPO_ROOT_ENV = "SOLAR_APPS_REPO_ROOT"


def validate_private_output_path(
    path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str],
) -> Path:
    """Resolve a runtime destination inside Local/ or outside the repository."""

    repository = Path(repo_root).expanduser().resolve(strict=False)
    resolved = Path(path).expanduser().resolve(strict=False)
    # Keep the allowed in-repository root lexical: resolving a Local symlink
    # into Apps/ must not make that public directory an allowed runtime root.
    local = repository / "Local"
    if resolved.is_relative_to(repository) and not resolved.is_relative_to(local):
        raise ValueError(
            "Runtime outputs must be inside Local/ or outside the repository: "
            f"{resolved}"
        )
    return resolved


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
        local = validate_private_output_path(
            local_value if local_value else repository / "Local",
            repo_root=repository,
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
        ).validate()

    @property
    def config_path(self) -> Path:
        """Default machine-local path configuration."""

        return self.config_dir / "paths.local.yaml"

    def _resolved_runtime_directories(self) -> tuple[Path, ...]:
        """Validate the entire layout before any directory can be created."""

        return tuple(
            validate_private_output_path(directory, repo_root=self.repo_root)
            for directory in (
                self.local_root,
                self.config_dir,
                self.state_dir,
                self.workspaces_dir,
                self.outputs_dir,
                self.observations_dir,
                self.logs_dir,
                self.tmp_dir,
            )
        )

    def validate(self) -> "RuntimeLayout":
        """Check current resolved destinations without creating runtime files."""

        self._resolved_runtime_directories()
        return self

    def ensure(self) -> "RuntimeLayout":
        """Create the complete ignored runtime directory contract."""

        directories = self._resolved_runtime_directories()
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return self


__all__ = [
    "LOCAL_ROOT_ENV",
    "REPO_ROOT_ENV",
    "RuntimeLayout",
    "validate_private_output_path",
]
