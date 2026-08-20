"""Validation helpers for the two supported Miniforge environments."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PRIMARY_ENVIRONMENT = "solarphysics_env_latest"
STANDBY_ENVIRONMENT = "solarphysics_env"
SUPPORTED_ENVIRONMENTS = (PRIMARY_ENVIRONMENT, STANDBY_ENVIRONMENT)
LOCK_CANDIDATE_ENVIRONMENT = "solarphysics_lock_candidate"
LOCK_CANDIDATE_OPT_IN = "SOLAR_APPS_ALLOW_LOCK_CANDIDATE"
LOCK_REPLAY_ENVIRONMENT = "solarphysics_replay"
LOCK_REPLAY_OPT_IN = "SOLAR_APPS_ALLOW_LOCK_REPLAY"


class UnsupportedPythonEnvironment(RuntimeError):
    """Raised when an application would escape the supported Miniforge envs."""


@dataclass(frozen=True, slots=True)
class MiniforgeRuntime:
    executable: Path
    environment_root: Path
    environment_name: str
    miniforge_root: Path


def launcher_program() -> str:
    """Return the public launcher name selected by the wrapper."""

    configured = os.environ.get("SOLAR_APPS_LAUNCHER", "").strip()
    if configured:
        return configured
    return "Apps/run.ps1" if os.name == "nt" else "Apps/run.sh"


def launcher_command(suffix: str = "") -> str:
    """Build a platform-appropriate public command name."""

    return f"{launcher_program()} {suffix}".rstrip()


def _environment_root(selected: Path) -> Path:
    if selected.name.casefold() == "python.exe":
        return selected.parent
    if selected.name.startswith("python") and selected.parent.name == "bin":
        return selected.parent.parent
    return selected.parent


def _has_miniforge_provenance(root: Path) -> bool:
    metadata = root / "conda-meta"
    if any(metadata.glob("miniforge_console_shortcut-*.json")):
        return True
    history = metadata / "history"
    if not history.is_file():
        return False
    try:
        opening = history.read_text(encoding="utf-8", errors="replace")[:8192]
    except OSError:
        return False
    return bool(re.search(r"(?im)^# cmd:.*\bMiniforge3?\b", opening))


def inspect_miniforge_runtime(
    executable: str | os.PathLike[str] | None = None,
    *,
    miniforge_root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    require_exists: bool = True,
) -> MiniforgeRuntime:
    """Validate one interpreter as latest or explicit formal standby."""

    env = os.environ if environ is None else environ
    selected = Path(executable or sys.executable).expanduser().resolve(strict=False)
    if require_exists and not selected.is_file():
        raise UnsupportedPythonEnvironment(f"Python interpreter not found: {selected}")
    environment_root = _environment_root(selected)
    environment_name = environment_root.name
    lock_candidate = (
        environment_name == LOCK_CANDIDATE_ENVIRONMENT
        and env.get(LOCK_CANDIDATE_OPT_IN, "").strip() == "1"
    )
    lock_replay = (
        environment_name == LOCK_REPLAY_ENVIRONMENT
        and env.get(LOCK_REPLAY_OPT_IN, "").strip() == "1"
    )
    lock_maintenance = lock_candidate or lock_replay
    if environment_name not in SUPPORTED_ENVIRONMENTS and not lock_maintenance:
        raise UnsupportedPythonEnvironment(
            "Solar applications require Miniforge environment "
            f"{PRIMARY_ENVIRONMENT!r}, or explicit standby {STANDBY_ENVIRONMENT!r}; "
            f"got {environment_name!r}."
        )
    if not lock_maintenance and environment_root.parent.name.casefold() != "envs":
        raise UnsupportedPythonEnvironment(
            f"Interpreter is not inside a Miniforge envs directory: {selected}"
        )
    trusted_root_value = miniforge_root or env.get("SOLAR_MINIFORGE_ROOT")
    trusted_root = (
        Path(trusted_root_value).expanduser().resolve(strict=False)
        if trusted_root_value
        else None
    )
    if lock_maintenance:
        if trusted_root is None:
            raise UnsupportedPythonEnvironment(
                "A disposable lock maintenance environment requires an explicit "
                "SOLAR_MINIFORGE_ROOT"
            )
        installation_root = trusted_root
    else:
        installation_root = environment_root.parent.parent.resolve(strict=False)
    if (
        not lock_maintenance
        and trusted_root is not None
        and trusted_root != installation_root
    ):
        raise UnsupportedPythonEnvironment(
            "Interpreter environment is not below the explicitly selected Conda root"
        )
    if require_exists and not _has_miniforge_provenance(installation_root):
        raise UnsupportedPythonEnvironment(
            f"Conda installation is not marked as Miniforge: {installation_root}"
        )
    if require_exists and not (environment_root / "conda-meta").is_dir():
        raise UnsupportedPythonEnvironment(
            f"Conda environment metadata is missing: {environment_root}"
        )
    return MiniforgeRuntime(
        executable=selected,
        environment_root=environment_root,
        environment_name=environment_name,
        miniforge_root=installation_root,
    )


__all__ = [
    "MiniforgeRuntime",
    "LOCK_CANDIDATE_ENVIRONMENT",
    "LOCK_CANDIDATE_OPT_IN",
    "LOCK_REPLAY_ENVIRONMENT",
    "LOCK_REPLAY_OPT_IN",
    "PRIMARY_ENVIRONMENT",
    "STANDBY_ENVIRONMENT",
    "SUPPORTED_ENVIRONMENTS",
    "UnsupportedPythonEnvironment",
    "inspect_miniforge_runtime",
    "launcher_command",
    "launcher_program",
]
