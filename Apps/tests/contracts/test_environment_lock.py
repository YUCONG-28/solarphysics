"""Fail-closed contracts for platform-specific environment evidence."""

from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_ROOT = REPO_ROOT / "environment" / "locks"
LOCK_TOOL = REPO_ROOT / "tools" / "environment_lock.py"
LOCK_MANUAL = REPO_ROOT / "environment" / "README.md"


@pytest.fixture(scope="module")
def lock_module():
    spec = importlib.util.spec_from_file_location("public_environment_lock", LOCK_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_environment_gate_matches_committed_lock_state() -> None:
    targets = sorted(path for path in LOCK_ROOT.glob("*") if path.is_dir())
    completed = subprocess.run(
        [
            sys.executable,
            str(LOCK_TOOL),
            "check",
            "--require-artifact-hashes",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if targets:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode == 2
        assert "no committed platform locks found" in completed.stderr
        normalized_manual = " ".join(LOCK_MANUAL.read_text(encoding="utf-8").split())
        assert "exact-environment gate is therefore **red**" in normalized_manual


def test_lock_workflow_preserves_artifact_and_source_boundaries() -> None:
    source = LOCK_TOOL.read_text(encoding="utf-8")
    manual = LOCK_MANUAL.read_text(encoding="utf-8")

    assert '"--explicit", "--sha256"' in source
    assert '"python", "-m", "pip", "check"' in source
    assert "pip_artifacts_sha256" in source
    assert "--require-hashes" in manual
    assert "--no-build-isolation" in manual
    assert "must not describe that environment as exact" in manual
    assert "/Users/" not in manual
    assert "C:\\Users\\" not in manual


def test_capture_requires_explicit_apply_and_has_no_install_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, str(LOCK_TOOL), "capture", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--apply" in completed.stdout

    capture_source = (
        LOCK_TOOL.read_text(encoding="utf-8")
        .split("def _capture", maxsplit=1)[1]
        .split("def _check_target", maxsplit=1)[0]
    )
    assert '"install"' not in capture_source
    assert '"download"' not in capture_source


def test_lock_profiles_include_local_build_backends(lock_module) -> None:
    requirements = {
        str(requirement) for requirement in lock_module._source_requirements()
    }

    assert "setuptools>=77" in requirements
    assert "wheel" in requirements or "wheel>=0.45" in requirements


def test_fresh_capture_invalidates_an_older_pip_seal(lock_module, tmp_path) -> None:
    for name in ("pip-hashed.txt", "pip-artifacts.json"):
        (tmp_path / name).write_text("stale", encoding="utf-8")
    unrelated = tmp_path / "conda-explicit.txt"
    unrelated.write_text("keep", encoding="utf-8")

    lock_module._invalidate_pip_seal(tmp_path)

    assert not (tmp_path / "pip-hashed.txt").exists()
    assert not (tmp_path / "pip-artifacts.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_pep660_editables_accept_only_this_reviewed_checkout(lock_module) -> None:
    payload = (
        "["
        f'{{"name":"solar-physics-toolkit","editable_project_location":'
        f'"{(REPO_ROOT / "Python").as_posix()}"}},'
        f'{{"name":"solarphysics-apps","editable_project_location":'
        f'"{(REPO_ROOT / "Apps").as_posix()}"}}'
        "]"
    )
    observed = lock_module._validate_editable_projects(
        payload, require_current_checkout=True
    )
    assert set(observed) == {"solar-physics-toolkit", "solarphysics-apps"}

    with pytest.raises(lock_module.LockError, match="exactly"):
        lock_module._validate_editable_projects(
            payload[:-1]
            + ',{"name":"third-party","editable_project_location":"/tmp/third"}]',
            require_current_checkout=True,
        )


def test_freeze_parser_ignores_only_known_local_editable_paths(lock_module) -> None:
    freeze = "\n".join(
        (
            f"-e {(REPO_ROOT / 'Python').as_posix()}",
            f"-e file://{(REPO_ROOT / 'Apps').as_posix()}",
            "numpy==2.5.1",
        )
    )
    assert lock_module._freeze_to_pins(freeze, set()) == {"numpy": "2.5.1"}

    with pytest.raises(lock_module.LockError, match="unrecognized editable"):
        lock_module._freeze_to_pins("-e /tmp/third-party", set())
