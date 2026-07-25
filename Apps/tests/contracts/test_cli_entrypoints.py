"""Contracts for the canonical Miniforge application CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from solar_apps.cli import router
from solar_apps.cli.admin import initialize_runtime
from solar_apps.platform.environment import inspect_miniforge_runtime
from solar_apps.platform.environment import UnsupportedPythonEnvironment
from solar_apps.platform.layout import RuntimeLayout

APPS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APPS_ROOT.parent
LAUNCHER = APPS_ROOT / "run.ps1"
SHELL_LAUNCHER = APPS_ROOT / "run.sh"


def test_canonical_cli_help_is_import_safe() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "solar_apps.cli", "--help"],
        cwd=APPS_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "frontend" in completed.stdout
    assert "workflow" in completed.stdout
    assert "admin" in completed.stdout
    expected_launcher = "Apps/run.ps1" if os.name == "nt" else "Apps/run.sh"
    assert completed.stdout.startswith(f"usage: {expected_launcher}")
    assert "python -m solar_apps" not in completed.stdout


def test_direct_cli_rejects_an_unsupported_python(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        router,
        "inspect_miniforge_runtime",
        lambda: (_ for _ in ()).throw(UnsupportedPythonEnvironment("not Miniforge")),
    )
    assert router.main(["--help"]) == 2
    assert "not Miniforge" in capsys.readouterr().err


def test_frontend_catalog_has_all_launchable_apps() -> None:
    assert set(router.FRONTEND_TARGETS) == {
        "aia-radio-composite",
        "app-v1",
        "app-v1-preview",
        "bad-frame-review",
        "dart-spectrogram",
        "image-composer",
        "image-viewer",
        "roi-lightcurve",
        "radio-composite",
        "source-map",
        "source-trajectory",
        "workbench",
    }


@pytest.mark.parametrize(
    ("legacy", "target", "forwarded"),
    [
        (
            ["webapp", "--port", "9000"],
            router.FRONTEND_TARGETS["workbench"],
            ["--port", "9000"],
        ),
        (
            ["image_viewer", "--help"],
            router.FRONTEND_TARGETS["image-viewer"],
            ["--help"],
        ),
        (
            ["radio", "source-map-app", "--help"],
            router.FRONTEND_TARGETS["source-map"],
            ["--help"],
        ),
        (
            ["radio", "dart-spectrogram", "--help"],
            router.FRONTEND_TARGETS["dart-spectrogram"],
            ["--help"],
        ),
        (
            ["radio", "roi-lightcurve", "--help"],
            router.FRONTEND_TARGETS["roi-lightcurve"],
            ["--help"],
        ),
        (
            ["radio", "source-trajectory-app", "--help"],
            router.FRONTEND_TARGETS["source-trajectory"],
            ["--help"],
        ),
        (["bad_frame_ml", "models"], router.TOOL_TARGETS["bad-frame-ml"], ["models"]),
    ],
)
def test_legacy_aliases_resolve_to_canonical_targets(
    monkeypatch: pytest.MonkeyPatch,
    legacy: list[str],
    target: str,
    forwarded: list[str],
) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        router,
        "forward_main",
        lambda module, arguments, **_kwargs: calls.append((module, list(arguments)))
        or 0,
    )
    assert router.main(legacy) == 0
    assert calls == [(target, forwarded)]


def test_admin_init_creates_only_private_runtime_contract(tmp_path: Path) -> None:
    repo = tmp_path / "workspace"
    template = repo / "Apps" / "configs" / "examples" / "paths.example.yaml"
    template.parent.mkdir(parents=True)
    template.write_text("apps:\n  allowed_roots: []\n", encoding="utf-8")
    (repo / "Python").mkdir()
    local = tmp_path / "private-runtime"
    layout = RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(local)},
    )

    created = initialize_runtime(layout=layout)

    assert layout.config_path in created
    assert (local / "run.ps1") in created
    assert (local / "run.sh") in created
    assert layout.config_path.read_text(encoding="utf-8") == template.read_text(
        encoding="utf-8"
    )
    for directory in (
        layout.state_dir,
        layout.workspaces_dir,
        layout.outputs_dir,
        layout.logs_dir,
        layout.tmp_dir,
    ):
        assert directory.is_dir()
    assert "Apps\\run.ps1" in (local / "run.ps1").read_text(encoding="utf-8")
    assert "Apps/run.sh" in (local / "run.sh").read_text(encoding="utf-8")
    assert os.access(local / "run.sh", os.X_OK)


def test_launcher_has_no_arbitrary_python_or_path_fallback() -> None:
    text = LAUNCHER.read_text(encoding="utf-8-sig")
    assert "PythonExecutable" not in text
    assert "solarphysics_backup" not in text
    assert ".venv" not in text
    assert "$sourceRoots" not in text
    assert "solarphysics_env_latest" in text
    assert "solarphysics_env" in text
    shell_text = SHELL_LAUNCHER.read_text(encoding="utf-8")
    assert "solarphysics_env_latest" in shell_text
    assert "solarphysics_backup" not in shell_text
    assert ".venv" not in shell_text
    probe = (APPS_ROOT / "solar_apps" / "platform" / "environment_probe.py").read_text(
        encoding="utf-8"
    )
    assert "solarphysics-apps" in probe
    assert "solar-physics-toolkit" in probe
    assert "solar_apps.launcher" in text
    assert "solar_apps.launcher" in shell_text


@pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-only")
def test_launcher_rejects_historical_backup_before_starting_python() -> None:
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-EnvironmentName",
            "solarphysics_backup",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "Unsupported environment" in completed.stderr


@pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-only")
@pytest.mark.parametrize(
    "working_directory",
    [REPO_ROOT, REPO_ROOT / "Python"],
    ids=["repository-root", "python-directory"],
)
def test_launcher_runs_from_the_current_supported_miniforge(
    working_directory: Path,
) -> None:
    runtime = inspect_miniforge_runtime(sys.executable)
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-MiniforgeRoot",
            str(runtime.miniforge_root),
            "frontend",
            "--help",
        ],
        cwd=working_directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "workbench" in completed.stdout
