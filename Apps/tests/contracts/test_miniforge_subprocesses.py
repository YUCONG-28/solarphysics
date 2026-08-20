"""Regression contracts for application-owned Python child processes."""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from solar_apps.frontends.radio.composite_figure import composite_figure_launcher
from solar_apps.frontends.radio.roi_lightcurve import roi_lightcurve_launcher
from solar_apps.frontends.radio.source_map import jobs as source_map_jobs
from solar_apps.frontends.radio.source_trajectory import source_app_launcher
from solar_apps.frontends.workbench import runner as workbench_runner
from solar_apps.frontends.workbench.radio_workspace import runner as radio_runner
from solar_apps.platform.environment import inspect_miniforge_runtime
from solar_apps.platform.processes import (
    PYTHON_EXECUTABLE_ENV,
    miniforge_subprocess_environment,
    selected_python_executable,
)


class _CompletedProcess:
    returncode = 0
    pid = 100

    def __init__(self, *, with_stdout: bool = False) -> None:
        self.stdout = io.StringIO("") if with_stdout else None

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def _assert_selected_miniforge_child(
    command: list[str], kwargs: dict[str, Any]
) -> None:
    selected = str(selected_python_executable())
    selected_environment = inspect_miniforge_runtime().environment_name
    environment = kwargs["env"]

    assert os.path.normcase(command[0]) == os.path.normcase(selected)
    assert os.path.normcase(environment[PYTHON_EXECUTABLE_ENV]) == os.path.normcase(
        selected
    )
    assert environment["SOLAR_APPS_ENVIRONMENT"] == selected_environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in environment
    assert os.path.normcase(environment["PATH"].split(os.pathsep)[0]) == (
        os.path.normcase(str(Path(selected).parent))
    )


@pytest.mark.parametrize(
    "launcher",
    [composite_figure_launcher, roi_lightcurve_launcher, source_app_launcher],
    ids=["radio-composite", "roi-lightcurve", "source-trajectory"],
)
def test_streamlit_launchers_use_selected_miniforge_environment(
    launcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "must-not-reach-child")
    monkeypatch.setattr(launcher, "_pick_port", lambda _preferred: 8765)
    launched: list[tuple[list[str], dict[str, Any]]] = []
    environment_calls: list[dict[str, str]] = []

    def build_environment() -> dict[str, str]:
        environment = miniforge_subprocess_environment()
        environment_calls.append(environment)
        return environment

    def fake_popen(command: list[str], **kwargs: Any) -> _CompletedProcess:
        launched.append((command, kwargs))
        return _CompletedProcess()

    monkeypatch.setattr(launcher, "miniforge_subprocess_environment", build_environment)
    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    assert launcher.main(["--port", "0", "--no-browser", "--no-auto-stop"]) == 0
    assert len(environment_calls) == 1
    assert len(launched) == 1
    command, kwargs = launched[0]
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert kwargs["env"] is environment_calls[0]
    _assert_selected_miniforge_child(command, kwargs)


def test_source_map_worker_job_uses_selected_miniforge_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "must-not-reach-child")
    workspace = tmp_path / "source-map-job"
    workspace.mkdir()
    launched: list[tuple[list[str], dict[str, Any]]] = []
    environment_calls: list[dict[str, str]] = []

    class FixedTemporaryDirectory:
        name = str(workspace)

        def __init__(self, **_kwargs: Any) -> None:
            pass

    class DormantThread:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

    def build_environment() -> dict[str, str]:
        environment = miniforge_subprocess_environment()
        environment_calls.append(environment)
        return environment

    def fake_popen(command: list[str], **kwargs: Any) -> _CompletedProcess:
        launched.append((command, kwargs))
        return _CompletedProcess()

    monkeypatch.setattr(
        source_map_jobs.tempfile, "TemporaryDirectory", FixedTemporaryDirectory
    )
    monkeypatch.setattr(source_map_jobs.threading, "Thread", DormantThread)
    monkeypatch.setattr(
        source_map_jobs, "miniforge_subprocess_environment", build_environment
    )
    monkeypatch.setattr(source_map_jobs.subprocess, "Popen", fake_popen)

    policy = SimpleNamespace()
    registry = source_map_jobs.JobRegistry(
        policy=policy,
        artifacts=source_map_jobs.ArtifactRegistry(policy),
    )
    started = registry.start({}, {"id": "candidate-1"})

    assert started["status"] == "running"
    assert len(environment_calls) == 1
    command, kwargs = launched[0]
    assert command[1:3] == [
        "-m",
        "solar_apps.frontends.radio.source_map.worker",
    ]
    assert kwargs["env"] is environment_calls[0]
    assert kwargs["shell"] is False
    _assert_selected_miniforge_child(command, kwargs)


def test_workbench_runner_uses_selected_miniforge_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "must-not-reach-child")
    launched: list[tuple[list[str], dict[str, Any]]] = []
    environment_calls: list[tuple[str | Path | None, dict[str, str]]] = []
    real_environment_builder = miniforge_subprocess_environment

    class Feature:
        id = "contract-job"
        title = "Contract job"

        @staticmethod
        def build_command(_payload: dict[str, Any], *, context) -> list[str]:
            return [str(context.python_executable), "-m", "contract.job"]

    class Registry:
        @staticmethod
        def get(_module_id: str) -> Feature:
            return Feature()

    def build_environment(
        env: dict[str, str] | None = None,
        python_executable: str | Path | None = None,
    ) -> dict[str, str]:
        environment = real_environment_builder(env, python_executable=python_executable)
        environment_calls.append((python_executable, environment))
        return environment

    def fake_popen(command: list[str], **kwargs: Any) -> _CompletedProcess:
        launched.append((command, kwargs))
        return _CompletedProcess(with_stdout=True)

    monkeypatch.setattr(
        workbench_runner, "miniforge_subprocess_environment", build_environment
    )
    context = workbench_runner.JobContext(repo_root=tmp_path)
    runner = workbench_runner.JobRunner(Registry(), context, popen_factory=fake_popen)

    job = runner.start("contract-job")
    result = runner.wait(job.id)

    assert result["status"] == "succeeded"
    assert len(environment_calls) == 1
    assert os.path.normcase(str(environment_calls[0][0])) == os.path.normcase(
        str(selected_python_executable())
    )
    command, kwargs = launched[0]
    assert command[1:] == ["-m", "contract.job"]
    assert kwargs["env"] is environment_calls[0][1]
    _assert_selected_miniforge_child(command, kwargs)


def test_radio_workspace_runner_uses_selected_miniforge_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "must-not-reach-child")
    selected = str(selected_python_executable())
    manifest = SimpleNamespace(
        status="queued",
        provenance={},
        command=[selected, "-m", "contract.radio_job"],
        cwd=str(tmp_path),
        returncode=None,
        artifacts=[],
        module_id="contract",
        action_id="run",
        progress=0.0,
        started_at=None,
        finished_at=None,
        error=None,
    )

    class Store:
        @staticmethod
        def load_run(_workspace_id: str, _run_id: str):
            return manifest

        @staticmethod
        def save_run(_manifest) -> None:
            pass

        @staticmethod
        def append_log(_workspace_id: str, _run_id: str, _line: str) -> None:
            pass

    launched: list[tuple[list[str], dict[str, Any]]] = []
    environment_calls: list[tuple[str | Path | None, dict[str, str]]] = []

    def build_environment(
        env: dict[str, str] | None = None,
        python_executable: str | Path | None = None,
    ) -> dict[str, str]:
        environment = miniforge_subprocess_environment(
            env, python_executable=python_executable
        )
        environment_calls.append((python_executable, environment))
        return environment

    def fake_popen(command: list[str], **kwargs: Any) -> _CompletedProcess:
        launched.append((command, kwargs))
        return _CompletedProcess()

    monkeypatch.setattr(
        radio_runner, "prepend_conda_dll_paths_to_env", build_environment
    )
    manager = radio_runner.RadioRunManager.__new__(radio_runner.RadioRunManager)
    manager.store = Store()
    manager.python_executable = selected
    manager.popen_factory = fake_popen
    manager._condition = threading.Condition(threading.RLock())
    manager._processes = {}
    manager._active_total = 1
    manager._active_by_workspace = {"workspace": 1}
    manager._index_artifacts = lambda *_args, **_kwargs: []

    manager._execute("workspace", "run-1")

    assert manifest.status == "succeeded"
    assert len(environment_calls) == 1
    assert os.path.normcase(str(environment_calls[0][0])) == os.path.normcase(selected)
    command, kwargs = launched[0]
    assert command[1:] == ["-m", "contract.radio_job"]
    assert kwargs["env"] is environment_calls[0][1]
    assert kwargs["shell"] is False
    _assert_selected_miniforge_child(command, kwargs)


def test_workbench_runners_reject_an_alternate_python(tmp_path: Path) -> None:
    alternate = tmp_path / "other-environment" / "python.exe"

    with pytest.raises(ValueError, match="selected by the public Apps launcher"):
        workbench_runner.JobContext(
            repo_root=tmp_path,
            python_executable=alternate,
        )
    with pytest.raises(ValueError, match="selected by the public Apps launcher"):
        radio_runner.RadioRunManager(
            SimpleNamespace(),
            repo_root=tmp_path,
            python_executable=alternate,
        )
