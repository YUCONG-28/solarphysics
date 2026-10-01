"""Tests for the bounded native App 1.0 task controller."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QProcess, QTimer

from solar_apps.frontends.app_v1 import tasks
from solar_apps.frontends.app_v1.tasks import TaskQueueController, TaskRecord
from solar_apps.platform.layout import RuntimeLayout


def _layout(tmp_path: Path) -> RuntimeLayout:
    repo = Path(__file__).resolve().parents[4]
    return RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )


def test_concurrency_is_bounded_between_one_and_four(tmp_path: Path) -> None:
    controller = TaskQueueController(_layout(tmp_path))

    with pytest.raises(ValueError, match="between 1 and 4"):
        controller.set_max_concurrency(0)
    with pytest.raises(ValueError, match="between 1 and 4"):
        controller.set_max_concurrency(5)

    controller.set_max_concurrency(4)
    assert controller.max_concurrency == 4


def test_two_tasks_can_run_concurrently_and_finish_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    monkeypatch.setattr(
        tasks, "selected_python_executable", lambda: Path(sys.executable)
    )
    controller = TaskQueueController(_layout(tmp_path))
    controller.set_max_concurrency(2)
    maximum_active = 0

    def observe(_task_id: str) -> None:
        nonlocal maximum_active
        active = sum(
            record.status in {"starting", "running"} for record in controller.records
        )
        maximum_active = max(maximum_active, active)

    controller.task_changed.connect(observe)
    loop = QEventLoop()
    controller.queue_idle.connect(loop.quit)
    for index in range(2):
        controller.enqueue_python_module(
            title=f"worker {index}",
            module_id="workbench",
            python_module="solar_apps.frontends.app_v1.task_worker",
            arguments=("--steps", "8", "--delay-ms", "30"),
        )
    QTimer.singleShot(15_000, loop.quit)
    loop.exec()
    app.processEvents()

    assert maximum_active == 2
    assert [record.status for record in controller.records] == [
        "succeeded",
        "succeeded",
    ]
    assert not controller.process_running
    assert controller.active_task_ids == ()
    cache_root = controller.layout.tmp_dir / "app_v1" / "worker-cache"
    assert {path.name for path in cache_root.iterdir()} == {
        "matplotlib",
        "sunpy",
        "xdg",
    }
    controller.shutdown()


def test_failed_to_start_finishes_once_and_continues_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    executables = iter((tmp_path / "missing-python", Path(sys.executable)))
    monkeypatch.setattr(tasks, "selected_python_executable", lambda: next(executables))
    controller = TaskQueueController(_layout(tmp_path))
    terminal_events: list[str] = []
    controller.task_changed.connect(
        lambda task_id: (
            terminal_events.append(task_id)
            if controller.task(task_id).status in {"succeeded", "failed", "cancelled"}
            else None
        )
    )
    failed = controller.enqueue_python_module(
        title="cannot start",
        module_id="workbench",
        python_module="solar_apps.frontends.app_v1.task_worker",
    )
    following = controller.enqueue_python_module(
        title="following",
        module_id="workbench",
        python_module="solar_apps.frontends.app_v1.task_worker",
        arguments=("--steps", "1", "--delay-ms", "0"),
    )
    loop = QEventLoop()
    controller.queue_idle.connect(loop.quit)
    QTimer.singleShot(3_000, loop.quit)
    loop.exec()
    app.processEvents()
    assert failed.status == "failed"
    # Some Qt backends may deliver finished after the error notification.
    controller._process_finished(failed.task_id, 0, QProcess.ExitStatus.NormalExit)

    assert failed.status == "failed"
    assert following.status == "succeeded"
    assert terminal_events.count(failed.task_id) == 1
    assert terminal_events.count(following.task_id) == 1
    assert controller.active_task_ids == ()
    assert controller._stdout_buffers == {}
    assert not controller._pending
    controller.shutdown()


def test_failed_to_start_blocks_dependents_and_finishes_independent_dag_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solar_apps.frontends.app_v1.flow_execution import FlowExecutionController
    from solar_apps.frontends.app_v1.function_catalog import DEFAULT_FUNCTION_CATALOG
    from solar_apps.frontends.app_v1.flows import AppV1FlowV1, FlowEdgeV1, FlowNodeV1

    app = QCoreApplication.instance() or QCoreApplication([])
    executables = iter((tmp_path / "missing-python", Path(sys.executable)))
    monkeypatch.setattr(tasks, "selected_python_executable", lambda: next(executables))
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    source = tmp_path / "input.csv"
    source.write_text("synthetic", encoding="utf-8")
    controller = FlowExecutionController(_layout(tmp_path), DEFAULT_FUNCTION_CATALOG)
    flow = AppV1FlowV1(
        "start-failure",
        "Start failure",
        nodes=(
            FlowNodeV1(
                "bad", "artifact-input", {"path": str(source), "artifact_type": "table"}
            ),
            FlowNodeV1(
                "independent",
                "artifact-input",
                {"path": str(source), "artifact_type": "table"},
            ),
            FlowNodeV1("blocked", "newkirk-diagnostics"),
        ),
        edges=(FlowEdgeV1("bad", "artifact", "blocked", "drift"),),
        concurrency=1,
    )
    results: list[dict[str, int]] = []
    loop = QEventLoop()
    controller.flow_finished.connect(
        lambda result: (results.append(result), loop.quit())
    )
    QTimer.singleShot(3_000, loop.quit)
    controller.run(flow)
    loop.exec()
    app.processEvents()

    assert results
    assert controller.states["bad"].status == "failed"
    assert controller.states["blocked"].status == "blocked"
    assert controller.states["independent"].status == "succeeded"
    assert all(lane.active_task_ids == () for lane in controller._lanes)
    controller.shutdown()


def test_retry_reuses_composer_snapshot_arguments(tmp_path: Path) -> None:
    controller = TaskQueueController(_layout(tmp_path))
    arguments = ("--project", "original.fic.json", "--project-sha256", "original-hash")
    original = controller.enqueue_python_module(
        title="composer",
        module_id="image-composer",
        python_module="solar_apps.frontends.app_v1.composer_worker",
        arguments=arguments,
    )
    controller.cancel(original.task_id)
    retried = controller.retry(original.task_id)
    assert retried.arguments == arguments
    assert retried.retry_of == original.task_id
    controller.shutdown()


def test_worker_artifact_event_preserves_source_port_and_content_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metadata.json"
    artifact.write_bytes(b"{}")
    controller = TaskQueueController(_layout(tmp_path))
    record = TaskRecord(
        task_id="task-1",
        title="worker",
        module_id="workbench",
        python_module="example.worker",
        arguments=(),
    )
    raw = json.dumps(
        {
            "schema_version": 1,
            "kind": "artifact",
            "payload": {
                "path": str(artifact),
                "source_port": "metadata",
            },
        }
    )

    assert controller._handle_worker_event(record, raw) is True
    controller._finalize_artifact_records(record)

    assert record.artifact_records == [
        {
            "path": str(artifact),
            "source_port": "metadata",
            "sha256": (
                "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            ),
            "bytes": 2,
        }
    ]


def test_working_directory_defaults_to_repo_root(tmp_path: Path) -> None:
    controller = TaskQueueController(_layout(tmp_path))

    assert controller.working_directory == _layout(tmp_path).repo_root


def test_set_working_directory_accepts_existing_subdirectory(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    controller = TaskQueueController(layout)
    target = tmp_path / "subdir"
    target.mkdir()

    result = controller.set_working_directory(target)

    assert result == target.resolve()
    assert controller.working_directory == target.resolve()


def test_set_working_directory_rejects_relative_path(tmp_path: Path) -> None:
    controller = TaskQueueController(_layout(tmp_path))

    with pytest.raises(ValueError, match="absolute"):
        controller.set_working_directory("relative/dir")


def test_set_working_directory_rejects_missing_path(tmp_path: Path) -> None:
    controller = TaskQueueController(_layout(tmp_path))

    with pytest.raises(NotADirectoryError):
        controller.set_working_directory(tmp_path / "does-not-exist")


def test_launch_uses_configured_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSignal:
        def connect(self, _slot: object) -> None:
            pass

    class FakeQProcess:
        ProcessChannelMode = type("ProcessChannelMode", (), {"MergedChannels": 1})
        ProcessError = type("ProcessError", (), {})
        ProcessState = type("ProcessState", (), {"NotRunning": 1})
        ExitStatus = type("ExitStatus", (), {})

        def __init__(self, parent: object | None = None) -> None:
            self.working_directory: str | None = None
            self.readyReadStandardOutput = FakeSignal()
            self.started = FakeSignal()
            self.errorOccurred = FakeSignal()
            self.finished = FakeSignal()

        def setProcessChannelMode(self, _mode: object) -> None:
            pass

        def setWorkingDirectory(self, path: object) -> None:
            self.working_directory = str(path)

        def setProcessEnvironment(self, _environment: object) -> None:
            pass

        def setProgram(self, _program: object) -> None:
            pass

        def setArguments(self, _arguments: object) -> None:
            pass

        def start(self) -> None:
            pass

    controller = TaskQueueController(_layout(tmp_path))
    monkeypatch.setattr(tasks, "QProcess", FakeQProcess)
    monkeypatch.setattr(
        tasks, "selected_python_executable", lambda: Path(sys.executable)
    )
    monkeypatch.setattr(tasks, "miniforge_subprocess_environment", lambda: {})
    record = TaskRecord(
        task_id="task-wd",
        title="worker",
        module_id="workbench",
        python_module="example.worker",
        arguments=(),
    )

    controller._launch(record)

    process = controller._processes["task-wd"]
    assert process.working_directory == str(controller.working_directory)
