"""Tests for the bounded native App 1.0 task controller."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer

from solar_apps.frontends.app_v1 import tasks
from solar_apps.frontends.app_v1.tasks import TaskQueueController
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
