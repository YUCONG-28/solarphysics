# SPDX-License-Identifier: GPL-3.0-only
"""Queued, cancellable PyQt6 subprocess tasks for the App 1.0 shell."""

from __future__ import annotations

import json
import re
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QTimer,
    pyqtSignal,
)

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.processes import (
    miniforge_subprocess_environment,
    selected_python_executable,
)

from .contracts import WorkerEventV1

_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_PROGRESS = re.compile(r"^PROGRESS\s+(\d{1,3})$")
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
_EVENT_PREFIX = "APP_V1_EVENT "


@dataclass(slots=True)
class TaskRecord:
    """In-memory presentation state for one controlled child process."""

    task_id: str
    title: str
    module_id: str
    python_module: str
    arguments: tuple[str, ...]
    output_dir: str | None = None
    retry_of: str | None = None
    status: str = "queued"
    progress: int = 0
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)
    events: list[WorkerEventV1] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    manifest_path: str | None = None


class TaskQueueController(QObject):
    """Execute queued Python modules with bounded supervised concurrency."""

    task_changed = pyqtSignal(str)
    log_line = pyqtSignal(str, str)
    worker_event = pyqtSignal(str, object)
    artifact_ready = pyqtSignal(str, str)
    queue_idle = pyqtSignal()

    def __init__(self, layout: RuntimeLayout, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.layout = layout
        self._records: dict[str, TaskRecord] = {}
        self._pending: deque[str] = deque()
        self._processes: dict[str, QProcess] = {}
        self._stdout_buffers: dict[str, str] = {}
        self._max_concurrency = 1
        self._shutdown = False

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        return tuple(self._records.values())

    @property
    def current_task_id(self) -> str | None:
        return next(iter(self._processes), None)

    @property
    def active_task_ids(self) -> tuple[str, ...]:
        return tuple(self._processes)

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def process_running(self) -> bool:
        return any(
            process.state() != QProcess.ProcessState.NotRunning
            for process in self._processes.values()
        )

    def set_max_concurrency(self, value: int) -> None:
        """Set the worker bound without interrupting already-running tasks."""

        clean = int(value)
        if not 1 <= clean <= 4:
            raise ValueError("Task concurrency must be between 1 and 4")
        self._max_concurrency = clean
        QTimer.singleShot(0, self._start_next)

    def task(self, task_id: str) -> TaskRecord:
        try:
            return self._records[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task: {task_id}") from exc

    def enqueue_python_module(
        self,
        *,
        title: str,
        module_id: str,
        python_module: str,
        arguments: tuple[str, ...] = (),
        output_dir: str | None = None,
        retry_of: str | None = None,
    ) -> TaskRecord:
        if self._shutdown:
            raise RuntimeError("The task queue is shutting down")
        if not title.strip() or not module_id.strip():
            raise ValueError("Task title and module ID are required")
        if not _MODULE.fullmatch(python_module):
            raise ValueError("python_module must be a dotted Python module")
        clean_arguments = tuple(str(item) for item in arguments)
        if any("\n" in item or "\r" in item for item in clean_arguments):
            raise ValueError("Task arguments may not contain newlines")
        record = TaskRecord(
            task_id=uuid.uuid4().hex,
            title=title.strip(),
            module_id=module_id.strip(),
            python_module=python_module,
            arguments=clean_arguments,
            output_dir=None if output_dir is None else str(output_dir),
            retry_of=retry_of,
        )
        self._records[record.task_id] = record
        self._pending.append(record.task_id)
        self.task_changed.emit(record.task_id)
        QTimer.singleShot(0, self._start_next)
        return record

    def enqueue_batch(
        self,
        launches: Iterable[object],
    ) -> tuple[TaskRecord, ...]:
        """Queue adapter launch records in deterministic input order."""

        records: list[TaskRecord] = []
        for launch in launches:
            records.append(
                self.enqueue_python_module(
                    title=str(getattr(launch, "title")),
                    module_id=str(getattr(launch, "module_id")),
                    python_module=str(getattr(launch, "python_module")),
                    arguments=tuple(getattr(launch, "arguments", ())),
                    output_dir=(
                        None
                        if getattr(launch, "output_dir", None) is None
                        else str(getattr(launch, "output_dir"))
                    ),
                )
            )
        return tuple(records)

    def cancel(self, task_id: str) -> TaskRecord:
        record = self.task(task_id)
        if record.status in _TERMINAL:
            return record
        if task_id not in self._processes:
            try:
                self._pending.remove(task_id)
            except ValueError:
                pass
            record.status = "cancelled"
            self.task_changed.emit(task_id)
            return record

        record.status = "cancelling"
        self.task_changed.emit(task_id)
        process = self._processes.get(task_id)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.terminate()
            QTimer.singleShot(1200, lambda: self._kill_if_running(task_id))
        return record

    def retry(self, task_id: str) -> TaskRecord:
        record = self.task(task_id)
        if record.status not in _TERMINAL:
            raise ValueError("Only terminal tasks can be retried")
        return self.enqueue_python_module(
            title=record.title,
            module_id=record.module_id,
            python_module=record.python_module,
            arguments=record.arguments,
            output_dir=record.output_dir,
            retry_of=record.task_id,
        )

    def shutdown(self, timeout_ms: int = 3000) -> None:
        self._shutdown = True
        for task_id in tuple(self._pending):
            self.cancel(task_id)
        task_ids = tuple(self._processes)
        for task_id in task_ids:
            self.cancel(task_id)
        wait_each = max(0, int(timeout_ms)) // max(1, len(task_ids))
        for task_id in task_ids:
            process = self._processes.get(task_id)
            if process is None:
                continue
            if not process.waitForFinished(wait_each):
                process.kill()
                process.waitForFinished(1000)

    def _start_next(self) -> None:
        if self._shutdown:
            return
        while self._pending and len(self._processes) < self._max_concurrency:
            task_id = self._pending.popleft()
            record = self._records[task_id]
            if record.status == "cancelled":
                continue
            self._launch(record)
        if not self._pending and not self._processes:
            self.queue_idle.emit()

    def _launch(self, record: TaskRecord) -> None:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(self.layout.apps_root))
        environment = QProcessEnvironment()
        for key, value in miniforge_subprocess_environment().items():
            environment.insert(str(key), str(value))
        cache_root = self.layout.tmp_dir / "app_v1" / "worker-cache"
        cache_dirs = {
            "MPLCONFIGDIR": cache_root / "matplotlib",
            "SUNPY_CONFIGDIR": cache_root / "sunpy",
            "XDG_CACHE_HOME": cache_root / "xdg",
        }
        for key, directory in cache_dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
            environment.insert(key, str(directory))
        environment.insert("APP_V1_RUN_ID", record.task_id)
        environment.insert("APP_V1_MODULE_ID", record.module_id)
        process.setProcessEnvironment(environment)
        process.setProgram(str(selected_python_executable()))
        process.setArguments(["-m", record.python_module, *record.arguments])
        task_id = record.task_id
        process.readyReadStandardOutput.connect(
            lambda task_id=task_id: self._read_output(task_id)
        )
        process.started.connect(lambda task_id=task_id: self._process_started(task_id))
        process.errorOccurred.connect(
            lambda error, task_id=task_id: self._process_error(task_id, error)
        )
        process.finished.connect(
            lambda exit_code, exit_status, task_id=task_id: self._process_finished(
                task_id,
                exit_code,
                exit_status,
            )
        )
        self._processes[task_id] = process
        self._stdout_buffers[task_id] = ""
        record.status = "starting"
        self.task_changed.emit(task_id)
        process.start()

    def _process_started(self, task_id: str) -> None:
        record = self._records.get(task_id)
        if record is None:
            return
        record.status = "running"
        self.task_changed.emit(record.task_id)

    def _read_output(self, task_id: str) -> None:
        process = self._processes.get(task_id)
        record = self._records.get(task_id)
        if process is None or record is None:
            return
        text = self._stdout_buffers.get(task_id, "") + bytes(
            process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        self._stdout_buffers[task_id] = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._stdout_buffers[task_id] = lines.pop()
        for line in lines:
            self._process_output_line(record, line.rstrip("\r\n"))

    def _process_output_line(self, record: TaskRecord, line: str) -> None:
        """Consume one complete worker protocol or legacy log line."""

        if line.startswith(_EVENT_PREFIX):
            if self._handle_worker_event(record, line.removeprefix(_EVENT_PREFIX)):
                self.task_changed.emit(record.task_id)
                return
        match = _PROGRESS.fullmatch(line.strip())
        if match:
            record.progress = min(100, int(match.group(1)))
        else:
            rendered = line.removeprefix("LOG ").strip()
            if rendered:
                record.logs.append(rendered)
                self.log_line.emit(record.task_id, rendered)
        self.task_changed.emit(record.task_id)

    def _handle_worker_event(self, record: TaskRecord, raw: str) -> bool:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("worker event must be a JSON object")
            event = WorkerEventV1(
                run_id=str(payload.get("run_id") or record.task_id),
                module_id=str(payload.get("module_id") or record.module_id),
                kind=payload["kind"],
                payload=payload.get("payload") or {},
                schema_version=int(payload.get("schema_version", 1)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rendered = f"Ignored invalid worker event: {exc}"
            record.logs.append(rendered)
            self.log_line.emit(record.task_id, rendered)
            return True
        record.events.append(event)
        if event.kind == "progress":
            value = event.payload.get("percent")
            if isinstance(value, (int, float)):
                record.progress = min(100, max(0, int(value)))
        elif event.kind in {"preview", "artifact"}:
            value = event.payload.get("path")
            if isinstance(value, str) and value and value not in record.artifacts:
                record.artifacts.append(value)
                self.artifact_ready.emit(record.task_id, value)
        elif event.kind == "result":
            value = event.payload.get("manifest_path")
            if isinstance(value, str) and value:
                record.manifest_path = value
        self.worker_event.emit(record.task_id, event)
        return True

    def _process_error(
        self,
        task_id: str,
        _error: QProcess.ProcessError,
    ) -> None:
        record = self._records.get(task_id)
        if record is None or record.status == "cancelling":
            return
        record.logs.append("The worker process could not be started or continued.")
        self.log_line.emit(record.task_id, record.logs[-1])

    def _process_finished(
        self,
        task_id: str,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_output(task_id)
        record = self._records.get(task_id)
        if record is not None:
            buffer = self._stdout_buffers.get(task_id, "")
            if buffer:
                self._process_output_line(record, buffer)
            record.return_code = int(exit_code)
            if record.status == "cancelling":
                record.status = "cancelled"
            elif exit_code == 0:
                record.status = "succeeded"
                record.progress = 100
                self._discover_output_artifacts(record)
            else:
                record.status = "failed"
            self.task_changed.emit(record.task_id)
        process = self._processes.pop(task_id, None)
        self._stdout_buffers.pop(task_id, None)
        if process is not None:
            process.deleteLater()
        QTimer.singleShot(0, self._start_next)

    def _discover_output_artifacts(self, record: TaskRecord) -> None:
        """Expose products from legacy-compatible headless workers."""

        if not record.output_dir:
            return
        root = Path(record.output_dir)
        if not root.is_dir():
            return
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            rendered = str(path)
            if path.name in {"manifest.json", "artifact-manifest.json"}:
                record.manifest_path = rendered
            if rendered in record.artifacts:
                continue
            record.artifacts.append(rendered)
            self.artifact_ready.emit(record.task_id, rendered)

    def _kill_if_running(self, task_id: str) -> None:
        process = self._processes.get(task_id)
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()


__all__ = ["TaskQueueController", "TaskRecord"]
