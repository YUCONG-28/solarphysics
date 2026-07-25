# SPDX-License-Identifier: GPL-3.0-only
"""Queued, cancellable PyQt6 subprocess tasks for the App 1.0 shell."""

from __future__ import annotations

import re
import uuid
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

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

_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_PROGRESS = re.compile(r"^PROGRESS\s+(\d{1,3})$")
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


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


class TaskQueueController(QObject):
    """Execute Python modules sequentially using the launcher-selected runtime."""

    task_changed = pyqtSignal(str)
    log_line = pyqtSignal(str, str)
    queue_idle = pyqtSignal()

    def __init__(self, layout: RuntimeLayout, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.layout = layout
        self._records: dict[str, TaskRecord] = {}
        self._pending: deque[str] = deque()
        self._current_id: str | None = None
        self._process: QProcess | None = None
        self._shutdown = False

    @property
    def records(self) -> tuple[TaskRecord, ...]:
        return tuple(self._records.values())

    @property
    def current_task_id(self) -> str | None:
        return self._current_id

    @property
    def process_running(self) -> bool:
        return (
            self._process is not None
            and self._process.state() != QProcess.ProcessState.NotRunning
        )

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
        if task_id != self._current_id:
            try:
                self._pending.remove(task_id)
            except ValueError:
                pass
            record.status = "cancelled"
            self.task_changed.emit(task_id)
            return record

        record.status = "cancelling"
        self.task_changed.emit(task_id)
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.terminate()
            QTimer.singleShot(1200, self._kill_if_running)
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
        if self._current_id is None:
            return
        self.cancel(self._current_id)
        process = self._process
        if process is None:
            return
        if not process.waitForFinished(max(0, int(timeout_ms))):
            process.kill()
            process.waitForFinished(1000)

    def _start_next(self) -> None:
        if self._shutdown or self._current_id is not None:
            return
        while self._pending:
            task_id = self._pending.popleft()
            record = self._records[task_id]
            if record.status == "cancelled":
                continue
            self._launch(record)
            return
        self.queue_idle.emit()

    def _launch(self, record: TaskRecord) -> None:
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setWorkingDirectory(str(self.layout.apps_root))
        environment = QProcessEnvironment()
        for key, value in miniforge_subprocess_environment().items():
            environment.insert(str(key), str(value))
        process.setProcessEnvironment(environment)
        process.setProgram(str(selected_python_executable()))
        process.setArguments(["-m", record.python_module, *record.arguments])
        process.readyReadStandardOutput.connect(self._read_output)
        process.started.connect(self._process_started)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._process_finished)
        self._current_id = record.task_id
        self._process = process
        record.status = "starting"
        self.task_changed.emit(record.task_id)
        process.start()

    def _process_started(self) -> None:
        record = self._current_record()
        if record is None:
            return
        record.status = "running"
        self.task_changed.emit(record.task_id)

    def _read_output(self) -> None:
        process = self._process
        record = self._current_record()
        if process is None or record is None:
            return
        text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            match = _PROGRESS.fullmatch(line.strip())
            if match:
                record.progress = min(100, int(match.group(1)))
            else:
                rendered = line.removeprefix("LOG ").strip()
                if rendered:
                    record.logs.append(rendered)
                    self.log_line.emit(record.task_id, rendered)
            self.task_changed.emit(record.task_id)

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        record = self._current_record()
        if record is None or record.status == "cancelling":
            return
        record.logs.append("The worker process could not be started or continued.")
        self.log_line.emit(record.task_id, record.logs[-1])

    def _process_finished(
        self,
        exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        self._read_output()
        record = self._current_record()
        if record is not None:
            record.return_code = int(exit_code)
            if record.status == "cancelling":
                record.status = "cancelled"
            elif exit_code == 0:
                record.status = "succeeded"
                record.progress = 100
            else:
                record.status = "failed"
            self.task_changed.emit(record.task_id)
        process = self._process
        self._current_id = None
        self._process = None
        if process is not None:
            process.deleteLater()
        QTimer.singleShot(0, self._start_next)

    def _kill_if_running(self) -> None:
        process = self._process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def _current_record(self) -> TaskRecord | None:
        if self._current_id is None:
            return None
        return self._records.get(self._current_id)


__all__ = ["TaskQueueController", "TaskRecord"]
