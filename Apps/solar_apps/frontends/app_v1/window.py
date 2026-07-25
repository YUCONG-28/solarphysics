# SPDX-License-Identifier: GPL-3.0-only
"""Phase 1 PyQt6 application shell and placeholder pages."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.ui.state import frontend_state_store

from .catalog import MODULES
from .contracts import AppV1ProjectV1, ModuleDescriptor
from .phase2a import Phase2AAdapter, TaskLaunch
from .phase2a_page import Phase2APanel
from .phase2b import Phase2BAdapter
from .phase2b_page import Phase2BPanel
from .phase2c import Phase2CAdapter
from .phase2c_page import Phase2CPanel
from .phase4 import Phase4ComposerAdapter
from .phase4_page import Phase4ComposerPanel
from .project_store import AppV1ProjectStore
from .project_ui import ProjectPanel
from .runtime import AppV1RuntimePaths
from .tasks import TaskQueueController, TaskRecord
from .theme import AppV1ThemeController
from .timeline import SQLiteTimelineIndex, TimeCoordinator
from .time_sync_ui import TimeSyncPanel


class PlotSurface(QWidget):
    """Theme-invariant preview surface reserved for scientific plots."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("appV1PlotSurface")
        self.setMinimumSize(420, 280)
        self.setToolTip("Scientific rendering is disabled in Phase 1.")

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#10151f"))
        pen = QPen(QColor("#273449"))
        pen.setWidth(1)
        painter.setPen(pen)
        spacing = 40
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Plot Preview\nScientific execution is disabled in Phase 1",
        )


class ModulePage(QWidget):
    """One registered interface page with an explicit Phase 1 placeholder."""

    launch_demo_requested = pyqtSignal(str)
    task_launch_requested = pyqtSignal(object)

    def __init__(
        self,
        descriptor: ModuleDescriptor,
        runtime_layout: RuntimeLayout,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.descriptor = descriptor
        self.time_status: QLabel | None = None
        self.phase4_panel: Phase4ComposerPanel | None = None
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        title = QLabel(descriptor.title)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        badge = QLabel(f"Target phase {descriptor.target_phase}")
        badge.setProperty("badge", True)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(badge)
        layout.addLayout(header)

        description = QLabel(self._description())
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(description)
        if descriptor.time_aware:
            self.time_status = QLabel("UTC sync: no current time selected")
            self.time_status.setWordWrap(True)
            self.time_status.setProperty("muted", True)
            layout.addWidget(self.time_status)
        active_phase = None
        if descriptor.module_id == "image-viewer":
            phase2a = Phase2APanel(Phase2AAdapter(runtime_layout))
            phase2a.task_requested.connect(self.task_launch_requested)
            layout.addWidget(phase2a, 1)
            active_phase = "2A"
        elif descriptor.module_id in {
            "bad-frame-review",
            "source-map",
            "roi-lightcurve",
            "radio-composite",
        }:
            phase2b = Phase2BPanel(
                Phase2BAdapter(runtime_layout),
                descriptor.module_id,
            )
            phase2b.task_requested.connect(self.task_launch_requested)
            layout.addWidget(phase2b, 1)
            active_phase = "2B"
        elif descriptor.module_id in {"dart-spectrogram", "source-trajectory"}:
            phase2c = Phase2CPanel(
                Phase2CAdapter(runtime_layout),
                descriptor.module_id,
            )
            phase2c.task_requested.connect(self.task_launch_requested)
            layout.addWidget(phase2c, 1)
            active_phase = "2C"
        elif descriptor.module_id == "image-composer":
            self.phase4_panel = Phase4ComposerPanel(
                Phase4ComposerAdapter(runtime_layout)
            )
            self.phase4_panel.task_requested.connect(self.task_launch_requested)
            layout.addWidget(self.phase4_panel, 1)
            active_phase = "4"
        else:
            layout.addWidget(PlotSurface(), 1)

        footer = QHBoxLayout()
        state = QLabel(
            f"Phase {active_phase} adapter active."
            if active_phase is not None
            else "Placeholder page — no scientific calculation is available."
        )
        state.setProperty("muted", True)
        demo = QPushButton("Run shell demo task")
        demo.clicked.connect(
            lambda: self.launch_demo_requested.emit(descriptor.module_id)
        )
        footer.addWidget(state)
        footer.addStretch(1)
        footer.addWidget(demo)
        layout.addLayout(footer)

    def apply_sync_selection(
        self,
        selection,
        *,
        source_modules: dict[str, str],
    ) -> None:  # type: ignore[no-untyped-def]
        if self.time_status is None:
            return
        relevant = {
            source_id: locator
            for source_id, locator in selection.matched_locators.items()
            if source_modules.get(source_id) == self.descriptor.module_id
        }
        current = selection.current_time_utc.isoformat().replace("+00:00", "Z")
        if relevant:
            matched = sum(locator is not None for locator in relevant.values())
            detail = f"{matched}/{len(relevant)} local source(s) matched"
        else:
            detail = "current time broadcast received"
        self.time_status.setText(f"UTC sync: {current} — {detail}")
        if self.phase4_panel is not None:
            self.phase4_panel.set_current_time(selection.current_time_utc)

    def _description(self) -> str:
        if self.descriptor.module_id == "workbench":
            return (
                "Application home, project context, recent tasks, and output summary. "
                "Scientific tools remain in their dedicated pages."
            )
        if self.descriptor.module_id == "radio-workspace":
            return (
                "Navigation and task aggregation for radio interfaces. "
                "It does not duplicate radio science implementations."
            )
        return (
            f"{self.descriptor.title} is registered for Phase "
            f"{self.descriptor.target_phase}. The current page validates navigation, "
            "layout, theme, and process orchestration only."
        )


class AppV1MainWindow(QMainWindow):
    """Native application shell shared by all later App 1.0 phases."""

    def __init__(
        self,
        layout: RuntimeLayout,
        *,
        initial_theme: object = "auto",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layout = layout
        self.runtime_paths = AppV1RuntimePaths.from_layout(layout)
        self.timeline_index = SQLiteTimelineIndex(self.runtime_paths.time_index_path)
        self.time_coordinator = TimeCoordinator(self.timeline_index)
        self.project_store = AppV1ProjectStore(self.runtime_paths)
        self.project_parameters: dict[str, dict[str, object]] = {}
        self._parameter_module_id: str | None = None
        self.timeline_config_path = (
            self.runtime_paths.workspaces_dir / "preview.timeline.json"
        )
        self.timeline_load_warning: str | None = None
        try:
            self.time_coordinator.load(self.timeline_config_path)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            self.timeline_load_warning = (
                f"Timeline configuration was not restored: {type(exc).__name__}: {exc}"
            )
        self.setObjectName("appV1MainWindow")
        self.setWindowTitle("Solar Physics App 1.0")
        self.resize(1320, 860)

        state_store = frontend_state_store("app-v1", layout=layout)
        self.theme_controller = AppV1ThemeController(
            self._application(),
            state_store=state_store,
            initial_mode=initial_theme,
        )
        self.task_controller = TaskQueueController(layout, self)
        self.module_pages: dict[str, ModulePage] = {}
        self._module_rows: dict[str, int] = {}

        self._build_toolbar()
        self._build_central()
        self._build_parameter_dock()
        self._build_project_dock()
        self._build_task_dock()
        self._build_log_dock()
        self._build_output_dock()
        self._build_time_sync_dock()
        self._connect_signals()
        self._select_module("workbench")
        if self.timeline_load_warning:
            self.log_output.appendPlainText(self.timeline_load_warning)
        self.statusBar().showMessage("App 1.0 ready")

    @property
    def registered_module_ids(self) -> tuple[str, ...]:
        return tuple(self.module_pages)

    def set_theme(self, mode: object) -> str:
        selected = self.theme_controller.set_mode(mode)
        index = self.theme_combo.findData(selected)
        if index >= 0 and index != self.theme_combo.currentIndex():
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)
        return self.theme_controller.effective_mode()

    def enqueue_demo_task(
        self,
        module_id: str = "workbench",
        *,
        steps: int = 8,
        delay_ms: int = 30,
    ) -> TaskRecord:
        return self.task_controller.enqueue_python_module(
            title=f"{self.module_pages[module_id].descriptor.title} shell demo",
            module_id=module_id,
            python_module="solar_apps.frontends.app_v1.task_worker",
            arguments=(
                "--steps",
                str(max(1, int(steps))),
                "--delay-ms",
                str(max(0, int(delay_ms))),
            ),
        )

    def demo_confirmation_summary(self, module_id: str) -> str:
        descriptor = self.module_pages[module_id].descriptor
        return "\n".join(
            (
                f"Module: {descriptor.title}",
                "Input: none",
                "Parameters: 8 progress steps, 30 ms per step",
                "Output: none; no scientific artifact will be created",
                "Workload: lightweight shell lifecycle check",
            )
        )

    def confirm_and_enqueue_demo(self, module_id: str) -> TaskRecord | None:
        decision = QMessageBox.question(
            self,
            "Confirm shell demo task",
            self.demo_confirmation_summary(module_id),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if decision != QMessageBox.StandardButton.Yes:
            return None
        return self.enqueue_demo_task(module_id)

    def cancel_current_task(self) -> TaskRecord | None:
        task_id = self.task_controller.current_task_id
        if task_id is None:
            return None
        return self.task_controller.cancel(task_id)

    def redraw_last_task(self) -> TaskRecord | None:
        """Re-run the most recent terminal task with identical parameters."""

        for record in reversed(self.task_controller.records):
            if record.status in {"succeeded", "failed", "cancelled"}:
                return self.task_controller.retry(record.task_id)
        return None

    def recover_failed_tasks(self) -> tuple[TaskRecord, ...]:
        """Queue every failed task again in its original order."""

        already_retried = {
            record.retry_of
            for record in self.task_controller.records
            if record.retry_of is not None
        }
        return tuple(
            self.task_controller.retry(record.task_id)
            for record in self.task_controller.records
            if record.status == "failed" and record.task_id not in already_retried
        )

    def save_current_project(self) -> Path:
        self._capture_parameter_document()
        artifact_manifests: list[str] = []
        for record in self.task_controller.records:
            if record.status != "succeeded" or not record.output_dir:
                continue
            output = Path(record.output_dir)
            candidates = (
                tuple(output.rglob("manifest.json")) if output.is_dir() else (output,)
            )
            for candidate in candidates:
                try:
                    reference = self.project_store.normalize_manifest_reference(
                        candidate
                    )
                except ValueError:
                    continue
                if reference not in artifact_manifests:
                    artifact_manifests.append(reference)
        project = AppV1ProjectV1(
            project_id=self.project_panel.project_id.text(),
            name=self.project_panel.project_name.text(),
            modules=self.registered_module_ids,
            parameters=self.project_parameters,
            timeline=self.time_coordinator.to_dict(),
            layout={
                "geometry_base64": bytes(self.saveGeometry().toBase64()).decode(
                    "ascii"
                ),
                "dock_state_base64": bytes(self.saveState().toBase64()).decode("ascii"),
            },
            artifact_manifests=tuple(artifact_manifests),
            saved_at_utc=datetime.now(timezone.utc),
        )
        target = self.project_store.save(project)
        self.project_panel.status.setText(f"Saved {target.name}")
        return target

    def load_current_project(self) -> AppV1ProjectV1:
        project = self.project_store.load(self.project_panel.project_id.text())
        self.project_panel.project_name.setText(project.name)
        self.project_parameters = {
            str(key): dict(value)
            for key, value in project.parameters.items()
            if isinstance(value, dict)
        }
        self.time_coordinator.load_dict(project.timeline)
        self.time_sync_panel.refresh_sources()
        geometry = project.layout.get("geometry_base64")
        dock_state = project.layout.get("dock_state_base64")
        if isinstance(geometry, str):
            self.restoreGeometry(base64.b64decode(geometry))
        if isinstance(dock_state, str):
            self.restoreState(base64.b64decode(dock_state))
        self._show_parameter_document(self._current_module_id())
        self.output_list.clear()
        for reference in project.artifact_manifests:
            self.output_list.addItem(f"Project artifact: {reference}")
        if self.time_coordinator.current_time_utc is not None:
            self.time_coordinator.select(self.time_coordinator.current_time_utc)
        self.project_panel.status.setText(f"Opened {project.project_id}.spapp.json")
        return project

    def save_current_preset(self) -> Path:
        module_id = self._current_module_id()
        parameters = self._read_parameter_document()
        target = self.project_store.save_preset(
            module_id,
            self.project_panel.preset_id.text(),
            parameters,
        )
        self.project_parameters[module_id] = parameters
        self.project_panel.status.setText(f"Saved preset {target.stem}")
        return target

    def load_current_preset(self) -> dict[str, object]:
        module_id = self._current_module_id()
        parameters = self.project_store.load_preset(
            module_id,
            self.project_panel.preset_id.text(),
        )
        self.project_parameters[module_id] = parameters
        self.parameter_document.setPlainText(
            json.dumps(parameters, indent=2, ensure_ascii=False, sort_keys=True)
        )
        self.project_panel.status.setText(
            f"Loaded preset {self.project_panel.preset_id.text()}"
        )
        return parameters

    def closeEvent(self, event: QCloseEvent) -> None:
        try:
            self._capture_parameter_document()
        except (TypeError, ValueError):
            pass
        self.task_controller.shutdown()
        try:
            self.time_coordinator.save(self.timeline_config_path)
        except OSError as exc:
            self.log_output.appendPlainText(
                f"Timeline configuration could not be saved: {exc}"
            )
        event.accept()

    def register_timeline_source(
        self,
        source_id: str,
        *,
        module_id: str | None = None,
        offset_seconds: float = 0.0,
        tolerance_seconds: float = 0.0,
    ):  # type: ignore[no-untyped-def]
        registration = self.time_coordinator.register_source(
            source_id,
            module_id=module_id,
            offset_seconds=offset_seconds,
            tolerance_seconds=tolerance_seconds,
        )
        self.time_sync_panel.refresh_sources()
        return registration

    def _application(self):  # type: ignore[no-untyped-def]
        from PyQt6.QtWidgets import QApplication

        application = QApplication.instance()
        if application is None:
            raise RuntimeError("Create QApplication before AppV1MainWindow")
        return application

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Application")
        toolbar.setMovable(False)
        toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        for label, value in (
            ("Auto", "auto"),
            ("Light", "light"),
            ("Dark", "dark"),
        ):
            self.theme_combo.addItem(label, value)
        current = self.theme_combo.findData(self.theme_controller.mode)
        self.theme_combo.setCurrentIndex(max(0, current))
        toolbar.addWidget(self.theme_combo)
        toolbar.addSeparator()
        self.demo_button = QPushButton("Run demo task")
        self.redraw_button = QPushButton("Redraw last")
        self.recover_button = QPushButton("Retry failed")
        self.cancel_button = QPushButton("Cancel current task")
        self.redraw_button.setEnabled(False)
        self.recover_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        toolbar.addWidget(self.demo_button)
        toolbar.addWidget(self.redraw_button)
        toolbar.addWidget(self.recover_button)
        toolbar.addWidget(self.cancel_button)

    def _build_central(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        self.navigation = QTreeWidget()
        self.navigation.setHeaderHidden(True)
        self.navigation.setMinimumWidth(220)
        self.navigation.setMaximumWidth(310)
        self.navigation.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.page_stack = QStackedWidget()
        layout.addWidget(self.navigation)
        layout.addWidget(self.page_stack, 1)
        self.setCentralWidget(root)

        categories: dict[str, QTreeWidgetItem] = {}
        for descriptor in MODULES:
            category = categories.get(descriptor.category)
            if category is None:
                category = QTreeWidgetItem([descriptor.category])
                category.setFlags(category.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                categories[descriptor.category] = category
                self.navigation.addTopLevelItem(category)
            item = QTreeWidgetItem([descriptor.title])
            item.setData(0, Qt.ItemDataRole.UserRole, descriptor.module_id)
            category.addChild(item)
            page = ModulePage(descriptor, self.layout)
            self.module_pages[descriptor.module_id] = page
            self.page_stack.addWidget(page)
        self.navigation.expandAll()

    def _build_parameter_dock(self) -> None:
        self.parameter_dock = QDockWidget("Parameters", self)
        self.parameter_dock.setObjectName("appV1ParametersDock")
        panel = QWidget()
        form = QFormLayout(panel)
        self.parameter_module = QLabel()
        self.parameter_input = QLabel("No input selected")
        self.parameter_output = QLabel("No output will be created")
        self.parameter_workload = QLabel("Shell-only preview")
        self.parameter_document = QPlainTextEdit("{}")
        self.parameter_document.setToolTip(
            "Versioned JSON parameters saved in projects and presets."
        )
        self.parameter_document.setMaximumHeight(150)
        for widget in (
            self.parameter_input,
            self.parameter_output,
            self.parameter_workload,
        ):
            widget.setWordWrap(True)
            widget.setProperty("muted", True)
        form.addRow("Module", self.parameter_module)
        form.addRow("Input", self.parameter_input)
        form.addRow("Output", self.parameter_output)
        form.addRow("Workload", self.parameter_workload)
        form.addRow("Parameter JSON", self.parameter_document)
        self.parameter_dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.parameter_dock)

    def _build_project_dock(self) -> None:
        self.project_dock = QDockWidget("Project", self)
        self.project_dock.setObjectName("appV1ProjectDock")
        self.project_panel = ProjectPanel()
        self.project_dock.setWidget(self.project_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.project_dock)
        self.tabifyDockWidget(self.parameter_dock, self.project_dock)

    def _build_task_dock(self) -> None:
        self.task_dock = QDockWidget("Task Queue", self)
        self.task_dock.setObjectName("appV1TaskDock")
        self.task_table = QTableWidget(0, 4)
        self.task_table.setHorizontalHeaderLabels(
            ["Task", "Module", "Status", "Progress"]
        )
        self.task_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.task_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self.task_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.task_dock.setWidget(self.task_table)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.task_dock)

    def _build_log_dock(self) -> None:
        self.log_dock = QDockWidget("Logs", self)
        self.log_dock.setObjectName("appV1LogDock")
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_dock.setWidget(self.log_output)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        self.tabifyDockWidget(self.task_dock, self.log_dock)

    def _build_output_dock(self) -> None:
        self.output_dock = QDockWidget("Outputs", self)
        self.output_dock.setObjectName("appV1OutputDock")
        self.output_list = QListWidget()
        self.output_list.addItem(
            "Phase 1 does not create scientific products or partial outputs."
        )
        self.output_dock.setWidget(self.output_list)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.output_dock)
        self.tabifyDockWidget(self.parameter_dock, self.output_dock)
        self.parameter_dock.raise_()

    def _build_time_sync_dock(self) -> None:
        self.time_sync_dock = QDockWidget("UTC Time Sync", self)
        self.time_sync_dock.setObjectName("appV1TimeSyncDock")
        self.time_sync_panel = TimeSyncPanel(self.time_coordinator)
        self.time_sync_dock.setWidget(self.time_sync_panel)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.time_sync_dock,
        )
        self.tabifyDockWidget(self.output_dock, self.time_sync_dock)

    def _connect_signals(self) -> None:
        self.navigation.currentItemChanged.connect(self._navigation_changed)
        self.theme_combo.currentIndexChanged.connect(
            lambda _index: self.set_theme(self.theme_combo.currentData())
        )
        self.demo_button.clicked.connect(
            lambda: self.confirm_and_enqueue_demo(self._current_module_id())
        )
        self.cancel_button.clicked.connect(self.cancel_current_task)
        self.redraw_button.clicked.connect(self.redraw_last_task)
        self.recover_button.clicked.connect(self.recover_failed_tasks)
        self.project_panel.save_project_requested.connect(
            lambda: self._project_action(self.save_current_project)
        )
        self.project_panel.load_project_requested.connect(
            lambda: self._project_action(self.load_current_project)
        )
        self.project_panel.save_preset_requested.connect(
            lambda: self._project_action(self.save_current_preset)
        )
        self.project_panel.load_preset_requested.connect(
            lambda: self._project_action(self.load_current_preset)
        )
        for page in self.module_pages.values():
            page.launch_demo_requested.connect(self.confirm_and_enqueue_demo)
            page.task_launch_requested.connect(self._enqueue_launch)
        self.task_controller.task_changed.connect(self._refresh_task)
        self.task_controller.log_line.connect(self._append_log)
        self.task_controller.queue_idle.connect(
            lambda: self.statusBar().showMessage("Task queue idle")
        )
        self.time_sync_panel.selection_changed.connect(self._timeline_selection_changed)

    def _navigation_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            return
        module_id = current.data(0, Qt.ItemDataRole.UserRole)
        if module_id:
            self._select_module(str(module_id))

    def _select_module(self, module_id: str) -> None:
        if self._parameter_module_id is not None:
            try:
                self._capture_parameter_document()
            except (TypeError, ValueError) as exc:
                self.project_panel.status.setText(str(exc))
        page = self.module_pages[module_id]
        self.page_stack.setCurrentWidget(page)
        self.parameter_module.setText(page.descriptor.title)
        matches = self.navigation.findItems(
            page.descriptor.title,
            Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive,
        )
        if matches and self.navigation.currentItem() is not matches[0]:
            self.navigation.setCurrentItem(matches[0])
        self._show_parameter_document(module_id)
        self.statusBar().showMessage(f"Selected {page.descriptor.title}")

    def _current_module_id(self) -> str:
        page = self.page_stack.currentWidget()
        if isinstance(page, ModulePage):
            return page.descriptor.module_id
        return "workbench"

    def _refresh_task(self, task_id: str) -> None:
        record = self.task_controller.task(task_id)
        row = self._module_rows.get(task_id)
        if row is None:
            row = self.task_table.rowCount()
            self.task_table.insertRow(row)
            self._module_rows[task_id] = row
        values = (
            record.title,
            record.module_id,
            record.status,
            f"{record.progress}%",
        )
        for column, value in enumerate(values):
            self.task_table.setItem(row, column, QTableWidgetItem(value))
        active = any(
            item.status in {"starting", "running", "cancelling"}
            for item in self.task_controller.records
        )
        self.cancel_button.setEnabled(active)
        terminal = any(
            item.status in {"succeeded", "failed", "cancelled"}
            for item in self.task_controller.records
        )
        self.redraw_button.setEnabled(terminal)
        self.recover_button.setEnabled(
            any(
                item.status == "failed"
                and item.task_id
                not in {
                    candidate.retry_of
                    for candidate in self.task_controller.records
                    if candidate.retry_of is not None
                }
                for item in self.task_controller.records
            )
        )
        self.statusBar().showMessage(f"{record.title}: {record.status}")
        if record.status == "succeeded":
            if record.output_dir:
                output = Path(record.output_dir)
                products = (
                    [path for path in output.rglob("*") if path.is_file()]
                    if output.is_dir()
                    else []
                )
                self.output_list.addItem(
                    f"{record.title}: {len(products)} product(s) — {output}"
                )
            else:
                self.output_list.addItem(
                    f"{record.title}: completed without scientific output"
                )

    def _append_log(self, task_id: str, line: str) -> None:
        record = self.task_controller.task(task_id)
        self.log_output.appendPlainText(f"[{record.title}] {line}")

    def _timeline_selection_changed(self, selection) -> None:  # type: ignore[no-untyped-def]
        source_modules = {
            source.source_id: source.module_id
            for source in self.time_coordinator.sources
        }
        for page in self.module_pages.values():
            page.apply_sync_selection(
                selection,
                source_modules=source_modules,
            )
        current = selection.current_time_utc.isoformat().replace("+00:00", "Z")
        self.statusBar().showMessage(f"UTC time synchronized: {current}")

    def _enqueue_launch(self, launch: TaskLaunch) -> None:
        self.task_controller.enqueue_python_module(
            title=launch.title,
            module_id=launch.module_id,
            python_module=launch.python_module,
            arguments=launch.arguments,
            output_dir=str(launch.output_dir),
        )

    def _read_parameter_document(self) -> dict[str, object]:
        payload = json.loads(self.parameter_document.toPlainText() or "{}")
        if not isinstance(payload, dict):
            raise TypeError("Parameter JSON must be an object")
        return payload

    def _capture_parameter_document(self) -> None:
        if self._parameter_module_id is None:
            return
        self.project_parameters[self._parameter_module_id] = (
            self._read_parameter_document()
        )

    def _show_parameter_document(self, module_id: str) -> None:
        self._parameter_module_id = module_id
        parameters = self.project_parameters.get(module_id, {})
        self.parameter_document.setPlainText(
            json.dumps(parameters, indent=2, ensure_ascii=False, sort_keys=True)
        )

    def _project_action(self, operation) -> None:  # type: ignore[no-untyped-def]
        try:
            operation()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.project_panel.status.setText(f"{type(exc).__name__}: {exc}")


__all__ = ["AppV1MainWindow", "ModulePage", "PlotSurface"]
