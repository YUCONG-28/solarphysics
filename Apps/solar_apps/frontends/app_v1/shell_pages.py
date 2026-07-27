# SPDX-License-Identifier: GPL-3.0-only
"""Native Workbench and Radio Workspace shell pages."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .catalog import MODULES
from .components import ArtifactBrowser, NativeModulePanel
from .function_catalog import DEFAULT_FUNCTION_CATALOG
from .phase2a import TaskLaunch
from .runtime import AppV1RuntimePaths
from .schema_form import SchemaForm


class WorkbenchNativePanel(NativeModulePanel):
    """Registry-backed native form for supervised command workflows."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        runtime: AppV1RuntimePaths,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("workbench", legacy_label="legacy Workbench", parent=parent)
        self.runtime = runtime
        self._specs = {
            item.function_id: item for item in DEFAULT_FUNCTION_CATALOG.functions
        }
        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workflows = QListWidget()
        self.workflows.setMinimumWidth(180)
        self.workflows.setMaximumWidth(260)
        for spec in sorted(
            self._specs.values(),
            key=lambda item: (item.category, item.title),
        ):
            item = QListWidgetItem(f"{spec.category} · {spec.title}")
            item.setData(Qt.ItemDataRole.UserRole, spec.function_id)
            self.workflows.addItem(item)
        self.workflows.currentItemChanged.connect(self._selection_changed)
        splitter.addWidget(self.workflows)

        editor = QWidget()
        editor_layout = QVBoxLayout(editor)
        self.title = QLabel("Select a workflow")
        self.title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setProperty("muted", True)
        try:
            configured = configured_allowed_roots()
        except (OSError, TypeError, ValueError):
            configured = ()
        self.allowed_roots = (
            *configured,
            self.runtime.outputs_dir,
            self.runtime.workspaces_dir,
            self.runtime.tmp_dir,
        )
        self.form = SchemaForm(
            allowed_roots=tuple(map(str, self.allowed_roots))
        )
        run = QPushButton("Review and run")
        run.setProperty("primary", True)
        run.clicked.connect(self._request_run)
        editor_layout.addWidget(self.title)
        editor_layout.addWidget(self.description)
        editor_layout.addWidget(self.form, 1)
        editor_layout.addWidget(run)
        self.artifacts = ArtifactBrowser()
        editor_layout.addWidget(self.artifacts, 1)
        splitter.addWidget(editor)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 650])
        root.addWidget(splitter, 1)
        if self.workflows.count():
            self.workflows.setCurrentRow(0)

    def _selection_changed(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        spec = self._specs[str(current.data(Qt.ItemDataRole.UserRole))]
        self.title.setText(spec.title)
        self.description.setText(spec.description)
        self.form.set_function(spec)

    def _request_run(self) -> None:
        item = self.workflows.currentItem()
        if item is None:
            return
        spec = self._specs[str(item.data(Qt.ItemDataRole.UserRole))]
        try:
            values = self.form.values()
            output = self._output_directory(spec.function_id, values)
            module, arguments, normalized = spec.build_arguments(
                values,
                variant_id=self.form.selected_variant_id(),
                default_output=str(output),
                allowed_roots=self.allowed_roots,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.record_diagnostic(exc)
            self.description.setText(str(exc))
            return
        summary = {
            "Module": spec.title,
            "Input": "Typed paths and connected artifacts from the shared schema",
            "Parameters": json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "Output": str(output),
            "Workload": "One supervised native Workbench task",
        }
        if not self.confirm(self, f"Run {spec.title}", summary):
            return
        self.task_requested.emit(
            TaskLaunch(
                spec.title,
                "workbench",
                module,
                arguments,
                output,
                "\n".join(f"{key}: {value}" for key, value in summary.items()),
            )
        )

    def _output_directory(
        self,
        workflow_id: str,
        values: dict[str, object],
    ) -> Path:
        directory = str(values.get("output_dir") or "").strip()
        if directory:
            return Path(directory).expanduser().resolve(strict=False)
        output_file = str(values.get("output_file") or "").strip()
        if output_file:
            return Path(output_file).expanduser().resolve(strict=False).parent
        return self.runtime.run_output_dir(
            "preview",
            f"run-{uuid.uuid4().hex[:12]}",
            workflow_id,
        )

    def handle_artifact(self, path: str) -> None:
        self.artifacts.open_path(path)


class RadioWorkspaceNativePanel(NativeModulePanel):
    """Native module/preset coordinator that keeps work inside App 1.0."""

    navigate_requested = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "radio-workspace",
            legacy_label="legacy Radio Workspace",
            parent=parent,
        )
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.workspace_name = QLineEdit("Radio analysis")
        self.preset = QComboBox()
        self.preset.addItem("All radio modules", "all")
        self.preset.addItem("Imaging and review", "imaging")
        self.preset.addItem("Spectra and trajectory", "spectra")
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 4)
        self.concurrency.setValue(1)
        self.allowed_root = QLineEdit()
        self.configure_path_field(self.allowed_root)
        form.addRow("Workspace", self.workspace_name)
        form.addRow("Preset", self.preset)
        form.addRow("Concurrent workers", self.concurrency)
        form.addRow("Allowed root", self.allowed_root)
        root.addLayout(form)

        self.modules = QListWidget()
        for descriptor in MODULES:
            if descriptor.category != "Radio" or descriptor.module_id == "radio-workspace":
                continue
            item = QListWidgetItem(descriptor.title)
            item.setData(Qt.ItemDataRole.UserRole, descriptor.module_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.modules.addItem(item)
        self.modules.itemDoubleClicked.connect(self._navigate_item)
        root.addWidget(self.modules, 1)
        actions = QHBoxLayout()
        apply_preset = QPushButton("Apply preset")
        apply_preset.clicked.connect(self._apply_preset)
        open_selected = QPushButton("Open selected module")
        open_selected.setProperty("primary", True)
        open_selected.clicked.connect(self._open_selected)
        figure_studio = QPushButton("Open Figure Studio")
        figure_studio.clicked.connect(
            lambda: self.navigate_requested.emit("image-composer")
        )
        actions.addWidget(apply_preset)
        actions.addWidget(open_selected)
        actions.addWidget(figure_studio)
        actions.addStretch(1)
        root.addLayout(actions)
        self.status = QLabel(
            "Double-click a module to configure it. Tasks remain FIFO by default."
        )
        self.status.setWordWrap(True)
        self.status.setProperty("muted", True)
        root.addWidget(self.status)

    def _apply_preset(self) -> None:
        selected = str(self.preset.currentData())
        imaging = {"bad-frame-review", "source-map", "roi-lightcurve", "radio-composite"}
        spectra = {"dart-spectrogram", "source-trajectory"}
        enabled = imaging if selected == "imaging" else spectra if selected == "spectra" else None
        for row in range(self.modules.count()):
            item = self.modules.item(row)
            module_id = str(item.data(Qt.ItemDataRole.UserRole))
            item.setCheckState(
                Qt.CheckState.Checked
                if enabled is None or module_id in enabled
                else Qt.CheckState.Unchecked
            )
        self.status.setText(f"Applied preset: {self.preset.currentText()}")

    def _open_selected(self) -> None:
        item = self.modules.currentItem()
        if item is not None:
            self._navigate_item(item)

    def _navigate_item(self, item: QListWidgetItem) -> None:
        self.navigate_requested.emit(str(item.data(Qt.ItemDataRole.UserRole)))


__all__ = ["RadioWorkspaceNativePanel", "WorkbenchNativePanel"]
