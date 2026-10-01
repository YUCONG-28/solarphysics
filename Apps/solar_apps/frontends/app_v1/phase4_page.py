# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 Image Composer built on the retained schema-1 model."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.models import ComposerProject, FolderSource

from .components import NativeModulePanel
from .phase4 import Phase4ComposerAdapter
from .phase4_canvas import (
    ComposerCanvas,
    ComposerCanvasControls,
    FolderList,
    SlotGraphicsItem,
    ThumbnailList as ThumbnailList,
)
from .phase4_export import ComposerExportControls
from .phase4_project import ComposerProjectControls


class Phase4ComposerPanel(
    ComposerCanvasControls,
    ComposerProjectControls,
    ComposerExportControls,
    NativeModulePanel,
):
    """PyQt6 canvas that reuses the legacy model, matching, and renderer."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase4ComposerAdapter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "image-composer",
            legacy_label="legacy Image Composer",
            parent=parent,
        )
        self.adapter = adapter
        self.project = ComposerProject()
        self.project_path: Path | None = None
        self._dirty = False
        self._items: dict[str, SlotGraphicsItem] = {}
        self._updating_controls = False
        root = QVBoxLayout(self)
        note_row = QHBoxLayout()
        note = QLabel(
            "Drag folders onto the canvas. Layout geometry is native PyQt6; "
            "schema-1 persistence, matching, and rendering reuse the existing composer."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        note_row.addWidget(note, 1)
        root.addLayout(note_row)
        splitter = QSplitter()
        splitter.addWidget(self._source_panel())
        splitter.addWidget(self._canvas_panel())
        splitter.addWidget(self._control_panel())
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _source_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Image folders"))
        self.folder_list = FolderList()
        self.folder_list.currentItemChanged.connect(self._folder_selection_changed)
        self.folder_list.itemDoubleClicked.connect(
            lambda item: self.add_slot(
                str(item.data(Qt.ItemDataRole.UserRole)),
                QPointF(40.0, 40.0),
            )
        )
        layout.addWidget(self.folder_list, 1)
        self.thumbnail_list = ThumbnailList()
        self.thumbnail_list.itemDoubleClicked.connect(
            lambda item: self.add_slot(
                str(item.data(Qt.ItemDataRole.UserRole)[0]),
                QPointF(40.0, 40.0),
                ordinal=int(item.data(Qt.ItemDataRole.UserRole)[1]),
            )
        )
        layout.addWidget(QLabel("Images (drag an exact frame)"))
        layout.addWidget(self.thumbnail_list, 2)
        folder_form = QFormLayout()
        self.folder_start = QSpinBox()
        self.folder_start.setRange(1, 1)
        self.folder_end = QSpinBox()
        self.folder_end.setRange(1, 1)
        self.folder_offset = QDoubleSpinBox()
        self.folder_offset.setRange(-86400.0, 86400.0)
        self.folder_offset.setDecimals(3)
        folder_form.addRow("First image", self.folder_start)
        folder_form.addRow("Last image", self.folder_end)
        folder_form.addRow("Clock offset (s)", self.folder_offset)
        layout.addLayout(folder_form)
        self.folder_start.valueChanged.connect(self._folder_settings_changed)
        self.folder_end.valueChanged.connect(self._folder_settings_changed)
        self.folder_offset.valueChanged.connect(self._folder_settings_changed)
        remove = QPushButton("Remove folder")
        remove.clicked.connect(self.remove_current_folder)
        layout.addLayout(
            self._button_row(
                "Add folder...",
                "Relink...",
                self.choose_folder,
                self.relink_current_folder,
            )
        )
        layout.addWidget(remove)
        layout.addLayout(
            self._button_row("New", "Open...", self.new_project, self.choose_project)
        )
        layout.addLayout(
            self._button_row(
                "Save",
                "Save as...",
                self.save_current_project,
                self.choose_save_project,
            )
        )
        return panel

    def _control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._append_canvas_layer_controls(layout)
        match_form = QFormLayout()
        self.match_master = QComboBox()
        self.match_mode = QComboBox()
        self.match_mode.addItems(["time", "relative"])
        self.match_tolerance = QDoubleSpinBox()
        self.match_tolerance.setRange(0.0, 86400.0)
        self.match_tolerance.setDecimals(3)
        self.match_tolerance.setValue(self.project.matching.tolerance_seconds)
        self.match_strict = QCheckBox("Require every source within tolerance")
        self.match_strict.setChecked(self.project.matching.strict)
        match_form.addRow("Master folder", self.match_master)
        match_form.addRow("Match mode", self.match_mode)
        match_form.addRow("Tolerance (s)", self.match_tolerance)
        match_form.addRow("", self.match_strict)
        layout.addLayout(match_form)
        self.match_master.currentIndexChanged.connect(self._matching_changed)
        self.match_mode.currentTextChanged.connect(self._matching_changed)
        self.match_tolerance.valueChanged.connect(self._matching_changed)
        self.match_strict.toggled.connect(self._matching_changed)
        self._append_export_controls(layout)
        self.sync_status = QLabel("UTC sync: waiting for current time")
        self.sync_status.setWordWrap(True)
        self.sync_status.setProperty("muted", True)
        layout.addWidget(self.sync_status)
        layout.addStretch(1)
        return panel

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float = 0.0,
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(3)
        control.setValue(value)
        return control

    @staticmethod
    def _button_row(
        first_label: str,
        second_label: str,
        first_callback,
        second_callback,
    ) -> QHBoxLayout:  # type: ignore[no-untyped-def]
        layout = QHBoxLayout()
        first = QPushButton(first_label)
        second = QPushButton(second_label)
        first.clicked.connect(first_callback)
        second.clicked.connect(second_callback)
        layout.addWidget(first)
        layout.addWidget(second)
        return layout

    def choose_folder(self) -> None:
        initial = (
            str(self.adapter.allowed_roots[0]) if self.adapter.allowed_roots else ""
        )
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select image folder",
            initial,
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.add_folder(selected)

    def add_folder(self, path: str | Path) -> FolderSource | None:
        try:
            resolved = Path(path).expanduser().resolve(strict=False)
            if not self.adapter._inside(resolved):
                raise PermissionError(
                    f"Image folder is outside configured allowed roots: {resolved}"
                )
            records = scan_folder(resolved)
            if not records:
                raise ValueError("The selected folder contains no supported images")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Image Composer input error", str(exc))
            return None
        folder = FolderSource.create(resolved, records)
        self.project.folders.append(folder)
        if not self.project.matching.master_folder_id:
            self.project.matching.master_folder_id = folder.id
        item = QListWidgetItem(f"{folder.name} ({len(records)})")
        item.setData(Qt.ItemDataRole.UserRole, folder.id)
        self.folder_list.addItem(item)
        self.folder_list.setCurrentItem(item)
        self._refresh_match_folders(select_id=folder.id)
        self._mark_dirty()
        return folder

    def _current_folder(self) -> FolderSource | None:
        item = self.folder_list.currentItem()
        if item is None:
            return None
        return self.project.folder_map().get(str(item.data(Qt.ItemDataRole.UserRole)))

    def _folder_selection_changed(self, *_args) -> None:
        folder = self._current_folder()
        self.thumbnail_list.clear()
        self._updating_controls = True
        try:
            if folder is None:
                self.folder_start.setRange(1, 1)
                self.folder_end.setRange(1, 1)
                self.folder_start.setValue(1)
                self.folder_end.setValue(1)
                self.folder_offset.setValue(0.0)
                return
            maximum = max(1, len(folder.records))
            self.folder_start.setRange(1, maximum)
            self.folder_end.setRange(1, maximum)
            self.folder_start.setValue(min(maximum, max(1, folder.start_index)))
            self.folder_end.setValue(min(maximum, max(1, folder.end_index)))
            self.folder_offset.setValue(folder.offset_seconds)
            for record in folder.records:
                item = QListWidgetItem(f"{record.ordinal}: {record.path.name}")
                item.setData(
                    Qt.ItemDataRole.UserRole,
                    (folder.id, record.ordinal),
                )
                self.thumbnail_list.addItem(item)
        finally:
            self._updating_controls = False

    def _folder_settings_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        folder = self._current_folder()
        if folder is None:
            return
        start = self.folder_start.value()
        end = self.folder_end.value()
        if end < start:
            end = start
            self._updating_controls = True
            self.folder_end.setValue(end)
            self._updating_controls = False
        folder.start_index = start
        folder.end_index = end
        folder.offset_seconds = self.folder_offset.value()
        self._mark_dirty()

    def relink_current_folder(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            f"Relink {folder.name}",
            str(folder.path),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        try:
            resolved = Path(selected).expanduser().resolve(strict=False)
            if not self.adapter._inside(resolved):
                raise PermissionError(
                    f"Image folder is outside configured allowed roots: {resolved}"
                )
            records = scan_folder(resolved)
            if not records:
                raise ValueError("The selected folder contains no supported images")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not relink folder", str(exc))
            return
        folder.path = resolved
        folder.name = resolved.name or str(resolved)
        folder.records = records
        folder.resolved = True
        folder.start_index = min(max(1, folder.start_index), len(records))
        folder.end_index = min(max(folder.start_index, folder.end_index), len(records))
        self._reload_project_ui(select_folder_id=folder.id)
        self._mark_dirty()

    def remove_current_folder(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        if (
            QMessageBox.question(
                self,
                "Remove image folder",
                f"Remove {folder.name} and its layers?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        removed_slot_ids = {
            slot.id for slot in self.project.slots if slot.folder_id == folder.id
        }
        self.project.folders = [
            item for item in self.project.folders if item.id != folder.id
        ]
        self.project.slots = [
            slot for slot in self.project.slots if slot.id not in removed_slot_ids
        ]
        if self.project.matching.master_folder_id == folder.id:
            self.project.matching.master_folder_id = (
                self.project.folders[0].id if self.project.folders else ""
            )
        self._reload_project_ui()
        self._mark_dirty()

    def _refresh_match_folders(self, *, select_id: str = "") -> None:
        current = select_id or self.project.matching.master_folder_id
        self.match_master.blockSignals(True)
        self.match_master.clear()
        for folder in self.project.folders:
            self.match_master.addItem(folder.name, folder.id)
        index = self.match_master.findData(current)
        self.match_master.setCurrentIndex(max(0, index))
        self.match_master.blockSignals(False)
        if self.match_master.count() and not self.project.matching.master_folder_id:
            self.project.matching.master_folder_id = str(
                self.match_master.currentData()
            )

    def _matching_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        self.project.matching.master_folder_id = str(
            self.match_master.currentData() or ""
        )
        self.project.matching.mode = self.match_mode.currentText()
        self.project.matching.tolerance_seconds = self.match_tolerance.value()
        self.project.matching.strict = self.match_strict.isChecked()
        self._mark_dirty()


__all__ = ["ComposerCanvas", "FolderList", "Phase4ComposerPanel", "SlotGraphicsItem"]
