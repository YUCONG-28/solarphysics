# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 Image Composer with versioned chronological match plans."""

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
    QLineEdit,
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
from solar_apps.frontends.image_composer.matching import freeze_match_plan, in_window_records

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
    timeline_sources_changed = pyqtSignal()

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
        self._matching_settings_key = None
        root = QVBoxLayout(self)
        note_row = QHBoxLayout()
        note = QLabel(
            "Drag folders onto the canvas. Layout geometry is native PyQt6; "
            "A fewest-frame root and successive full-coverage maps preserve every output frame while flagging timing gaps."
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
        self.folder_offset.setDecimals(9)
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
        rescan = QPushButton("Rescan sources and rebuild matching")
        rescan.clicked.connect(self.rescan_sources)
        layout.addWidget(rescan)
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
        self.match_master.setEnabled(False)
        self.match_master.setToolTip(
            "Automatically selects the enabled source with the fewest valid frames inside the image window as the root."
        )
        self.match_output = QComboBox()
        self.match_output.setEnabled(False)
        self.match_output.setToolTip(
            "The final source in the hierarchy supplies every output frame and its real observation UTC."
        )
        self.match_mode = QComboBox()
        self.match_mode.addItems(["time", "relative"])
        self.match_tolerance = QDoubleSpinBox()
        self.match_tolerance.setRange(0.0, 86400.0)
        self.match_tolerance.setDecimals(9)
        self.match_tolerance.setValue(self.project.matching.tolerance_seconds)
        self.match_strict = QCheckBox("Keep every master frame; flag timing gaps")
        self.match_strict.setChecked(True)
        self.match_strict.setEnabled(False)
        self.match_time_start = QLineEdit()
        self.match_time_end = QLineEdit()
        self.match_time_start.setPlaceholderText(
            "Unbounded or YYYY-MM-DDTHH:MM:SS.fffffffffZ"
        )
        self.match_time_end.setPlaceholderText(
            "Unbounded or YYYY-MM-DDTHH:MM:SS.fffffffffZ"
        )
        self.match_time_start.setToolTip(
            "Keep observations at or after this UTC time, before applying clock offsets."
        )
        self.match_time_end.setToolTip(
            "Keep observations strictly before this UTC time."
        )
        match_form.addRow("Root source (fewest frames)", self.match_master)
        match_form.addRow("Output source (most frames)", self.match_output)
        match_form.addRow("Match mode", self.match_mode)
        match_form.addRow("Tolerance (s)", self.match_tolerance)
        match_form.addRow("Start UTC (inclusive)", self.match_time_start)
        match_form.addRow("End UTC (exclusive)", self.match_time_end)
        match_form.addRow("", self.match_strict)
        export_preview = QPushButton("Show Composer export preview")
        export_preview.clicked.connect(self.show_export_preview)
        match_form.addRow("", export_preview)
        layout.addLayout(match_form)
        self.match_master.currentIndexChanged.connect(self._matching_changed)
        self.match_mode.currentTextChanged.connect(self._matching_changed)
        self.match_tolerance.valueChanged.connect(self._matching_changed)
        self.match_strict.toggled.connect(self._matching_changed)
        self.match_time_start.editingFinished.connect(self._matching_changed)
        self.match_time_end.editingFinished.connect(self._matching_changed)
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
        self.project.sequence_plan = None
        self.project._match_plan_cache = None
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

    def rescan_sources(self) -> bool:
        """Explicitly replace a stale preview snapshot after rechecking inputs."""
        try:
            scanned = []
            for folder in self.project.folders:
                if not self.adapter._inside(folder.path):
                    raise PermissionError(
                        f"Image folder is outside configured allowed roots: {folder.path}"
                    )
                records = scan_folder(folder.path)
                end = (
                    len(records)
                    if folder.end_index == len(folder.records)
                    else folder.end_index
                )
                if not records or folder.start_index > end or end > len(records):
                    raise ValueError(
                        f"The selected range is unavailable in {folder.name}; adjust the range before rebuilding."
                    )
                scanned.append((folder, records, end))
            for folder, records, end in scanned:
                folder.records, folder.end_index, folder.resolved = records, end, True
            self.project.sequence_plan = None
            self.project._match_plan_cache = None
            self._reload_project_ui()
            self._matching_settings_key = None
            self._mark_dirty()
            if self.project.slots:
                plan = freeze_match_plan(self.project)
                self.sync_status.setText(
                    f"Rebuilt UTC matching: {len(plan)} output frame(s), all selected source frames covered through hierarchical parent maps."
                )
            return True
        except (OSError, ValueError) as exc:
            self.record_diagnostic(exc)
            self.sync_status.setText(str(exc))
            return False


    def _refresh_match_folders(self, *, select_id: str = "") -> None:
        current = select_id or self.project.matching.master_folder_id
        output_id = ""
        used = {slot.folder_id for slot in self.project.slots}
        enabled = [folder for folder in self.project.folders if folder.id in used]
        if enabled:
            try:
                counts = {
                    folder.id: len(in_window_records(self.project, folder))
                    for folder in enabled
                }
                ordered = sorted(
                    (folder for folder in enabled if counts[folder.id]),
                    key=lambda folder: (counts[folder.id], folder.id),
                )
                current = self.project.matching.master_folder_id = (
                    ordered[0].id if ordered else ""
                )
                output_id = ordered[-1].id if ordered else ""
            except ValueError as exc:
                self.sync_status.setText(str(exc))
        self.match_master.blockSignals(True)
        self.match_output.blockSignals(True)
        self.match_master.clear()
        self.match_output.clear()
        for folder in self.project.folders:
            self.match_master.addItem(folder.name, folder.id)
            self.match_output.addItem(folder.name, folder.id)
        index = self.match_master.findData(current)
        self.match_master.setCurrentIndex(index)
        self.match_output.setCurrentIndex(self.match_output.findData(output_id))
        self.match_master.blockSignals(False)
        self.match_output.blockSignals(False)

    def _matching_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        self.project.matching.master_folder_id = str(
            self.match_master.currentData() or ""
        )
        self.project.matching.mode = self.match_mode.currentText()
        self.project.matching.tolerance_seconds = self.match_tolerance.value()
        self.project.matching.strict = False
        self.project.matching.time_start = self.match_time_start.text().strip() or None
        self.project.matching.time_end = self.match_time_end.text().strip() or None
        self._mark_dirty()
        if self.project.slots:
            self.show_export_preview()


__all__ = ["ComposerCanvas", "FolderList", "Phase4ComposerPanel", "SlotGraphicsItem"]
