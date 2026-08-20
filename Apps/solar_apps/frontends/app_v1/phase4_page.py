# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 Image Composer built on the retained schema-1 model."""

from __future__ import annotations

import math
from datetime import timezone
from pathlib import Path

from PyQt6.QtCore import (
    QByteArray,
    QDataStream,
    QIODevice,
    QPointF,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QDrag, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.matching import nearest_record
from solar_apps.frontends.image_composer.models import (
    ComposerProject,
    FolderSource,
    LayoutSlot,
)
from solar_apps.frontends.image_composer.project import load_project, save_project

from .phase4 import Phase4ComposerAdapter
from .components import NativeModulePanel, load_preview_pixmap

_FOLDER_MIME = "application/x-app-v1-composer-folder"
_THUMBNAIL_MIME = "application/x-app-v1-composer-thumbnail"


class FolderList(QListWidget):
    """Drag registered folder IDs onto the canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def startDrag(self, _actions) -> None:  # type: ignore[no-untyped-def]
        item = self.currentItem()
        if item is None:
            return
        payload = QByteArray()
        stream = QDataStream(payload, QIODevice.OpenModeFlag.WriteOnly)
        stream.writeQString(str(item.data(Qt.ItemDataRole.UserRole)))
        mime = self.mimeData([item])
        mime.setData(_FOLDER_MIME, payload)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class ThumbnailList(QListWidget):
    """Drag an exact folder record ordinal onto the canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def startDrag(self, _actions) -> None:  # type: ignore[no-untyped-def]
        item = self.currentItem()
        if item is None:
            return
        folder_id, ordinal = item.data(Qt.ItemDataRole.UserRole)
        payload = QByteArray()
        stream = QDataStream(payload, QIODevice.OpenModeFlag.WriteOnly)
        stream.writeQString(str(folder_id))
        stream.writeInt32(int(ordinal))
        mime = self.mimeData([item])
        mime.setData(_THUMBNAIL_MIME, payload)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class ComposerCanvas(QGraphicsView):
    """Drop-enabled scene with a scientific-theme-independent background."""

    def __init__(
        self,
        panel: "Phase4ComposerPanel",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.panel = panel
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_FOLDER_MIME) or event.mimeData().hasFormat(
            _THUMBNAIL_MIME
        ):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_FOLDER_MIME) or event.mimeData().hasFormat(
            _THUMBNAIL_MIME
        ):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not (
            event.mimeData().hasFormat(_FOLDER_MIME)
            or event.mimeData().hasFormat(_THUMBNAIL_MIME)
        ):
            super().dropEvent(event)
            return
        exact = event.mimeData().hasFormat(_THUMBNAIL_MIME)
        payload = event.mimeData().data(_THUMBNAIL_MIME if exact else _FOLDER_MIME)
        stream = QDataStream(payload, QIODevice.OpenModeFlag.ReadOnly)
        folder_id = stream.readQString()
        ordinal = stream.readInt32() if exact else None
        position = self.mapToScene(event.position().toPoint())
        self.panel.add_slot(folder_id, position, ordinal=ordinal)
        event.acceptProposedAction()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class SlotGraphicsItem(QGraphicsRectItem):
    """Movable layer whose geometry is stored in one retained LayoutSlot."""

    def __init__(
        self,
        panel: "Phase4ComposerPanel",
        slot: LayoutSlot,
        pixmap: QPixmap,
    ) -> None:
        super().__init__(0.0, 0.0, slot.width, slot.height)
        self.panel = panel
        self.slot = slot
        self.image = QGraphicsPixmapItem(self)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPen(QPen(QColor("#38bdf8"), 2.0))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setPos(slot.x, slot.y)
        self.setZValue(slot.z_index)
        self.setTransformOriginPoint(slot.width / 2.0, slot.height / 2.0)
        self.setRotation(slot.rotation)
        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        width = max(1, round(self.slot.width))
        height = max(1, round(self.slot.height))
        if pixmap.isNull():
            pixmap = QPixmap(width, height)
            pixmap.fill(QColor("#1f2937"))
        if self.slot.fit == "stretch":
            rendered = pixmap.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = y = 0.0
        else:
            mode = (
                Qt.AspectRatioMode.KeepAspectRatioByExpanding
                if self.slot.fit == "cover"
                else Qt.AspectRatioMode.KeepAspectRatio
            )
            rendered = pixmap.scaled(
                width,
                height,
                mode,
                Qt.TransformationMode.SmoothTransformation,
            )
            if self.slot.fit == "cover":
                x = max(0, (rendered.width() - width) // 2)
                y = max(0, (rendered.height() - height) // 2)
                rendered = rendered.copy(x, y, width, height)
                x = y = 0.0
            else:
                x = (width - rendered.width()) / 2.0
                y = (height - rendered.height()) / 2.0
        self.image.setPixmap(rendered)
        self.image.setPos(x, y)
        self.image.setOpacity(self.slot.opacity)
        self.setRect(0.0, 0.0, self.slot.width, self.slot.height)
        self.setTransformOriginPoint(self.slot.width / 2.0, self.slot.height / 2.0)

    def itemChange(self, change, value):  # type: ignore[no-untyped-def]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            position = QPointF(value)
            if self.panel.snap_to_grid.isChecked():
                grid = max(1, self.panel.grid_size.value())
                position = QPointF(
                    round(position.x() / grid) * grid,
                    round(position.y() / grid) * grid,
                )
            return position
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            position = self.pos()
            self.slot.x = float(position.x())
            self.slot.y = float(position.y())
            self.panel.slot_geometry_changed(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if bool(value):
                self.panel.load_selected_controls(self)
        return super().itemChange(change, value)


class Phase4ComposerPanel(NativeModulePanel):
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

    def _canvas_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.scene = QGraphicsScene(self)
        self.canvas = ComposerCanvas(self)
        self.canvas.setScene(self.scene)
        layout.addWidget(self.canvas, 1)
        self._refresh_scene_rect()
        return panel

    def _control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        form = QFormLayout()
        self.canvas_width = QSpinBox()
        self.canvas_width.setRange(2, 16384)
        self.canvas_width.setValue(self.project.canvas.width)
        self.canvas_height = QSpinBox()
        self.canvas_height.setRange(2, 16384)
        self.canvas_height.setValue(self.project.canvas.height)
        self.grid_size = QSpinBox()
        self.grid_size.setRange(1, 512)
        self.grid_size.setValue(20)
        self.snap_to_grid = QCheckBox("Snap")
        self.snap_to_grid.setChecked(True)
        self.lock_aspect = QCheckBox("Lock aspect ratio")
        self.lock_aspect.setChecked(True)
        self.canvas_background = QPushButton(self.project.canvas.background)
        self.canvas_background.clicked.connect(self.choose_background)
        form.addRow("Canvas width", self.canvas_width)
        form.addRow("Canvas height", self.canvas_height)
        form.addRow("Background", self.canvas_background)
        form.addRow("Grid size", self.grid_size)
        form.addRow("", self.snap_to_grid)
        form.addRow("", self.lock_aspect)
        self.slot_x = self._double_spin(-100000.0, 100000.0)
        self.slot_y = self._double_spin(-100000.0, 100000.0)
        self.slot_width = self._double_spin(1.0, 100000.0, 420.0)
        self.slot_height = self._double_spin(1.0, 100000.0, 280.0)
        self.slot_rotation = self._double_spin(-360.0, 360.0)
        self.slot_opacity = self._double_spin(0.0, 1.0, 1.0)
        self.slot_opacity.setSingleStep(0.05)
        self.slot_fit = QComboBox()
        self.slot_fit.addItems(["contain", "cover", "stretch"])
        for label, control in (
            ("X", self.slot_x),
            ("Y", self.slot_y),
            ("Width", self.slot_width),
            ("Height", self.slot_height),
            ("Rotation", self.slot_rotation),
            ("Opacity", self.slot_opacity),
            ("Fit", self.slot_fit),
        ):
            form.addRow(label, control)
        layout.addLayout(form)
        for control in (
            self.slot_x,
            self.slot_y,
            self.slot_width,
            self.slot_height,
            self.slot_rotation,
            self.slot_opacity,
        ):
            control.valueChanged.connect(self.apply_selected_controls)
        self.slot_fit.currentTextChanged.connect(self.apply_selected_controls)
        self.canvas_width.valueChanged.connect(self._canvas_changed)
        self.canvas_height.valueChanged.connect(self._canvas_changed)
        layout.addLayout(
            self._button_row("Align left", "Align top", self.align_left, self.align_top)
        )
        layout.addLayout(
            self._button_row(
                "Center H",
                "Center V",
                self.center_horizontal,
                self.center_vertical,
            )
        )
        layout.addLayout(
            self._button_row(
                "Equal size",
                "Auto grid",
                self.equal_size,
                self.auto_grid,
            )
        )
        layout.addLayout(
            self._button_row(
                "Bring front",
                "Send back",
                lambda: self.change_layer(1),
                lambda: self.change_layer(-1),
            )
        )
        layout.addLayout(
            self._button_row(
                "Duplicate layer",
                "Delete layer",
                self.duplicate_selected_slot,
                self.delete_selected_slot,
            )
        )
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
        export_form = QFormLayout()
        self.output_path = QLineEdit()
        self.output_format = QComboBox()
        self.output_format.addItems(["mp4", "avi"])
        choose_output = QPushButton("Choose output...")
        choose_output.clicked.connect(self.choose_output_path)
        self.export_scale = QSpinBox()
        self.export_scale.setRange(1, 8)
        self.export_scale.setValue(2)
        self.export_fps = QDoubleSpinBox()
        self.export_fps.setRange(0.1, 60.0)
        self.export_fps.setValue(5.0)
        self.export_frames = QCheckBox("Save PNG frames")
        export_form.addRow("Sequence output", self.output_path)
        export_form.addRow("Format", self.output_format)
        export_form.addRow("", choose_output)
        export_form.addRow("Export scale", self.export_scale)
        export_form.addRow("Sequence FPS", self.export_fps)
        export_form.addRow("", self.export_frames)
        self.output_path.textChanged.connect(self._export_settings_changed)
        self.output_format.currentTextChanged.connect(self._output_format_changed)
        self.export_fps.valueChanged.connect(self._export_settings_changed)
        self.export_frames.toggled.connect(self._export_settings_changed)
        layout.addLayout(export_form)
        static = QPushButton("Confirm high-resolution PNG")
        static.clicked.connect(self.request_static_export)
        sequence = QPushButton("Confirm synchronized sequence")
        sequence.clicked.connect(self.request_sequence_export)
        layout.addWidget(static)
        layout.addWidget(sequence)
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

    def add_slot(
        self,
        folder_id: str,
        position: QPointF,
        *,
        ordinal: int | None = None,
    ) -> LayoutSlot | None:
        folder = self.project.folder_map().get(folder_id)
        if folder is None or not folder.records:
            return None
        record = (
            folder.record_by_ordinal(int(ordinal)) if ordinal is not None else None
        ) or folder.records[0]
        slot = LayoutSlot.create(
            folder.id,
            record.ordinal,
            x=max(0.0, position.x()),
            y=max(0.0, position.y()),
            width=min(420.0, self.project.canvas.width),
            height=min(280.0, self.project.canvas.height),
            z_index=len(self.project.slots),
        )
        slot.preview_relative_path = record.path.name
        self.project.slots.append(slot)
        self._add_graphics_item(slot)
        self._mark_dirty()
        return slot

    def _add_graphics_item(self, slot: LayoutSlot) -> None:
        folder = self.project.folder_map().get(slot.folder_id)
        record = folder.record_by_ordinal(slot.preview_ordinal) if folder else None
        pixmap = self._slot_pixmap(slot, record.path if record is not None else None)
        item = SlotGraphicsItem(self, slot, pixmap)
        self.scene.addItem(item)
        self._items[slot.id] = item

    def _selected_items(self) -> list[SlotGraphicsItem]:
        return [
            item
            for item in self.scene.selectedItems()
            if isinstance(item, SlotGraphicsItem)
        ]

    def load_selected_controls(self, item: SlotGraphicsItem) -> None:
        self._updating_controls = True
        try:
            for control, value in (
                (self.slot_x, item.slot.x),
                (self.slot_y, item.slot.y),
                (self.slot_width, item.slot.width),
                (self.slot_height, item.slot.height),
                (self.slot_rotation, item.slot.rotation),
                (self.slot_opacity, item.slot.opacity),
            ):
                control.setValue(value)
            self.slot_fit.setCurrentText(item.slot.fit)
        finally:
            self._updating_controls = False

    def apply_selected_controls(self, *_args) -> None:
        if self._updating_controls:
            return
        selected = self._selected_items()
        if not selected:
            return
        item = selected[0]
        old_ratio = item.slot.width / max(item.slot.height, 1e-9)
        item.slot.x = self.slot_x.value()
        item.slot.y = self.slot_y.value()
        width = self.slot_width.value()
        height = self.slot_height.value()
        if self.lock_aspect.isChecked() and abs(width - item.slot.width) > 1e-6:
            height = width / old_ratio
            self._updating_controls = True
            self.slot_height.setValue(height)
            self._updating_controls = False
        item.slot.width = width
        item.slot.height = height
        item.slot.rotation = self.slot_rotation.value()
        item.slot.opacity = self.slot_opacity.value()
        item.slot.fit = self.slot_fit.currentText()
        item.setPos(item.slot.x, item.slot.y)
        item.setRotation(item.slot.rotation)
        folder = self.project.folder_map()[item.slot.folder_id]
        record = folder.record_by_ordinal(item.slot.preview_ordinal)
        item.set_pixmap(
            self._slot_pixmap(item.slot, record.path if record is not None else None)
        )
        self._mark_dirty()

    def slot_geometry_changed(self, item: SlotGraphicsItem) -> None:
        if item.isSelected():
            self.load_selected_controls(item)
        self._mark_dirty()

    def _canvas_changed(self) -> None:
        width = self.canvas_width.value()
        height = self.canvas_height.value()
        if width % 2:
            width += 1
            self.canvas_width.setValue(width)
        if height % 2:
            height += 1
            self.canvas_height.setValue(height)
        self.project.canvas.width = width
        self.project.canvas.height = height
        self._refresh_scene_rect()
        self._mark_dirty()

    def choose_background(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self.project.canvas.background),
            self,
            "Choose canvas background",
        )
        if not selected.isValid():
            return
        self.project.canvas.background = selected.name()
        self.canvas_background.setText(self.project.canvas.background)
        self._refresh_scene_rect()
        self._mark_dirty()

    def _refresh_scene_rect(self) -> None:
        self.scene.setSceneRect(
            0.0,
            0.0,
            float(self.project.canvas.width),
            float(self.project.canvas.height),
        )
        self.scene.setBackgroundBrush(QColor(self.project.canvas.background))
        self.canvas.fitInView(
            self.scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def align_left(self) -> None:
        items = self._selected_items()
        if items:
            target = min(item.pos().x() for item in items)
            for item in items:
                item.setX(target)
            self._mark_dirty()

    def align_top(self) -> None:
        items = self._selected_items()
        if items:
            target = min(item.pos().y() for item in items)
            for item in items:
                item.setY(target)
            self._mark_dirty()

    def center_horizontal(self) -> None:
        for item in self._selected_items():
            item.setX((self.project.canvas.width - item.slot.width) / 2.0)
        if self._selected_items():
            self._mark_dirty()

    def center_vertical(self) -> None:
        for item in self._selected_items():
            item.setY((self.project.canvas.height - item.slot.height) / 2.0)
        if self._selected_items():
            self._mark_dirty()

    def equal_size(self) -> None:
        items = self._selected_items()
        if len(items) < 2:
            return
        width, height = items[0].slot.width, items[0].slot.height
        for item in items[1:]:
            item.slot.width = width
            item.slot.height = height
            folder = self.project.folder_map()[item.slot.folder_id]
            record = folder.record_by_ordinal(item.slot.preview_ordinal)
            item.set_pixmap(
                self._slot_pixmap(
                    item.slot, record.path if record is not None else None
                )
            )
        self._mark_dirty()

    def auto_grid(self) -> None:
        items = list(self._items.values())
        if not items:
            return
        columns = math.ceil(math.sqrt(len(items)))
        rows = math.ceil(len(items) / columns)
        cell_width = self.project.canvas.width / columns
        cell_height = self.project.canvas.height / rows
        for index, item in enumerate(
            sorted(items, key=lambda value: value.slot.z_index)
        ):
            column = index % columns
            row = index // columns
            item.slot.width = cell_width
            item.slot.height = cell_height
            item.setPos(column * cell_width, row * cell_height)
            folder = self.project.folder_map()[item.slot.folder_id]
            record = folder.record_by_ordinal(item.slot.preview_ordinal)
            item.set_pixmap(
                self._slot_pixmap(
                    item.slot, record.path if record is not None else None
                )
            )
        self._mark_dirty()

    def change_layer(self, direction: int) -> None:
        items = self._selected_items()
        if not items:
            return
        item = items[0]
        ordered = sorted(self.project.slots, key=lambda slot: slot.z_index)
        current = ordered.index(item.slot)
        target = max(0, min(len(ordered) - 1, current + int(direction)))
        ordered[current], ordered[target] = ordered[target], ordered[current]
        for index, slot in enumerate(ordered):
            slot.z_index = index
            self._items[slot.id].setZValue(index)
        self._mark_dirty()

    def duplicate_selected_slot(self) -> None:
        selected = self._selected_items()
        if not selected:
            return
        original = selected[0].slot
        duplicate = LayoutSlot.create(
            original.folder_id,
            original.preview_ordinal,
            x=original.x + 20.0,
            y=original.y + 20.0,
            width=original.width,
            height=original.height,
            z_index=len(self.project.slots),
        )
        duplicate.preview_relative_path = original.preview_relative_path
        duplicate.rotation = original.rotation
        duplicate.opacity = original.opacity
        duplicate.fit = original.fit
        self.project.slots.append(duplicate)
        self._add_graphics_item(duplicate)
        self.scene.clearSelection()
        self._items[duplicate.id].setSelected(True)
        self._mark_dirty()

    def delete_selected_slot(self) -> None:
        selected = self._selected_items()
        if not selected:
            return
        for item in selected:
            self.scene.removeItem(item)
            self._items.pop(item.slot.id, None)
            self.project.slots = [
                slot for slot in self.project.slots if slot.id != item.slot.id
            ]
        self.project.normalize_z_indexes()
        for slot in self.project.slots:
            self._items[slot.id].setZValue(slot.z_index)
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

    def choose_project(self) -> None:
        if not self.confirm_discard_changes():
            return
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Image Composer project",
            str(self.adapter.runtime.workspaces_dir),
            "Image Composer project (*.fic.json);;JSON (*.json)",
        )
        if selected:
            self.import_project(selected)

    def import_project(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            if not self.adapter._inside(candidate):
                raise PermissionError(
                    f"Project is outside configured allowed roots: {candidate}"
                )
            project = load_project(candidate)
            for folder in project.folders:
                resolved = folder.path.expanduser().resolve(strict=False)
                folder.path = resolved
                folder.records = (
                    scan_folder(resolved)
                    if self.adapter._inside(resolved) and resolved.is_dir()
                    else []
                )
                folder.resolved = bool(folder.records)
                if folder.records:
                    folder.start_index = min(
                        max(1, folder.start_index), len(folder.records)
                    )
                    folder.end_index = min(
                        max(folder.start_index, folder.end_index),
                        len(folder.records),
                    )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not import project", str(exc))
            return False
        self.project = project
        self._reload_project_ui()
        self.project_path = candidate
        self._dirty = False
        return True

    def _reload_project_ui(self, *, select_folder_id: str = "") -> None:
        self._updating_controls = True
        self.folder_list.clear()
        self.thumbnail_list.clear()
        self.scene.clear()
        self._items.clear()
        selected_item: QListWidgetItem | None = None
        try:
            self.canvas_width.setValue(self.project.canvas.width)
            self.canvas_height.setValue(self.project.canvas.height)
            self.canvas_background.setText(self.project.canvas.background)
            for folder in self.project.folders:
                state = "" if folder.resolved else " — unresolved"
                item = QListWidgetItem(f"{folder.name} ({len(folder.records)}){state}")
                item.setData(Qt.ItemDataRole.UserRole, folder.id)
                self.folder_list.addItem(item)
                if folder.id == select_folder_id:
                    selected_item = item
            for slot in sorted(self.project.slots, key=lambda item: item.z_index):
                self._add_graphics_item(slot)
            self._refresh_match_folders()
            self.match_mode.setCurrentText(self.project.matching.mode)
            self.match_tolerance.setValue(self.project.matching.tolerance_seconds)
            self.match_strict.setChecked(self.project.matching.strict)
            self.output_path.setText(self.project.export.output_path)
            self.output_format.setCurrentText(self.project.export.output_format)
            self.export_fps.setValue(self.project.export.fps)
            self.export_frames.setChecked(self.project.export.save_png_frames)
            self._refresh_scene_rect()
        finally:
            self._updating_controls = False
        if selected_item is None and self.folder_list.count():
            selected_item = self.folder_list.item(0)
        if selected_item is not None:
            self.folder_list.setCurrentItem(selected_item)

    def choose_save_project(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Image Composer project",
            str(self.adapter.runtime.workspaces_dir / "composition.fic.json"),
            "Image Composer project (*.fic.json)",
        )
        if not selected:
            return
        self._save_project_to(selected)

    def save_current_project(self) -> bool:
        if self.project_path is None:
            self.choose_save_project()
            return not self._dirty
        return self._save_project_to(self.project_path)

    def _save_project_to(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            if not self.adapter._inside(candidate):
                raise PermissionError(
                    f"Project is outside configured allowed roots: {candidate}"
                )
            saved = save_project(candidate, self.project)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return False
        self.project_path = saved
        self._dirty = False
        QMessageBox.information(self, "Project saved", str(saved))
        return True

    def new_project(self) -> bool:
        if not self.confirm_discard_changes():
            return False
        self.project = ComposerProject()
        self.project_path = None
        self._dirty = False
        self._reload_project_ui()
        return True

    def _mark_dirty(self) -> None:
        if not self._updating_controls:
            self._dirty = True

    def confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        result = QMessageBox.warning(
            self,
            "Unsaved Image Composer project",
            "Save changes to the current .fic.json project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if result == QMessageBox.StandardButton.Cancel:
            return False
        if result == QMessageBox.StandardButton.Save:
            return self.save_current_project()
        return True

    def set_current_time(self, current_time_utc) -> None:  # type: ignore[no-untyped-def]
        current = current_time_utc.astimezone(timezone.utc).replace(tzinfo=None)
        updated = 0
        for slot in self.project.slots:
            folder = self.project.folder_map().get(slot.folder_id)
            if folder is None or not folder.selected_records():
                continue
            record = nearest_record(folder, folder.selected_records(), current)
            slot.preview_ordinal = record.ordinal
            slot.preview_relative_path = record.path.name
            self._items[slot.id].set_pixmap(self._slot_pixmap(slot, record.path))
            updated += 1
        rendered = current_time_utc.isoformat().replace("+00:00", "Z")
        self.sync_status.setText(f"UTC sync: {rendered}; updated {updated} layer(s)")

    @staticmethod
    def _slot_pixmap(slot: LayoutSlot, path: Path | None) -> QPixmap:
        if path is None:
            return QPixmap()
        return load_preview_pixmap(
            path,
            QSize(max(1, round(slot.width)), max(1, round(slot.height))),
        )

    def request_static_export(self) -> None:
        default = self.adapter.runtime.outputs_dir / "composition.png"
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export high-resolution PNG",
            str(default),
            "PNG image (*.png)",
        )
        if not selected or not self._confirm_overwrite(Path(selected)):
            return
        self._request(
            lambda: self.adapter.build_static_export(
                self.project,
                scale=self.export_scale.value(),
                output_path=selected,
            )
        )

    def request_sequence_export(self) -> None:
        output = self.output_path.text().strip() or None
        if output is not None and not self._confirm_overwrite(Path(output)):
            return
        self._request(
            lambda: self.adapter.build_sequence_export(
                self.project,
                scale=self.export_scale.value(),
                fps=self.export_fps.value(),
                save_png_frames=self.export_frames.isChecked(),
                output_path=output,
            )
        )

    def choose_output_path(self) -> None:
        suffix = self.output_format.currentText()
        current = self.output_path.text().strip()
        initial = current or str(
            self.adapter.runtime.outputs_dir / f"composition.{suffix}"
        )
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Choose sequence output",
            initial,
            f"{suffix.upper()} video (*.{suffix})",
        )
        if selected:
            self.output_path.setText(selected)

    def _output_format_changed(self, value: str) -> None:
        current = self.output_path.text().strip()
        if current:
            self.output_path.setText(str(Path(current).with_suffix(f".{value}")))
        self._export_settings_changed()

    def _export_settings_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        self.project.export.output_path = self.output_path.text().strip()
        self.project.export.output_format = self.output_format.currentText()
        self.project.export.fps = self.export_fps.value()
        self.project.export.save_png_frames = self.export_frames.isChecked()
        self._mark_dirty()

    def _confirm_overwrite(self, path: Path) -> bool:
        candidate = path.expanduser().resolve(strict=False)
        related = (
            candidate,
            candidate.with_name(f"{candidate.stem}_matches.csv"),
            candidate.with_name(f"{candidate.stem}_frames"),
        )
        existing = [item for item in related if item.exists()]
        if not existing:
            return True
        names = "\n".join(str(item) for item in existing)
        return (
            QMessageBox.warning(
                self,
                "Replace existing output?",
                f"The following output paths already exist:\n{names}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        )

    def _request(self, builder) -> None:  # type: ignore[no-untyped-def]
        try:
            launch = builder()
        except (OSError, ValueError) as exc:
            self.record_diagnostic(exc)
            QMessageBox.critical(self, "Image Composer export error", str(exc))
            return
        if self.confirm(
            self,
            "Confirm Image Composer export",
            launch.summary,
        ):
            self.task_requested.emit(launch)


__all__ = [
    "ComposerCanvas",
    "FolderList",
    "Phase4ComposerPanel",
    "SlotGraphicsItem",
]
