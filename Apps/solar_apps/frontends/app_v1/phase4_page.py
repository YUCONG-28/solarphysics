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
        if event.mimeData().hasFormat(_FOLDER_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_FOLDER_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not event.mimeData().hasFormat(_FOLDER_MIME):
            super().dropEvent(event)
            return
        payload = event.mimeData().data(_FOLDER_MIME)
        stream = QDataStream(payload, QIODevice.OpenModeFlag.ReadOnly)
        folder_id = stream.readQString()
        position = self.mapToScene(event.position().toPoint())
        self.panel.add_slot(folder_id, position)
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
        self.folder_list.itemDoubleClicked.connect(
            lambda item: self.add_slot(
                str(item.data(Qt.ItemDataRole.UserRole)),
                QPointF(40.0, 40.0),
            )
        )
        layout.addWidget(self.folder_list, 1)
        add = QPushButton("Add folder...")
        add.clicked.connect(self.choose_folder)
        open_project = QPushButton("Import .fic.json...")
        open_project.clicked.connect(self.choose_project)
        save = QPushButton("Save .fic.json...")
        save.clicked.connect(self.choose_save_project)
        layout.addWidget(add)
        layout.addWidget(open_project)
        layout.addWidget(save)
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
        form.addRow("Canvas width", self.canvas_width)
        form.addRow("Canvas height", self.canvas_height)
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
        export_form = QFormLayout()
        self.export_scale = QSpinBox()
        self.export_scale.setRange(1, 8)
        self.export_scale.setValue(2)
        self.export_fps = QDoubleSpinBox()
        self.export_fps.setRange(0.1, 60.0)
        self.export_fps.setValue(5.0)
        self.export_frames = QCheckBox("Save PNG frames")
        export_form.addRow("Export scale", self.export_scale)
        export_form.addRow("Sequence FPS", self.export_fps)
        export_form.addRow("", self.export_frames)
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
        return folder

    def add_slot(self, folder_id: str, position: QPointF) -> LayoutSlot | None:
        folder = self.project.folder_map().get(folder_id)
        if folder is None or not folder.records:
            return None
        slot = LayoutSlot.create(
            folder.id,
            folder.records[0].ordinal,
            x=max(0.0, position.x()),
            y=max(0.0, position.y()),
            width=min(420.0, self.project.canvas.width),
            height=min(280.0, self.project.canvas.height),
            z_index=len(self.project.slots),
        )
        slot.preview_relative_path = folder.records[0].path.name
        self.project.slots.append(slot)
        self._add_graphics_item(slot)
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

    def slot_geometry_changed(self, item: SlotGraphicsItem) -> None:
        if item.isSelected():
            self.load_selected_controls(item)

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

    def align_top(self) -> None:
        items = self._selected_items()
        if items:
            target = min(item.pos().y() for item in items)
            for item in items:
                item.setY(target)

    def center_horizontal(self) -> None:
        for item in self._selected_items():
            item.setX((self.project.canvas.width - item.slot.width) / 2.0)

    def center_vertical(self) -> None:
        for item in self._selected_items():
            item.setY((self.project.canvas.height - item.slot.height) / 2.0)

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

    def choose_project(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Image Composer project",
            str(self.adapter.runtime.workspaces_dir),
            "Image Composer project (*.fic.json);;JSON (*.json)",
        )
        if selected:
            self.import_project(selected)

    def import_project(self, path: str | Path) -> bool:
        try:
            project = load_project(path)
            for folder in project.folders:
                if not self.adapter._inside(folder.path):
                    raise PermissionError(
                        f"Project folder is outside allowed roots: {folder.path}"
                    )
                folder.records = scan_folder(folder.path)
                folder.resolved = bool(folder.records)
                folder.end_index = min(
                    max(folder.start_index, folder.end_index),
                    max(1, len(folder.records)),
                )
            self.adapter.validate_project_inputs(project)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not import project", str(exc))
            return False
        self.project = project
        self._reload_project_ui()
        return True

    def _reload_project_ui(self) -> None:
        self.folder_list.clear()
        self.scene.clear()
        self._items.clear()
        self.canvas_width.setValue(self.project.canvas.width)
        self.canvas_height.setValue(self.project.canvas.height)
        for folder in self.project.folders:
            item = QListWidgetItem(f"{folder.name} ({len(folder.records)})")
            item.setData(Qt.ItemDataRole.UserRole, folder.id)
            self.folder_list.addItem(item)
        for slot in sorted(self.project.slots, key=lambda item: item.z_index):
            self._add_graphics_item(slot)
        self._refresh_scene_rect()

    def choose_save_project(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Image Composer project",
            str(self.adapter.runtime.workspaces_dir / "composition.fic.json"),
            "Image Composer project (*.fic.json)",
        )
        if not selected:
            return
        try:
            saved = save_project(selected, self.project)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return
        QMessageBox.information(self, "Project saved", str(saved))

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
        self._request(
            lambda: self.adapter.build_static_export(
                self.project,
                scale=self.export_scale.value(),
            )
        )

    def request_sequence_export(self) -> None:
        self._request(
            lambda: self.adapter.build_sequence_export(
                self.project,
                scale=self.export_scale.value(),
                fps=self.export_fps.value(),
                save_png_frames=self.export_frames.isChecked(),
            )
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
