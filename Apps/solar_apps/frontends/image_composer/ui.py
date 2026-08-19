"""PySide6 desktop interface for the local free image composer."""

from __future__ import annotations

import copy
import io
import json
import threading
import time
from pathlib import Path
from uuid import uuid4

from PIL import ImageOps
from PySide6.QtCore import (
    QByteArray,
    QMimeData,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QBrush,
    QColor,
    QDrag,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)

from solar_apps.platform.paths import (
    PathMemoryContext,
    RecentPathMemory,
    configured_allowed_roots,
    validate_allowed_path,
)
from solar_apps.platform.paths.native_dialog import NativeDialogError
from solar_apps.ui.state import frontend_path_memory, frontend_state_store
from solar_apps.ui.theme import QtThemeController
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .catalog import scan_folder
from .matching import MatchPlanError, validate_project
from .models import ComposerProject, FolderSource, LayoutSlot
from .project import ProjectFormatError, load_project, save_project
from .rendering import (
    ExportCancelled,
    ExportError,
    ExportResult,
    export_project,
    load_oriented_rgba,
    render_slot_tile,
)

THUMBNAIL_MIME = "application/x-solar-image-composer-thumbnail"


class _ScanSignals(QObject):
    finished = Signal(str, object, object)
    failed = Signal(str, str)


class _ScanTask(QRunnable):
    def __init__(self, request_id: str, path: Path) -> None:
        super().__init__()
        self.request_id = request_id
        self.path = path
        self.signals = _ScanSignals()

    def run(self) -> None:
        try:
            records = scan_folder(self.path)
        except Exception as exc:  # surfaced verbatim in the desktop UI
            self.signals.failed.emit(self.request_id, str(exc))
            return
        self.signals.finished.emit(self.request_id, self.path, records)


class _ThumbnailSignals(QObject):
    finished = Signal(str, str, int, QByteArray)
    settled = Signal(str)


class _ThumbnailTask(QRunnable):
    def __init__(self, folder_id: str, ordinal: int, path: Path) -> None:
        super().__init__()
        self.task_id = uuid4().hex
        self.folder_id = folder_id
        self.ordinal = ordinal
        self.path = path
        self.signals = _ThumbnailSignals()
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()
        # The window owns the Python wrapper until the queued ``settled`` signal
        # is handled.  Disabling Qt auto-deletion keeps that ownership explicit.
        self.setAutoDelete(False)

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    def cancel(self) -> None:
        self._cancel_event.set()

    def wait(self, timeout: float) -> bool:
        return self._done_event.wait(max(0.0, timeout))

    def mark_dequeued(self) -> None:
        """Settle a cancelled task removed before its worker started."""

        _settle_thumbnail_task(self)

    def run(self) -> None:
        payload: QByteArray | None = None
        try:
            if self._cancel_event.is_set():
                return
            image = load_oriented_rgba(self.path)
            image = ImageOps.contain(image, (160, 100))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            payload = QByteArray(stream.getvalue())
        except ExportError:
            payload = QByteArray()
        finally:
            try:
                if payload is not None and not self._cancel_event.is_set():
                    self.signals.finished.emit(
                        self.task_id, self.folder_id, self.ordinal, payload
                    )
            except RuntimeError:
                # Defensive fallback if Qt is already tearing down at process exit.
                pass
            finally:
                try:
                    self.signals.settled.emit(self.task_id)
                except RuntimeError:
                    pass
                _settle_thumbnail_task(self)


_RETIRED_THUMBNAIL_TASKS: set[_ThumbnailTask] = set()
_RETIRED_THUMBNAIL_TASKS_LOCK = threading.Lock()


def _retain_running_thumbnail_task(task: _ThumbnailTask) -> None:
    """Keep a detached worker and its signal source alive until ``run`` exits."""

    with _RETIRED_THUMBNAIL_TASKS_LOCK:
        if not task.done:
            _RETIRED_THUMBNAIL_TASKS.add(task)


def _settle_thumbnail_task(task: _ThumbnailTask) -> None:
    """Publish completion only after retired-task ownership is released."""

    with _RETIRED_THUMBNAIL_TASKS_LOCK:
        _RETIRED_THUMBNAIL_TASKS.discard(task)
        task._done_event.set()


class _ExportSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)


class _ExportTask(QRunnable):
    def __init__(self, project: ComposerProject, cancel_event: threading.Event) -> None:
        super().__init__()
        self.project = project
        self.cancel_event = cancel_event
        self.signals = _ExportSignals()

    def run(self) -> None:
        try:
            result = export_project(
                self.project,
                cancelled=self.cancel_event.is_set,
                progress=self.signals.progress.emit,
            )
        except ExportCancelled as exc:
            self.signals.cancelled.emit(str(exc))
        except Exception as exc:  # the UI owns user-facing error presentation
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class ThumbnailList(QListWidget):
    """Icon list that emits a stable folder/ordinal MIME payload."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setIconSize(QSize(120, 80))
        self.setGridSize(QSize(146, 116))
        self.setWordWrap(True)
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def startDrag(self, supported_actions: Qt.DropAction) -> None:
        item = self.currentItem()
        payload = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not payload:
            return
        mime = QMimeData()
        mime.setData(THUMBNAIL_MIME, json.dumps(payload).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(self.iconSize()))
        drag.exec(Qt.DropAction.CopyAction)


class ComposerView(QGraphicsView):
    imageDropped = Signal(str, int, QPointF)

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setAcceptDrops(True)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(THUMBNAIL_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(THUMBNAIL_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasFormat(THUMBNAIL_MIME):
            super().dropEvent(event)
            return
        try:
            payload = json.loads(bytes(event.mimeData().data(THUMBNAIL_MIME)))
            folder_id = str(payload["folder_id"])
            ordinal = int(payload["ordinal"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            event.ignore()
            return
        point = self.mapToScene(event.position().toPoint())
        self.imageDropped.emit(folder_id, ordinal, point)
        event.acceptProposedAction()

    def wheelEvent(self, event) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class SlotItem(QGraphicsObject):
    changed = Signal(str)
    selected = Signal(str)

    HANDLE_SIZE = 18.0

    def __init__(self, slot: LayoutSlot, preview_path: Path | None) -> None:
        super().__init__()
        self.slot = slot
        self.preview_path = preview_path
        self._pixmap = QPixmap()
        self._resizing = False
        self._resize_start = (slot.width, slot.height)
        self._press_pos = QPointF()
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.sync_from_model(refresh=True)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.slot.width, self.slot.height)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        del option, widget
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._pixmap.isNull():
            painter.fillRect(self.boundingRect(), QColor("#28313d"))
            painter.setPen(QColor("#d5dde8"))
            painter.drawText(
                self.boundingRect().adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Preview unavailable\nThe folder may need to be relinked.",
            )
        else:
            painter.drawPixmap(
                self.boundingRect(), self._pixmap, QRectF(self._pixmap.rect())
            )
        if self.isSelected():
            painter.setPen(QPen(QColor("#4da3ff"), 2.0, Qt.PenStyle.DashLine))
            painter.drawRect(self.boundingRect().adjusted(1, 1, -1, -1))
            painter.fillRect(self._handle_rect(), QColor("#4da3ff"))

    def sync_from_model(self, *, refresh: bool = False) -> None:
        self.prepareGeometryChange()
        self.setPos(self.slot.x, self.slot.y)
        self.setTransformOriginPoint(self.slot.width / 2, self.slot.height / 2)
        self.setRotation(self.slot.rotation)
        self.setOpacity(self.slot.opacity)
        self.setZValue(self.slot.z_index)
        if refresh:
            self.refresh_pixmap()
        self.update()

    def refresh_pixmap(self) -> None:
        self._pixmap = QPixmap()
        if self.preview_path is None:
            self.update()
            return
        try:
            image = load_oriented_rgba(self.preview_path)
            tile = render_slot_tile(
                image,
                max(1, round(self.slot.width)),
                max(1, round(self.slot.height)),
                self.slot.fit,
            )
        except ExportError:
            self.update()
            return
        raw = tile.tobytes("raw", "RGBA")
        qimage = QImage(
            raw,
            tile.width,
            tile.height,
            tile.width * 4,
            QImage.Format.Format_RGBA8888,
        ).copy()
        self._pixmap = QPixmap.fromImage(qimage)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._handle_rect().contains(
            event.pos()
        ):
            self._resizing = True
            self._resize_start = (self.slot.width, self.slot.height)
            self._press_pos = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self._resizing:
            super().mouseMoveEvent(event)
            return
        width = max(40.0, self._resize_start[0] + event.pos().x() - self._press_pos.x())
        height = max(
            40.0, self._resize_start[1] + event.pos().y() - self._press_pos.y()
        )
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            ratio = self._resize_start[0] / max(1.0, self._resize_start[1])
            if width / max(1.0, height) > ratio:
                height = width / ratio
            else:
                width = height * ratio
        self.prepareGeometryChange()
        self.slot.width = width
        self.slot.height = height
        self.setTransformOriginPoint(width / 2, height / 2)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._resizing:
            self._resizing = False
            self.refresh_pixmap()
            self.changed.emit(self.slot.id)
            event.accept()
            return
        super().mouseReleaseEvent(event)
        self.slot.x = self.pos().x()
        self.slot.y = self.pos().y()
        self.changed.emit(self.slot.id)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.slot.x = value.x()
            self.slot.y = value.y()
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged and bool(
            value
        ):
            self.selected.emit(self.slot.id)
        return super().itemChange(change, value)

    def _handle_rect(self) -> QRectF:
        return QRectF(
            self.slot.width - self.HANDLE_SIZE,
            self.slot.height - self.HANDLE_SIZE,
            self.HANDLE_SIZE,
            self.HANDLE_SIZE,
        )


class ImageComposerWindow(QMainWindow):
    """English Windows-first image composition workbench."""

    def __init__(
        self,
        project_path: str | Path | None = None,
        *,
        allowed_roots: tuple[str | Path, ...] | None = None,
        path_memory: RecentPathMemory | None = None,
    ) -> None:
        super().__init__()
        self._allowed_roots = tuple(
            configured_allowed_roots()
            if allowed_roots is None
            else (
                Path(root).expanduser().resolve(strict=False) for root in allowed_roots
            )
        )
        self._ui_store = frontend_state_store("image-composer")
        self._path_memory = path_memory or frontend_path_memory(self._allowed_roots)
        self._theme_controller = QtThemeController(
            QApplication.instance(),
            state_store=self._ui_store,
            path_memory=self._path_memory,
            frontend_id="image-composer",
        )
        self.project = ComposerProject()
        self.project_path: Path | None = None
        self._dirty = False
        self._updating = False
        self._project_generation = 0
        self._scan_requests: dict[str, tuple[str | None, bool, int]] = {}
        self._thumbnail_items: dict[tuple[str, int], QListWidgetItem] = {}
        self._thumbnail_tasks: dict[str, _ThumbnailTask] = {}
        self._slot_items: dict[str, SlotItem] = {}
        self._selected_slot_id = ""
        self._export_task: _ExportTask | None = None
        self._cancel_event: threading.Event | None = None
        self._closing = False
        self.thread_pool = QThreadPool.globalInstance()
        self.setWindowTitle("Free Image Composer")
        self.resize(1500, 900)
        self._build_ui()
        self._connect_signals()
        self._apply_project_to_ui()
        if project_path is not None:
            self.open_project(Path(project_path))

    def _build_ui(self) -> None:
        self.scene = QGraphicsScene(self)
        self.view = ComposerView(self.scene, self)

        root = QWidget(self)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()
        layout.addWidget(self.left_panel, 0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.right_panel, 0)
        self.setCentralWidget(root)

        file_menu = self.menuBar().addMenu("&File")
        self.new_action = QAction("&New", self)
        self.new_action.setShortcut(QKeySequence.StandardKey.New)
        self.open_action = QAction("&Open Project...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.save_action = QAction("&Save Project", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_as_action = QAction("Save Project &As...", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        for action in (
            self.new_action,
            self.open_action,
            self.save_action,
            self.save_as_action,
        ):
            file_menu.addAction(action)

        view_menu = self.menuBar().addMenu("&View")
        self.fit_view_action = QAction("Fit Canvas", self)
        self.reset_zoom_action = QAction("100% Zoom", self)
        view_menu.addActions((self.fit_view_action, self.reset_zoom_action))
        theme_menu = view_menu.addMenu("Theme")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for mode in ("auto", "light", "dark"):
            action = QAction(mode.title(), self)
            action.setCheckable(True)
            action.setChecked(self._theme_controller.mode == mode)
            action.triggered.connect(
                lambda _checked=False, selected=mode: self._set_theme_mode(selected)
            )
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[mode] = action
        view_menu.addSeparator()
        self.reset_ui_state_action = QAction("Reset UI State", self)
        self.reset_ui_state_action.triggered.connect(self._reset_ui_state)
        view_menu.addAction(self.reset_ui_state_action)

    def _set_theme_mode(self, mode: str) -> None:
        selected = self._theme_controller.set_mode(mode)
        for name, action in self.theme_actions.items():
            action.setChecked(name == selected)

    def _reset_ui_state(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset UI State",
            "Reset the theme and remembered dialog locations? Project content is not changed.",
            QMessageBox.StandardButton.Reset | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Reset:
            return
        self._theme_controller.reset()
        for name, action in self.theme_actions.items():
            action.setChecked(name == "auto")

    @staticmethod
    def _path_context(operation: str, field: str) -> PathMemoryContext:
        return PathMemoryContext(
            frontend="image-composer", operation=operation, field=field
        )

    def _dialog_initial_path(
        self,
        *,
        operation: str,
        field: str,
        dialog_mode: str,
        current_value: str | Path = "",
    ) -> str:
        return self._path_memory.resolve_initial(
            context=self._path_context(operation, field),
            dialog_mode=dialog_mode,
            current_value=str(current_value or ""),
        )

    def _accept_dialog_path(
        self,
        value: str | Path,
        *,
        operation: str,
        field: str,
        dialog_mode: str,
        kind: str,
        default_suffix: str = "",
    ) -> Path | None:
        try:
            selected = validate_allowed_path(
                value,
                allowed_roots=self._allowed_roots,
                kind=kind,
                default_suffix=default_suffix,
            )
        except NativeDialogError as exc:
            QMessageBox.warning(self, "Path Not Allowed", str(exc))
            return None
        self._path_memory.remember(
            context=self._path_context(operation, field),
            dialog_mode=dialog_mode,
            paths=(selected,),
        )
        return selected

    def _build_left_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel)
        title = QLabel("Image Folders")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        row = QHBoxLayout()
        self.add_folders_button = QPushButton("Add Folders...")
        self.relink_folder_button = QPushButton("Relink...")
        self.remove_folder_button = QPushButton("Remove")
        row.addWidget(self.add_folders_button)
        row.addWidget(self.relink_folder_button)
        row.addWidget(self.remove_folder_button)
        layout.addLayout(row)

        self.folder_combo = QComboBox()
        layout.addWidget(self.folder_combo)
        form = QFormLayout()
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(1, 1_000_000)
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(1, 1_000_000)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-86_400.0, 86_400.0)
        self.offset_spin.setDecimals(6)
        self.offset_spin.setSuffix(" s")
        form.addRow("Inclusive start", self.start_index_spin)
        form.addRow("Inclusive end", self.end_index_spin)
        form.addRow("Clock offset", self.offset_spin)
        layout.addLayout(form)
        self.folder_summary = QLabel("Add one or more image folders to begin.")
        self.folder_summary.setWordWrap(True)
        layout.addWidget(self.folder_summary)
        self.thumbnail_list = ThumbnailList()
        layout.addWidget(self.thumbnail_list, 1)
        hint = QLabel(
            "Drag any thumbnail onto the canvas. The image is only a layout preview."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #657080;")
        layout.addWidget(hint)
        return panel

    def _build_right_panel(self) -> QWidget:
        content = QWidget(self)
        content.setMinimumWidth(320)
        content.setMaximumWidth(390)
        layout = QVBoxLayout(content)
        layout.addWidget(self._canvas_group())
        layout.addWidget(self._slot_group())
        layout.addWidget(self._matching_group())
        layout.addWidget(self._export_group())
        layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumWidth(340)
        scroll.setMaximumWidth(410)
        return scroll

    def _canvas_group(self) -> QGroupBox:
        group = QGroupBox("Canvas")
        form = QFormLayout(group)
        self.canvas_width_spin = QSpinBox()
        self.canvas_width_spin.setRange(2, 8192)
        self.canvas_width_spin.setSingleStep(2)
        self.canvas_height_spin = QSpinBox()
        self.canvas_height_spin.setRange(2, 8192)
        self.canvas_height_spin.setSingleStep(2)
        self.background_button = QPushButton("Choose...")
        form.addRow("Width", self.canvas_width_spin)
        form.addRow("Height", self.canvas_height_spin)
        form.addRow("Background", self.background_button)
        return group

    def _slot_group(self) -> QGroupBox:
        group = QGroupBox("Selected Slot")
        layout = QVBoxLayout(group)
        self.slot_status = QLabel("Select a slot on the canvas.")
        self.slot_status.setWordWrap(True)
        layout.addWidget(self.slot_status)
        grid = QGridLayout()
        self.slot_x_spin = self._geometry_spin(-20_000, 20_000)
        self.slot_y_spin = self._geometry_spin(-20_000, 20_000)
        self.slot_width_spin = self._geometry_spin(1, 20_000)
        self.slot_height_spin = self._geometry_spin(1, 20_000)
        for row, (label, control) in enumerate(
            (
                ("X", self.slot_x_spin),
                ("Y", self.slot_y_spin),
                ("Width", self.slot_width_spin),
                ("Height", self.slot_height_spin),
            )
        ):
            grid.addWidget(QLabel(label), row // 2, (row % 2) * 2)
            grid.addWidget(control, row // 2, (row % 2) * 2 + 1)
        layout.addLayout(grid)
        form = QFormLayout()
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-360.0, 360.0)
        self.rotation_spin.setDecimals(2)
        self.rotation_spin.setSuffix(" deg")
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setDecimals(2)
        self.fit_combo = QComboBox()
        self.fit_combo.addItems(("contain", "cover", "stretch"))
        form.addRow("Rotation", self.rotation_spin)
        form.addRow("Opacity", self.opacity_spin)
        form.addRow("Fit", self.fit_combo)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.duplicate_slot_button = QPushButton("Duplicate")
        self.delete_slot_button = QPushButton("Delete")
        self.send_back_button = QPushButton("Send Back")
        self.bring_front_button = QPushButton("Bring Front")
        for button in (
            self.duplicate_slot_button,
            self.delete_slot_button,
            self.send_back_button,
            self.bring_front_button,
        ):
            row.addWidget(button)
        layout.addLayout(row)
        return group

    def _matching_group(self) -> QGroupBox:
        group = QGroupBox("Sequence Matching")
        form = QFormLayout(group)
        self.master_folder_combo = QComboBox()
        self.match_mode_combo = QComboBox()
        self.match_mode_combo.addItem("Nearest time", "time")
        self.match_mode_combo.addItem("Relative index", "relative")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 86_400.0)
        self.tolerance_spin.setDecimals(6)
        self.tolerance_spin.setSuffix(" s")
        self.strict_check = QCheckBox(
            "Skip a frame when any used folder exceeds tolerance"
        )
        form.addRow("Master timeline", self.master_folder_combo)
        form.addRow("Mode", self.match_mode_combo)
        form.addRow("Tolerance", self.tolerance_spin)
        form.addRow(self.strict_check)
        return group

    def _export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        layout = QVBoxLayout(group)
        path_row = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_button = QPushButton("Browse...")
        path_row.addWidget(self.output_path_edit, 1)
        path_row.addWidget(self.output_path_button)
        layout.addLayout(path_row)
        form = QFormLayout()
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("MP4 (mp4v)", "mp4")
        self.output_format_combo.addItem("AVI (MJPG)", "avi")
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(5.0)
        self.fps_spin.setDecimals(2)
        self.save_png_check = QCheckBox("Also save PNG frame sequence")
        form.addRow("Format", self.output_format_combo)
        form.addRow("FPS", self.fps_spin)
        form.addRow(self.save_png_check)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.export_button = QPushButton("Render Video")
        self.export_button.setStyleSheet("font-weight: 600;")
        self.cancel_export_button = QPushButton("Cancel")
        self.cancel_export_button.setEnabled(False)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.cancel_export_button)
        layout.addLayout(buttons)
        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 1)
        self.export_progress.setValue(0)
        layout.addWidget(self.export_progress)
        self.export_status = QLabel("CSV is always written; PNG frames are optional.")
        self.export_status.setWordWrap(True)
        layout.addWidget(self.export_status)
        return group

    @staticmethod
    def _geometry_spin(minimum: int, maximum: int) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(1)
        return control

    def _connect_signals(self) -> None:
        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.choose_project)
        self.save_action.triggered.connect(self.save_current_project)
        self.save_as_action.triggered.connect(self.save_project_as)
        self.fit_view_action.triggered.connect(self.fit_canvas)
        self.reset_zoom_action.triggered.connect(self.view.resetTransform)
        self.add_folders_button.clicked.connect(self.choose_folders)
        self.relink_folder_button.clicked.connect(self.relink_current_folder)
        self.remove_folder_button.clicked.connect(self.remove_current_folder)
        self.folder_combo.currentIndexChanged.connect(self._folder_selection_changed)
        self.start_index_spin.valueChanged.connect(self._folder_settings_changed)
        self.end_index_spin.valueChanged.connect(self._folder_settings_changed)
        self.offset_spin.valueChanged.connect(self._folder_settings_changed)
        self.view.imageDropped.connect(self.add_slot)
        self.scene.selectionChanged.connect(self._scene_selection_changed)
        self.canvas_width_spin.valueChanged.connect(self._canvas_changed)
        self.canvas_height_spin.valueChanged.connect(self._canvas_changed)
        self.background_button.clicked.connect(self.choose_background)
        for control in (
            self.slot_x_spin,
            self.slot_y_spin,
            self.slot_width_spin,
            self.slot_height_spin,
            self.rotation_spin,
            self.opacity_spin,
        ):
            control.valueChanged.connect(self._slot_inspector_changed)
        self.fit_combo.currentTextChanged.connect(self._slot_inspector_changed)
        self.duplicate_slot_button.clicked.connect(self.duplicate_selected_slot)
        self.delete_slot_button.clicked.connect(self.delete_selected_slot)
        self.send_back_button.clicked.connect(
            lambda: self.move_selected_slot(front=False)
        )
        self.bring_front_button.clicked.connect(
            lambda: self.move_selected_slot(front=True)
        )
        self.master_folder_combo.currentIndexChanged.connect(self._matching_changed)
        self.match_mode_combo.currentIndexChanged.connect(self._matching_changed)
        self.tolerance_spin.valueChanged.connect(self._matching_changed)
        self.strict_check.toggled.connect(self._matching_changed)
        self.output_path_edit.textChanged.connect(self._export_settings_changed)
        self.output_format_combo.currentIndexChanged.connect(
            self._export_settings_changed
        )
        self.fps_spin.valueChanged.connect(self._export_settings_changed)
        self.save_png_check.toggled.connect(self._export_settings_changed)
        self.output_path_button.clicked.connect(self.choose_output_path)
        self.export_button.clicked.connect(self.start_export)
        self.cancel_export_button.clicked.connect(self.cancel_export)

    def choose_folders(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Add Image Folder",
            self._dialog_initial_path(
                operation="add-folder",
                field="source-folder",
                dialog_mode="select_directory",
            ),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        path = self._accept_dialog_path(
            selected,
            operation="add-folder",
            field="source-folder",
            dialog_mode="select_directory",
            kind="directory",
        )
        if path is None:
            return
        existing = {folder.path for folder in self.project.folders}
        if path not in existing:
            self._start_scan(path, existing_folder_id=None, mark_dirty=True)

    def add_folder_source(self, folder: FolderSource) -> None:
        """Add an already scanned source; useful for tests and integrations."""

        self.project.folders.append(folder)
        if not self.project.matching.master_folder_id:
            self.project.matching.master_folder_id = folder.id
        self._refresh_folder_combos(select_id=folder.id)
        self._mark_dirty()

    def _start_scan(
        self, path: Path, existing_folder_id: str | None, *, mark_dirty: bool
    ) -> None:
        request_id = f"scan_{uuid4().hex}"
        self._scan_requests[request_id] = (
            existing_folder_id,
            mark_dirty,
            self._project_generation,
        )
        self.folder_summary.setText(f"Scanning {path} ...")
        task = _ScanTask(request_id, path)
        task.signals.finished.connect(self._scan_finished)
        task.signals.failed.connect(self._scan_failed)
        self.thread_pool.start(task)

    def _scan_finished(self, request_id: str, path: Path, records: list) -> None:
        folder_id, mark_dirty, generation = self._scan_requests.pop(
            request_id, (None, False, -1)
        )
        if generation != self._project_generation:
            return
        if folder_id:
            folder = self.project.folder_map().get(folder_id)
            if folder is None:
                return
            folder.path = Path(path)
            folder.name = Path(path).name or str(path)
            folder.records = list(records)
            folder.resolved = True
        else:
            folder = FolderSource.create(path, list(records))
            self.project.folders.append(folder)
            if not self.project.matching.master_folder_id:
                self.project.matching.master_folder_id = folder.id
        self._refresh_folder_combos(select_id=folder.id)
        self._refresh_slot_previews(folder.id)
        if mark_dirty:
            self._mark_dirty()

    def _scan_failed(self, request_id: str, message: str) -> None:
        folder_id, _mark_dirty, generation = self._scan_requests.pop(
            request_id, (None, False, -1)
        )
        if generation != self._project_generation:
            return
        if folder_id:
            folder = self.project.folder_map().get(folder_id)
            if folder is not None:
                folder.records = []
                folder.resolved = False
        self.folder_summary.setText(f"Scan failed: {message}")
        QMessageBox.warning(self, "Folder Scan Failed", message)
        self._refresh_folder_combos(select_id=folder_id or "")

    def relink_current_folder(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Relink Image Folder",
            self._dialog_initial_path(
                operation="relink-folder",
                field="source-folder",
                dialog_mode="select_directory",
                current_value=folder.path,
            ),
        )
        if not selected:
            return
        path = self._accept_dialog_path(
            selected,
            operation="relink-folder",
            field="source-folder",
            dialog_mode="select_directory",
            kind="directory",
        )
        if path is None:
            return
        folder.path = path
        folder.name = folder.path.name or str(folder.path)
        folder.records = []
        folder.resolved = False
        self._start_scan(folder.path, existing_folder_id=folder.id, mark_dirty=True)

    def remove_current_folder(self) -> None:
        folder = self._current_folder()
        if folder is None:
            return
        bound = [slot for slot in self.project.slots if slot.folder_id == folder.id]
        if bound:
            answer = QMessageBox.question(
                self,
                "Remove Folder",
                f"Remove {folder.name} and its {len(bound)} canvas slot(s)?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        for slot in list(bound):
            self._remove_slot(slot.id)
        self.project.folders = [
            item for item in self.project.folders if item.id != folder.id
        ]
        if self.project.matching.master_folder_id == folder.id:
            self.project.matching.master_folder_id = (
                self.project.folders[0].id if self.project.folders else ""
            )
        self._refresh_folder_combos()
        self._mark_dirty()

    def _refresh_folder_combos(self, *, select_id: str = "") -> None:
        current_id = select_id or self.folder_combo.currentData() or ""
        master_id = self.project.matching.master_folder_id
        self._updating = True
        self.folder_combo.clear()
        self.master_folder_combo.clear()
        for folder in self.project.folders:
            label = folder.name if folder.resolved else f"{folder.name} (unresolved)"
            self.folder_combo.addItem(label, folder.id)
            self.master_folder_combo.addItem(label, folder.id)
        self._set_combo_data(self.folder_combo, current_id)
        self._set_combo_data(self.master_folder_combo, master_id)
        self._updating = False
        self._folder_selection_changed()

    def _folder_selection_changed(self) -> None:
        if self._updating:
            return
        folder = self._current_folder()
        self._updating = True
        self._detach_thumbnail_tasks()
        self.thumbnail_list.clear()
        self._thumbnail_items.clear()
        enabled = folder is not None
        for control in (
            self.start_index_spin,
            self.end_index_spin,
            self.offset_spin,
            self.relink_folder_button,
            self.remove_folder_button,
        ):
            control.setEnabled(enabled)
        if folder is None:
            self.folder_summary.setText("Add one or more image folders to begin.")
            self._updating = False
            return
        self.start_index_spin.setValue(max(1, folder.start_index))
        self.end_index_spin.setValue(max(1, folder.end_index))
        self.offset_spin.setValue(folder.offset_seconds)
        sources: dict[str, int] = {}
        for record in folder.records:
            sources[record.time_source] = sources.get(record.time_source, 0) + 1
            item = QListWidgetItem(f"{record.ordinal}. {record.path.name}")
            item.setData(
                Qt.ItemDataRole.UserRole,
                {"folder_id": folder.id, "ordinal": record.ordinal},
            )
            item.setToolTip(
                f"{record.path}\n{record.timestamp.isoformat(timespec='microseconds')}\n{record.time_source}"
            )
            self.thumbnail_list.addItem(item)
            self._thumbnail_items[(folder.id, record.ordinal)] = item
            task = _ThumbnailTask(folder.id, record.ordinal, record.path)
            task.signals.finished.connect(self._thumbnail_finished)
            task.signals.settled.connect(self._thumbnail_task_settled)
            self._thumbnail_tasks[task.task_id] = task
            self.thread_pool.start(task)
        source_text = ", ".join(
            f"{key}: {value}" for key, value in sorted(sources.items())
        )
        self.folder_summary.setText(
            f"{len(folder.records)} image(s). Range {folder.start_index}-{folder.end_index}.\n{source_text or 'No timestamps.'}"
        )
        self._updating = False

    def _thumbnail_finished(
        self,
        task_id: str,
        folder_id: str,
        ordinal: int,
        payload: QByteArray,
    ) -> None:
        if self._closing or task_id not in self._thumbnail_tasks:
            return
        item = self._thumbnail_items.get((folder_id, ordinal))
        if item is None or payload.isEmpty():
            return
        pixmap = QPixmap()
        pixmap.loadFromData(payload, "PNG")
        item.setIcon(pixmap)

    def _thumbnail_task_settled(self, task_id: str) -> None:
        self._thumbnail_tasks.pop(task_id, None)

    def _detach_thumbnail_tasks(self, *, wait_timeout: float = 0.0) -> None:
        """Cancel thumbnail work and safely detach any running workers.

        Queued tasks are removed immediately.  Running image decodes receive a
        cancellation request and are retained outside the window until they
        exit.  Closing may wait briefly, but never beyond ``wait_timeout``.
        """

        tasks = tuple(self._thumbnail_tasks.values())
        self._thumbnail_tasks.clear()
        if not tasks:
            return

        running: list[_ThumbnailTask] = []
        for task in tasks:
            task.cancel()
            for signal, slot in (
                (task.signals.finished, self._thumbnail_finished),
                (task.signals.settled, self._thumbnail_task_settled),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            if self.thread_pool.tryTake(task):
                task.mark_dequeued()
            else:
                running.append(task)

        deadline = time.monotonic() + max(0.0, wait_timeout)
        for task in running:
            if wait_timeout > 0.0:
                task.wait(deadline - time.monotonic())
            if not task.done:
                _retain_running_thumbnail_task(task)

    def _folder_settings_changed(self) -> None:
        if self._updating:
            return
        folder = self._current_folder()
        if folder is None:
            return
        folder.start_index = self.start_index_spin.value()
        folder.end_index = self.end_index_spin.value()
        folder.offset_seconds = self.offset_spin.value()
        self.folder_summary.setText(
            f"{len(folder.records)} image(s). Range {folder.start_index}-{folder.end_index}."
        )
        self._mark_dirty()

    def add_slot(
        self, folder_id: str, ordinal: int, point: QPointF
    ) -> LayoutSlot | None:
        folder = self.project.folder_map().get(folder_id)
        if folder is None or folder.record_by_ordinal(ordinal) is None:
            return None
        width = min(420.0, self.project.canvas.width * 0.6)
        height = min(280.0, self.project.canvas.height * 0.6)
        slot = LayoutSlot.create(
            folder_id,
            ordinal,
            x=point.x() - width / 2,
            y=point.y() - height / 2,
            width=width,
            height=height,
            z_index=len(self.project.slots),
        )
        record = folder.record_by_ordinal(ordinal)
        slot.preview_relative_path = record.path.name if record else ""
        self.project.slots.append(slot)
        self._create_slot_item(slot)
        self._select_slot(slot.id)
        self._mark_dirty()
        return slot

    def _create_slot_item(self, slot: LayoutSlot) -> SlotItem:
        preview = self._preview_path(slot)
        item = SlotItem(slot, preview)
        item.changed.connect(self._slot_item_changed)
        item.selected.connect(self._select_slot)
        self.scene.addItem(item)
        self._slot_items[slot.id] = item
        return item

    def _slot_item_changed(self, slot_id: str) -> None:
        self._selected_slot_id = slot_id
        self._update_slot_inspector()
        self._mark_dirty()

    def _scene_selection_changed(self) -> None:
        selected = [
            item for item in self.scene.selectedItems() if isinstance(item, SlotItem)
        ]
        self._selected_slot_id = selected[0].slot.id if selected else ""
        self._update_slot_inspector()

    def _select_slot(self, slot_id: str) -> None:
        item = self._slot_items.get(slot_id)
        if item is None:
            return
        if item.isSelected():
            self._selected_slot_id = slot_id
            self._update_slot_inspector()
            return
        self.scene.clearSelection()
        item.setSelected(True)
        self._selected_slot_id = slot_id
        self._update_slot_inspector()

    def _update_slot_inspector(self) -> None:
        slot = self.project.slot_by_id(self._selected_slot_id)
        self._updating = True
        controls = (
            self.slot_x_spin,
            self.slot_y_spin,
            self.slot_width_spin,
            self.slot_height_spin,
            self.rotation_spin,
            self.opacity_spin,
            self.fit_combo,
            self.duplicate_slot_button,
            self.delete_slot_button,
            self.send_back_button,
            self.bring_front_button,
        )
        for control in controls:
            control.setEnabled(slot is not None)
        if slot is None:
            self.slot_status.setText("Select a slot on the canvas.")
            self._updating = False
            return
        folder = self.project.folder_map().get(slot.folder_id)
        self.slot_status.setText(
            f"{folder.name if folder else 'Missing folder'} / preview #{slot.preview_ordinal}"
        )
        self.slot_x_spin.setValue(slot.x)
        self.slot_y_spin.setValue(slot.y)
        self.slot_width_spin.setValue(slot.width)
        self.slot_height_spin.setValue(slot.height)
        self.rotation_spin.setValue(slot.rotation)
        self.opacity_spin.setValue(slot.opacity)
        self.fit_combo.setCurrentText(slot.fit)
        self._updating = False

    def _slot_inspector_changed(self) -> None:
        if self._updating:
            return
        slot = self.project.slot_by_id(self._selected_slot_id)
        item = self._slot_items.get(self._selected_slot_id)
        if slot is None or item is None:
            return
        old_size = (slot.width, slot.height)
        old_fit = slot.fit
        slot.x = self.slot_x_spin.value()
        slot.y = self.slot_y_spin.value()
        slot.width = self.slot_width_spin.value()
        slot.height = self.slot_height_spin.value()
        slot.rotation = self.rotation_spin.value()
        slot.opacity = self.opacity_spin.value()
        slot.fit = self.fit_combo.currentText()
        item.sync_from_model(
            refresh=old_size != (slot.width, slot.height) or old_fit != slot.fit
        )
        self._mark_dirty()

    def duplicate_selected_slot(self) -> None:
        source = self.project.slot_by_id(self._selected_slot_id)
        if source is None:
            return
        slot = LayoutSlot.create(
            source.folder_id,
            source.preview_ordinal,
            x=source.x + 24,
            y=source.y + 24,
            width=source.width,
            height=source.height,
            z_index=len(self.project.slots),
        )
        slot.rotation = source.rotation
        slot.opacity = source.opacity
        slot.fit = source.fit
        slot.preview_relative_path = source.preview_relative_path
        self.project.slots.append(slot)
        self._create_slot_item(slot)
        self._select_slot(slot.id)
        self._mark_dirty()

    def delete_selected_slot(self) -> None:
        if self._selected_slot_id:
            self._remove_slot(self._selected_slot_id)
            self._mark_dirty()

    def _remove_slot(self, slot_id: str) -> None:
        item = self._slot_items.pop(slot_id, None)
        if item is not None:
            self.scene.removeItem(item)
        self.project.slots = [slot for slot in self.project.slots if slot.id != slot_id]
        self.project.normalize_z_indexes()
        for current in self._slot_items.values():
            current.sync_from_model()
        self._selected_slot_id = ""
        self._update_slot_inspector()

    def move_selected_slot(self, *, front: bool) -> None:
        slot = self.project.slot_by_id(self._selected_slot_id)
        if slot is None:
            return
        ordered = sorted(self.project.slots, key=lambda item: item.z_index)
        ordered.remove(slot)
        if front:
            ordered.append(slot)
        else:
            ordered.insert(0, slot)
        for index, current in enumerate(ordered):
            current.z_index = index
            self._slot_items[current.id].setZValue(index)
        self._mark_dirty()

    def _canvas_changed(self) -> None:
        if self._updating:
            return
        width = self._force_even(self.canvas_width_spin)
        height = self._force_even(self.canvas_height_spin)
        self.project.canvas.width = width
        self.project.canvas.height = height
        self._update_scene_canvas()
        self._mark_dirty()

    @staticmethod
    def _force_even(control: QSpinBox) -> int:
        value = control.value()
        if value % 2:
            control.blockSignals(True)
            control.setValue(min(control.maximum(), value + 1))
            control.blockSignals(False)
            value = control.value()
        return value

    def choose_background(self) -> None:
        color = QColorDialog.getColor(QColor(self.project.canvas.background), self)
        if not color.isValid():
            return
        self.project.canvas.background = color.name()
        self._update_scene_canvas()
        self._mark_dirty()

    def _update_scene_canvas(self) -> None:
        self.scene.setSceneRect(
            0, 0, self.project.canvas.width, self.project.canvas.height
        )
        self.scene.setBackgroundBrush(QBrush(QColor(self.project.canvas.background)))
        self.background_button.setStyleSheet(
            f"background: {self.project.canvas.background}; color: white;"
        )

    def fit_canvas(self) -> None:
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _matching_changed(self) -> None:
        if self._updating:
            return
        self.project.matching.master_folder_id = (
            self.master_folder_combo.currentData() or ""
        )
        self.project.matching.mode = self.match_mode_combo.currentData() or "time"
        self.project.matching.tolerance_seconds = self.tolerance_spin.value()
        self.project.matching.strict = self.strict_check.isChecked()
        time_mode = self.project.matching.mode == "time"
        self.tolerance_spin.setEnabled(time_mode)
        self.strict_check.setEnabled(time_mode)
        self._mark_dirty()

    def choose_output_path(self) -> None:
        output_format = self.output_format_combo.currentData() or "mp4"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Video",
            self._dialog_initial_path(
                operation="export-video",
                field="output-file",
                dialog_mode="save_file",
                current_value=self.output_path_edit.text(),
            ),
            "MP4 Video (*.mp4)" if output_format == "mp4" else "AVI Video (*.avi)",
        )
        if not selected:
            return
        suffix = f".{output_format}"
        path = self._accept_dialog_path(
            selected,
            operation="export-video",
            field="output-file",
            dialog_mode="save_file",
            kind="save_file",
            default_suffix=suffix,
        )
        if path is None:
            return
        self.output_path_edit.setText(str(path))

    def _export_settings_changed(self) -> None:
        if self._updating:
            return
        output_format = self.output_format_combo.currentData() or "mp4"
        self.project.export.output_path = self.output_path_edit.text().strip()
        self.project.export.output_format = output_format
        self.project.export.fps = self.fps_spin.value()
        self.project.export.save_png_frames = self.save_png_check.isChecked()
        path = (
            Path(self.project.export.output_path)
            if self.project.export.output_path
            else None
        )
        if path is not None and path.suffix.casefold() not in {".mp4", ".avi"}:
            self.output_path_edit.setToolTip(
                f"The output path must end with .{output_format}."
            )
        else:
            self.output_path_edit.setToolTip("")
        self._mark_dirty()

    def start_export(self) -> None:
        self._export_settings_changed()
        try:
            validate_project(self.project, require_output=True)
            self._validate_output_suffix()
        except (MatchPlanError, ExportError) as exc:
            QMessageBox.warning(self, "Export Preflight Failed", str(exc))
            return
        output = self._accept_dialog_path(
            self.project.export.output_path,
            operation="export-video",
            field="output-file",
            dialog_mode="save_file",
            kind="save_file",
        )
        if output is None:
            return
        self.project.export.output_path = str(output)
        csv_path = output.with_name(f"{output.stem}_matches.csv")
        frames_path = output.with_name(f"{output.stem}_frames")
        existing = [path for path in (output, csv_path, frames_path) if path.exists()]
        if existing:
            answer = QMessageBox.question(
                self,
                "Replace Existing Output",
                "The following output already exists and will be replaced only after a successful render:\n\n"
                + "\n".join(str(path) for path in existing),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._cancel_event = threading.Event()
        self._export_task = _ExportTask(copy.deepcopy(self.project), self._cancel_event)
        self._export_task.signals.progress.connect(self._export_progressed)
        self._export_task.signals.finished.connect(self._export_finished)
        self._export_task.signals.failed.connect(self._export_failed)
        self._export_task.signals.cancelled.connect(self._export_cancelled)
        self.export_button.setEnabled(False)
        self.cancel_export_button.setEnabled(True)
        self.export_progress.setRange(0, 0)
        self.export_status.setText("Preparing match plan and video writer...")
        self.thread_pool.start(self._export_task)

    def _validate_output_suffix(self) -> None:
        output = Path(self.project.export.output_path)
        expected = f".{self.project.export.output_format}"
        if output.suffix.casefold() != expected:
            raise ExportError(f"Output path must end with {expected}.")

    def cancel_export(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.export_status.setText(
                "Cancellation requested; finishing the current image operation..."
            )
            self.cancel_export_button.setEnabled(False)

    def _export_progressed(self, current: int, total: int, message: str) -> None:
        self.export_progress.setRange(0, max(1, total))
        self.export_progress.setValue(current)
        self.export_status.setText(message)

    def _export_finished(self, result: ExportResult) -> None:
        self._finish_export_ui()
        if result.status == "no_frames":
            self.export_status.setText(
                f"No video was produced. All {result.attempted_frames} master frames were skipped; CSV saved to {result.csv_path}."
            )
            QMessageBox.information(
                self, "No Matched Frames", self.export_status.text()
            )
            return
        self.export_status.setText(
            f"Saved {result.emitted_frames}/{result.attempted_frames} frame(s) to {result.video_path}. CSV: {result.csv_path}"
        )

    def _export_failed(self, message: str) -> None:
        self._finish_export_ui()
        self.export_status.setText(f"Export failed: {message}")
        QMessageBox.critical(self, "Export Failed", message)

    def _export_cancelled(self, message: str) -> None:
        self._finish_export_ui()
        self.export_status.setText(message)

    def _finish_export_ui(self) -> None:
        self.export_button.setEnabled(True)
        self.cancel_export_button.setEnabled(False)
        self._export_task = None
        self._cancel_event = None

    def new_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        self._project_generation += 1
        self.project = ComposerProject()
        self.project_path = None
        self._dirty = False
        self._apply_project_to_ui()

    def choose_project(self) -> None:
        if not self._confirm_discard_changes():
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open Composer Project",
            self._dialog_initial_path(
                operation="open-project",
                field="project-file",
                dialog_mode="open_file",
            ),
            "Image Composer Project (*.fic.json)",
        )
        if not selected:
            return
        path = self._accept_dialog_path(
            selected,
            operation="open-project",
            field="project-file",
            dialog_mode="open_file",
            kind="file",
        )
        if path is not None:
            self.open_project(path, validate_path=False)

    def open_project(self, path: str | Path, *, validate_path: bool = True) -> None:
        if validate_path:
            validated = self._accept_dialog_path(
                path,
                operation="open-project",
                field="project-file",
                dialog_mode="open_file",
                kind="file",
            )
            if validated is None:
                return
            path = validated
        try:
            project = load_project(path)
        except ProjectFormatError as exc:
            QMessageBox.critical(self, "Project Could Not Be Opened", str(exc))
            return
        self._project_generation += 1
        self.project = project
        self.project_path = Path(path).expanduser().resolve()
        self._dirty = False
        self._apply_project_to_ui()
        for folder in self.project.folders:
            try:
                safe_folder = validate_allowed_path(
                    folder.path,
                    allowed_roots=self._allowed_roots,
                    kind="directory",
                )
            except NativeDialogError:
                folder.resolved = False
                continue
            folder.path = safe_folder
            if safe_folder.is_dir():
                self._start_scan(
                    safe_folder, existing_folder_id=folder.id, mark_dirty=False
                )
            else:
                folder.resolved = False
        self._refresh_folder_combos()
        self._update_window_title()

    def save_current_project(self) -> bool:
        if self.project_path is None:
            return self.save_project_as()
        try:
            self.project_path = save_project(self.project_path, self.project)
        except OSError as exc:
            QMessageBox.critical(self, "Project Save Failed", str(exc))
            return False
        self._dirty = False
        self._update_window_title()
        return True

    def save_project_as(self) -> bool:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Composer Project",
            self._dialog_initial_path(
                operation="save-project",
                field="project-file",
                dialog_mode="save_file",
                current_value=str(self.project_path or "untitled.fic.json"),
            ),
            "Image Composer Project (*.fic.json)",
        )
        if not selected:
            return False
        path = self._accept_dialog_path(
            selected,
            operation="save-project",
            field="project-file",
            dialog_mode="save_file",
            kind="save_file",
            default_suffix=".fic.json",
        )
        if path is None:
            return False
        if path.exists():
            answer = QMessageBox.question(
                self,
                "Replace Existing Project",
                f"Replace the existing project file?\n\n{path}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.project_path = path
        return self.save_current_project()

    def _apply_project_to_ui(self) -> None:
        self._updating = True
        self.scene.clear()
        self._slot_items.clear()
        self._selected_slot_id = ""
        self.canvas_width_spin.setValue(self.project.canvas.width)
        self.canvas_height_spin.setValue(self.project.canvas.height)
        self.tolerance_spin.setValue(self.project.matching.tolerance_seconds)
        self.strict_check.setChecked(self.project.matching.strict)
        self._set_combo_data(self.match_mode_combo, self.project.matching.mode)
        self.output_path_edit.setText(self.project.export.output_path)
        self._set_combo_data(
            self.output_format_combo, self.project.export.output_format
        )
        self.fps_spin.setValue(self.project.export.fps)
        self.save_png_check.setChecked(self.project.export.save_png_frames)
        self._updating = False
        self._refresh_folder_combos()
        self._update_scene_canvas()
        for slot in self.project.slots:
            self._create_slot_item(slot)
        self._update_slot_inspector()
        self._matching_changed()
        self._dirty = False
        self._update_window_title()

    def _refresh_slot_previews(self, folder_id: str) -> None:
        for slot in self.project.slots:
            if slot.folder_id != folder_id:
                continue
            item = self._slot_items.get(slot.id)
            if item is None:
                continue
            item.preview_path = self._preview_path(slot)
            item.refresh_pixmap()

    def _preview_path(self, slot: LayoutSlot) -> Path | None:
        folder = self.project.folder_map().get(slot.folder_id)
        if folder is None:
            return None
        record = None
        if slot.preview_relative_path:
            record = next(
                (
                    item
                    for item in folder.records
                    if item.path.name == slot.preview_relative_path
                ),
                None,
            )
            if record is not None:
                slot.preview_ordinal = record.ordinal
        if record is None:
            record = folder.record_by_ordinal(slot.preview_ordinal)
        return record.path if record else None

    def _current_folder(self) -> FolderSource | None:
        folder_id = self.folder_combo.currentData()
        return self.project.folder_map().get(folder_id)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _mark_dirty(self) -> None:
        if self._updating:
            return
        self._dirty = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        name = self.project_path.name if self.project_path else "Untitled"
        marker = " *" if self._dirty else ""
        self.setWindowTitle(f"Free Image Composer — {name}{marker}")

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved Project",
            "Save changes to the current project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_current_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if self._export_task is not None:
            answer = QMessageBox.question(
                self,
                "Export In Progress",
                "Cancel the export? Temporary outputs will be removed. The window will remain open until cancellation finishes.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.cancel_export()
            event.ignore()
            return
        if self._confirm_discard_changes():
            self._closing = True
            self._detach_thumbnail_tasks(wait_timeout=0.1)
            event.accept()
        else:
            event.ignore()


def create_window(
    project_path: str | Path | None = None,
    *,
    allowed_roots: tuple[str | Path, ...] | None = None,
) -> ImageComposerWindow:
    """Create the main window after QApplication has been initialized."""

    if QApplication.instance() is None:
        raise RuntimeError(
            "Create QApplication before creating the image composer window."
        )
    return ImageComposerWindow(
        project_path=project_path,
        allowed_roots=allowed_roots,
    )
