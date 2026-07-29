# SPDX-License-Identifier: GPL-3.0-only
"""Reusable native widgets for the App 1.0 module pages."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QImageReader,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

_SUMMARY_FIELDS = ("Module", "Input", "Parameters", "Output", "Workload")


def load_preview_pixmap(
    value: str | Path,
    target_size: QSize,
    *,
    minimum_size: QSize = QSize(320, 240),
    maximum_size: QSize = QSize(2048, 2048),
) -> QPixmap:
    """Decode an image near its on-screen size to avoid large QImage allocations."""

    reader = QImageReader(str(value))
    reader.setAutoTransform(True)
    source_size = reader.size()
    width = min(max(target_size.width(), minimum_size.width()), maximum_size.width())
    height = min(
        max(target_size.height(), minimum_size.height()),
        maximum_size.height(),
    )
    if source_size.isValid():
        scaled = source_size.scaled(
            QSize(width, height),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        if (
            scaled.width() < source_size.width()
            or scaled.height() < source_size.height()
        ):
            reader.setScaledSize(scaled)
    image = reader.read()
    return QPixmap.fromImage(image) if not image.isNull() else QPixmap()


def parse_confirmation_summary(summary: str) -> dict[str, str]:
    """Parse established confirmation text without exposing legacy endpoints."""

    result = {field: "" for field in _SUMMARY_FIELDS}
    current: str | None = None
    for raw_line in str(summary).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Endpoint:"):
            continue
        matched = False
        for field in _SUMMARY_FIELDS:
            prefix = f"{field}:"
            if line.startswith(prefix):
                result[field] = line.removeprefix(prefix).strip()
                current = field
                matched = True
                break
        if not matched and current is not None:
            result[current] = f"{result[current]} {line}".strip()
    return result


class RunConfirmationDialog(QDialog):
    """Readable, copyable confirmation for paths and scientific parameters."""

    def __init__(
        self,
        title: str,
        summary: str | Mapping[str, object],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("appV1RunConfirmationDialog")
        self.setWindowTitle(title)
        self.setModal(True)
        self.setSizeGripEnabled(True)

        values = (
            parse_confirmation_summary(summary)
            if isinstance(summary, str)
            else {field: str(summary.get(field, "")) for field in _SUMMARY_FIELDS}
        )
        root = QVBoxLayout(self)
        heading = QLabel("Review operation")
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(heading)
        help_text = QLabel(
            "Nothing will run until you select Run. Long values can be selected "
            "and copied."
        )
        help_text.setWordWrap(True)
        help_text.setProperty("muted", True)
        root.addWidget(help_text)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        self.value_labels: dict[str, QLabel] = {}
        for field in _SUMMARY_FIELDS:
            label = QLabel(values[field] or "—")
            label.setObjectName(f"confirmation{field}Value")
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            label.setWordWrap(True)
            label.setMinimumWidth(420)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Minimum,
            )
            self.value_labels[field] = label
            form.addRow(f"{field}:", label)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        run_button = QPushButton("Run")
        run_button.setProperty("primary", True)
        cancel_button.clicked.connect(self.reject)
        run_button.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(cancel_button)
        buttons.addWidget(run_button)
        root.addLayout(buttons)

        screen = self.screen() or (parent.screen() if parent is not None else None)
        available = (
            screen.availableGeometry()
            if screen is not None
            else QRectF(0, 0, 1200, 800)
        )
        width = max(660, min(920, int(available.width() * 0.68)))
        height = max(420, min(680, int(available.height() * 0.65)))
        self.resize(width, height)

    @classmethod
    def confirm(
        cls,
        parent: QWidget,
        title: str,
        summary: str | Mapping[str, object],
    ) -> bool:
        return cls(title, summary, parent).exec() == QDialog.DialogCode.Accepted


class NativeModulePanel(QWidget):
    """Base panel with consistent diagnostics and an explicit legacy escape hatch."""

    legacy_interface_requested = pyqtSignal(str)
    reset_requested = pyqtSignal()

    def __init__(
        self,
        module_id: str,
        *,
        legacy_label: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.module_id = module_id
        self.legacy_label = legacy_label or "legacy interface"
        self._diagnostics: list[str] = []

    def build_more_button(self) -> QToolButton:
        button = QToolButton()
        button.setText("More")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        reset = QAction("Reset page state", menu)
        reset.triggered.connect(self.reset_requested)
        copy = QAction("Copy diagnostics", menu)
        copy.triggered.connect(self.copy_diagnostics)
        legacy = QAction(f"Open {self.legacy_label}", menu)
        legacy.setObjectName("openLegacyInterfaceAction")
        legacy.triggered.connect(
            lambda: self.legacy_interface_requested.emit(self.module_id)
        )
        menu.addAction(reset)
        menu.addAction(copy)
        menu.addSeparator()
        menu.addAction(legacy)
        button.setMenu(menu)
        return button

    def record_diagnostic(self, message: object) -> None:
        rendered = str(message).strip()
        if rendered:
            self._diagnostics.append(rendered)
            del self._diagnostics[:-100]

    def copy_diagnostics(self) -> None:
        from PyQt6.QtWidgets import QApplication

        QApplication.clipboard().setText("\n".join(self._diagnostics))

    @staticmethod
    def confirm(
        parent: QWidget,
        title: str,
        summary: str | Mapping[str, object],
    ) -> bool:
        return RunConfirmationDialog.confirm(parent, title, summary)

    @staticmethod
    def configure_path_field(field: QWidget) -> None:
        field.setMinimumWidth(320)
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if isinstance(field, QLineEdit):
            field.setToolTip(field.text())
            field.textChanged.connect(field.setToolTip)


class ScientificImageCanvas(QGraphicsView):
    """Native image surface with pan, zoom, and rectangle/lasso overlays."""

    roi_created = pyqtSignal(object)
    coordinates_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("scientificImageCanvas")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._tool = "pan"
        self._origin: QPointF | None = None
        self._draft_rect: QGraphicsRectItem | None = None
        self._draft_path: QGraphicsPathItem | None = None
        self._lasso_points: list[QPointF] = []
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setMinimumSize(250, 240)
        self.setBackgroundBrush(QBrush(QColor("#10151f")))

    def set_image(self, source: str | Path | QPixmap) -> bool:
        pixmap = source if isinstance(source, QPixmap) else QPixmap(str(source))
        if pixmap.isNull():
            return False
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_image()
        return True

    def clear_image(self) -> None:
        self.clear_rois()
        self._pixmap_item.setPixmap(QPixmap())

    def set_tool(self, tool: str) -> None:
        if tool not in {"pan", "rectangle", "lasso"}:
            raise ValueError("Canvas tool must be pan, rectangle, or lasso")
        self._tool = tool
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if tool == "pan"
            else QGraphicsView.DragMode.NoDrag
        )

    def fit_image(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(
                self._pixmap_item.boundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def add_rectangle(
        self,
        rect: QRectF,
        *,
        color: str = "#00d4ff",
        width: float = 2.0,
    ) -> QGraphicsRectItem:
        item = QGraphicsRectItem(rect.normalized())
        item.setPen(QPen(QColor(color), width))
        item.setZValue(10)
        self._scene.addItem(item)
        return item

    def add_lasso(
        self,
        points: Sequence[QPointF | Sequence[float]],
        *,
        color: str = "#00d4ff",
        width: float = 2.0,
    ) -> QGraphicsPathItem:
        clean = [
            (
                point
                if isinstance(point, QPointF)
                else QPointF(float(point[0]), float(point[1]))
            )
            for point in points
        ]
        path = QPainterPath()
        if clean:
            path.moveTo(clean[0])
            for point in clean[1:]:
                path.lineTo(point)
            path.closeSubpath()
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor(color), width))
        item.setZValue(10)
        self._scene.addItem(item)
        return item

    def clear_rois(self) -> None:
        for item in tuple(self._scene.items()):
            if item is not self._pixmap_item:
                self._scene.removeItem(item)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        point = self.mapToScene(event.position().toPoint())
        self.coordinates_changed.emit(point.x(), point.y())
        if self._tool == "rectangle" and self._origin is not None:
            if self._draft_rect is not None:
                self._draft_rect.setRect(QRectF(self._origin, point).normalized())
        elif self._tool == "lasso" and self._origin is not None:
            self._lasso_points.append(point)
            path = QPainterPath(self._lasso_points[0])
            for item in self._lasso_points[1:]:
                path.lineTo(item)
            if self._draft_path is not None:
                self._draft_path.setPath(path)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._tool != "pan":
            self._origin = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor("#00d4ff"), 2)
            if self._tool == "rectangle":
                self._draft_rect = self._scene.addRect(
                    QRectF(self._origin, self._origin),
                    pen,
                )
                self._draft_rect.setZValue(20)
            else:
                self._lasso_points = [self._origin]
                self._draft_path = self._scene.addPath(
                    QPainterPath(self._origin),
                    pen,
                )
                self._draft_path.setZValue(20)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            if self._tool == "rectangle" and self._draft_rect is not None:
                rect = self._draft_rect.rect().normalized()
                if rect.width() >= 2 and rect.height() >= 2:
                    self.roi_created.emit(
                        {
                            "type": "rectangle",
                            "geometry": {
                                "left": rect.left(),
                                "right": rect.right(),
                                "top": rect.top(),
                                "bottom": rect.bottom(),
                            },
                        }
                    )
                else:
                    self._scene.removeItem(self._draft_rect)
            elif self._tool == "lasso" and self._draft_path is not None:
                if len(self._lasso_points) >= 3:
                    path = self._draft_path.path()
                    path.closeSubpath()
                    self._draft_path.setPath(path)
                    self.roi_created.emit(
                        {
                            "type": "lasso",
                            "geometry": {
                                "points": [
                                    [point.x(), point.y()]
                                    for point in self._lasso_points
                                ]
                            },
                        }
                    )
                else:
                    self._scene.removeItem(self._draft_path)
            self._origin = None
            self._draft_rect = None
            self._draft_path = None
            self._lasso_points = []
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ArtifactBrowser(QWidget):
    """Preview common App 1.0 artifacts without opening another frontend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        self.empty = QLabel("No artifact selected")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(280, 220)
        self.image.setScaledContents(False)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.table = QTableWidget()
        for widget in (self.empty, self.image, self.text, self.table):
            self.stack.addWidget(widget)
        layout.addWidget(self.stack)
        self.current_path: Path | None = None

    def open_path(self, value: str | Path) -> bool:
        path = Path(value)
        if not path.is_file():
            self.empty.setText(f"Artifact does not exist:\n{path}")
            self.stack.setCurrentWidget(self.empty)
            return False
        self.current_path = path
        suffix = path.suffix.casefold()
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            pixmap = load_preview_pixmap(path, self.image.size())
            if not pixmap.isNull():
                target = self.image.size()
                self.image.setPixmap(
                    pixmap.scaled(
                        target,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.stack.setCurrentWidget(self.image)
                return True
        if suffix in {".csv", ".tsv"}:
            return self._open_table(path, "\t" if suffix == ".tsv" else ",")
        if suffix in {".json", ".txt", ".log", ".html"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if suffix == ".json":
                try:
                    text = json.dumps(
                        json.loads(text),
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                except json.JSONDecodeError:
                    pass
            self.text.setPlainText(text)
            self.stack.setCurrentWidget(self.text)
            return True
        self.text.setPlainText(f"{path.name}\n\nThis artifact is available at:\n{path}")
        self.stack.setCurrentWidget(self.text)
        return True

    def choose_and_open(
        self,
        *,
        directory: str | Path | None = None,
    ) -> bool:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open artifact",
            "" if directory is None else str(directory),
            "Artifacts (*.png *.jpg *.jpeg *.gif *.csv *.tsv *.json *.txt *.log "
            "*.html *.mp4 *.webm);;All files (*)",
        )
        return bool(selected) and self.open_path(selected)

    def _open_table(self, path: Path, delimiter: str) -> bool:
        with path.open(
            "r", encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            rows = list(csv.reader(handle, delimiter=delimiter))
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        else:
            self.table.setColumnCount(len(rows[0]))
            self.table.setHorizontalHeaderLabels(rows[0])
            body = rows[1:1001]
            self.table.setRowCount(len(body))
            for row_index, row in enumerate(body):
                for column, value in enumerate(row[: self.table.columnCount()]):
                    self.table.setItem(
                        row_index,
                        column,
                        QTableWidgetItem(value),
                    )
            self.table.resizeColumnsToContents()
        self.stack.setCurrentWidget(self.table)
        return True


def first_artifact(
    paths: Iterable[str | Path],
    suffixes: set[str] | None = None,
) -> Path | None:
    allowed = None if suffixes is None else {item.casefold() for item in suffixes}
    for value in paths:
        path = Path(value)
        if path.is_file() and (allowed is None or path.suffix.casefold() in allowed):
            return path
    return None


__all__ = [
    "ArtifactBrowser",
    "NativeModulePanel",
    "RunConfirmationDialog",
    "ScientificImageCanvas",
    "first_artifact",
    "load_preview_pixmap",
    "parse_confirmation_summary",
]
