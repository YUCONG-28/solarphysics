# SPDX-License-Identifier: GPL-3.0-only
"""Single shared implementations of App 1.0 foundation capabilities."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QWidget,
)


def validate_allowed_path(
    value: str | Path,
    allowed_roots: Sequence[str | Path],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve one path and prove it is below an explicitly allowed root."""

    candidate = Path(value).expanduser().resolve(strict=False)
    roots = tuple(Path(item).expanduser().resolve(strict=False) for item in allowed_roots)
    if not roots:
        raise ValueError("At least one allowed root must be configured")
    if not any(candidate == root or candidate.is_relative_to(root) for root in roots):
        raise ValueError(f"Path is outside the configured allowed roots: {candidate}")
    if must_exist and not candidate.exists():
        raise ValueError(f"Path does not exist: {candidate}")
    return candidate


class AllowedPathField(QWidget):
    """The only App 1.0 path picker, constrained to configured roots."""

    path_changed = pyqtSignal(str)

    def __init__(
        self,
        allowed_roots: Sequence[str | Path],
        *,
        directory: bool = False,
        extensions: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.allowed_roots = tuple(Path(item) for item in allowed_roots)
        self.directory = bool(directory)
        self.extensions = tuple(str(item) for item in extensions)
        self._last_directory: Path | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit()
        self.line_edit.setMinimumWidth(320)
        self.line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.line_edit.textChanged.connect(self._text_changed)
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self.browse)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, value: str) -> None:  # noqa: N802 - Qt compatibility
        self.line_edit.setText(str(value))

    def validated_path(self, *, must_exist: bool = False) -> Path:
        return validate_allowed_path(
            self.text(),
            self.allowed_roots,
            must_exist=must_exist,
        )

    def browse(self) -> None:
        initial = str(
            self._last_directory
            or (self.allowed_roots[0] if self.allowed_roots else Path())
        )
        if self.directory:
            selected = QFileDialog.getExistingDirectory(self, "Select folder", initial)
        else:
            file_filter = "All files (*)"
            if self.extensions:
                patterns = " ".join(f"*{item}" for item in self.extensions)
                file_filter = f"Supported files ({patterns});;All files (*)"
            selected, _chosen = QFileDialog.getOpenFileName(
                self,
                "Select file",
                initial,
                file_filter,
            )
        if not selected:
            return
        try:
            path = validate_allowed_path(selected, self.allowed_roots, must_exist=True)
        except ValueError:
            return
        self._last_directory = path if path.is_dir() else path.parent
        self.setText(str(path))

    def _text_changed(self, value: str) -> None:
        self.line_edit.setToolTip(value)
        self.path_changed.emit(value)


def validate_roi(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared rectangle/lasso ROI schema."""

    roi_type = str(value.get("type", "")).strip()
    geometry = value.get("geometry")
    if roi_type not in {"rectangle", "lasso"} or not isinstance(geometry, Mapping):
        raise ValueError("ROI must be a rectangle or lasso with geometry")
    clean_geometry: dict[str, Any]
    if roi_type == "rectangle":
        names = ("left", "right", "top", "bottom")
        try:
            clean_geometry = {name: float(geometry[name]) for name in names}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Rectangle ROI requires finite bounds") from exc
        if clean_geometry["left"] >= clean_geometry["right"]:
            raise ValueError("Rectangle ROI left must be below right")
        if clean_geometry["top"] == clean_geometry["bottom"]:
            raise ValueError("Rectangle ROI top and bottom must differ")
    else:
        points = geometry.get("points")
        if not isinstance(points, Sequence) or len(points) < 3:
            raise ValueError("Lasso ROI requires at least three points")
        try:
            clean_geometry = {
                "points": [[float(point[0]), float(point[1])] for point in points]
            }
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Invalid lasso ROI point") from exc
    clean = dict(value)
    clean.update(
        {
            "schema_version": 1,
            "type": roi_type,
            "geometry": clean_geometry,
        }
    )
    return clean


class RoiController(QObject):
    """The only ROI state/history/import/export controller."""

    changed = pyqtSignal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[dict[str, Any]] = []
        self._history: list[list[dict[str, Any]]] = []
        self._future: list[list[dict[str, Any]]] = []

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._items)

    def add(self, value: Mapping[str, Any]) -> dict[str, Any]:
        clean = validate_roi(value)
        self._remember()
        self._items.append(clean)
        self.changed.emit(self.items)
        return clean

    def replace(self, values: Sequence[Mapping[str, Any]]) -> None:
        clean = [validate_roi(item) for item in values]
        self._remember()
        self._items = clean
        self.changed.emit(self.items)

    def clear(self) -> None:
        if not self._items:
            return
        self._remember()
        self._items.clear()
        self.changed.emit(self.items)

    def undo(self) -> tuple[dict[str, Any], ...] | None:
        if not self._history:
            return None
        self._future.append(self._copy(self._items))
        self._items = self._history.pop()
        self.changed.emit(self.items)
        return self.items

    def redo(self) -> tuple[dict[str, Any], ...] | None:
        if not self._future:
            return None
        self._history.append(self._copy(self._items))
        self._items = self._future.pop()
        self.changed.emit(self.items)
        return self.items

    def load(self, path: str | Path) -> tuple[dict[str, Any], ...]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("rois", ()) if isinstance(payload, Mapping) else payload
        if not isinstance(values, Sequence):
            raise TypeError("ROI document must contain a list")
        self._remember()
        self._items = [validate_roi(item) for item in values]
        self.changed.emit(self.items)
        return self.items

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"schema_version": 1, "rois": self._items},
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def _remember(self) -> None:
        self._history.append(self._copy(self._items))
        del self._history[:-100]
        self._future.clear()

    @staticmethod
    def _copy(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return json.loads(json.dumps(list(values), allow_nan=False))


class PlaybackController(QObject):
    """The only frame cursor and playback timer implementation."""

    frame_changed = pyqtSignal(int, object)
    playing_changed = pyqtSignal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frames: tuple[Any, ...] = ()
        self._index = -1
        self._fps = 5.0
        self._loop = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step_forward)

    def set_frames(self, frames: Sequence[Any]) -> None:
        self.pause()
        self._frames = tuple(frames)
        self._index = 0 if self._frames else -1
        if self._index >= 0:
            self.frame_changed.emit(self._index, self._frames[self._index])

    @property
    def frames(self) -> tuple[Any, ...]:
        return self._frames

    @property
    def index(self) -> int:
        return self._index

    def add_frame(
        self,
        frame: Any,
        *,
        select: bool = True,
        emit: bool = True,
    ) -> int:
        if frame not in self._frames:
            self._frames = (*self._frames, frame)
        index = self._frames.index(frame)
        if select:
            self._index = index
            if emit:
                self.frame_changed.emit(index, frame)
        return index

    def configure(self, *, fps: float, loop: bool = True) -> None:
        if not 0.1 <= float(fps) <= 120:
            raise ValueError("Frame rate must be between 0.1 and 120 fps")
        self._fps = float(fps)
        self._loop = bool(loop)
        if self._timer.isActive():
            self._timer.start(round(1000 / self._fps))

    def play(self) -> None:
        if not self._frames:
            return
        self._timer.start(round(1000 / self._fps))
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playing_changed.emit(False)

    def step_forward(self) -> None:
        if not self._frames:
            return
        next_index = self._index + 1
        if next_index >= len(self._frames):
            if not self._loop:
                self.pause()
                return
            next_index = 0
        self._select(next_index)

    def step_backward(self) -> None:
        if self._frames:
            self._select((self._index - 1) % len(self._frames))

    def _select(self, index: int) -> None:
        self._index = int(index)
        self.frame_changed.emit(self._index, self._frames[self._index])


class ArtifactExportService:
    """The only typed artifact copier/serializer used by App 1.0."""

    SUPPORTED_FORMATS = frozenset(
        {"png", "csv", "json", "gif", "mp4", "webm", "zip"}
    )

    def export(
        self,
        payload: Any,
        target: str | Path,
        *,
        format_name: str,
    ) -> Path:
        format_id = format_name.lower().lstrip(".")
        if format_id not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported export format: {format_name}")
        destination = Path(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if format_id == "json":
            destination.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
        elif format_id == "csv":
            rows = list(payload)
            with destination.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)
        elif format_id == "zip":
            source = Path(payload)
            archive = shutil.make_archive(
                str(destination.with_suffix("")),
                "zip",
                root_dir=source,
            )
            destination = Path(archive)
        else:
            source = Path(payload)
            if source.resolve(strict=False) != destination.resolve(strict=False):
                shutil.copy2(source, destination)
        return destination


__all__ = [
    "AllowedPathField",
    "ArtifactExportService",
    "PlaybackController",
    "RoiController",
    "validate_allowed_path",
    "validate_roi",
]
