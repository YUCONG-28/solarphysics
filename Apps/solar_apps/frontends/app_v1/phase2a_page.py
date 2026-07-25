# SPDX-License-Identifier: GPL-3.0-only
"""PyQt6 Phase 2A page for existing image, AIA, and HMI workflows."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .phase2a import ImageSequenceSelection, Phase2AAdapter, TaskLaunch


class Phase2APanel(QWidget):
    """One native page that adapts, but never reimplements, Phase 2A science."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2AAdapter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.adapter = adapter
        self._selection: ImageSequenceSelection | None = None
        self._image_index = 0
        layout = QVBoxLayout(self)
        note = QLabel(
            "Phase 2A adapters call the existing AIA, HMI, and Image Viewer "
            "implementations. Inputs remain constrained to configured allowed roots."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_viewer_tab(), "Image Viewer")
        self.tabs.addTab(self._build_aia_tab(), "AIA Processing")
        self.tabs.addTab(self._build_hmi_tab(), "HMI Overlay")
        layout.addWidget(self.tabs, 1)

    def _build_viewer_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        row = QHBoxLayout()
        self.viewer_folder = QLineEdit()
        self.viewer_folder.setPlaceholderText("Select an image folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(self.viewer_folder))
        self.viewer_recursive = QCheckBox("Recursive")
        load = QPushButton("Confirm and load")
        load.clicked.connect(self._load_images)
        row.addWidget(self.viewer_folder, 1)
        row.addWidget(browse)
        row.addWidget(self.viewer_recursive)
        row.addWidget(load)
        layout.addLayout(row)

        self.viewer_preview = QLabel("No image sequence loaded")
        self.viewer_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer_preview.setMinimumSize(480, 300)
        self.viewer_preview.setStyleSheet(
            "background: #10151f; color: #cbd5e1; border: 1px solid #334155;"
        )
        layout.addWidget(self.viewer_preview, 1)
        controls = QHBoxLayout()
        previous = QPushButton("Previous")
        next_button = QPushButton("Next")
        previous.clicked.connect(lambda: self._step_image(-1))
        next_button.clicked.connect(lambda: self._step_image(1))
        self.viewer_status = QLabel("0 / 0")
        self.viewer_status.setProperty("muted", True)
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(self.viewer_status)
        controls.addStretch(1)
        layout.addLayout(controls)
        return tab

    def _build_aia_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.aia_input = QLineEdit()
        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(self.aia_input, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(self.aia_input))
        input_layout.addWidget(browse)
        self.aia_mode = QComboBox()
        self.aia_mode.addItems(["test", "single", "mosaic"])
        self.aia_waves = QLineEdit("171")
        self.aia_waves.setPlaceholderText("171 193 304")
        run = QPushButton("Confirm and run AIA")
        run.clicked.connect(self._request_aia)
        form.addRow("AIA folder", input_row)
        form.addRow("Mode", self.aia_mode)
        form.addRow("Wavelengths", self.aia_waves)
        form.addRow("", run)
        return tab

    def _build_hmi_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.hmi_aia_input = QLineEdit()
        self.hmi_input = QLineEdit()
        form.addRow("AIA folder", self._path_row(self.hmi_aia_input))
        form.addRow("HMI folder", self._path_row(self.hmi_input))
        self.hmi_dpi = QSpinBox()
        self.hmi_dpi.setRange(72, 1200)
        self.hmi_dpi.setValue(300)
        self.hmi_tolerance = QSpinBox()
        self.hmi_tolerance.setRange(0, 3600)
        self.hmi_tolerance.setValue(24)
        self.hmi_tolerance.setSuffix(" s")
        run = QPushButton("Confirm and run overlay")
        run.clicked.connect(self._request_hmi)
        form.addRow("DPI", self.hmi_dpi)
        form.addRow("Time tolerance", self.hmi_tolerance)
        form.addRow("", run)
        return tab

    def _path_row(self, field: QLineEdit) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(field))
        layout.addWidget(browse)
        return container

    def _browse_into(self, field: QLineEdit) -> None:
        initial = field.text().strip()
        if not initial and self.adapter.allowed_roots:
            initial = str(self.adapter.allowed_roots[0])
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            initial,
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            field.setText(selected)

    def _load_images(self) -> None:
        try:
            selection = self.adapter.select_images(
                self.viewer_folder.text(),
                recursive=self.viewer_recursive.isChecked(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        if not self._confirm("Confirm image loading", selection.summary):
            return
        self._selection = selection
        self._image_index = 0
        self._render_current_image()

    def _step_image(self, delta: int) -> None:
        if self._selection is None or not self._selection.images:
            return
        self._image_index = (self._image_index + delta) % len(self._selection.images)
        self._render_current_image()

    def _render_current_image(self) -> None:
        selection = self._selection
        if selection is None or not selection.images:
            self.viewer_preview.setText("No supported images found")
            self.viewer_preview.setPixmap(QPixmap())
            self.viewer_status.setText("0 / 0")
            return
        path = selection.images[self._image_index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.viewer_preview.setText(f"Could not decode {path.name}")
            self.viewer_preview.setPixmap(QPixmap())
        else:
            self.viewer_preview.setPixmap(
                pixmap.scaled(
                    self.viewer_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.viewer_status.setText(
            f"{self._image_index + 1} / {len(selection.images)} — {path.name}"
        )

    def _request_aia(self) -> None:
        try:
            waves = tuple(
                int(item) for item in self.aia_waves.text().replace(",", " ").split()
            )
            launch = self.adapter.build_aia(
                self.aia_input.text(),
                mode=self.aia_mode.currentText(),
                waves=waves,
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)

    def _request_hmi(self) -> None:
        try:
            launch = self.adapter.build_hmi_overlay(
                self.hmi_aia_input.text(),
                self.hmi_input.text(),
                dpi=self.hmi_dpi.value(),
                max_time_diff_seconds=float(self.hmi_tolerance.value()),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)

    def _confirm_and_emit(self, launch: TaskLaunch) -> None:
        if self._confirm("Confirm scientific task", launch.summary):
            self.task_requested.emit(launch)

    def _confirm(self, title: str, summary: str) -> bool:
        decision = QMessageBox.question(
            self,
            title,
            summary,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return decision == QMessageBox.StandardButton.Yes

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "App 1.0 input error", message)


__all__ = ["Phase2APanel"]
