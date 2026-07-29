# SPDX-License-Identifier: GPL-3.0-only
"""PyQt6 Phase 2A page for existing image, AIA, and HMI workflows."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .basic_services import PlaybackController
from .components import NativeModulePanel, load_preview_pixmap
from .phase2a import (
    ImageGroupSelection,
    ImageSequenceSelection,
    Phase2AAdapter,
    TaskLaunch,
)


class Phase2APanel(NativeModulePanel):
    """One native page that adapts, but never reimplements, Phase 2A science."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2AAdapter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "image-viewer",
            legacy_label="legacy Image Viewer",
            parent=parent,
        )
        self.adapter = adapter
        self._selection: ImageSequenceSelection | None = None
        self._group_selection: ImageGroupSelection | None = None
        self._image_index = 0
        self._preview_labels: list[QLabel] = []
        self.playback = PlaybackController(self)
        self.playback.frame_changed.connect(self._playback_frame_changed)
        layout = QVBoxLayout(self)
        note_row = QHBoxLayout()
        note = QLabel(
            "Phase 2A adapters call the existing AIA, HMI, and Image Viewer "
            "implementations. Inputs remain constrained to configured allowed roots."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        note_row.addWidget(note, 1)
        layout.addLayout(note_row)
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
        self.configure_path_field(self.viewer_folder)
        self.viewer_folder.setPlaceholderText("Select an image folder")
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(self.viewer_folder))
        add = QPushButton("Add folder")
        add.clicked.connect(self._add_viewer_folder)
        self.viewer_recursive = QCheckBox("Recursive")
        load = QPushButton("Confirm and load")
        load.clicked.connect(self._load_images)
        row.addWidget(self.viewer_folder, 1)
        row.addWidget(browse)
        row.addWidget(add)
        row.addWidget(self.viewer_recursive)
        row.addWidget(load)
        layout.addLayout(row)

        folder_row = QHBoxLayout()
        self.viewer_folders = QListWidget()
        self.viewer_folders.setMaximumHeight(90)
        self.viewer_folders.setToolTip(
            "Folders are synchronized by image index during preview and export."
        )
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_viewer_folders)
        folder_row.addWidget(self.viewer_folders, 1)
        folder_row.addWidget(remove)
        layout.addLayout(folder_row)

        self.viewer_preview_area = QScrollArea()
        self.viewer_preview_area.setWidgetResizable(True)
        self.viewer_preview_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.viewer_preview_container = QWidget()
        self.viewer_preview_layout = QHBoxLayout(self.viewer_preview_container)
        self.viewer_preview = self._new_preview_label("No image sequence loaded")
        self._preview_labels.append(self.viewer_preview)
        self.viewer_preview_layout.addWidget(self.viewer_preview, 1)
        self.viewer_preview_area.setWidget(self.viewer_preview_container)
        layout.addWidget(self.viewer_preview_area, 1)

        controls = QHBoxLayout()
        previous = QPushButton("Previous")
        self.viewer_play = QPushButton("Play")
        next_button = QPushButton("Next")
        previous.clicked.connect(self.playback.step_backward)
        self.viewer_play.clicked.connect(self._toggle_playback)
        next_button.clicked.connect(self.playback.step_forward)
        self.viewer_fps = QDoubleSpinBox()
        self.viewer_fps.setRange(0.2, 120.0)
        self.viewer_fps.setDecimals(1)
        self.viewer_fps.setValue(5.0)
        self.viewer_fps.setSuffix(" fps")
        self.viewer_fps.valueChanged.connect(self._update_playback_rate)
        self.viewer_status = QLabel("0 / 0")
        self.viewer_status.setProperty("muted", True)
        controls.addWidget(previous)
        controls.addWidget(self.viewer_play)
        controls.addWidget(next_button)
        controls.addWidget(self.viewer_fps)
        controls.addWidget(self.viewer_status)
        controls.addStretch(1)
        layout.addLayout(controls)

        export_row = QHBoxLayout()
        self.viewer_export_mode = QComboBox()
        self.viewer_export_mode.addItem("Side-by-side composite", "composite")
        self.viewer_export_mode.addItem("One file per folder", "separate")
        self.viewer_export_format = QComboBox()
        self.viewer_export_format.addItems(["mp4", "gif", "webm"])
        self.viewer_export_workers = QSpinBox()
        self.viewer_export_workers.setRange(1, 16)
        self.viewer_export_workers.setValue(1)
        export = QPushButton("Confirm and export media")
        export.clicked.connect(self._request_image_export)
        export_row.addWidget(QLabel("Export"))
        export_row.addWidget(self.viewer_export_mode)
        export_row.addWidget(self.viewer_export_format)
        export_row.addWidget(QLabel("Workers"))
        export_row.addWidget(self.viewer_export_workers)
        export_row.addWidget(export)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        return tab

    @staticmethod
    def _new_preview_label(text: str = "") -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(300, 260)
        label.setStyleSheet(
            "background: #10151f; color: #cbd5e1; border: 1px solid #334155;"
        )
        return label

    def _build_aia_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.aia_input = QLineEdit()
        self.configure_path_field(self.aia_input)
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
        self.aia_start = QSpinBox()
        self.aia_start.setRange(0, 10_000_000)
        self.aia_end = QSpinBox()
        self.aia_end.setRange(-1, 10_000_000)
        self.aia_end.setSpecialValueText("All")
        self.aia_end.setValue(-1)
        self.aia_test_index = QSpinBox()
        self.aia_test_index.setRange(0, 10_000_000)
        self.aia_test_index.setValue(0)
        self.aia_workers = QSpinBox()
        self.aia_workers.setRange(1, 64)
        self.aia_workers.setValue(1)
        run = QPushButton("Confirm and run AIA")
        run.clicked.connect(self._request_aia)
        form.addRow("AIA folder", input_row)
        form.addRow("Mode", self.aia_mode)
        form.addRow("Wavelengths", self.aia_waves)
        form.addRow("Start index", self.aia_start)
        form.addRow("End index", self.aia_end)
        form.addRow("Test index", self.aia_test_index)
        form.addRow("Workers", self.aia_workers)
        form.addRow("", run)
        return tab

    def _build_hmi_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self.hmi_aia_input = QLineEdit()
        self.hmi_input = QLineEdit()
        self.configure_path_field(self.hmi_aia_input)
        self.configure_path_field(self.hmi_input)
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
        folders = tuple(
            self.viewer_folders.item(index).text()
            for index in range(self.viewer_folders.count())
        )
        current = self.viewer_folder.text().strip()
        if not folders and current:
            folders = (current,)
        try:
            selection = self.adapter.select_image_groups(
                folders,
                recursive=self.viewer_recursive.isChecked(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        if not self._confirm("Confirm image loading", selection.summary):
            return
        self._group_selection = selection
        self._selection = selection.groups[0]
        self._image_index = 0
        self._rebuild_preview_labels()
        frame_count = max((len(group.images) for group in selection.groups), default=0)
        self.playback.configure(fps=self.viewer_fps.value())
        self.playback.set_frames(tuple(range(frame_count)))

    def _step_image(self, delta: int) -> None:
        if delta < 0:
            self.playback.step_backward()
        else:
            self.playback.step_forward()

    def _render_current_image(self) -> None:
        selection = self._group_selection
        if selection is None or not selection.groups:
            self.viewer_preview.setText("No supported images found")
            self.viewer_preview.setPixmap(QPixmap())
            self.viewer_status.setText("0 / 0")
            return
        frame_count = max((len(group.images) for group in selection.groups), default=0)
        names: list[str] = []
        for label, group in zip(self._preview_labels, selection.groups, strict=True):
            if self._image_index >= len(group.images):
                label.setText(f"{group.folder.name}\nNo frame {self._image_index + 1}")
                label.setPixmap(QPixmap())
                continue
            path = group.images[self._image_index]
            names.append(path.name)
            pixmap = load_preview_pixmap(path, label.size())
            if pixmap.isNull():
                label.setText(f"{group.folder.name}\nCould not decode {path.name}")
                label.setPixmap(QPixmap())
            else:
                label.setPixmap(
                    pixmap.scaled(
                        label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                label.setToolTip(str(path))
        self.viewer_status.setText(
            f"{self._image_index + 1} / {frame_count} — " + " | ".join(names)
        )

    def _add_viewer_folder(self) -> None:
        value = self.viewer_folder.text().strip()
        try:
            selected = self.adapter.validate_input_directory(value)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        existing = {
            self.viewer_folders.item(index).text()
            for index in range(self.viewer_folders.count())
        }
        if str(selected) not in existing:
            self.viewer_folders.addItem(str(selected))

    def _remove_viewer_folders(self) -> None:
        for item in self.viewer_folders.selectedItems():
            self.viewer_folders.takeItem(self.viewer_folders.row(item))

    def _rebuild_preview_labels(self) -> None:
        while self.viewer_preview_layout.count():
            item = self.viewer_preview_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._preview_labels = []
        selection = self._group_selection
        groups = selection.groups if selection is not None else ()
        for group in groups:
            label = self._new_preview_label(group.folder.name)
            label.setToolTip(str(group.folder))
            self._preview_labels.append(label)
            self.viewer_preview_layout.addWidget(label, 1)
        if not self._preview_labels:
            self._preview_labels.append(
                self._new_preview_label("No image sequence loaded")
            )
            self.viewer_preview_layout.addWidget(self._preview_labels[0], 1)
        self.viewer_preview = self._preview_labels[0]

    def _playback_frame_changed(self, index: int, _frame: object) -> None:
        self._image_index = index
        self._render_current_image()

    def _toggle_playback(self) -> None:
        if self.viewer_play.text() == "Pause":
            self.playback.pause()
            self.viewer_play.setText("Play")
            return
        self.playback.play()
        self.viewer_play.setText("Pause")

    def _update_playback_rate(self, value: float) -> None:
        self.playback.configure(fps=value)

    def _request_image_export(self) -> None:
        selection = self._group_selection
        if selection is None:
            self._show_error("Load at least one image folder before exporting")
            return
        try:
            launch = self.adapter.build_image_export(
                selection,
                output_format=self.viewer_export_format.currentText(),
                composite=self.viewer_export_mode.currentData() == "composite",
                fps=self.viewer_fps.value(),
                workers=self.viewer_export_workers.value(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)

    def handle_artifact(self, value: str) -> None:
        """Return generated AIA/HMI image products to this native page."""

        path = Path(value)
        if path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
            return
        pixmap = load_preview_pixmap(path, self.viewer_preview.size())
        if pixmap.isNull():
            return
        self.tabs.setCurrentIndex(0)
        self.viewer_preview.setPixmap(
            pixmap.scaled(
                self.viewer_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.viewer_status.setText(f"Generated artifact — {path.name}")

    def _request_aia(self) -> None:
        try:
            waves = tuple(
                int(item) for item in self.aia_waves.text().replace(",", " ").split()
            )
            launch = self.adapter.build_aia(
                self.aia_input.text(),
                mode=self.aia_mode.currentText(),
                waves=waves,
                start=self.aia_start.value(),
                end=None if self.aia_end.value() < 0 else self.aia_end.value(),
                test_index=self.aia_test_index.value(),
                workers=self.aia_workers.value(),
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
        return self.confirm(self, title, summary)

    def _show_error(self, message: str) -> None:
        from PyQt6.QtWidgets import QMessageBox

        self.record_diagnostic(message)
        QMessageBox.critical(self, "App 1.0 input error", message)


__all__ = ["Phase2APanel"]
