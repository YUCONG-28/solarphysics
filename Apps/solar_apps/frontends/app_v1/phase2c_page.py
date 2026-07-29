# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 controls for Phase 2C retained workflows."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
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

from .phase2a import TaskLaunch
from .phase2c import Phase2CAdapter
from .basic_services import PlaybackController
from .components import ArtifactBrowser, NativeModulePanel


class Phase2CPanel(NativeModulePanel):
    """DART, drift, Newkirk, trajectory, and DEM controls."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2CAdapter,
        module_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            module_id,
            legacy_label=f"legacy {module_id.replace('-', ' ').title()}",
            parent=parent,
        )
        self.adapter = adapter
        self.module_id = module_id
        self.playback = PlaybackController(self)
        self.playback.frame_changed.connect(self._trajectory_frame_changed)
        self.playback.playing_changed.connect(self._trajectory_playing_changed)
        layout = QVBoxLayout(self)
        note_row = QHBoxLayout()
        note = QLabel(
            "Inputs are validated locally. Scientific work remains in the existing "
            "workflow and runs only after the confirmation summary is accepted."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        note_row.addWidget(note, 1)
        layout.addLayout(note_row)
        tabs = QTabWidget()
        if module_id == "dart-spectrogram":
            tabs.addTab(self._dart_tab(), "DART")
            tabs.addTab(self._drift_tab(), "Drift Rate")
            tabs.addTab(self._newkirk_tab(), "Newkirk")
        else:
            tabs.addTab(self._trajectory_tab(), "Trajectory")
            tabs.addTab(self._dem_tab(), "DEM Overlay")
        layout.addWidget(tabs, 1)
        self.artifacts = ArtifactBrowser()
        layout.addWidget(self.artifacts, 1)

    def _dart_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.dart_dir = QLineEdit()
        self.configure_path_field(self.dart_dir)
        form.addRow("DART folder", self._path_row(self.dart_dir, directory=True))
        self.dart_centers = QLineEdit("149, 164, 190")
        self.dart_centers.setPlaceholderText("Optional center frequencies")
        self.dart_bandwidth = QDoubleSpinBox()
        self.dart_bandwidth.setRange(0.001, 10000.0)
        self.dart_bandwidth.setValue(2.0)
        self.dart_bandwidth.setSuffix(" MHz")
        self.dart_display = QComboBox()
        self.dart_display.addItems(["db", "linear"])
        self.dart_samples = QSpinBox()
        self.dart_samples.setRange(32, 10000)
        self.dart_samples.setValue(1200)
        self.dart_dpi = QSpinBox()
        self.dart_dpi.setRange(72, 600)
        self.dart_dpi.setValue(150)
        form.addRow("Center frequencies", self.dart_centers)
        form.addRow("Bandwidth", self.dart_bandwidth)
        form.addRow("Display", self.dart_display)
        form.addRow("Maximum samples", self.dart_samples)
        form.addRow("DPI", self.dart_dpi)
        launch = QPushButton("Confirm and launch DART Spectrogram")
        launch.clicked.connect(self._request_dart)
        buttons.addWidget(launch)
        return page

    def _drift_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.t_start = QLineEdit("2025-01-24T04:48:30Z")
        self.t_end = QLineEdit("2025-01-24T04:48:35Z")
        self.f_start = QDoubleSpinBox()
        self.f_start.setRange(0.001, 100000.0)
        self.f_start.setValue(300.0)
        self.f_end = QDoubleSpinBox()
        self.f_end.setRange(0.001, 100000.0)
        self.f_end.setValue(149.0)
        form.addRow("Start time (UTC)", self.t_start)
        form.addRow("Start frequency (MHz)", self.f_start)
        form.addRow("End time (UTC)", self.t_end)
        form.addRow("End frequency (MHz)", self.f_end)
        launch = QPushButton("Confirm and calculate drift rate")
        launch.clicked.connect(self._request_drift)
        buttons.addWidget(launch)
        return page

    def _newkirk_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.gaussian_csv = QLineEdit()
        self.drift_csv = QLineEdit()
        self.configure_path_field(self.gaussian_csv)
        self.configure_path_field(self.drift_csv)
        form.addRow(
            "Gaussian diagnostics CSV",
            self._path_row(self.gaussian_csv, directory=False),
        )
        form.addRow(
            "Drift diagnostics CSV",
            self._path_row(self.drift_csv, directory=False),
        )
        launch = QPushButton("Confirm and run Newkirk diagnostics")
        launch.clicked.connect(self._request_newkirk)
        buttons.addWidget(launch)
        return page

    def _trajectory_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.centers = QLineEdit()
        self.aia_dir = QLineEdit()
        self.configure_path_field(self.centers)
        self.configure_path_field(self.aia_dir)
        self.tail_n = QSpinBox()
        self.tail_n.setRange(1, 10000)
        self.tail_n.setValue(5)
        self.trajectory_mode = QComboBox()
        self.trajectory_mode.addItems(["tail", "current", "all"])
        self.trajectory_theme = QComboBox()
        self.trajectory_theme.addItems(["light", "dark"])
        self.trajectory_width = QSpinBox()
        self.trajectory_width.setRange(320, 4096)
        self.trajectory_width.setValue(960)
        self.trajectory_height = QSpinBox()
        self.trajectory_height.setRange(240, 4096)
        self.trajectory_height.setValue(720)
        self.trajectory_max_frames = QSpinBox()
        self.trajectory_max_frames.setRange(1, 10000)
        self.trajectory_max_frames.setValue(300)
        self.trajectory_fps = QDoubleSpinBox()
        self.trajectory_fps.setRange(0.2, 120.0)
        self.trajectory_fps.setValue(6.0)
        self.trajectory_fps.setSuffix(" fps")
        self.trajectory_fps.valueChanged.connect(
            lambda value: self.playback.configure(fps=value)
        )
        self.trajectory_format = QComboBox()
        self.trajectory_format.addItems(["mp4", "gif", "webm"])
        form.addRow("Center table", self._path_row(self.centers, directory=False))
        form.addRow("Optional AIA folder", self._path_row(self.aia_dir, directory=True))
        form.addRow("Frame mode", self.trajectory_mode)
        form.addRow("Tail length", self.tail_n)
        form.addRow("Theme", self.trajectory_theme)
        form.addRow("Frame width", self.trajectory_width)
        form.addRow("Frame height", self.trajectory_height)
        form.addRow("Preview frame limit", self.trajectory_max_frames)
        form.addRow("Playback / export rate", self.trajectory_fps)
        form.addRow("Media format", self.trajectory_format)
        interactive = QPushButton("Confirm and prepare native playback")
        interactive.clicked.connect(self._request_trajectory)
        media = QPushButton("Confirm and export media")
        media.clicked.connect(self._request_trajectory_media)
        export = QPushButton("Confirm and export trajectory HTML")
        export.clicked.connect(self._request_trajectory_export)
        buttons.addWidget(interactive)
        buttons.addWidget(media)
        buttons.addWidget(export)

        playback = QHBoxLayout()
        previous = QPushButton("Previous")
        self.trajectory_play = QPushButton("Play")
        next_button = QPushButton("Next")
        previous.clicked.connect(self.playback.step_backward)
        self.trajectory_play.clicked.connect(self._toggle_trajectory_playback)
        next_button.clicked.connect(self.playback.step_forward)
        self.trajectory_status = QLabel("No native trajectory frames prepared")
        self.trajectory_status.setProperty("muted", True)
        playback.addWidget(previous)
        playback.addWidget(self.trajectory_play)
        playback.addWidget(next_button)
        playback.addWidget(self.trajectory_status, 1)
        form.addRow("Native playback", playback)
        return page

    def _dem_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.dem_aia = QLineEdit()
        self.dem_tb = QLineEdit()
        self.dem_radio = QLineEdit()
        self.configure_path_field(self.dem_aia)
        self.configure_path_field(self.dem_tb)
        self.configure_path_field(self.dem_radio)
        form.addRow("AIA FITS", self._path_row(self.dem_aia, directory=False))
        form.addRow("Tb data", self._path_row(self.dem_tb, directory=False))
        form.addRow("Radio FITS", self._path_row(self.dem_radio, directory=False))
        launch = QPushButton("Confirm and run DEM radio overlay")
        launch.clicked.connect(self._request_dem)
        buttons.addWidget(launch)
        return page

    @staticmethod
    def _page() -> tuple[QWidget, QFormLayout, QHBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return page, form, buttons

    def _path_row(self, field: QLineEdit, *, directory: bool) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse_into(field, directory=directory))
        layout.addWidget(browse)
        return container

    def _browse_into(self, field: QLineEdit, *, directory: bool) -> None:
        initial = field.text().strip()
        if not initial and self.adapter.allowed_roots:
            initial = str(self.adapter.allowed_roots[0])
        if directory:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select folder",
                initial,
                QFileDialog.Option.ShowDirsOnly,
            )
        else:
            selected, _filter = QFileDialog.getOpenFileName(
                self,
                "Select input file",
                initial,
                "Scientific data (*.csv *.xlsx *.fits *.fit *.fts *.npy *.npz);;"
                "All files (*)",
            )
        if selected:
            field.setText(selected)

    def _request_dart(self) -> None:
        self._build(
            lambda: self.adapter.build_dart_spectrogram(
                self.dart_dir.text(),
                center_frequencies=self.dart_centers.text(),
                bandwidth_mhz=self.dart_bandwidth.value(),
                display_mode=self.dart_display.currentText(),
                max_samples=self.dart_samples.value(),
                dpi=self.dart_dpi.value(),
            )
        )

    def _request_drift(self) -> None:
        self._build(
            lambda: self.adapter.build_drift_rate(
                t_start=self.t_start.text(),
                f_start_mhz=self.f_start.value(),
                t_end=self.t_end.text(),
                f_end_mhz=self.f_end.value(),
            )
        )

    def _request_newkirk(self) -> None:
        self._build(
            lambda: self.adapter.build_newkirk_diagnostics(
                gaussian_csv=self.gaussian_csv.text(),
                drift_csv=self.drift_csv.text(),
            )
        )

    def _request_trajectory(self) -> None:
        self.playback.set_frames(())
        self._build(
            lambda: self.adapter.build_source_trajectory(
                self.centers.text(),
                aia_dir=self.aia_dir.text(),
                frame_mode=self.trajectory_mode.currentText(),
                tail_n=self.tail_n.value(),
                width=self.trajectory_width.value(),
                height=self.trajectory_height.value(),
                theme=self.trajectory_theme.currentText(),
                max_frames=self.trajectory_max_frames.value(),
            )
        )

    def _request_trajectory_media(self) -> None:
        self._build(
            lambda: self.adapter.build_trajectory_media(
                self.centers.text(),
                aia_dir=self.aia_dir.text(),
                output_format=self.trajectory_format.currentText(),
                frame_mode=self.trajectory_mode.currentText(),
                tail_n=self.tail_n.value(),
                fps=self.trajectory_fps.value(),
                width=self.trajectory_width.value(),
                height=self.trajectory_height.value(),
                theme=self.trajectory_theme.currentText(),
            )
        )

    def _request_trajectory_export(self) -> None:
        self._build(
            lambda: self.adapter.build_trajectory_export(
                self.centers.text(),
                aia_dir=self.aia_dir.text(),
                tail_n=self.tail_n.value(),
            )
        )

    def _request_dem(self) -> None:
        self._build(
            lambda: self.adapter.build_dem_radio_overlay(
                aia_fits=self.dem_aia.text(),
                tb_data=self.dem_tb.text(),
                radio_file=self.dem_radio.text(),
            )
        )

    def _build(self, builder) -> None:  # type: ignore[no-untyped-def]
        try:
            launch = builder()
        except (OSError, ValueError) as exc:
            self.record_diagnostic(exc)
            QMessageBox.critical(self, "App 1.0 input error", str(exc))
            return
        self._confirm_and_emit(launch)

    def _confirm_and_emit(self, launch: TaskLaunch) -> None:
        if self.confirm(self, "Confirm Phase 2C task", launch.summary):
            self.task_requested.emit(launch)

    def handle_artifact(self, path: str) -> None:
        artifact = Path(path)
        if artifact.name.startswith("trajectory-frame-") and artifact.suffix == ".png":
            self.playback.add_frame(artifact)
            return
        if artifact.name == "trajectory-playback.json":
            return
        self.artifacts.open_path(path)

    def _trajectory_frame_changed(self, index: int, frame: object) -> None:
        path = Path(str(frame))
        self.artifacts.open_path(path)
        self.trajectory_status.setText(
            f"{index + 1} / {len(self.playback.frames)} — {path.name}"
        )

    def _toggle_trajectory_playback(self) -> None:
        if self.trajectory_play.text() == "Pause":
            self.playback.pause()
        else:
            self.playback.configure(fps=self.trajectory_fps.value())
            self.playback.play()

    def _trajectory_playing_changed(self, playing: bool) -> None:
        self.trajectory_play.setText("Pause" if playing else "Play")


__all__ = ["Phase2CPanel"]
