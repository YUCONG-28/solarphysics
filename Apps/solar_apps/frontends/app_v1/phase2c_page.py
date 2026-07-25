# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 controls for Phase 2C retained workflows."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
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


class Phase2CPanel(QWidget):
    """DART, drift, Newkirk, trajectory, and DEM controls."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2CAdapter,
        module_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.adapter = adapter
        self.module_id = module_id
        layout = QVBoxLayout(self)
        note = QLabel(
            "Inputs are validated locally. Scientific work remains in the existing "
            "workflow and runs only after the confirmation summary is accepted."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)
        tabs = QTabWidget()
        if module_id == "dart-spectrogram":
            tabs.addTab(self._dart_tab(), "DART")
            tabs.addTab(self._drift_tab(), "Drift Rate")
            tabs.addTab(self._newkirk_tab(), "Newkirk")
        else:
            tabs.addTab(self._trajectory_tab(), "Trajectory")
            tabs.addTab(self._dem_tab(), "DEM Overlay")
        layout.addWidget(tabs, 1)

    def _dart_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.dart_dir = QLineEdit()
        form.addRow("DART folder", self._path_row(self.dart_dir, directory=True))
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
        self.tail_n = QSpinBox()
        self.tail_n.setRange(1, 10000)
        self.tail_n.setValue(5)
        form.addRow("Center table", self._path_row(self.centers, directory=False))
        form.addRow("Optional AIA folder", self._path_row(self.aia_dir, directory=True))
        form.addRow("Static export tail length", self.tail_n)
        interactive = QPushButton("Confirm and launch trajectory app")
        interactive.clicked.connect(self._request_trajectory)
        export = QPushButton("Confirm and export trajectory HTML")
        export.clicked.connect(self._request_trajectory_export)
        buttons.addWidget(interactive)
        buttons.addWidget(export)
        return page

    def _dem_tab(self) -> QWidget:
        page, form, buttons = self._page()
        self.dem_aia = QLineEdit()
        self.dem_tb = QLineEdit()
        self.dem_radio = QLineEdit()
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
        self._build(lambda: self.adapter.build_dart_spectrogram(self.dart_dir.text()))

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
        self._build(
            lambda: self.adapter.build_source_trajectory(
                self.centers.text(),
                aia_dir=self.aia_dir.text(),
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
            QMessageBox.critical(self, "App 1.0 input error", str(exc))
            return
        self._confirm_and_emit(launch)

    def _confirm_and_emit(self, launch: TaskLaunch) -> None:
        decision = QMessageBox.question(
            self,
            "Confirm Phase 2C task",
            launch.summary,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if decision == QMessageBox.StandardButton.Yes:
            self.task_requested.emit(launch)


__all__ = ["Phase2CPanel"]
