# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 controls for Phase 2B radio adapters."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .phase2a import TaskLaunch
from .phase2b import Phase2BAdapter
from .components import ArtifactBrowser, NativeModulePanel


class Phase2BPanel(NativeModulePanel):
    """Module-specific launch controls with mandatory confirmation."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2BAdapter,
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
        layout = QVBoxLayout(self)
        note_row = QHBoxLayout()
        note = QLabel(
            "This native page validates inputs and runs the retained scientific "
            "implementation in a supervised process."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        note_row.addWidget(note, 1)
        layout.addLayout(note_row)
        form = QFormLayout()
        layout.addLayout(form)
        self.primary = QLineEdit()
        self.configure_path_field(self.primary)
        form.addRow(self._primary_label(), self._path_row(self.primary))
        self.secondary: QLineEdit | None = None
        self.option: QComboBox | None = None
        self.roi_bounds: QLineEdit | None = None
        self.frequencies: QLineEdit | None = None

        if module_id == "radio-composite":
            self.secondary = QLineEdit()
            self.configure_path_field(self.secondary)
            form.addRow("DART folder", self._path_row(self.secondary))
        elif module_id == "source-map":
            self.option = QComboBox()
            self.option.addItems(["1", "2", "3"])
            form.addRow("Gaussian sources", self.option)
        elif module_id == "roi-lightcurve":
            self.option = QComboBox()
            self.option.addItems(["L+R", "LCP", "RCP", "all"])
            form.addRow("Polarization", self.option)
            self.roi_bounds = QLineEdit("-300,-300,300,300")
            self.roi_bounds.setToolTip(
                "HPLN/HPLT bounds: left,bottom,right,top in arcseconds"
            )
            self.frequencies = QLineEdit()
            self.frequencies.setPlaceholderText("All frequencies")
            form.addRow("ROI bounds (arcsec)", self.roi_bounds)
            form.addRow("Frequencies (MHz)", self.frequencies)

        buttons = QHBoxLayout()
        launch = QPushButton(self._launch_label())
        launch.clicked.connect(self._request_primary)
        buttons.addWidget(launch)
        if module_id == "source-map":
            gaussian = QPushButton("Confirm and run one-frame Gaussian fit")
            gaussian.clicked.connect(self._request_gaussian)
            buttons.addWidget(gaussian)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.artifacts = ArtifactBrowser()
        layout.addWidget(self.artifacts, 1)

    def _primary_label(self) -> str:
        return (
            "Allowed input root"
            if self.module_id == "bad-frame-review"
            else "Radio folder"
        )

    def _launch_label(self) -> str:
        names = {
            "bad-frame-review": "Confirm and launch reviewer",
            "source-map": "Confirm and launch Source Map",
            "roi-lightcurve": "Confirm and launch ROI Light Curve",
            "radio-composite": "Confirm and launch Radio Composite",
        }
        return names[self.module_id]

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

    def _request_primary(self) -> None:
        try:
            if self.module_id == "bad-frame-review":
                launch = self.adapter.build_bad_frame_review(self.primary.text())
            elif self.module_id == "source-map":
                launch = self.adapter.build_source_map_app(self.primary.text())
            elif self.module_id == "roi-lightcurve":
                launch = self.adapter.build_roi_lightcurve(
                    self.primary.text(),
                    polarization=self.option.currentText() if self.option else "L+R",
                    roi_bounds=(
                        self.roi_bounds.text()
                        if self.roi_bounds is not None
                        else "-300,-300,300,300"
                    ),
                    frequencies=(
                        self.frequencies.text() if self.frequencies is not None else ""
                    ),
                )
            else:
                launch = self.adapter.build_radio_composite(
                    self.primary.text(),
                    self.secondary.text() if self.secondary else "",
                )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)

    def _request_gaussian(self) -> None:
        try:
            launch = self.adapter.build_gaussian_fit(
                self.primary.text(),
                source_count=int(self.option.currentText()) if self.option else 1,
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)

    def _confirm_and_emit(self, launch: TaskLaunch) -> None:
        decision = self.confirm(self, "Confirm radio task", launch.summary)
        if decision:
            self.task_requested.emit(launch)

    def _show_error(self, message: str) -> None:
        self.record_diagnostic(message)
        QMessageBox.critical(self, "App 1.0 input error", message)

    def handle_artifact(self, path: str) -> None:
        self.artifacts.open_path(path)


__all__ = ["Phase2BPanel"]
