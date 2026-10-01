# SPDX-License-Identifier: GPL-3.0-only
"""Image Composer export controls, confirmations, and draft ownership."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class ComposerExportControls:
    """Prepare adapter launches and transfer only confirmed drafts to the queue."""

    def _append_export_controls(self, layout: QVBoxLayout) -> None:
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
            self.adapter.confirm_prepared_export(launch)
            self.task_requested.emit(launch)
        else:
            self.adapter.discard_prepared_export(launch)


__all__ = ["ComposerExportControls"]
