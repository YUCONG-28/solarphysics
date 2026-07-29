# SPDX-License-Identifier: GPL-3.0-only
"""Native PyQt6 controls for Phase 2B radio adapters."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
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
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
        self._review_manifest: Path | None = None
        self._review_payload: dict[str, object] | None = None

        if module_id == "radio-composite":
            self.secondary = QLineEdit()
            self.configure_path_field(self.secondary)
            form.addRow("DART folder", self._path_row(self.secondary))
            self.composite_frequencies = QLineEdit()
            self.composite_frequencies.setPlaceholderText(
                "All discovered radio frequencies"
            )
            self.composite_polarization = QComboBox()
            self.composite_polarization.addItems(["RR+LL", "RR", "LL"])
            self.composite_roi = QLineEdit("-300,-300,300,300")
            self.composite_bandwidth = QDoubleSpinBox()
            self.composite_bandwidth.setRange(0.001, 10000.0)
            self.composite_bandwidth.setValue(2.0)
            self.composite_bandwidth.setSuffix(" MHz")
            self.composite_fps = QDoubleSpinBox()
            self.composite_fps.setRange(0.2, 120.0)
            self.composite_fps.setValue(10.0)
            self.composite_fps.setSuffix(" fps")
            self.composite_stride = QSpinBox()
            self.composite_stride.setRange(1, 100000)
            self.composite_dpi = QSpinBox()
            self.composite_dpi.setRange(72, 600)
            self.composite_dpi.setValue(160)
            self.composite_transform = QComboBox()
            self.composite_transform.addItems(["linear", "log10"])
            self.composite_video = QCheckBox("Generate MP4")
            self.composite_video.setChecked(True)
            self.composite_frames = QCheckBox("Generate PNG sequence")
            self.composite_frames.setChecked(True)
            outputs = QWidget()
            output_layout = QHBoxLayout(outputs)
            output_layout.setContentsMargins(0, 0, 0, 0)
            output_layout.addWidget(self.composite_video)
            output_layout.addWidget(self.composite_frames)
            output_layout.addStretch(1)
            form.addRow("Frequencies (MHz)", self.composite_frequencies)
            form.addRow("Polarization", self.composite_polarization)
            form.addRow("ROI bounds (arcsec)", self.composite_roi)
            form.addRow("DART bandwidth", self.composite_bandwidth)
            form.addRow("Sequence FPS", self.composite_fps)
            form.addRow("Frame stride", self.composite_stride)
            form.addRow("DPI", self.composite_dpi)
            form.addRow("Map transform", self.composite_transform)
            form.addRow("Sequence products", outputs)
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
        elif module_id == "bad-frame-review":
            self.review_frequencies = QLineEdit()
            self.review_frequencies.setPlaceholderText("All discovered frequencies")
            self.review_polarizations = QLineEdit("RR,LL")
            self.review_start = QSpinBox()
            self.review_start.setRange(0, 10_000_000)
            self.review_end = QSpinBox()
            self.review_end.setRange(-1, 10_000_000)
            self.review_end.setSpecialValueText("All")
            self.review_end.setValue(-1)
            self.review_strategy = QComboBox()
            self.review_strategy.addItems(["rules", "labeling", "shadow"])
            self.review_scope = QComboBox()
            self.review_scope.addItems(["candidates", "all_scanned"])
            self.review_sample = QSpinBox()
            self.review_sample.setRange(1, 1_000_000)
            self.review_sample.setValue(1200)
            form.addRow("Frequencies (MHz)", self.review_frequencies)
            form.addRow("Polarizations", self.review_polarizations)
            form.addRow("Start index", self.review_start)
            form.addRow("End index", self.review_end)
            form.addRow("Sampling strategy", self.review_strategy)
            form.addRow("Review scope", self.review_scope)
            form.addRow("Labeling sample count", self.review_sample)

        buttons = QHBoxLayout()
        launch = QPushButton(self._launch_label())
        launch.clicked.connect(self._request_primary)
        buttons.addWidget(launch)
        if module_id == "source-map":
            gaussian = QPushButton("Confirm and run one-frame Gaussian fit")
            gaussian.clicked.connect(self._request_gaussian)
            buttons.addWidget(gaussian)
        elif module_id == "bad-frame-review":
            restore = QPushButton("Open existing review…")
            restore.clicked.connect(self._open_review_manifest)
            buttons.addWidget(restore)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        if module_id == "bad-frame-review":
            self._build_review_controls(layout)
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
                launch = self.adapter.build_bad_frame_review(
                    self.primary.text(),
                    frequencies=self.review_frequencies.text(),
                    polarizations=self.review_polarizations.text(),
                    start_index=self.review_start.value(),
                    end_index=(
                        None if self.review_end.value() < 0 else self.review_end.value()
                    ),
                    strategy=self.review_strategy.currentText(),
                    scope=self.review_scope.currentText(),
                    sample_count=self.review_sample.value(),
                )
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
                    frequencies=self.composite_frequencies.text(),
                    polarization=self.composite_polarization.currentText(),
                    roi_bounds=self.composite_roi.text(),
                    dart_bandwidth_mhz=self.composite_bandwidth.value(),
                    fps=self.composite_fps.value(),
                    stride=self.composite_stride.value(),
                    dpi=self.composite_dpi.value(),
                    transform=self.composite_transform.currentText(),
                    save_video=self.composite_video.isChecked(),
                    save_frames=self.composite_frames.isChecked(),
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
        artifact = Path(path)
        if self.module_id == "bad-frame-review" and artifact.name == "review.json":
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._show_error(f"Could not restore review: {exc}")
            else:
                if isinstance(payload, dict):
                    self._load_review_payload(artifact, payload)
        self.artifacts.open_path(path)

    def _build_review_controls(self, layout: QVBoxLayout) -> None:
        self.review_items = QListWidget()
        self.review_items.setMinimumHeight(150)
        layout.addWidget(self.review_items, 1)
        label_row = QHBoxLayout()
        self.review_quality = QComboBox()
        self.review_quality.addItems(["good", "degraded", "bad", "uncertain"])
        self.review_event_tags = QLineEdit()
        self.review_event_tags.setPlaceholderText(
            "Event tags, e.g. solar_burst,strong_polarization"
        )
        self.review_artifact_tags = QLineEdit()
        self.review_artifact_tags.setPlaceholderText(
            "Artifact tags, e.g. stripe,sidelobe,noise"
        )
        self.review_cmap = QComboBox()
        self.review_cmap.addItems(
            [
                "coolwarm",
                "hot",
                "inferno",
                "magma",
                "viridis",
                "plasma",
                "jet",
                "cividis",
            ]
        )
        self.review_transform = QComboBox()
        self.review_transform.addItems(["robust_asinh", "linear"])
        label_row.addWidget(QLabel("Quality"))
        label_row.addWidget(self.review_quality)
        label_row.addWidget(self.review_event_tags, 1)
        label_row.addWidget(self.review_artifact_tags, 1)
        label_row.addWidget(self.review_cmap)
        label_row.addWidget(self.review_transform)
        layout.addLayout(label_row)
        action_row = QHBoxLayout()
        preview = QPushButton("Preview selected")
        preview.clicked.connect(lambda: self._request_review_action("preview"))
        label = QPushButton("Save selected label")
        label.clicked.connect(lambda: self._request_review_action("label"))
        self.review_final_status = QComboBox()
        self.review_final_status.addItems(["completed", "skipped"])
        finalize = QPushButton("Finalize review")
        finalize.clicked.connect(lambda: self._request_review_action("finalize"))
        archive = QPushButton("Export JSON/CSV/Audit ZIP")
        archive.clicked.connect(lambda: self._request_review_action("archive"))
        action_row.addWidget(preview)
        action_row.addWidget(label)
        action_row.addWidget(self.review_final_status)
        action_row.addWidget(finalize)
        action_row.addWidget(archive)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        self.review_status = QLabel(
            "Create a review or open an existing review manifest."
        )
        self.review_status.setWordWrap(True)
        self.review_status.setProperty("muted", True)
        layout.addWidget(self.review_status)

    def _open_review_manifest(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open bad-frame review",
            str(self.adapter.runtime.outputs_dir),
            "Bad-frame review (review.json);;JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve(strict=False)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._show_error(f"Could not open review: {exc}")
            return
        if not isinstance(payload, dict) or payload.get("kind") != (
            "radio-bad-frame-review"
        ):
            self._show_error("The selected file is not a radio bad-frame review")
            return
        self._load_review_payload(path, payload)
        self.artifacts.open_path(path)

    def _load_review_payload(
        self,
        path: Path,
        payload: dict[str, object],
    ) -> None:
        self._review_manifest = path
        self._review_payload = payload
        self.review_items.clear()
        review_input = payload.get("input")
        scope = (
            str(review_input.get("review_scope", "candidates"))
            if isinstance(review_input, dict)
            else "candidates"
        )
        candidates = payload.get("candidates")
        candidate_by_file = (
            {
                str(item.get("file_id")): item
                for item in candidates
                if isinstance(item, dict)
            }
            if isinstance(candidates, list)
            else {}
        )
        records = payload.get("files") if scope == "all_scanned" else candidates
        for raw in records if isinstance(records, list) else []:
            if not isinstance(raw, dict):
                continue
            kind = "frame" if scope == "all_scanned" else "candidate"
            target_id = raw.get("file_id" if kind == "frame" else "candidate_id")
            candidate = candidate_by_file.get(str(raw.get("file_id")), raw)
            human = candidate.get("human_label")
            quality = (
                human.get("quality_label") if isinstance(human, dict) else "pending"
            )
            item = QListWidgetItem(
                f"{quality} · {raw.get('frequency_mhz', '?')} MHz · "
                f"{raw.get('polarization', '?')} · "
                f"{raw.get('relative_path', target_id)}"
            )
            item.setData(
                256,
                {"kind": kind, "id": str(target_id)},
            )
            self.review_items.addItem(item)
        if self.review_items.count():
            self.review_items.setCurrentRow(0)
        summary = payload.get("summary")
        self.review_status.setText(
            f"Review {payload.get('review_id')} · {payload.get('status')} · "
            f"{self.review_items.count()} visible item(s) · summary={summary}"
        )

    def _request_review_action(self, action: str) -> None:
        payload = self._review_payload
        manifest = self._review_manifest
        if payload is None or manifest is None:
            self._show_error("Create or open a bad-frame review first")
            return
        item = self.review_items.currentItem()
        target = item.data(256) if item is not None else None
        if action in {"preview", "label"} and not isinstance(target, dict):
            self._show_error("Select a review item first")
            return
        try:
            launch = self.adapter.build_bad_frame_action(
                manifest.parent.parent,
                str(payload.get("review_id")),
                action=action,
                target_kind=(str(target["kind"]) if isinstance(target, dict) else None),
                target_id=str(target["id"]) if isinstance(target, dict) else None,
                quality=self.review_quality.currentText(),
                event_tags=self.review_event_tags.text(),
                artifact_tags=self.review_artifact_tags.text(),
                final_status=self.review_final_status.currentText(),
                cmap=self.review_cmap.currentText(),
                transform=self.review_transform.currentText(),
            )
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self._confirm_and_emit(launch)


__all__ = ["Phase2BPanel"]
