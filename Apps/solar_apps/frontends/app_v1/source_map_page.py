# SPDX-License-Identifier: GPL-3.0-only
"""Native Source Map workspace for App 1.0."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPainter
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
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from solar_apps.workflows.radio.artifacts import (
    image_pixel_to_data,
    sidecar_path_for,
    validate_roi_set,
)

from .components import NativeModulePanel, ScientificImageCanvas
from .basic_services import PlaybackController, RoiController
from .phase2a import TaskLaunch
from .phase2b import Phase2BAdapter


class SourceMapNativePanel(NativeModulePanel):
    """Complete in-window Source Map preparation, rendering, and ROI surface."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: Phase2BAdapter,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            "source-map",
            legacy_label="legacy Source Map",
            parent=parent,
        )
        self.adapter = adapter
        self._discovery_file: Path | None = None
        self._metadata: dict[str, Any] | None = None
        self._image_path: Path | None = None
        self.playback = PlaybackController(self)
        self.playback.frame_changed.connect(
            lambda _index, path: self._load_image(Path(path))
        )
        self.roi_controller = RoiController(self)
        self.roi_controller.changed.connect(lambda _items: self._refresh_roi_list())

        root = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._configuration_panel())
        splitter.addWidget(self._canvas_panel())
        splitter.addWidget(self._roi_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 520, 190])
        root.addWidget(splitter, 1)

    def _configuration_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)

        self.config_name = QLineEdit(
            "solar_apps.workflows.radio.configs.radio_20250124_config"
        )
        self.config_name.setCursorPosition(0)
        self.source_path = QLineEdit()
        self.output_path = QLineEdit()
        for field in (self.config_name, self.source_path, self.output_path):
            self.configure_path_field(field)
            field.setMinimumWidth(135)
        form.addRow("Event config", self.config_name)
        form.addRow("Source", self._source_path_row())
        form.addRow("Output folder", self._directory_row(self.output_path))

        self.mode = QComboBox()
        self.mode.addItem("Single band", "single_band")
        self.mode.addItem("Multi band", "multi_band")
        self.mode.setMinimumContentsLength(14)
        self.frequencies = QLineEdit("149, 164, 190")
        self.polarization = QComboBox()
        self.polarization.addItems(["RR+LL", "RR", "LL"])
        self.polarization.setMinimumContentsLength(8)
        form.addRow("Map mode", self.mode)
        form.addRow("Frequencies (MHz)", self.frequencies)
        form.addRow("Polarization", self.polarization)
        self.input_start = QSpinBox()
        self.input_start.setRange(0, 10_000_000)
        self.input_start.setValue(0)
        self.input_end = QSpinBox()
        self.input_end.setRange(-1, 10_000_000)
        self.input_end.setSpecialValueText("All")
        self.input_end.setValue(-1)
        form.addRow("Input start index", self.input_start)
        form.addRow("Input end index", self.input_end)

        self.cmap = QComboBox()
        self.cmap.addItems(
            ["hot", "inferno", "magma", "viridis", "plasma", "jet", "cividis"]
        )
        self.cmap.setMinimumContentsLength(10)
        self.range_mode = QComboBox()
        self.range_mode.addItems(["auto", "fixed", "global"])
        self.range_mode.setMinimumContentsLength(10)
        self.fixed_min = QDoubleSpinBox()
        self.fixed_min.setRange(-1e30, 1e30)
        self.fixed_min.setDecimals(6)
        self.fixed_max = QDoubleSpinBox()
        self.fixed_max.setRange(-1e30, 1e30)
        self.fixed_max.setDecimals(6)
        self.fixed_max.setValue(1.0)
        self.gaussian_overlay = QCheckBox("Gaussian overlay")
        self.gaussian_overlay.setChecked(True)
        self.spectrogram = QCheckBox("Spectrogram panel")
        form.addRow("Color map", self.cmap)
        form.addRow("Color range", self.range_mode)
        form.addRow("Fixed minimum", self.fixed_min)
        form.addRow("Fixed maximum", self.fixed_max)
        form.addRow("", self.gaussian_overlay)
        form.addRow("", self.spectrogram)

        self.background = QComboBox()
        self.background.addItems(
            ["off", "noise_map_only", "local_mesh", "local_median"]
        )
        self.background.setMinimumContentsLength(16)
        self.background_display = QCheckBox("Use background for display")
        self.background_fit = QCheckBox("Use background for fit")
        self.advanced = QTextEdit("{}")
        self.advanced.setMaximumHeight(100)
        form.addRow("Background", self.background)
        form.addRow("", self.background_display)
        form.addRow("", self.background_fit)
        form.addRow("Advanced JSON", self.advanced)

        self.discover_button = QPushButton("Discover")
        self.discover_button.setProperty("primary", True)
        self.discover_button.clicked.connect(self._request_discovery)
        self.candidate = QComboBox()
        self.candidate.setEnabled(False)
        self.candidate.setMinimumContentsLength(22)
        self.render_button = QPushButton("Render current")
        self.render_button.setEnabled(False)
        self.render_button.clicked.connect(self._request_render)
        self.sequence_button = QPushButton("Prepare sequence")
        self.sequence_button.setEnabled(False)
        self.sequence_button.clicked.connect(self._request_sequence)
        self.sequence_start = QSpinBox()
        self.sequence_start.setMinimum(1)
        self.sequence_end = QSpinBox()
        self.sequence_end.setMinimum(1)
        form.addRow("", self.discover_button)
        form.addRow("Candidate", self.candidate)
        form.addRow("Start frame", self.sequence_start)
        form.addRow("End frame", self.sequence_end)
        form.addRow("", self.render_button)
        form.addRow("", self.sequence_button)
        self.status = QLabel("No candidates discovered")
        self.status.setWordWrap(True)
        self.status.setProperty("muted", True)
        form.addRow("Status", self.status)
        scroll.setWidget(content)
        scroll.setMinimumWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    def _source_path_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.source_path, 1)
        file_button = QPushButton("File…")
        file_button.clicked.connect(self._choose_source_file)
        folder_button = QPushButton("Folder…")
        folder_button.clicked.connect(lambda: self._choose_directory(self.source_path))
        layout.addWidget(file_button)
        layout.addWidget(folder_button)
        return row

    def _directory_row(self, field: QLineEdit) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda: self._choose_directory(field))
        layout.addWidget(button)
        return row

    def _canvas_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        drawing_tools = QHBoxLayout()
        self.rectangle_button = QPushButton("Rectangle")
        self.lasso_button = QPushButton("Lasso")
        self.pan_button = QPushButton("Pan")
        self.fit_button = QPushButton("Fit")
        self.open_button = QPushButton("Open artifact…")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.rectangle_button.clicked.connect(lambda: self.canvas.set_tool("rectangle"))
        self.lasso_button.clicked.connect(lambda: self.canvas.set_tool("lasso"))
        self.pan_button.clicked.connect(lambda: self.canvas.set_tool("pan"))
        self.fit_button.clicked.connect(self._fit_canvas)
        self.open_button.clicked.connect(self._open_artifact)
        self.previous_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        for button in (
            self.rectangle_button,
            self.lasso_button,
            self.pan_button,
            self.fit_button,
        ):
            drawing_tools.addWidget(button)
        drawing_tools.addStretch(1)
        layout.addLayout(drawing_tools)
        navigation_tools = QHBoxLayout()
        for button in (self.open_button, self.previous_button, self.next_button):
            navigation_tools.addWidget(button)
        navigation_tools.addStretch(1)
        layout.addLayout(navigation_tools)
        self.canvas = ScientificImageCanvas()
        self.canvas.roi_created.connect(self._canvas_roi_created)
        self.canvas.coordinates_changed.connect(self._coordinate_changed)
        layout.addWidget(self.canvas, 1)
        status_row = QHBoxLayout()
        self.frame_status = QLabel("No source map loaded")
        self.coordinate_status = QLabel("HPLN — / HPLT —")
        self.frame_status.setProperty("muted", True)
        self.coordinate_status.setProperty("muted", True)
        status_row.addWidget(self.frame_status, 1)
        status_row.addWidget(self.coordinate_status)
        layout.addLayout(status_row)
        return panel

    def _roi_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Regions"))
        self.roi_list = QListWidget()
        layout.addWidget(self.roi_list, 1)
        buttons = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.clear_button = QPushButton("Clear")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button.clicked.connect(self._redo)
        self.clear_button.clicked.connect(self._clear_rois)
        buttons.addWidget(self.undo_button)
        buttons.addWidget(self.redo_button)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)
        self.save_roi_button = QPushButton("Save ROI JSON…")
        self.save_roi_button.clicked.connect(self._save_roi_json)
        self.export_image_button = QPushButton("Export annotated PNG…")
        self.export_image_button.clicked.connect(self._export_annotated_png)
        layout.addWidget(self.save_roi_button)
        layout.addWidget(self.export_image_button)
        self.roi_help = QLabel(
            "Draw rectangles or lassos on a rendered map. Coordinates are stored "
            "as HPLN/HPLT arcseconds."
        )
        self.roi_help.setWordWrap(True)
        self.roi_help.setProperty("muted", True)
        layout.addWidget(self.roi_help)
        panel.setMinimumWidth(175)
        return panel

    def _choose_source_file(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select radio FITS",
            self._initial_path(self.source_path),
            "FITS files (*.fits *.fit *.fts);;All files (*)",
        )
        if selected:
            self.source_path.setText(selected)

    def _choose_directory(self, field: QLineEdit) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select folder",
            self._initial_path(field),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            field.setText(selected)

    def _initial_path(self, field: QLineEdit) -> str:
        value = field.text().strip()
        if value:
            return value
        return str(self.adapter.allowed_roots[0]) if self.adapter.allowed_roots else ""

    def _ensure_output(self) -> Path:
        value = self.output_path.text().strip()
        if value:
            return Path(value).expanduser().resolve(strict=False)
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        output = self.adapter.runtime.run_output_dir(
            "preview",
            run_id,
            "source-map",
        )
        self.output_path.setText(str(output))
        return output

    def _config_payload(self) -> dict[str, Any]:
        advanced = json.loads(self.advanced.toPlainText() or "{}")
        if not isinstance(advanced, dict):
            raise TypeError("Advanced JSON must be an object")
        frequencies = [
            float(value.strip())
            for value in self.frequencies.text().split(",")
            if value.strip()
        ]
        payload: dict[str, Any] = {
            "config": self.config_name.text().strip(),
            "mode": self.mode.currentData(),
            "source_path": self.source_path.text().strip(),
            "output_dir": str(self._ensure_output()),
            "frequencies": frequencies,
            "polarization": self.polarization.currentText(),
            "start_idx": self.input_start.value(),
            "end_idx": (None if self.input_end.value() < 0 else self.input_end.value()),
            "cmap": self.cmap.currentText(),
            "color_range_mode": self.range_mode.currentText(),
            "gaussian_overlay": self.gaussian_overlay.isChecked(),
            "spectrogram_panel": self.spectrogram.isChecked(),
            "background_mode": self.background.currentText(),
            "background_display": self.background_display.isChecked(),
            "background_fit": self.background_fit.isChecked(),
            "advanced": advanced,
        }
        if self.range_mode.currentText() == "fixed":
            payload["fixed_vmin"] = self.fixed_min.value()
            payload["fixed_vmax"] = self.fixed_max.value()
        return payload

    def _request_file(self, stem: str, payload: dict[str, Any]) -> tuple[Path, Path]:
        root = self.adapter.runtime.tmp_dir / "source-map-native"
        root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        request = root / f"{stem}-{token}.request.json"
        result = root / f"{stem}-{token}.result.json"
        request.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return request, result

    def _request_discovery(self) -> None:
        try:
            config = self._config_payload()
            source = self.adapter.validate_input_directory(
                config["source_path"]
                if Path(config["source_path"]).is_dir()
                else Path(config["source_path"]).parent
            )
            output = self._ensure_output()
            request, result = self._request_file(
                "discover",
                {
                    "allowed_roots": [
                        *map(str, self.adapter.allowed_roots),
                        str(output),
                    ],
                    "config": config,
                },
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._show_error(exc)
            return
        summary = "\n".join(
            (
                "Module: Source Map Discovery",
                f"Input: {source}",
                (
                    f"Parameters: mode={config['mode']}; "
                    f"polarization={config['polarization']}"
                ),
                f"Output: {output}",
                "Workload: inspect compatible FITS candidates",
            )
        )
        if not self.confirm(self, "Discover Source Map candidates", summary):
            return
        self.status.setText("Discovery queued…")
        self.task_requested.emit(
            TaskLaunch(
                "Source Map discovery",
                "source-map",
                "solar_apps.frontends.app_v1.source_map_worker",
                (
                    "--operation",
                    "discover",
                    "--request-file",
                    str(request),
                    "--result-file",
                    str(result),
                ),
                output,
                summary,
            )
        )

    def _request_render(self) -> None:
        if self._discovery_file is None:
            return
        output = self._ensure_output()
        request, result = self._request_file(
            "render",
            {
                "discovery_file": str(self._discovery_file),
                "candidate_index": self.candidate.currentIndex(),
            },
        )
        summary = "\n".join(
            (
                "Module: Source Map",
                f"Input: {self.candidate.currentText()}",
                (
                    f"Parameters: cmap={self.cmap.currentText()}; "
                    f"gaussian={self.gaussian_overlay.isChecked()}"
                ),
                f"Output: {output}",
                "Workload: one selected Source Map frame",
            )
        )
        if self.confirm(self, "Render Source Map", summary):
            self.task_requested.emit(
                TaskLaunch(
                    "Source Map render",
                    "source-map",
                    "solar_apps.frontends.app_v1.source_map_worker",
                    (
                        "--operation",
                        "render",
                        "--request-file",
                        str(request),
                        "--result-file",
                        str(result),
                    ),
                    output,
                    summary,
                )
            )

    def _request_sequence(self) -> None:
        if self._discovery_file is None:
            return
        output = self._ensure_output()
        request, result = self._request_file(
            "sequence",
            {
                "discovery_file": str(self._discovery_file),
                "start_frame": self.sequence_start.value(),
                "end_frame": self.sequence_end.value(),
            },
        )
        summary = "\n".join(
            (
                "Module: Source Map Sequence",
                f"Input: {self.source_path.text().strip()}",
                (
                    f"Parameters: frames={self.sequence_start.value()}–"
                    f"{self.sequence_end.value()}"
                ),
                f"Output: {output}",
                (
                    "Workload: "
                    f"{self.sequence_end.value() - self.sequence_start.value() + 1} "
                    "Source Map frame(s)"
                ),
            )
        )
        if self.confirm(self, "Prepare Source Map sequence", summary):
            self.task_requested.emit(
                TaskLaunch(
                    "Source Map sequence",
                    "source-map",
                    "solar_apps.frontends.app_v1.source_map_worker",
                    (
                        "--operation",
                        "sequence",
                        "--request-file",
                        str(request),
                        "--result-file",
                        str(result),
                    ),
                    output,
                    summary,
                )
            )

    def handle_artifact(self, path: str | Path) -> None:
        artifact = Path(path)
        if artifact.suffix.casefold() == ".json":
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except OSError, json.JSONDecodeError:
                return
            if isinstance(payload, dict) and "public_candidates" in payload:
                self._load_discovery(artifact, payload)
            return
        if artifact.suffix.casefold() == ".png":
            if sidecar_path_for(artifact).is_file():
                self._load_image(artifact)

    def _load_discovery(self, path: Path, payload: dict[str, Any]) -> None:
        candidates = payload.get("public_candidates")
        if not isinstance(candidates, list):
            return
        self._discovery_file = path
        self.candidate.clear()
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            observed = candidate.get("observed_at_utc") or candidate.get("time_utc")
            label = (
                f"{index}: {observed}"
                if observed
                else f"{index}: {candidate.get('id', 'candidate')}"
            )
            self.candidate.addItem(label, candidate)
        count = self.candidate.count()
        self.candidate.setEnabled(count > 0)
        self.render_button.setEnabled(count > 0)
        self.sequence_button.setEnabled(count > 0)
        self.sequence_start.setMaximum(max(1, count))
        self.sequence_end.setMaximum(max(1, count))
        self.sequence_end.setValue(max(1, count))
        self.status.setText(f"{count} candidate(s) discovered")

    def _load_image(self, path: Path) -> None:
        sidecar = sidecar_path_for(path)
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._show_error(f"Could not load Source Map sidecar: {exc}")
            return
        if not self.canvas.set_image(path):
            self._show_error(f"Could not decode Source Map image: {path}")
            return
        self._metadata = metadata
        self._image_path = path
        self.playback.add_frame(path, select=True, emit=False)
        self._update_frame_status()
        self._render_rois()

    def _open_artifact(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Source Map PNG",
            self.output_path.text().strip(),
            "PNG image (*.png)",
        )
        if selected:
            self._load_image(Path(selected))

    def _fit_canvas(self) -> None:
        self.canvas.fit_image()

    def _step_frame(self, delta: int) -> None:
        if delta < 0:
            self.playback.step_backward()
        else:
            self.playback.step_forward()

    def _update_frame_status(self) -> None:
        count = len(self.playback.frames)
        if count:
            path = Path(self.playback.frames[self.playback.index])
            self.frame_status.setText(
                f"{self.playback.index + 1} / {count} — {path.name}"
            )
        self.previous_button.setEnabled(count > 1)
        self.next_button.setEnabled(count > 1)

    def _panel_id(self) -> str | None:
        if self._metadata is None:
            return None
        panels = self._metadata.get("panels")
        if not isinstance(panels, list) or not panels:
            return None
        panel = panels[0]
        return str(panel.get("id")) if isinstance(panel, dict) else None

    def _coordinate_changed(self, x: float, y: float) -> None:
        panel_id = self._panel_id()
        if self._metadata is None or panel_id is None:
            return
        try:
            hpln, hplt = image_pixel_to_data(self._metadata, panel_id, x, y)
        except KeyError, TypeError, ValueError, ZeroDivisionError:
            return
        self.coordinate_status.setText(f"HPLN {hpln:.2f} / HPLT {hplt:.2f}")

    def _canvas_roi_created(self, raw: dict[str, Any]) -> None:
        panel_id = self._panel_id()
        if self._metadata is None or panel_id is None:
            self._show_error(
                "Load a Source Map with a valid sidecar before drawing ROI."
            )
            self.canvas.clear_rois()
            return
        geometry = raw["geometry"]
        if raw["type"] == "rectangle":
            left, top = image_pixel_to_data(
                self._metadata,
                panel_id,
                geometry["left"],
                geometry["top"],
            )
            right, bottom = image_pixel_to_data(
                self._metadata,
                panel_id,
                geometry["right"],
                geometry["bottom"],
            )
            converted = {
                "left": min(left, right),
                "right": max(left, right),
                "bottom": min(bottom, top),
                "top": max(bottom, top),
            }
        else:
            converted = {
                "points": [
                    list(
                        image_pixel_to_data(
                            self._metadata,
                            panel_id,
                            point[0],
                            point[1],
                        )
                    )
                    for point in geometry["points"]
                ]
            }
        index = len(self.roi_controller.items) + 1
        self.roi_controller.add(
            {
                "id": f"roi-{uuid.uuid4().hex[:12]}",
                "name": f"ROI {index}",
                "type": raw["type"],
                "geometry": converted,
                "visible": True,
                "style": {
                    "color": "#00d4ff",
                    "line_width": 3,
                    "show_label": True,
                },
            }
        )

    def _render_rois(self) -> None:
        self.canvas.clear_rois()
        if self._metadata is None:
            return
        from solar_apps.workflows.radio.artifacts import data_to_image_pixel

        panel_id = self._panel_id()
        if panel_id is None:
            return
        for roi in self.roi_controller.items:
            geometry = roi["geometry"]
            if roi["type"] == "rectangle":
                first = data_to_image_pixel(
                    self._metadata,
                    panel_id,
                    geometry["left"],
                    geometry["top"],
                )
                second = data_to_image_pixel(
                    self._metadata,
                    panel_id,
                    geometry["right"],
                    geometry["bottom"],
                )
                self.canvas.add_rectangle(
                    QRectF(QPointF(*first), QPointF(*second)),
                    color=roi["style"]["color"],
                    width=float(roi["style"]["line_width"]),
                )
            else:
                points = [
                    QPointF(
                        *data_to_image_pixel(
                            self._metadata,
                            panel_id,
                            point[0],
                            point[1],
                        )
                    )
                    for point in geometry["points"]
                ]
                self.canvas.add_lasso(
                    points,
                    color=roi["style"]["color"],
                    width=float(roi["style"]["line_width"]),
                )

    def _refresh_roi_list(self) -> None:
        self.roi_list.clear()
        for roi in self.roi_controller.items:
            self.roi_list.addItem(f"{roi['name']} — {roi['type']}")
        self._render_rois()

    def _undo(self) -> None:
        self.roi_controller.undo()

    def _redo(self) -> None:
        self.roi_controller.redo()

    def _clear_rois(self) -> None:
        self.roi_controller.clear()

    def _roi_payload(self) -> dict[str, Any]:
        if self._metadata is None:
            raise ValueError("No Source Map artifact is loaded")
        return validate_roi_set(
            {
                "schema_version": 1,
                "coordinate_system": "HPLN/HPLT arcsec",
                "image_sha256": self._metadata["image"]["sha256"],
                "rois": list(self.roi_controller.items),
            },
            expected_image_sha256=self._metadata["image"]["sha256"],
        )

    def _save_roi_json(self) -> None:
        try:
            payload = self._roi_payload()
        except (KeyError, TypeError, ValueError) as exc:
            self._show_error(exc)
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save ROI JSON",
            str(self._ensure_output() / "source-map.roi-set.json"),
            "JSON (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.casefold() != ".json":
            path = path.with_suffix(".json")
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.status.setText(f"Saved {path.name}")

    def _export_annotated_png(self) -> None:
        if self._image_path is None:
            self._show_error("Load a Source Map before exporting.")
            return
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export annotated PNG",
            str(self._ensure_output() / f"{self._image_path.stem}.annotated.png"),
            "PNG (*.png)",
        )
        if not selected:
            return
        path = Path(selected)
        if path.suffix.casefold() != ".png":
            path = path.with_suffix(".png")
        rect = self.canvas.scene().sceneRect()
        image = QImage(
            max(1, round(rect.width())),
            max(1, round(rect.height())),
            QImage.Format.Format_ARGB32,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        self.canvas.scene().render(painter, QRectF(image.rect()), rect)
        painter.end()
        if not image.save(str(path), "PNG"):
            self._show_error(f"Could not save {path}")
            return
        self.status.setText(f"Exported {path.name}")

    def _show_error(self, error: object) -> None:
        message = str(error)
        self.record_diagnostic(message)
        QMessageBox.critical(self, "Source Map", message)


__all__ = ["SourceMapNativePanel"]
