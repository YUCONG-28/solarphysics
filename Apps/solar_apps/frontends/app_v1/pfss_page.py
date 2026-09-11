# SPDX-License-Identifier: GPL-3.0-only
"""Native PFSS result explorer. Model recalculation remains on the compute host."""

import json

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from solar_apps.platform.paths import configured_allowed_roots
from solar_apps.ui.state import frontend_state_store
from .components import NativeModulePanel
from .pfss_adapter import PFSSAdapter


class PFSSPanel(NativeModulePanel):
    def __init__(self, runtime_layout, *, allowed_roots=None, parent=None):
        super().__init__("pfss", legacy_enabled=False, parent=parent)
        self.adapter = PFSSAdapter(
            allowed_roots
            if allowed_roots is not None
            else configured_allowed_roots(config_path=runtime_layout.config_path)
        )
        self.store = frontend_state_store("pfss", layout=runtime_layout)
        self.metadata = {}
        self.records = []
        self.sources = None
        self.path = None
        self.aia = None
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.directory = QLineEdit()
        self.directory.setPlaceholderText("Choose a completed synchronized PFSS run")
        self.directory.setText(
            str(self.store.load(default={}).get("fields", {}).get("directory", ""))
        )
        browse = QPushButton("Choose results")
        browse.clicked.connect(self._browse)
        reload = QPushButton("Load / verify")
        reload.clicked.connect(self._load_clicked)
        header.addWidget(self.directory, 1)
        header.addWidget(browse)
        header.addWidget(reload)
        layout.addLayout(header)
        self.status = QLabel(
            "PFSS and Newkirk positions are conditional model results, not independent 3D measurements."
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        split = QSplitter()
        controls = QWidget()
        form = QVBoxLayout(controls)
        self.classification = QComboBox()
        self.classification.addItems(
            ["all", "open_positive", "open_negative", "closed", "failed"]
        )
        self.classification.currentTextChanged.connect(self._filter_lines)
        form.addWidget(self.classification)
        self.lines = QListWidget()
        self.lines.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lines.itemSelectionChanged.connect(self.redraw)
        form.addWidget(self.lines, 1)
        self.frequency = QComboBox()
        self.frequency.addItem("all frequencies")
        self.frequency.currentTextChanged.connect(self.redraw)
        form.addWidget(self.frequency)
        self.timestamp = QComboBox()
        self.timestamp.addItem("all UTC times")
        self.timestamp.currentTextChanged.connect(self.redraw)
        form.addWidget(self.timestamp)
        params = QFormLayout()
        self.rss = QDoubleSpinBox()
        self.rss.setRange(1.01, 10)
        self.rss.setValue(2.5)
        params.addRow("Source surface / R☉", self.rss)
        self.grid = {}
        for key, default in [("nphi", 360), ("ns", 180), ("nr", 35)]:
            spin = QSpinBox()
            spin.setRange(4, 4096)
            spin.setValue(default)
            params.addRow(key, spin)
            self.grid[key] = spin
        self.seed_box = QLineEdit()
        self.seed_box.setPlaceholderText("[xmin,xmax,ymin,ymax] arcsec, or null")
        params.addRow("Candidate surface seed box", self.seed_box)
        form.addLayout(params)
        export = QPushButton("Export remote-run parameters")
        export.clicked.connect(self._export)
        form.addWidget(export)
        split.addWidget(controls)
        tabs = QTabWidget()
        self.figure = None
        plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(plot_widget)
        self.plot_placeholder = QLabel(
            "Load a verified run to display AIA and magnetic field lines."
        )
        self.plot_layout.addWidget(self.plot_placeholder)
        tabs.addTab(plot_widget, "Projected / 3D views")
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tabs.addTab(self.table, "Selected lines · whole-window residuals")
        split.addWidget(tabs)
        split.setStretchFactor(1, 1)
        layout.addWidget(split, 1)

    def _ensure_canvas(self):
        if self.figure is not None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

        self.plot_placeholder.hide()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        self.plot_layout.addWidget(self.canvas)

    def set_current_time(self, instant):
        if self.sources is None:
            return
        import pandas as pd

        times = pd.to_datetime(self.sources.time_utc, utc=True, format="ISO8601")
        matching = self.sources.loc[times == pd.Timestamp(instant), "time_utc"]
        if len(matching):
            self.timestamp.setCurrentText(str(matching.iloc[0]))
        else:
            self.status.setText(
                "Selected UTC has no exact source frame in this run; current view retained."
            )

    def _browse(self):
        value = QFileDialog.getExistingDirectory(
            self, "Select synchronized PFSS results", self.directory.text()
        )
        if value:
            self.directory.setText(value)
            self._load_clicked()

    def _load_clicked(self):
        try:
            self.load_result(self.directory.text())
        except Exception as exc:
            self.status.setText(f"Results unavailable: {exc}")

    def load_result(self, directory):
        import pandas as pd
        import sunpy.map

        self.path, self.metadata, self.records = self.adapter.load(directory)
        self.directory.setText(str(self.path))
        self._ensure_canvas()
        self.aia = sunpy.map.Map(self.path / "aia_display.fits")
        self.original = json.loads((self.path / "resolved_config.json").read_text())
        self.sources = (
            pd.read_csv(self.path / "sources.csv")
            if (self.path / "sources.csv").exists()
            else None
        )
        for combo, first, values in [
            (
                self.frequency,
                "all frequencies",
                (
                    sorted(self.sources.frequency_mhz.unique())
                    if self.sources is not None
                    else []
                ),
            ),
            (
                self.timestamp,
                "all UTC times",
                (
                    sorted(self.sources.time_utc.unique())
                    if self.sources is not None
                    else []
                ),
            ),
        ]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first)
            combo.addItems([str(v) for v in values])
            combo.blockSignals(False)
        self.rss.setValue(self.metadata["config"]["rss"])
        for key, spin in self.grid.items():
            spin.setValue(self.metadata["config"][key])
        self.seed_box.setText(json.dumps(self.original.get("seed_hpc_box")))
        self.seed_box.setCursorPosition(0)
        self.seed_box.setToolTip(self.seed_box.text())
        self.store.update({"fields": {"directory": str(self.path)}})
        self.status.setText(
            f"Verified {len(self.records)} field lines; {self.metadata.get('source_count',0)} radio records. Independent source heights: 0. EUV-to-radio association is unverified; distant PFSS segments are conditional extensions. Select lines; drag the 3D view to rotate. Parameters only affect the next remote run."
        )
        self._filter_lines()
        ranked = self.path / "newkirk_forward_ranking.csv"
        self.table.clear()
        self.table.setRowCount(0)
        if ranked.exists() and ranked.stat().st_size > 2:
            frame = pd.read_csv(ranked)
            self.table.setColumnCount(len(frame.columns))
            self.table.setHorizontalHeaderLabels(list(frame.columns))
            self.table.setRowCount(len(frame))
            for i, row in enumerate(frame.itertuples(index=False, name=None)):
                for j, value in enumerate(row):
                    self.table.setItem(i, j, QTableWidgetItem(str(value)))
        self.redraw()

    def _filter_lines(self):
        self.lines.blockSignals(True)
        self.lines.clear()
        chosen = self.classification.currentText()
        for index, record in enumerate(self.records):
            if chosen != "all" and record["classification"] != chosen:
                continue
            item = QListWidgetItem(
                f"{record['fieldline_id']} · {record['classification']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.lines.addItem(item)
            if record["fieldline_id"] in self.metadata.get(
                "selected_fieldline_ids", []
            ):
                item.setSelected(True)
        self.lines.blockSignals(False)
        if self.lines.selectedItems():
            self.lines.scrollToItem(self.lines.selectedItems()[0])
        self.redraw()

    def redraw(self, *_):
        import numpy as np
        from matplotlib.colors import AsinhNorm

        if self.figure is None:
            return
        self.figure.clear()
        a = (
            self.figure.add_subplot(121, projection=self.aia)
            if self.aia is not None
            else self.figure.add_subplot(121)
        )
        b = self.figure.add_subplot(122, projection="3d")
        transform = {}
        scale = 1
        if self.aia is not None:
            normalized = self.aia.data / float(self.aia.meta["exptime"])
            a.imshow(
                normalized,
                origin="lower",
                cmap="sdoaia171",
                norm=AsinhNorm(
                    linear_width=30,
                    vmin=0,
                    vmax=float(np.nanpercentile(normalized, 99.7)),
                ),
            )
            transform = {"transform": a.get_transform("world")}
            scale = 3600
        colors = {
            "open_positive": "#d94842",
            "open_negative": "#2877b5",
            "closed": "#6a9b45",
            "failed": "#999999",
        }
        selected = self.lines.selectedItems()[:200]
        for item in selected:
            record = self.records[item.data(Qt.ItemDataRole.UserRole)]
            xyz, xy = record["xyz_carrington_rsun"], record["hpc_arcsec"]
            if not len(xyz):
                continue
            stride = max(1, int(np.ceil(len(xyz) / 250)))
            a.plot(
                xy[:, 0] / scale,
                xy[:, 1] / scale,
                color=colors[record["classification"]],
                lw=0.8,
                **transform,
            )
            b.plot(*xyz[::stride].T, color=colors[record["classification"]], lw=0.8)
        if self.sources is not None:
            frame = self.sources
            if self.frequency.currentIndex() > 0:
                frame = frame[
                    frame.frequency_mhz == float(self.frequency.currentText())
                ]
            if self.timestamp.currentIndex() > 0:
                frame = frame[frame.time_utc == self.timestamp.currentText()]
            a.scatter(
                frame.center_x_arcsec / scale,
                frame.center_y_arcsec / scale,
                c=frame.frequency_mhz,
                s=10,
                cmap="viridis",
                **transform,
            )
        if self.aia is not None:
            xlim = [-0.5, self.aia.data.shape[1] - 0.5]
            ylim = [-0.5, self.aia.data.shape[0] - 0.5]
            if self.sources is not None and len(self.sources):
                import astropy.units as u
                from astropy.coordinates import SkyCoord

                px, py = self.aia.world_to_pixel(
                    SkyCoord(
                        self.sources.center_x_arcsec.to_numpy() * u.arcsec,
                        self.sources.center_y_arcsec.to_numpy() * u.arcsec,
                        frame=self.aia.coordinate_frame,
                    )
                )
                xlim = [
                    min(xlim[0], np.nanmin(px.value) - 15),
                    max(xlim[1], np.nanmax(px.value) + 15),
                ]
                ylim = [
                    min(ylim[0], np.nanmin(py.value) - 15),
                    max(ylim[1], np.nanmax(py.value) + 15),
                ]
            a.set_xlim(xlim)
            a.set_ylim(ylim)
        a.set(
            xlabel="Solar X (arcsec)",
            ylabel="Solar Y (arcsec)",
            title="Model projection and apparent radio sources",
        )
        a.set_aspect("equal", adjustable="datalim")
        b.set(
            xlabel="X / R☉",
            ylabel="Y / R☉",
            zlabel="Z / R☉",
            title="Conditional PFSS · rotate to inspect",
        )
        b.set_box_aspect([1, 1, 1])
        selected_ids = {
            self.records[item.data(Qt.ItemDataRole.UserRole)]["fieldline_id"]
            for item in selected
        }
        for row in range(self.table.rowCount()):
            cell = self.table.item(row, 0)
            self.table.setRowHidden(
                row, cell is not None and cell.text() not in selected_ids
            )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _export(self):
        if not self.path:
            self.status.setText("Load a completed run before exporting parameters.")
            return
        value, _ = QFileDialog.getSaveFileName(
            self,
            "Save new remote-run configuration",
            str(self.path.parent / "pfss_request.json"),
            "JSON (*.json)",
        )
        if not value:
            return
        try:
            target = self.adapter.export_config(
                value,
                self.original,
                pfss={
                    **{k: s.value() for k, s in self.grid.items()},
                    "rss": self.rss.value(),
                    "max_seeds": 512,
                },
                seed_hpc_box=json.loads(self.seed_box.text() or "null"),
            )
            self.status.setText(
                f"Parameters saved: {target}. Run on SEVERUS; results return through the existing sync session."
            )
        except Exception as exc:
            self.status.setText(f"Configuration not saved: {exc}")
