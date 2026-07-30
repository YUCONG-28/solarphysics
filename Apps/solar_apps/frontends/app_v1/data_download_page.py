# SPDX-License-Identifier: GPL-3.0-only
"""Native two-stage observation search and download page."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from PyQt6.QtCore import QDateTime, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots
from solar_toolkit.net.observations import PRODUCTS, RemoteObservationV1

from .basic_services import AllowedPathField
from .components import NativeModulePanel
from .phase2a import TaskLaunch
from .runtime import AppV1RuntimePaths

_WORKER = "solar_apps.frontends.app_v1.data_download_worker"


class DataDownloadPanel(NativeModulePanel):
    """Search, select, and download observations without leaving App 1.0."""

    task_requested = pyqtSignal(object)

    def __init__(
        self,
        runtime: AppV1RuntimePaths,
        parent=None,  # type: ignore[no-untyped-def]
    ) -> None:
        super().__init__("data-download", legacy_enabled=False, parent=parent)
        self.runtime = runtime
        self._records: list[RemoteObservationV1] = []
        self._search_result: Path | None = None
        observations = runtime.observations_dir or runtime.outputs_dir / "observations"
        try:
            configured = configured_allowed_roots()
        except (OSError, TypeError, ValueError):
            configured = ()
        self.allowed_roots = (observations, *configured)

        root = QVBoxLayout(self)
        form = QFormLayout()
        self.product = QComboBox()
        for spec in PRODUCTS.values():
            self.product.addItem(spec.title, spec.product_id)
        self.product.currentIndexChanged.connect(self._product_changed)

        now = QDateTime.currentDateTimeUtc()
        self.start = QDateTimeEdit(now.addSecs(-3600))
        self.end = QDateTimeEdit(now)
        for editor in (self.start, self.end):
            editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
            editor.setTimeSpec(Qt.TimeSpec.UTC)
            editor.setCalendarPopup(True)

        self.spacecraft = QComboBox()
        self.detectors = QLineEdit()
        self.detectors.setPlaceholderText("Comma-separated detector IDs")
        self.historical_c1 = QCheckBox("Include historical LASCO C1")
        self.historical_c1.setToolTip(
            "C1 observations are historical and may legitimately return no results."
        )
        self.historical_c1.toggled.connect(self._historical_c1_changed)
        self.wavelengths = QLineEdit()
        self.wavelengths.setPlaceholderText("Comma-separated Angstrom values")
        self.level = QComboBox()
        self.sample = QSpinBox()
        self.sample.setRange(0, 86_400)
        self.sample.setSpecialValueText("Native cadence")
        self.sample.setSuffix(" s")

        self.observation_root = AllowedPathField(
            self.allowed_roots,
            directory=True,
        )
        self.observation_root.setText(str(observations))

        form.addRow("Product", self.product)
        form.addRow("Start", self.start)
        form.addRow("End", self.end)
        form.addRow("Spacecraft", self.spacecraft)
        form.addRow("Detectors", self.detectors)
        form.addRow("Historical data", self.historical_c1)
        form.addRow("Wavelengths", self.wavelengths)
        form.addRow("Level", self.level)
        form.addRow("Sample every", self.sample)
        form.addRow("Observation root", self.observation_root)
        root.addLayout(form)

        actions = QHBoxLayout()
        self.search_button = QPushButton("Search remote archive")
        self.search_button.setProperty("primary", True)
        self.search_button.clicked.connect(self._request_search)
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none = QPushButton("Select none")
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        self.download_button = QPushButton("Download selected")
        self.download_button.setProperty("primary", True)
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._request_download)
        actions.addWidget(self.search_button)
        actions.addWidget(select_all)
        actions.addWidget(select_none)
        actions.addWidget(self.download_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.summary = QLabel("Search an archive to preview remote observations.")
        self.summary.setWordWrap(True)
        self.summary.setProperty("muted", True)
        root.addWidget(self.summary)

        self.results = QTableWidget(0, 10)
        self.results.setHorizontalHeaderLabels(
            (
                "Use",
                "Source",
                "Spacecraft",
                "Start UTC",
                "Product",
                "Detector",
                "Wavelength",
                "Level",
                "Size",
                "Target",
            )
        )
        self.results.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results.itemChanged.connect(self._selection_changed)
        root.addWidget(self.results, 1)
        self._product_changed()

    def _product_changed(self) -> None:
        spec = PRODUCTS[str(self.product.currentData())]
        current = self.spacecraft.currentData()
        self.spacecraft.clear()
        if len(spec.spacecraft) > 1:
            self.spacecraft.addItem("All available", ",".join(spec.spacecraft))
        for item in spec.spacecraft:
            self.spacecraft.addItem(item, item)
        if current is not None:
            index = self.spacecraft.findData(current)
            if index >= 0:
                self.spacecraft.setCurrentIndex(index)
        detectors = spec.detectors
        if spec.product_id == "soho-lasco":
            detectors = ("c2", "c3")
            self.detectors.setToolTip(
                "C2 and C3 are the default. Add c1 for historical observations."
            )
        else:
            self.detectors.setToolTip("")
        self.historical_c1.blockSignals(True)
        self.historical_c1.setChecked(False)
        self.historical_c1.blockSignals(False)
        self.historical_c1.setVisible(spec.product_id == "soho-lasco")
        self.detectors.setText(",".join(detectors))
        defaults = {
            "sdo-aia-euv": "171",
            "sdo-aia-uv": "1600",
            "stereo-euvi": "171",
            "goes-suvi": "171",
        }
        self.wavelengths.setText(defaults.get(spec.product_id, ""))
        self.level.clear()
        self.level.addItem("Provider default", "")
        for level in spec.levels:
            self.level.addItem(level, level)

    def _historical_c1_changed(self, checked: bool) -> None:
        detectors = list(self._csv(self.detectors.text()))
        if checked and "c1" not in detectors:
            detectors.append("c1")
        elif not checked:
            detectors = [item for item in detectors if item != "c1"]
        self.detectors.setText(",".join(detectors))

    @staticmethod
    def _csv(value: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in value.split(",") if item.strip())

    def _request_search(self) -> None:
        try:
            start = self.start.dateTime().toPyDateTime()
            end = self.end.dateTime().toPyDateTime()
            if end <= start:
                raise ValueError("End time must be after start time")
            output = self.runtime.run_output_dir(
                "preview",
                f"run-{uuid.uuid4().hex[:12]}",
                "data-download",
            )
            spacecraft = str(self.spacecraft.currentData() or "")
            detectors = ",".join(self._csv(self.detectors.text()))
            wavelengths = ",".join(self._csv(self.wavelengths.text()))
            parameters = {
                "product": str(self.product.currentData()),
                "start": start.isoformat(),
                "end": end.isoformat(),
                "spacecraft": spacecraft,
                "detectors": detectors or "provider default",
                "wavelengths": wavelengths or "provider default",
                "sample": self.sample.value() or "native",
            }
        except (TypeError, ValueError) as exc:
            self.record_diagnostic(exc)
            self.summary.setText(str(exc))
            return
        if not self.confirm(
            self,
            "Search remote observations",
            {
                "Module": "Data Download",
                "Input": f"{start.isoformat()} to {end.isoformat()}",
                "Parameters": json.dumps(parameters, sort_keys=True),
                "Output": str(output),
                "Workload": "One read-only remote archive search",
            },
        ):
            return
        arguments = [
            "search",
            "--product-id",
            str(self.product.currentData()),
            "--start-utc",
            start.isoformat(),
            "--end-utc",
            end.isoformat(),
            "--output-dir",
            str(output),
        ]
        for flag, value in (
            ("--spacecraft", spacecraft),
            ("--detectors", detectors),
            ("--wavelengths", wavelengths),
            ("--level", str(self.level.currentData() or "")),
        ):
            if value:
                arguments.extend((flag, value))
        if self.sample.value():
            arguments.extend(("--sample-seconds", str(self.sample.value())))
        self.task_requested.emit(
            TaskLaunch(
                "Observation search",
                "data-download",
                _WORKER,
                tuple(arguments),
                output,
                json.dumps(parameters, sort_keys=True),
            )
        )
        self.summary.setText("Search queued. Existing results remain until it succeeds.")

    def _request_download(self) -> None:
        selected = self.selected_records()
        if not selected:
            self.summary.setText("Select at least one remote observation.")
            return
        try:
            observation_root = self.observation_root.validated_path(must_exist=False)
            output = self.runtime.run_output_dir(
                "preview",
                f"run-{uuid.uuid4().hex[:12]}",
                "data-download",
            )
            selection_dir = self.runtime.tmp_dir / "data-download"
            selection_dir.mkdir(parents=True, exist_ok=True)
            selection = selection_dir / f"selection-{uuid.uuid4().hex}.json"
            selection.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "records": [item.to_dict() for item in selected],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as exc:
            self.record_diagnostic(exc)
            self.summary.setText(str(exc))
            return
        total = sum(item.size_bytes or 0 for item in selected)
        unknown = sum(item.size_bytes is None for item in selected)
        if not self.confirm(
            self,
            "Download selected observations",
            {
                "Module": "Data Download",
                "Input": f"{len(selected)} selected remote record(s)",
                "Parameters": (
                    f"known size={_format_size(total)}; unknown size={unknown}; "
                    "workers=2"
                ),
                "Output": str(observation_root),
                "Workload": "Atomic downloads with retry and SHA-256 receipts",
            },
        ):
            return
        self.task_requested.emit(
            TaskLaunch(
                "Observation download",
                "data-download",
                _WORKER,
                (
                    "download",
                    "--selection",
                    str(selection),
                    "--observation-root",
                    str(observation_root),
                    "--output-dir",
                    str(output),
                    "--max-workers",
                    "2",
                ),
                output,
                f"{len(selected)} observations to {observation_root}",
            )
        )
        self.summary.setText(f"Queued {len(selected)} observation(s) for download.")

    def selected_records(self) -> tuple[RemoteObservationV1, ...]:
        selected: list[RemoteObservationV1] = []
        for row, record in enumerate(self._records):
            item = self.results.item(row, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(record)
        return tuple(selected)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.results.blockSignals(True)
        for row in range(self.results.rowCount()):
            item = self.results.item(row, 0)
            if item is not None:
                item.setCheckState(state)
        self.results.blockSignals(False)
        self._selection_changed()

    def _selection_changed(self, _item=None) -> None:  # type: ignore[no-untyped-def]
        count = len(self.selected_records())
        self.download_button.setEnabled(count > 0)
        if self._records:
            self.summary.setText(
                f"{len(self._records)} result(s); {count} selected."
            )

    def handle_artifact(self, path: str | Path) -> None:
        artifact = Path(path)
        if artifact.name == "search-results.json":
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                records = [
                    RemoteObservationV1.from_dict(item)
                    for item in payload.get("records", ())
                ]
                self._load_records(records)
                self._search_result = artifact
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.record_diagnostic(exc)
                self.summary.setText(f"Could not load search results: {exc}")
        elif artifact.name == "download-receipt.json":
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                items = payload.get("items") or ()
                failed = sum(item.get("status") == "failed" for item in items)
                self.summary.setText(
                    f"Download receipt: {len(items)} item(s), {failed} failed."
                )
            except (OSError, TypeError, json.JSONDecodeError) as exc:
                self.record_diagnostic(exc)

    def _load_records(self, records: list[RemoteObservationV1]) -> None:
        self._records = records
        self.results.blockSignals(True)
        self.results.setRowCount(len(records))
        for row, record in enumerate(records):
            check = QTableWidgetItem()
            check.setFlags(check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Unchecked)
            self.results.setItem(row, 0, check)
            values = (
                record.provider,
                record.spacecraft,
                record.start_utc.isoformat().replace("+00:00", "Z"),
                PRODUCTS[record.product_id].title,
                record.detector,
                "" if record.wavelength is None else str(record.wavelength),
                record.level or "",
                (
                    "unknown"
                    if record.size_bytes is None
                    else _format_size(record.size_bytes)
                ),
                record.target_relative_path,
            )
            for column, value in enumerate(values, start=1):
                self.results.setItem(row, column, QTableWidgetItem(value))
        self.results.blockSignals(False)
        self.results.resizeColumnsToContents()
        known = sum(item.size_bytes or 0 for item in records)
        unknown = sum(item.size_bytes is None for item in records)
        self.summary.setText(
            f"{len(records)} result(s); known size {_format_size(known)}; "
            f"{unknown} unknown-size record(s). Nothing is selected."
        )
        self.download_button.setEnabled(False)


def _format_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


__all__ = ["DataDownloadPanel"]
