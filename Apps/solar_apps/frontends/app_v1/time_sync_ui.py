# SPDX-License-Identifier: GPL-3.0-only
"""Thin PyQt6 controls around the import-safe timeline coordinator."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .contracts import SyncSelection
from .timeline import TimeCoordinator


class TimelineBridge(QObject):
    """Convert coordinator callbacks into a Qt-safe signal."""

    selection_changed = pyqtSignal(object)
    plan_changed = pyqtSignal()

    def __init__(
        self,
        coordinator: TimeCoordinator,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.coordinator = coordinator
        coordinator.subscribe(self.selection_changed.emit)
        coordinator.subscribe_state(self.plan_changed.emit)


class TimeSyncPanel(QWidget):
    """Global base-source and UTC stepping controls."""

    selection_changed = pyqtSignal(object)
    plan_changed = pyqtSignal(object)

    def __init__(
        self,
        coordinator: TimeCoordinator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.coordinator = coordinator
        self.bridge = TimelineBridge(coordinator, self)
        self.bridge.selection_changed.connect(self._selection_received)
        self.bridge.plan_changed.connect(self._plan_received)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.base_source = QComboBox()
        self.output_source = QLabel("No output source")
        self.current_time = QLabel("No UTC time selected")
        self.current_time.setWordWrap(True)
        self.time_start = QLineEdit()
        self.time_end = QLineEdit()
        for control in (self.time_start, self.time_end):
            control.setPlaceholderText("YYYY-MM-DDTHH:MM:SS.fffffffffZ; blank = open")
            control.setClearButtonEnabled(True)
        self.window_status = QLabel("All observed image times")
        self.window_status.setWordWrap(True)
        self.window_status.setProperty("muted", True)
        self.match_summary = QLabel("No timeline sources registered")
        self.match_summary.setWordWrap(True)
        self.match_summary.setProperty("muted", True)
        form.addRow("Automatic root (fewest frames)", self.base_source)
        form.addRow("Output source (most frames)", self.output_source)
        form.addRow("Image start UTC (inclusive)", self.time_start)
        form.addRow("Image end UTC (exclusive)", self.time_end)
        form.addRow("Current UTC", self.current_time)
        layout.addLayout(form)
        window_controls = QHBoxLayout()
        self.apply_window = QPushButton("Apply image window")
        self.clear_window = QPushButton("All image times")
        self.apply_window.clicked.connect(self._apply_window)
        self.clear_window.clicked.connect(self._clear_window)
        window_controls.addWidget(self.apply_window)
        window_controls.addWidget(self.clear_window)
        window_controls.addStretch(1)
        layout.addLayout(window_controls)
        layout.addWidget(self.window_status)
        controls = QHBoxLayout()
        previous = QPushButton("Previous")
        next_button = QPushButton("Next")
        previous.clicked.connect(lambda: self._step(-1))
        next_button.clicked.connect(lambda: self._step(1))
        self.base_source.currentIndexChanged.connect(self._base_changed)
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.match_summary)
        layout.addStretch(1)
        self.refresh_sources()

    def refresh_sources(self) -> None:
        selected = self.coordinator.base_source_id
        self.base_source.blockSignals(True)
        self.base_source.clear()
        for source in self.coordinator.sources:
            self.base_source.addItem(source.source_id, source.source_id)
        index = self.base_source.findData(selected)
        self.base_source.setCurrentIndex(index)
        self.base_source.blockSignals(False)
        self.base_source.setEnabled(False)
        self.base_source.setToolTip(
            "The fewest-frame source is the root. Successive full-coverage maps expand to every frame of the output source."
        )
        self.output_source.setText(
            self.coordinator.output_source_id or "No output source"
        )
        self.time_start.setText(self.coordinator.time_start or "")
        self.time_end.setText(self.coordinator.time_end or "")
        start = self.coordinator.time_start or "open start"
        end = self.coordinator.time_end or "open end"
        plan = self.coordinator.current_plan
        rows = len(plan.rows) if plan is not None else 0
        self.window_status.setText(
            f"Observed UTC: [{start}, {end}); {rows} output image(s). "
            "Browsing and spectrum display do not change this window. "
            + (
                "Hierarchy: " + " → ".join(plan.source_order)
                if plan is not None and plan.source_order
                else ""
            )
        )
        self.current_time.setText(
            self.coordinator.current_time_text or "No UTC time selected"
        )
        if not rows:
            self.match_summary.setText(
                "No data inside the image window; no images can be exported."
                if self.coordinator.sources
                else "No timeline sources registered"
            )
        elif self.coordinator.current_time_utc is None:
            self.match_summary.setText(f"{rows} images ready; select Previous or Next")

    def _apply_window(self) -> None:
        try:
            self.coordinator.set_time_window(
                self.time_start.text().strip() or None,
                self.time_end.text().strip() or None,
            )
        except (TypeError, ValueError) as exc:
            self.match_summary.setText(str(exc))
        else:
            self.refresh_sources()

    def _clear_window(self) -> None:
        self.time_start.clear()
        self.time_end.clear()
        self._apply_window()

    def _plan_received(self) -> None:
        self.refresh_sources()
        self.plan_changed.emit(self.coordinator.current_plan)

    def _base_changed(self, _index: int) -> None:
        source_id = self.base_source.currentData()
        if source_id:
            self.coordinator.set_base_source(str(source_id))

    def _step(self, delta: int) -> None:
        try:
            self.coordinator.step(delta)
        except (KeyError, ValueError) as exc:
            self.match_summary.setText(str(exc))

    def _selection_received(self, selection: SyncSelection) -> None:
        self.current_time.setText(
            self.coordinator.current_time_text
            or selection.current_time_utc.isoformat().replace("+00:00", "Z")
        )
        matched = sum(
            locator is not None for locator in selection.matched_locators.values()
        )
        self.refresh_sources()
        plan = self.coordinator.current_plan
        flagged = 0
        if plan is not None and self.coordinator._cursor_index is not None:
            flagged = sum(
                item.within_tolerance is False
                for item in plan.rows[
                    self.coordinator._cursor_index
                ].selections.values()
            )
        self.match_summary.setText(
            f"Matched {matched}/{len(selection.matched_locators)} source(s); {flagged} timing gap(s) flagged"
        )
        self.selection_changed.emit(selection)


__all__ = ["TimelineBridge", "TimeSyncPanel"]
