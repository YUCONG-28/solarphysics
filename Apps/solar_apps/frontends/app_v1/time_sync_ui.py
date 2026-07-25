# SPDX-License-Identifier: GPL-3.0-only
"""Thin PyQt6 controls around the import-safe timeline coordinator."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .contracts import SyncSelection
from .timeline import TimeCoordinator


class TimelineBridge(QObject):
    """Convert coordinator callbacks into a Qt-safe signal."""

    selection_changed = pyqtSignal(object)

    def __init__(
        self,
        coordinator: TimeCoordinator,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.coordinator = coordinator
        coordinator.subscribe(self.selection_changed.emit)


class TimeSyncPanel(QWidget):
    """Global base-source and UTC stepping controls."""

    selection_changed = pyqtSignal(object)

    def __init__(
        self,
        coordinator: TimeCoordinator,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.coordinator = coordinator
        self.bridge = TimelineBridge(coordinator, self)
        self.bridge.selection_changed.connect(self._selection_received)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.base_source = QComboBox()
        self.current_time = QLabel("No UTC time selected")
        self.current_time.setWordWrap(True)
        self.match_summary = QLabel("No timeline sources registered")
        self.match_summary.setWordWrap(True)
        self.match_summary.setProperty("muted", True)
        form.addRow("Base source", self.base_source)
        form.addRow("Current UTC", self.current_time)
        layout.addLayout(form)
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
        if index >= 0:
            self.base_source.setCurrentIndex(index)
        self.base_source.blockSignals(False)
        enabled = self.base_source.count() > 0
        self.base_source.setEnabled(enabled)

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
            selection.current_time_utc.isoformat().replace("+00:00", "Z")
        )
        matched = sum(
            locator is not None for locator in selection.matched_locators.values()
        )
        self.match_summary.setText(
            f"Matched {matched}/{len(selection.matched_locators)} source(s)"
        )
        self.selection_changed.emit(selection)


__all__ = ["TimelineBridge", "TimeSyncPanel"]
