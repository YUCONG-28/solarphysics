# SPDX-License-Identifier: GPL-3.0-only
"""Small native controls for App 1.0 project and parameter-preset actions."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProjectPanel(QWidget):
    """Editable project identity plus explicit save/load operations."""

    save_project_requested = pyqtSignal()
    load_project_requested = pyqtSignal()
    save_preset_requested = pyqtSignal()
    load_preset_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.project_id = QLineEdit("preview")
        self.project_name = QLineEdit("Solar Physics Project")
        self.preset_id = QLineEdit("default")
        form.addRow("Project ID", self.project_id)
        form.addRow("Project name", self.project_name)
        form.addRow("Preset ID", self.preset_id)
        layout.addLayout(form)

        project_actions = QHBoxLayout()
        self.save_project = QPushButton("Save project")
        self.load_project = QPushButton("Open project")
        project_actions.addWidget(self.save_project)
        project_actions.addWidget(self.load_project)
        layout.addLayout(project_actions)

        preset_actions = QHBoxLayout()
        self.save_preset = QPushButton("Save preset")
        self.load_preset = QPushButton("Load preset")
        preset_actions.addWidget(self.save_preset)
        preset_actions.addWidget(self.load_preset)
        layout.addLayout(preset_actions)

        self.status = QLabel("Project metadata has not been saved.")
        self.status.setWordWrap(True)
        self.status.setProperty("muted", True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.save_project.clicked.connect(self.save_project_requested)
        self.load_project.clicked.connect(self.load_project_requested)
        self.save_preset.clicked.connect(self.save_preset_requested)
        self.load_preset.clicked.connect(self.load_preset_requested)


__all__ = ["ProjectPanel"]
