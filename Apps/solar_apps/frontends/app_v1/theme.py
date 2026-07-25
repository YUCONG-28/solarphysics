# SPDX-License-Identifier: GPL-3.0-only
"""PyQt6-only semantic chrome theme for the App 1.0 process."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from solar_apps.ui.theme import ThemeMode, normalize_theme_mode

_COLORS = {
    "light": {
        "window": "#f4f7fb",
        "surface": "#ffffff",
        "alternate": "#edf2f8",
        "text": "#172033",
        "muted": "#5d6b80",
        "border": "#c9d4e3",
        "accent": "#2563eb",
        "highlighted": "#ffffff",
    },
    "dark": {
        "window": "#0b1120",
        "surface": "#111a2c",
        "alternate": "#182338",
        "text": "#e7edf7",
        "muted": "#a9b5c8",
        "border": "#33435d",
        "accent": "#60a5fa",
        "highlighted": "#071120",
    },
}


class AppV1ThemeController(QObject):
    """Apply and persist Auto/Light/Dark without touching plot rendering."""

    theme_changed = pyqtSignal(str, str)

    def __init__(
        self,
        application: QApplication,
        *,
        state_store: Any | None = None,
        initial_mode: object = "auto",
    ) -> None:
        super().__init__(application)
        self.application = application
        self.state_store = state_store
        saved = state_store.load(default={}) if state_store is not None else {}
        self.mode: ThemeMode = normalize_theme_mode(
            saved.get("theme", initial_mode)
            if isinstance(saved, dict)
            else initial_mode
        )
        hints = self.application.styleHints()
        changed = getattr(hints, "colorSchemeChanged", None)
        if changed is not None:
            changed.connect(self._system_scheme_changed)
        self.apply()

    def effective_mode(self) -> ThemeMode:
        if self.mode != "auto":
            return self.mode
        scheme = self.application.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"

    def set_mode(self, mode: object) -> ThemeMode:
        self.mode = normalize_theme_mode(mode)
        if self.state_store is not None:
            self.state_store.update({"theme": self.mode})
        self.apply()
        return self.mode

    def apply(self) -> str:
        effective = self.effective_mode()
        colors = _COLORS[effective]
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors["alternate"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors["accent"]))
        palette.setColor(
            QPalette.ColorRole.HighlightedText, QColor(colors["highlighted"])
        )
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(colors["surface"]))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(colors["text"]))
        palette.setColor(QPalette.ColorRole.Link, QColor(colors["accent"]))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors["muted"]))
        self.application.setPalette(palette)
        self.application.setStyleSheet("""
            QMainWindow, QDialog { background: %(window)s; }
            QDockWidget::title {
                background: %(alternate)s;
                border: 1px solid %(border)s;
                padding: 6px;
            }
            QGroupBox, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
            QListView, QTreeView, QTableView, QPlainTextEdit {
                background: %(surface)s;
                border: 1px solid %(border)s;
                border-radius: 5px;
            }
            QPushButton {
                background: %(surface)s;
                border: 1px solid %(border)s;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover { border-color: %(accent)s; }
            QPushButton:disabled { color: %(muted)s; }
            QHeaderView::section {
                background: %(alternate)s;
                border: 0;
                border-right: 1px solid %(border)s;
                padding: 5px;
            }
            QLabel[muted="true"] { color: %(muted)s; }
            QLabel[badge="true"] {
                color: %(accent)s;
                border: 1px solid %(accent)s;
                border-radius: 8px;
                padding: 2px 7px;
            }
            """ % colors)
        self.theme_changed.emit(self.mode, effective)
        return effective

    def _system_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        if self.mode == "auto":
            self.apply()


__all__ = ["AppV1ThemeController"]
