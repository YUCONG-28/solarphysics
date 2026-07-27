# SPDX-License-Identifier: GPL-3.0-only
"""Primer-inspired PyQt6 chrome themes for the App 1.0 process."""

from __future__ import annotations

from typing import Any, Literal, cast

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QStyleFactory

AppV1ThemeMode = Literal["auto", "light", "dark", "dark_dimmed"]
APP_V1_THEME_MODES: tuple[AppV1ThemeMode, ...] = (
    "auto",
    "light",
    "dark",
    "dark_dimmed",
)

_COLORS = {
    "light": {
        "window": "#ffffff",
        "surface": "#ffffff",
        "alternate": "#f6f8fa",
        "inset": "#f6f8fa",
        "text": "#1f2328",
        "muted": "#59636e",
        "disabled": "#818b98",
        "border": "#d1d9e0",
        "border_emphasis": "#818b98",
        "accent": "#0969da",
        "accent_hover": "#0860ca",
        "highlighted": "#ffffff",
        "success": "#1f883d",
        "danger": "#d1242f",
        "warning": "#9a6700",
        "button": "#f6f8fa",
        "button_hover": "#eff2f5",
    },
    "dark": {
        "window": "#0d1117",
        "surface": "#0d1117",
        "alternate": "#151b23",
        "inset": "#010409",
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "disabled": "#656c76",
        "border": "#3d444d",
        "border_emphasis": "#656c76",
        "accent": "#4493f8",
        "accent_hover": "#58a6ff",
        "highlighted": "#ffffff",
        "success": "#3fb950",
        "danger": "#f85149",
        "warning": "#d29922",
        "button": "#212830",
        "button_hover": "#2a313c",
    },
    "dark_dimmed": {
        "window": "#22272e",
        "surface": "#22272e",
        "alternate": "#2d333b",
        "inset": "#1c2128",
        "text": "#cdd9e5",
        "muted": "#768390",
        "disabled": "#636e7b",
        "border": "#444c56",
        "border_emphasis": "#636e7b",
        "accent": "#539bf5",
        "accent_hover": "#6cb6ff",
        "highlighted": "#ffffff",
        "success": "#57ab5a",
        "danger": "#e5534b",
        "warning": "#c69026",
        "button": "#373e47",
        "button_hover": "#444c56",
    },
}


def normalize_app_v1_theme_mode(value: object) -> AppV1ThemeMode:
    candidate = str(value or "auto").strip().casefold().replace("-", "_")
    if candidate not in APP_V1_THEME_MODES:
        return "auto"
    return cast(AppV1ThemeMode, candidate)


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
        self.mode: AppV1ThemeMode = normalize_app_v1_theme_mode(
            saved.get("theme", initial_mode)
            if isinstance(saved, dict)
            else initial_mode
        )
        hints = self.application.styleHints()
        changed = getattr(hints, "colorSchemeChanged", None)
        if changed is not None:
            changed.connect(self._system_scheme_changed)
        self.apply()

    def effective_mode(self) -> AppV1ThemeMode:
        if self.mode != "auto":
            return self.mode
        scheme = self.application.styleHints().colorScheme()
        return "dark" if scheme == Qt.ColorScheme.Dark else "light"

    def set_mode(self, mode: object) -> AppV1ThemeMode:
        self.mode = normalize_app_v1_theme_mode(mode)
        if self.state_store is not None:
            self.state_store.update({"theme": self.mode})
        self.apply()
        return self.mode

    def apply(self) -> str:
        effective = self.effective_mode()
        colors = _COLORS[effective]
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None and self.application.style().objectName() != "fusion":
            self.application.setStyle(fusion)
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
        palette.setColor(QPalette.ColorRole.BrightText, QColor(colors["danger"]))
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.Text,
            QColor(colors["disabled"]),
        )
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.ButtonText,
            QColor(colors["disabled"]),
        )
        self.application.setPalette(palette)
        self.application.setStyleSheet(
            """
            * {
                selection-background-color: %(accent)s;
                selection-color: %(highlighted)s;
            }
            QMainWindow, QDialog, QWidget#appV1PageViewport {
                background: %(window)s;
                color: %(text)s;
            }
            QToolBar {
                background: %(alternate)s;
                border: 0;
                border-bottom: 1px solid %(border)s;
                spacing: 6px;
                padding: 5px 8px;
            }
            QDockWidget::title {
                background: %(alternate)s;
                border-bottom: 1px solid %(border)s;
                padding: 7px 8px;
            }
            QGroupBox, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
            QListView, QTreeView, QTableView, QPlainTextEdit, QTextEdit {
                background: %(surface)s;
                border: 1px solid %(border)s;
                border-radius: 6px;
                padding: 4px 6px;
                min-height: 22px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
            QDoubleSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
                border: 2px solid %(accent)s;
                padding: 3px 5px;
            }
            QComboBox::drop-down {
                border: 0;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: %(surface)s;
                border: 1px solid %(border_emphasis)s;
                selection-background-color: %(accent)s;
            }
            QPushButton {
                background: %(button)s;
                border: 1px solid %(border)s;
                border-radius: 6px;
                padding: 6px 12px;
                min-height: 22px;
            }
            QPushButton:hover {
                background: %(button_hover)s;
                border-color: %(border_emphasis)s;
            }
            QPushButton:focus { border: 2px solid %(accent)s; }
            QPushButton[primary="true"] {
                color: #ffffff;
                background: %(success)s;
                border-color: %(success)s;
            }
            QPushButton[danger="true"] {
                color: %(danger)s;
                background: %(button)s;
                border-color: %(border)s;
            }
            QPushButton:disabled { color: %(muted)s; }
            QMenu {
                background: %(surface)s;
                color: %(text)s;
                border: 1px solid %(border)s;
                padding: 4px;
            }
            QMenu::item { padding: 6px 24px 6px 10px; border-radius: 4px; }
            QMenu::item:selected { background: %(alternate)s; }
            QTabWidget::pane {
                border: 1px solid %(border)s;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: %(alternate)s;
                border: 1px solid %(border)s;
                padding: 7px 12px;
            }
            QTabBar::tab:selected {
                color: %(accent)s;
                background: %(surface)s;
                border-bottom-color: %(surface)s;
            }
            QTreeView::item { min-height: 25px; padding: 1px 4px; }
            QTreeView::item:selected {
                background: %(accent)s;
                color: %(highlighted)s;
            }
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
                padding: 3px 8px;
            }
            QLabel[status="success"] { color: %(success)s; }
            QLabel[status="warning"] { color: %(warning)s; }
            QLabel[status="danger"] { color: %(danger)s; }
            QScrollArea {
                background: %(window)s;
                border: 0;
            }
            QScrollBar:vertical {
                background: %(alternate)s;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: %(border_emphasis)s;
                border-radius: 5px;
                min-height: 28px;
            }
            """
            % colors
        )
        self.theme_changed.emit(self.mode, effective)
        return effective

    def _system_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        if self.mode == "auto":
            self.apply()


__all__ = [
    "APP_V1_THEME_MODES",
    "AppV1ThemeController",
    "AppV1ThemeMode",
    "normalize_app_v1_theme_mode",
]
