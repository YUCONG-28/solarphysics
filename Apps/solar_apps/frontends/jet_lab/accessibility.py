"""Stable native control identities and user-requested window screenshots."""

from datetime import datetime
import hashlib
from pathlib import Path
import re

from PyQt6.QtCore import QObject, QEvent, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QAbstractSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QWidget,
)


def _belongs_to(owner, widget):
    # QWidget.isAncestorOf deliberately stops at a separate window; parented
    # QDialogs must still count as this application's own windows.
    while widget is not None:
        if widget is owner:
            return True
        widget = widget.parentWidget()
    return False


def _label(widget):
    if widget.accessibleName():
        return widget.accessibleName()
    if isinstance(widget, QAbstractButton):
        return widget.text().replace("&", "")
    if isinstance(widget, QDialog):
        return widget.windowTitle()
    parent = widget.parentWidget()
    if parent and isinstance(parent.layout(), QFormLayout):
        label = parent.layout().labelForField(widget)
        if isinstance(label, QLabel):
            return label.text().replace("&", "")
    if isinstance(widget, QLineEdit) and isinstance(parent, QAbstractSpinBox):
        return (parent.accessibleName() or parent.objectName()) + "数值"
    return ""


def identify_widgets(root, prefix="jet_lab"):
    """Give missing names and distinct AX identifiers without changing old names.

    Identifiers derive from the widget tree, not memory addresses, displayed
    values or annotation content. They remain unchanged while a widget lives.
    Existing Qt object names may legitimately repeat; full-path identifiers do
    not. This improves discovery but does not assume any AX backend is perfect.
    """
    # Explicit frontend attributes produce semantic IDs even after layout moves.
    for holder in [root, *root.findChildren(QWidget)]:
        for name, widget in vars(holder).items():
            if (
                isinstance(widget, QWidget)
                and not widget.objectName()
                and not name.startswith("_")
            ):
                widget.setObjectName(name)

    def visit(widget, parent_path, ordinal):
        original_name = widget.objectName()
        component = original_name or widget.metaObject().className()
        component = re.sub(r"[^\w.-]+", "_", component).strip("_") or "widget"
        path = f"{parent_path}.{component}_{ordinal}"
        if not original_name:
            widget.setObjectName("jet_" + hashlib.sha1(path.encode()).hexdigest()[:14])
        if not widget.accessibleName():
            name = _label(widget)
            if name:
                widget.setAccessibleName(name)
        if (
            hasattr(widget, "accessibleIdentifier")
            and not widget.accessibleIdentifier()
        ):
            widget.setAccessibleIdentifier(path)
        counts = {}
        for child in widget.findChildren(
            QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly
        ):
            key = child.objectName() or child.metaObject().className()
            counts[key] = counts.get(key, 0) + 1
            visit(child, path, counts[key])

    visit(root, prefix, 1)


def save_window_screenshot(owner, target=None):
    """Save only this application's chosen window after a user-selected path."""
    target = target or QApplication.activeWindow() or owner
    if not _belongs_to(owner, target):
        target = owner
    target = target.window()
    filename = "jet_lab_window_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    selected, _ = QFileDialog.getSaveFileName(
        target,
        "保存当前窗口截图",
        str(Path(owner.allowed_roots[0]) / filename),
        "PNG (*.png)",
    )
    if not selected:
        return None
    try:
        candidate = Path(selected)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".png")
        if candidate.suffix.lower() != ".png":
            raise ValueError("窗口截图请使用 .png 文件名")
        destination = owner.validate(candidate, "save_file")
        if not target.grab().save(str(destination), "PNG"):
            raise OSError("窗口截图写入失败")
        owner.last_screenshot = str(destination)
        owner.statusBar().showMessage("窗口截图已保存：" + str(destination), 15000)
        return destination
    except Exception as error:
        owner.statusBar().showMessage("截图未保存：" + str(error), 15000)
        return None


def screenshot_button(owner, target=None):
    button = QPushButton("保存窗口截图")
    button.setObjectName("save_window_screenshot")
    button.setAccessibleName(button.text())
    button.setToolTip("只保存当前 Jet Lab 窗口，供记录操作或反馈问题；不截取整个屏幕。")
    button.clicked.connect(lambda: save_window_screenshot(owner, target))
    return button


class _WindowAccessibility(QObject):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.closed = False
        identify_widgets(owner)
        QApplication.instance().installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(self.close)

    def eventFilter(self, watched, event):
        if self.closed:
            return False
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget):
            if _belongs_to(self.owner, watched) and not watched.accessibleIdentifier():
                # Scan the owner's tree once per newly populated dialog. Giving
                # each just-shown child its own namespace would repeat IDs.
                identify_widgets(self.owner)
        return False

    def prepare(self, dialog):
        identify_widgets(self.owner if _belongs_to(self.owner, dialog) else dialog)

    def close(self):
        if self.closed:
            return
        self.closed = True
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self)
            try:
                app.aboutToQuit.disconnect(self.close)
            except (TypeError, RuntimeError):
                pass
        self.owner = None


def install_accessibility(owner):
    """Install once after constructing the main window and its auxiliary panel."""
    existing = getattr(owner, "_jet_accessibility", None)
    if existing:
        return existing
    menu = owner.menuBar().addMenu("帮助")
    action = QAction("保存窗口截图", owner)
    action.setObjectName("save_window_screenshot_action")
    action.setShortcut(QKeySequence("Ctrl+Shift+P"))
    action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
    action.triggered.connect(lambda: save_window_screenshot(owner))
    menu.addAction(action)
    owner.addAction(action)
    panel = getattr(owner, "parameter_panel", None)
    if isinstance(panel, QStackedWidget) and panel.count() > 1:
        auxiliary = panel.widget(1)
        auxiliary.layout().insertWidget(1, screenshot_button(owner, owner))
    owner._jet_accessibility = _WindowAccessibility(owner)
    return owner._jet_accessibility
