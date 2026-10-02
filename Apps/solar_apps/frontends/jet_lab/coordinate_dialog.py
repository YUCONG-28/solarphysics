"""Native pixel entry with explicit accessible controls and remembered values."""

import math

from PyQt6.QtCore import QSignalBlocker, QLocale
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QWidget,
)

from .accessibility import identify_widgets, screenshot_button


class NativeCoordinateDialog(QDialog):
    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.number = owner.tie_id.value()
        self.setObjectName("native_coordinate_dialog")
        self.setAccessibleName("原图精确坐标")
        self.setWindowTitle(f"编号 {self.number} · 原图精确坐标 · 零起点像素")
        self.form = QFormLayout(self)
        self.side_group = QButtonGroup(self)
        sides = QWidget(self)
        row = QHBoxLayout(sides)
        row.setContentsMargins(0, 0, 0, 0)
        self.side_buttons = []
        for index, title in enumerate(("AIA / 左侧", "EUVI / 右侧")):
            button = QRadioButton(title)
            button.setObjectName(
                "coordinate_side_aia" if index == 0 else "coordinate_side_euvi"
            )
            button.setAccessibleName(title)
            button.setEnabled(owner.panes[index].doc is not None)
            self.side_group.addButton(button, index)
            row.addWidget(button)
            self.side_buttons.append(button)
        self.x = QLineEdit(self)
        self.y = QLineEdit(self)
        for axis, widget in (("x", self.x), ("y", self.y)):
            widget.setObjectName("coordinate_" + axis)
            widget.setAccessibleName("原生 " + axis + " 数值输入")
            validator = QDoubleValidator(0, 100000, 8, widget)
            validator.setNotation(QDoubleValidator.Notation.StandardNotation)
            locale = QLocale.c()
            locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
            validator.setLocale(locale)
            widget.setValidator(validator)
        self.form.addRow("原始视角", sides)
        self.form.addRow("原生 x（像素）", self.x)
        self.form.addRow("原生 y（像素）", self.y)
        self.note = QLabel("坐标使用当前原图；修改只在点击“添加/移动标点”后应用。")
        self.note.setWordWrap(True)
        self.form.addRow(self.note)
        actions = QWidget(self)
        actions_row = QHBoxLayout(actions)
        actions_row.setContentsMargins(0, 0, 0, 0)
        self.apply_button = QPushButton("添加/移动标点")
        self.apply_button.setObjectName("coordinate_apply")
        self.apply_button.setAccessibleName(self.apply_button.text())
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("coordinate_cancel")
        self.cancel_button.setAccessibleName(self.cancel_button.text())
        self.cancel_button.clicked.connect(self.reject)
        actions_row.addWidget(self.apply_button)
        actions_row.addWidget(self.cancel_button)
        self.form.addRow(actions)
        self.form.addRow(screenshot_button(owner, self))
        self._local_values = {}
        self._bounds = None
        self.index = None
        chosen = owner.active.currentIndex()
        if owner.panes[chosen].doc is None:
            chosen = next((i for i, p in enumerate(owner.panes) if p.doc), chosen)
        self.side_buttons[chosen].setChecked(True)
        self.switch_side(chosen)
        # AX may set a radio's checked property without sending a mouse click.
        # Toggled tracks the actual selected state for mouse, keyboard and AX.
        for index, button in enumerate(self.side_buttons):
            button.toggled.connect(
                lambda checked, i=index: self.switch_side(i) if checked else None
            )
        self.x.textChanged.connect(self.validate_inputs)
        self.y.textChanged.connect(self.validate_inputs)
        identify_widgets(self, "jet_native_coordinate")
        self.x.setFocus()
        self.x.selectAll()

    def key_for(self, index):
        document = self.owner.panes[index].doc
        return (document.sha256 if document else None, self.number, index)

    def switch_side(self, index):
        if self.index is not None:
            self._local_values[self.key_for(self.index)] = (
                self.x.text(),
                self.y.text(),
            )
        self.index = index
        document = self.owner.panes[index].doc
        self.apply_button.setEnabled(document is not None)
        self.x.setEnabled(document is not None)
        self.y.setEnabled(document is not None)
        if document is None:
            self._bounds = None
            self.note.setText("此侧没有原图；请先加载观测。")
            return
        height, width = document.raw.shape
        self._bounds = (width - 1, height - 1)
        existing = next(
            (
                p
                for p in document.state.get("tiepoints", [])
                if p["number"] == self.number
            ),
            None,
        )
        key = self.key_for(index)
        remembered = getattr(self.owner, "_coordinate_dialog_values", {})
        values = self._local_values.get(key)
        if values is None:
            values = (
                existing["pixel_xy"]
                if existing
                else remembered.get(key, ((width - 1) / 2, (height - 1) / 2))
            )
        blockers = [QSignalBlocker(self.x), QSignalBlocker(self.y)]
        for widget, value, maximum in zip((self.x, self.y), values, self._bounds):
            widget.validator().setRange(0, maximum, 8)
            widget.setText(
                value
                if isinstance(value, str)
                else format(float(value), ".8f").rstrip("0").rstrip(".") or "0"
            )
        del blockers
        self.note.setText(
            f"编号 {self.number} · 原图范围 x=0–{width-1}，y=0–{height-1}。"
            "点击“添加/移动标点”后应用；取消保留原标注。"
        )
        self.validate_inputs()

    def coordinate_values(self):
        if self._bounds is None or not all(
            widget.hasAcceptableInput() for widget in (self.x, self.y)
        ):
            return None
        try:
            values = [float(self.x.text()), float(self.y.text())]
        except ValueError:
            return None
        if any(
            not math.isfinite(v) or not 0 <= v <= maximum
            for v, maximum in zip(values, self._bounds)
        ):
            return None
        return values

    def validate_inputs(self, *_):
        valid = self.coordinate_values() is not None
        self.apply_button.setEnabled(valid)
        for widget in (self.x, self.y):
            widget.setProperty("invalidCoordinate", not widget.hasAcceptableInput())
        return valid

    def accept(self):
        if self.owner.panes[self.index].doc is not None and self.validate_inputs():
            super().accept()
        else:
            self.note.setText(
                "坐标尚未应用：请输入原图范围内的有限数字，使用小数点，不自动裁剪越界坐标。"
            )


def edit_native_coordinate(owner):
    if not owner.ensure_point_metadata_applied("输入精确坐标"):
        return False
    dialog = NativeCoordinateDialog(owner)
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return False
        index, xy = dialog.index, dialog.coordinate_values()
        if not hasattr(owner, "_coordinate_dialog_values"):
            owner._coordinate_dialog_values = {}
        owner._coordinate_dialog_values[dialog.key_for(index)] = tuple(xy)
        owner.perform(lambda: owner.mark_native_point(index, xy), index)
        return True
    finally:
        dialog.deleteLater()
