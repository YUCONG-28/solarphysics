# SPDX-License-Identifier: GPL-3.0-only
"""Qt parameter forms generated exclusively from :mod:`function_specs`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QLayout,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .basic_services import AllowedPathField
from .function_specs import FunctionSpec, ParameterSpec

_MINIMUM = -2_147_483_647
_MAXIMUM = 2_147_483_647


class SchemaForm(QWidget):
    """The only typed business-parameter editor in App 1.0."""

    values_changed = pyqtSignal()

    def __init__(
        self,
        *,
        allowed_roots: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.allowed_roots = tuple(allowed_roots)
        self.setMinimumHeight(300)
        self.function_spec: FunctionSpec | None = None
        self._editors: dict[str, QWidget] = {}
        self._rows: dict[str, tuple[QLabel, QWidget]] = {}
        root = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search parameters…")
        self.search.textChanged.connect(self._filter_rows)
        root.addWidget(self.search)
        self.variant_label = QLabel("Scientific algorithm")
        self.variant = QComboBox()
        self.variant.currentIndexChanged.connect(self.values_changed)
        root.addWidget(self.variant_label)
        root.addWidget(self.variant)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setSizeConstraint(
            QLayout.SizeConstraint.SetMinAndMaxSize
        )
        self.common_group = QGroupBox("Common")
        self.common_form = QFormLayout(self.common_group)
        self.common_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.common_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.advanced_group = QGroupBox("Advanced")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.toggled.connect(self._advanced_toggled)
        self.advanced_form = QFormLayout(self.advanced_group)
        self.advanced_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.advanced_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapLongRows
        )
        self.content_layout.addWidget(self.common_group)
        self.content_layout.addWidget(self.advanced_group)
        self.content_layout.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)
        self.set_function(None)

    def set_function(
        self,
        function_spec: FunctionSpec | None,
        values: Mapping[str, Any] | None = None,
        *,
        variant_id: str | None = None,
    ) -> None:
        self._clear_form(self.common_form)
        self._clear_form(self.advanced_form)
        self.function_spec = function_spec
        self._editors.clear()
        self._rows.clear()
        self.variant.clear()
        if function_spec is None:
            self.variant_label.hide()
            self.variant.hide()
            self.common_group.hide()
            self.advanced_group.hide()
            return
        family = function_spec.variant_family
        if family is None:
            self.variant_label.hide()
            self.variant.hide()
        else:
            self.variant_label.show()
            self.variant.show()
            for variant in family.variants:
                suffix = " · Recommended" if variant.is_primary else ""
                self.variant.addItem(f"{variant.title}{suffix}", variant.variant_id)
                index = self.variant.count() - 1
                self.variant.setItemData(
                    index,
                    (
                        f"{variant.description}\nCompatibility: "
                        f"{variant.compatibility}\nTests: {variant.test_status}"
                    ),
                    Qt.ItemDataRole.ToolTipRole,
                )
            selected = self.variant.findData(variant_id or family.primary.variant_id)
            self.variant.setCurrentIndex(max(0, selected))
        supplied = dict(values or {})
        for parameter in function_spec.parameters:
            editor = self._make_editor(parameter)
            value = supplied.get(parameter.parameter_id, parameter.default)
            self._set_editor_value(parameter, editor, value)
            label = QLabel(self._label_text(parameter))
            label.setToolTip(parameter.help_text)
            editor.setToolTip(parameter.help_text)
            form = (
                self.advanced_form
                if parameter.group == "advanced"
                else self.common_form
            )
            form.addRow(label, editor)
            self._editors[parameter.parameter_id] = editor
            self._rows[parameter.parameter_id] = (label, editor)
        self.common_group.show()
        self.advanced_group.setVisible(
            any(item.group == "advanced" for item in function_spec.parameters)
        )
        self._filter_rows(self.search.text())

    def values(self) -> dict[str, Any]:
        if self.function_spec is None:
            return {}
        raw = {
            parameter.parameter_id: self._editor_value(
                parameter,
                self._editors[parameter.parameter_id],
            )
            for parameter in self.function_spec.parameters
        }
        return self.function_spec.normalize_parameters(raw)

    def selected_variant_id(self) -> str | None:
        if self.function_spec is None or self.function_spec.variant_family is None:
            return None
        value = self.variant.currentData()
        return None if value in (None, "") else str(value)

    def reset_defaults(self) -> None:
        self.set_function(self.function_spec)
        self.values_changed.emit()

    @staticmethod
    def _clear_form(form: QFormLayout) -> None:
        while form.count():
            item = form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _make_editor(self, spec: ParameterSpec) -> QWidget:
        if spec.kind == "boolean":
            checkbox = QCheckBox()
            if spec.default is None and not spec.required:
                checkbox.setTristate(True)
            editor: QWidget = checkbox
        elif spec.kind == "integer" and (
            spec.default is not None or spec.required
        ):
            integer = QSpinBox()
            integer.setRange(
                _MINIMUM if spec.minimum is None else int(spec.minimum),
                _MAXIMUM if spec.maximum is None else int(spec.maximum),
            )
            integer.setSuffix(f" {spec.unit}" if spec.unit else "")
            editor = integer
        elif spec.kind == "number" and (
            spec.default is not None or spec.required
        ):
            number = QDoubleSpinBox()
            number.setDecimals(8)
            number.setRange(
                -1e100 if spec.minimum is None else float(spec.minimum),
                1e100 if spec.maximum is None else float(spec.maximum),
            )
            number.setSuffix(f" {spec.unit}" if spec.unit else "")
            editor = number
        elif spec.kind == "enum":
            combo = QComboBox()
            if spec.default is None and not spec.required:
                combo.addItem("(unset)", None)
            combo.setMinimumContentsLength(
                max((len(str(item)) for item in spec.choices), default=8)
            )
            for choice in spec.choices:
                combo.addItem(str(choice), choice)
            editor = combo
        elif spec.kind in {"directory", "file"}:
            editor = AllowedPathField(
                self.allowed_roots,
                directory=spec.kind == "directory",
                extensions=spec.path_extensions,
            )
        else:
            line = QLineEdit()
            if spec.kind in {"list", "object", "roi"}:
                line.setPlaceholderText("JSON value")
            editor = line
        self._connect_editor(editor)
        return editor

    def _connect_editor(self, editor: QWidget) -> None:
        if isinstance(editor, QCheckBox):
            editor.toggled.connect(self.values_changed)
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.valueChanged.connect(self.values_changed)
        elif isinstance(editor, QComboBox):
            editor.currentIndexChanged.connect(self.values_changed)
        elif isinstance(editor, AllowedPathField):
            editor.path_changed.connect(self.values_changed)
        elif isinstance(editor, QLineEdit):
            editor.textChanged.connect(self.values_changed)

    @staticmethod
    def _set_editor_value(
        spec: ParameterSpec,
        editor: QWidget,
        value: Any,
    ) -> None:
        if isinstance(editor, QCheckBox):
            if editor.isTristate() and value is None:
                editor.setCheckState(Qt.CheckState.PartiallyChecked)
            else:
                editor.setChecked(bool(value))
        elif isinstance(editor, QSpinBox):
            editor.setValue(0 if value is None else int(value))
        elif isinstance(editor, QDoubleSpinBox):
            editor.setValue(0.0 if value is None else float(value))
        elif isinstance(editor, QComboBox):
            selected = editor.findData(value)
            editor.setCurrentIndex(max(0, selected))
        elif isinstance(editor, AllowedPathField):
            editor.setText("" if value is None else str(value))
        elif isinstance(editor, QLineEdit):
            if spec.kind in {"list", "object", "roi"}:
                rendered = (
                    ""
                    if value is None
                    else json.dumps(value, ensure_ascii=False, allow_nan=False)
                )
            else:
                rendered = "" if value is None else str(value)
            editor.setText(rendered)

    @staticmethod
    def _editor_value(spec: ParameterSpec, editor: QWidget) -> Any:
        if isinstance(editor, QCheckBox):
            if (
                editor.isTristate()
                and editor.checkState() == Qt.CheckState.PartiallyChecked
            ):
                return None
            return editor.isChecked()
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            return editor.value()
        if isinstance(editor, QComboBox):
            return editor.currentData()
        if isinstance(editor, AllowedPathField):
            text = editor.text().strip()
            return "" if not text else str(editor.validated_path())
        if isinstance(editor, QLineEdit):
            text = editor.text().strip()
            if spec.kind in {"list", "object", "roi"}:
                if not text:
                    return [] if spec.kind == "list" else {}
                return json.loads(text)
            return text
        raise TypeError(f"Unsupported schema editor: {type(editor).__name__}")

    @staticmethod
    def _label_text(spec: ParameterSpec) -> str:
        required = " *" if spec.required else ""
        unit = f" ({spec.unit})" if spec.unit else ""
        return f"{spec.label}{unit}{required}"

    def _filter_rows(self, text: str) -> None:
        query = text.strip().casefold()
        if self.function_spec is None:
            return
        for parameter in self.function_spec.parameters:
            visible = not query or query in (
                f"{parameter.parameter_id} {parameter.label} "
                f"{parameter.help_text}"
            ).casefold()
            if parameter.group == "advanced" and not self.advanced_group.isChecked():
                visible = False
            label, editor = self._rows[parameter.parameter_id]
            label.setVisible(visible)
            editor.setVisible(visible)
        self.common_group.setMinimumHeight(0)
        self.advanced_group.setMinimumHeight(0)
        self.common_group.setMinimumHeight(self.common_group.sizeHint().height())
        self.advanced_group.setMinimumHeight(self.advanced_group.sizeHint().height())

    def _advanced_toggled(self, _checked: bool) -> None:
        self._filter_rows(self.search.text())


__all__ = ["SchemaForm"]
