# SPDX-License-Identifier: GPL-3.0-only
"""Native visual editor for typed App 1.0 workflow graphs."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QMimeData, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QDrag, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .components import ArtifactBrowser, RunConfirmationDialog
from .flow_execution import FlowExecutionController
from .flow_store import AppV1FlowStore
from .flows import AppV1FlowV1, FlowEdgeV1, FlowNodeV1, FunctionCatalog
from .function_catalog import DEFAULT_FUNCTION_CATALOG
from .runtime import AppV1RuntimePaths
from .schema_form import SchemaForm

_FUNCTION_MIME = "application/x-solar-app-v1-function"


class _FunctionList(QListWidget):
    def startDrag(self, supported_actions) -> None:  # type: ignore[no-untyped-def]
        item = self.currentItem()
        if item is None or not item.data(Qt.ItemDataRole.UserRole):
            return
        mime = QMimeData()
        mime.setData(
            _FUNCTION_MIME,
            str(item.data(Qt.ItemDataRole.UserRole)).encode("utf-8"),
        )
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(supported_actions)


class _GraphView(QGraphicsView):
    function_dropped = pyqtSignal(str, float, float)

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setAcceptDrops(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_FUNCTION_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_FUNCTION_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not event.mimeData().hasFormat(_FUNCTION_MIME):
            super().dropEvent(event)
            return
        function_id = bytes(event.mimeData().data(_FUNCTION_MIME)).decode("utf-8")
        point = self.mapToScene(event.position().toPoint())
        self.function_dropped.emit(function_id, point.x(), point.y())
        event.acceptProposedAction()


class _NodeItem(QGraphicsRectItem):
    def __init__(self, node: FlowNodeV1, title: str) -> None:
        super().__init__(0, 0, 210, 76)
        self.node_id = node.node_id
        self.setPos(QPointF(node.x, node.y))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setPen(QPen(Qt.GlobalColor.gray, 1.2))
        label = QGraphicsTextItem(
            f"{title}\n{node.node_id}"
            + ("\nDisabled" if node.disabled else ""),
            self,
        )
        label.setTextWidth(194)
        label.setPos(8, 6)


class WorkflowBuilder(QWidget):
    """Compose, validate, save, restore, and execute typed native workflows."""

    flow_saved = pyqtSignal(str)
    artifact_ready = pyqtSignal(str)

    def __init__(
        self,
        layout: RuntimeLayout,
        *,
        catalog: FunctionCatalog = DEFAULT_FUNCTION_CATALOG,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.layout = layout
        self.runtime = AppV1RuntimePaths.from_layout(layout)
        self.catalog = catalog
        self.store = AppV1FlowStore(self.runtime)
        self.executor = FlowExecutionController(layout, catalog, self)
        try:
            roots = configured_allowed_roots(workspace_root=layout.repo_root)
        except (OSError, TypeError, ValueError):
            roots = ()
        self._flow = AppV1FlowV1("untitled-flow", "Untitled Flow")
        self._undo: list[dict[str, object]] = []
        self._redo: list[dict[str, object]] = []
        self._node_items: dict[str, _NodeItem] = {}
        self._selected_node_id: str | None = None
        self._loading_form = False

        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.flow_name = QLineEdit("Untitled Flow")
        self.flow_name.setMinimumWidth(220)
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 4)
        save = QPushButton("Save")
        save.clicked.connect(self.save)
        self.saved_flows = QComboBox()
        self.saved_flows.setMinimumContentsLength(18)
        open_button = QPushButton("Open")
        open_button.clicked.connect(self.open_selected)
        undo = QPushButton("Undo")
        undo.clicked.connect(self.undo)
        redo = QPushButton("Redo")
        redo.clicked.connect(self.redo)
        validate = QPushButton("Validate")
        validate.clicked.connect(self.validate)
        self.run_button = QPushButton("Review and run flow")
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(self.run_flow)
        toolbar.addWidget(QLabel("Flow"))
        toolbar.addWidget(self.flow_name, 1)
        toolbar.addWidget(QLabel("Workers"))
        toolbar.addWidget(self.concurrency)
        toolbar.addWidget(save)
        toolbar.addWidget(self.saved_flows)
        toolbar.addWidget(open_button)
        toolbar.addWidget(undo)
        toolbar.addWidget(redo)
        toolbar.addWidget(validate)
        toolbar.addWidget(self.run_button)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.functions = _FunctionList()
        self.functions.setDragEnabled(True)
        self.functions.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        for category in sorted({item.category for item in catalog.functions}):
            header = QListWidgetItem(category)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.functions.addItem(header)
            for function in catalog.functions:
                if function.category != category:
                    continue
                item = QListWidgetItem(f"  {function.title}")
                item.setData(Qt.ItemDataRole.UserRole, function.function_id)
                item.setToolTip(function.description)
                self.functions.addItem(item)
        self.functions.itemDoubleClicked.connect(self._add_from_item)
        splitter.addWidget(self.functions)

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-1000, -1000, 4000, 3000)
        self.scene.selectionChanged.connect(self._selection_changed)
        self.view = _GraphView(self.scene)
        self.view.function_dropped.connect(self.add_node)
        splitter.addWidget(self.view)

        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        self.selected_title = QLabel("Select a node")
        self.selected_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        self.selected_help = QLabel()
        self.selected_help.setWordWrap(True)
        self.selected_help.setProperty("muted", True)
        self.disabled = QCheckBox("Disable this node")
        self.disabled.toggled.connect(self._node_options_changed)
        self.group = QLineEdit()
        self.group.setPlaceholderText("Optional group")
        self.group.editingFinished.connect(self._node_options_changed)
        self.form = SchemaForm(allowed_roots=tuple(map(str, roots)))
        self.form.values_changed.connect(self._form_changed)
        node_actions = QHBoxLayout()
        duplicate = QPushButton("Duplicate")
        duplicate.clicked.connect(self.duplicate_selected)
        delete = QPushButton("Delete")
        delete.clicked.connect(self.delete_selected)
        node_actions.addWidget(duplicate)
        node_actions.addWidget(delete)
        self.source_port = QComboBox()
        self.target_node = QComboBox()
        self.target_port = QComboBox()
        self.target_node.currentIndexChanged.connect(self._refresh_target_ports)
        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.connect_selected)
        inspector_layout.addWidget(self.selected_title)
        inspector_layout.addWidget(self.selected_help)
        inspector_layout.addWidget(self.disabled)
        inspector_layout.addWidget(self.group)
        inspector_layout.addLayout(node_actions)
        inspector_layout.addWidget(QLabel("Output port"))
        inspector_layout.addWidget(self.source_port)
        inspector_layout.addWidget(QLabel("Target node"))
        inspector_layout.addWidget(self.target_node)
        inspector_layout.addWidget(QLabel("Target input"))
        inspector_layout.addWidget(self.target_port)
        inspector_layout.addWidget(connect_button)
        inspector_layout.addWidget(self.form, 1)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 720, 360])
        root.addWidget(splitter, 1)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.status_table = QTableWidget(0, 3)
        self.status_table.setHorizontalHeaderLabels(["Node", "Function", "Status"])
        self.artifacts = ArtifactBrowser()
        bottom.addWidget(self.status_table)
        bottom.addWidget(self.artifacts)
        bottom.setSizes([480, 620])
        root.addWidget(bottom)
        self.status = QLabel("Drag functions onto the canvas or double-click them.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.executor.node_changed.connect(self._node_execution_changed)
        self.executor.log_line.connect(self._execution_log)
        self.executor.artifact_ready.connect(self._execution_artifact)
        self.executor.flow_finished.connect(self._execution_finished)
        self._refresh_saved_flows()
        self._render_graph()

    @property
    def flow(self) -> AppV1FlowV1:
        self._commit_selected_form()
        self._capture_positions()
        return replace(
            self._flow,
            name=self.flow_name.text().strip() or self._flow.name,
            concurrency=self.concurrency.value(),
        )

    def load_flow(
        self,
        flow: AppV1FlowV1,
        *,
        reset_history: bool = True,
    ) -> None:
        if self.executor.running:
            raise RuntimeError("Cannot replace a running workflow")
        self._flow = flow
        if reset_history:
            self._undo.clear()
            self._redo.clear()
        self.flow_name.setText(flow.name)
        self.concurrency.setValue(flow.concurrency)
        self._selected_node_id = None
        self._render_graph()
        self.status.setText(f"Loaded {flow.name}")

    def add_node(
        self,
        function_id: str,
        x: float = 0.0,
        y: float = 0.0,
    ) -> None:
        self.catalog.get(function_id)
        self._checkpoint()
        node = FlowNodeV1(
            node_id=f"node-{uuid.uuid4().hex[:8]}",
            function_id=function_id,
            x=float(x),
            y=float(y),
        )
        self._flow = replace(self._flow, nodes=(*self._flow.nodes, node))
        self._render_graph(select=node.node_id)

    def duplicate_selected(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        self._checkpoint()
        copy = replace(
            node,
            node_id=f"node-{uuid.uuid4().hex[:8]}",
            x=node.x + 30,
            y=node.y + 100,
        )
        self._flow = replace(self._flow, nodes=(*self._flow.nodes, copy))
        self._render_graph(select=copy.node_id)

    def delete_selected(self) -> None:
        node = self._selected_node()
        if node is None:
            return
        self._checkpoint()
        self._flow = replace(
            self._flow,
            nodes=tuple(item for item in self._flow.nodes if item.node_id != node.node_id),
            edges=tuple(
                edge
                for edge in self._flow.edges
                if edge.source_node != node.node_id and edge.target_node != node.node_id
            ),
        )
        self._selected_node_id = None
        self._render_graph()

    def connect_selected(self) -> None:
        source = self._selected_node()
        target_id = self.target_node.currentData()
        if source is None or not target_id:
            return
        try:
            edge = FlowEdgeV1(
                source.node_id,
                str(self.source_port.currentData()),
                str(target_id),
                str(self.target_port.currentData()),
            )
            candidate = replace(self._flow, edges=(*self._flow.edges, edge))
            source_function = self.catalog.get(source.function_id)
            target = next(
                item for item in candidate.nodes if item.node_id == edge.target_node
            )
            target_function = self.catalog.get(target.function_id)
            source_spec = next(
                item
                for item in source_function.outputs
                if item.port_id == edge.source_port
            )
            target_spec = next(
                item
                for item in target_function.inputs
                if item.port_id == edge.target_port
            )
            if not target_spec.accepts(source_spec.artifact_types):
                raise ValueError(
                    f"{source_spec.label} is incompatible with {target_spec.label}"
                )
            if not target_spec.multiple and any(
                item.target_node == edge.target_node
                and item.target_port == edge.target_port
                for item in self._flow.edges
            ):
                raise ValueError(f"{target_spec.label} accepts one connection")
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            self.status.setText(f"{type(exc).__name__}: {exc}")
            return
        self._checkpoint()
        self._flow = candidate
        self._render_graph(select=source.node_id)

    def validate(self) -> bool:
        try:
            flow = self.flow
            self.catalog.validate_flow(flow)
        except (KeyError, TypeError, ValueError) as exc:
            self.status.setText(f"{type(exc).__name__}: {exc}")
            return False
        self.status.setText(
            f"Valid schema-1 workflow: {len(flow.nodes)} nodes, "
            f"{len(flow.edges)} connections"
        )
        return True

    def save(self) -> Path | None:
        if not self.validate():
            return None
        flow = self.flow
        target = self.store.save(flow)
        self._flow = flow
        self._refresh_saved_flows(flow.flow_id)
        self.flow_saved.emit(flow.flow_id)
        self.status.setText(f"Saved {target}")
        return target

    def open_selected(self) -> None:
        flow_id = self.saved_flows.currentData()
        if flow_id:
            self.load_flow(self.store.load(str(flow_id)))

    def run_flow(self) -> None:
        if not self.validate():
            return
        flow = self.flow
        summary = {
            "Module": "Workflow Builder",
            "Input": f"{len(flow.edges)} typed artifact connection(s)",
            "Parameters": (
                f"{len(flow.nodes)} node(s); concurrency={flow.concurrency}; "
                "all values are schema validated"
            ),
            "Output": str(self.runtime.outputs_dir / "preview"),
            "Workload": "Independent branches continue if another branch fails",
        }
        if not RunConfirmationDialog.confirm(
            self,
            f"Run {flow.name}",
            summary,
        ):
            return
        self.executor.run(flow)
        self._refresh_status_table()
        self.status.setText("Workflow started")

    def undo(self) -> None:
        if not self._undo or self.executor.running:
            return
        self._redo.append(self.flow.to_dict())
        payload = self._undo.pop()
        self.load_flow(AppV1FlowStore.from_dict(payload), reset_history=False)

    def redo(self) -> None:
        if not self._redo or self.executor.running:
            return
        self._undo.append(self.flow.to_dict())
        payload = self._redo.pop()
        self.load_flow(AppV1FlowStore.from_dict(payload), reset_history=False)

    def shutdown(self) -> None:
        self.executor.shutdown()

    def _add_from_item(self, item: QListWidgetItem) -> None:
        function_id = item.data(Qt.ItemDataRole.UserRole)
        if function_id:
            offset = len(self._flow.nodes) * 35
            self.add_node(str(function_id), offset, offset)

    def _checkpoint(self) -> None:
        self._commit_selected_form()
        self._capture_positions()
        self._undo.append(self.flow.to_dict())
        del self._undo[:-100]
        self._redo.clear()

    def _selected_node(self) -> FlowNodeV1 | None:
        return next(
            (
                node
                for node in self._flow.nodes
                if node.node_id == self._selected_node_id
            ),
            None,
        )

    def _selection_changed(self) -> None:
        selected = [
            item for item in self.scene.selectedItems() if isinstance(item, _NodeItem)
        ]
        new_id = selected[0].node_id if selected else None
        if new_id == self._selected_node_id:
            return
        self._commit_selected_form()
        self._selected_node_id = new_id
        self._load_selected_form()

    def _load_selected_form(self) -> None:
        node = self._selected_node()
        self._loading_form = True
        try:
            if node is None:
                self.selected_title.setText("Select a node")
                self.selected_help.clear()
                self.disabled.setChecked(False)
                self.group.clear()
                self.form.set_function(None)
                self.source_port.clear()
                self.target_node.clear()
                self.target_port.clear()
                return
            function = self.catalog.get(node.function_id)
            self.selected_title.setText(function.title)
            self.selected_help.setText(function.description)
            self.disabled.setChecked(node.disabled)
            self.group.setText(node.group)
            self.form.set_function(
                function,
                node.parameters,
                variant_id=node.variant_id,
            )
            self.source_port.clear()
            for port in function.outputs:
                self.source_port.addItem(port.label, port.port_id)
            self.target_node.clear()
            for candidate in self._flow.nodes:
                if candidate.node_id != node.node_id:
                    target = self.catalog.get(candidate.function_id)
                    self.target_node.addItem(
                        f"{target.title} · {candidate.node_id}",
                        candidate.node_id,
                    )
            self._refresh_target_ports()
        finally:
            self._loading_form = False

    def _commit_selected_form(self) -> None:
        if self._loading_form:
            return
        node = self._selected_node()
        if node is None or self.form.function_spec is None:
            return
        try:
            values = self.form.values()
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        updated = replace(
            node,
            parameters=values,
            variant_id=self.form.selected_variant_id(),
            disabled=self.disabled.isChecked(),
            group=self.group.text().strip(),
        )
        self._flow = replace(
            self._flow,
            nodes=tuple(
                updated if item.node_id == node.node_id else item
                for item in self._flow.nodes
            ),
        )

    def _form_changed(self) -> None:
        if not self._loading_form:
            self._commit_selected_form()

    def _node_options_changed(self) -> None:
        if not self._loading_form:
            self._commit_selected_form()
            self._render_graph(select=self._selected_node_id)

    def _refresh_target_ports(self) -> None:
        self.target_port.clear()
        target_id = self.target_node.currentData()
        target = next(
            (node for node in self._flow.nodes if node.node_id == target_id),
            None,
        )
        if target is None:
            return
        for port in self.catalog.get(target.function_id).inputs:
            self.target_port.addItem(port.label, port.port_id)

    def _capture_positions(self) -> None:
        if not self._node_items:
            return
        self._flow = replace(
            self._flow,
            nodes=tuple(
                replace(
                    node,
                    x=self._node_items[node.node_id].pos().x(),
                    y=self._node_items[node.node_id].pos().y(),
                )
                for node in self._flow.nodes
            ),
        )

    def _render_graph(self, *, select: str | None = None) -> None:
        self.scene.clear()
        self._node_items.clear()
        for node in self._flow.nodes:
            function = self.catalog.get(node.function_id)
            item = _NodeItem(node, function.title)
            self.scene.addItem(item)
            self._node_items[node.node_id] = item
        for edge in self._flow.edges:
            source = self._node_items[edge.source_node]
            target = self._node_items[edge.target_node]
            start = source.sceneBoundingRect().center()
            end = target.sceneBoundingRect().center()
            line = QGraphicsLineItem(start.x(), start.y(), end.x(), end.y())
            line.setPen(QPen(Qt.GlobalColor.darkGray, 2))
            line.setZValue(-1)
            self.scene.addItem(line)
        self._selected_node_id = select
        if select in self._node_items:
            self._node_items[select].setSelected(True)
        self._load_selected_form()
        self._refresh_status_table()

    def _refresh_saved_flows(self, selected: str | None = None) -> None:
        self.saved_flows.clear()
        for path in self.store.list_flows():
            flow_id = path.name.removesuffix(".spflow.json")
            self.saved_flows.addItem(flow_id, flow_id)
        if selected:
            index = self.saved_flows.findData(selected)
            self.saved_flows.setCurrentIndex(max(0, index))

    def _refresh_status_table(self) -> None:
        self.status_table.setRowCount(len(self._flow.nodes))
        for row, node in enumerate(self._flow.nodes):
            function = self.catalog.get(node.function_id)
            state = self.executor.states.get(node.node_id)
            values = (node.node_id, function.title, state.status if state else "not run")
            for column, value in enumerate(values):
                self.status_table.setItem(row, column, QTableWidgetItem(value))
        self.status_table.resizeColumnsToContents()

    def _node_execution_changed(self, node_id: str) -> None:
        self._refresh_status_table()
        state = self.executor.states[node_id]
        self.status.setText(f"{node_id}: {state.status} {state.error}".strip())

    def _execution_log(self, node_id: str, line: str) -> None:
        self.status.setText(f"[{node_id}] {line}")

    def _execution_artifact(self, _node_id: str, path: str) -> None:
        self.artifacts.open_path(path)
        self.artifact_ready.emit(path)

    def _execution_finished(self, summary: object) -> None:
        self._refresh_status_table()
        self.status.setText(f"Workflow complete: {summary}")


__all__ = ["WorkflowBuilder"]
