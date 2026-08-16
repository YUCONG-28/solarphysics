# SPDX-License-Identifier: GPL-3.0-only
"""Concurrent, dependency-aware execution for version-one workflow graphs."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .flows import AppV1FlowV1, FlowNodeV1, FunctionCatalog
from .runtime import AppV1RuntimePaths
from .tasks import TaskQueueController

_TERMINAL = frozenset({"succeeded", "failed", "blocked", "cancelled", "skipped"})


@dataclass(slots=True)
class FlowNodeExecution:
    node_id: str
    status: str = "pending"
    task_id: str | None = None
    output_dir: str | None = None
    artifacts: list[str] = field(default_factory=list)
    artifact_identities: dict[str, list[dict[str, object]]] = field(
        default_factory=dict
    )
    error: str = ""


class FlowExecutionController(QObject):
    """Run ready graph nodes FIFO across one to four supervised process lanes."""

    node_changed = pyqtSignal(str)
    log_line = pyqtSignal(str, str)
    artifact_ready = pyqtSignal(str, str)
    flow_finished = pyqtSignal(object)

    def __init__(
        self,
        layout: RuntimeLayout,
        catalog: FunctionCatalog,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.layout = layout
        self.runtime = AppV1RuntimePaths.from_layout(layout)
        try:
            configured = configured_allowed_roots(workspace_root=layout.repo_root)
        except OSError, TypeError, ValueError:
            configured = ()
        self.allowed_roots = (
            *configured,
            self.runtime.outputs_dir,
            self.runtime.workspaces_dir,
            self.runtime.tmp_dir,
        )
        self.catalog = catalog
        self.flow: AppV1FlowV1 | None = None
        self.states: dict[str, FlowNodeExecution] = {}
        self._lanes: list[TaskQueueController] = []
        self._lane_nodes: dict[TaskQueueController, str] = {}
        self._task_nodes: dict[str, str] = {}
        self._run_id = ""
        self._project_id = "preview"
        self._cancelled = False

    @property
    def running(self) -> bool:
        return any(state.status == "running" for state in self.states.values())

    def run(
        self,
        flow: AppV1FlowV1,
        *,
        project_id: str = "preview",
    ) -> None:
        if self.running:
            raise RuntimeError("A workflow is already running")
        self.catalog.validate_flow(flow)
        self.shutdown()
        self.flow = flow
        self._run_id = f"run-{uuid.uuid4().hex[:12]}"
        self._project_id = project_id
        self._cancelled = False
        self.states = {
            node.node_id: FlowNodeExecution(
                node.node_id,
                status="skipped" if node.disabled else "pending",
            )
            for node in flow.nodes
        }
        for _index in range(flow.concurrency):
            lane = TaskQueueController(self.layout, self)
            lane.task_changed.connect(
                lambda task_id, worker=lane: self._task_changed(worker, task_id)
            )
            lane.log_line.connect(self._task_log)
            lane.artifact_ready.connect(self._task_artifact)
            self._lanes.append(lane)
        for node in flow.nodes:
            if node.disabled:
                self.node_changed.emit(node.node_id)
        QTimer.singleShot(0, self._schedule)

    def cancel(self) -> None:
        self._cancelled = True
        for lane in self._lanes:
            current = lane.current_task_id
            if current is not None:
                lane.cancel(current)
        for state in self.states.values():
            if state.status == "pending":
                state.status = "cancelled"
                self.node_changed.emit(state.node_id)
        self._finish_if_complete()

    def retry_failed(self) -> None:
        if self.flow is None or self.running:
            return
        retry_nodes = {
            node_id
            for node_id, state in self.states.items()
            if state.status in {"failed", "blocked", "cancelled"}
        }
        if not retry_nodes:
            return
        for node_id in retry_nodes:
            self.states[node_id] = FlowNodeExecution(node_id)
            self.node_changed.emit(node_id)
        self._cancelled = False
        QTimer.singleShot(0, self._schedule)

    def shutdown(self) -> None:
        for lane in self._lanes:
            lane.shutdown()
            lane.deleteLater()
        self._lanes.clear()
        self._lane_nodes.clear()
        self._task_nodes.clear()

    def summary(self) -> dict[str, int]:
        result = {status: 0 for status in _TERMINAL}
        for state in self.states.values():
            if state.status in result:
                result[state.status] += 1
        return result

    def _schedule(self) -> None:
        flow = self.flow
        if flow is None or self._cancelled:
            self._finish_if_complete()
            return
        nodes = {node.node_id: node for node in flow.nodes}
        dependencies = flow.dependencies()
        changed = True
        while changed:
            changed = False
            for node_id in flow.topological_order():
                state = self.states[node_id]
                if state.status != "pending":
                    continue
                parent_states = [
                    self.states[item].status for item in dependencies[node_id]
                ]
                if any(
                    status in {"failed", "blocked", "cancelled", "skipped"}
                    for status in parent_states
                ):
                    state.status = "blocked"
                    state.error = "A dependency did not produce a usable artifact"
                    self.node_changed.emit(node_id)
                    changed = True
        available = [
            lane
            for lane in self._lanes
            if lane not in self._lane_nodes and lane.current_task_id is None
        ]
        for node_id in flow.topological_order():
            if not available:
                break
            state = self.states[node_id]
            if state.status != "pending":
                continue
            if any(
                self.states[parent].status != "succeeded"
                for parent in dependencies[node_id]
            ):
                continue
            lane = available.pop(0)
            try:
                self._launch_node(lane, nodes[node_id])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                state.status = "failed"
                state.error = str(exc)
                self.log_line.emit(node_id, str(exc))
                self.node_changed.emit(node_id)
                QTimer.singleShot(0, self._schedule)
        self._finish_if_complete()

    def _launch_node(
        self,
        lane: TaskQueueController,
        node: FlowNodeV1,
    ) -> None:
        if self.flow is None:
            return
        function = self.catalog.get(node.function_id)
        parameters = dict(node.parameters)
        edges = [edge for edge in self.flow.edges if edge.target_node == node.node_id]
        inputs_by_id = {item.port_id: item for item in function.inputs}
        for edge in edges:
            port = inputs_by_id[edge.target_port]
            if port.parameter_id is None:
                continue
            source_function = self.catalog.get(
                next(
                    item.function_id
                    for item in self.flow.nodes
                    if item.node_id == edge.source_node
                )
            )
            source_output = next(
                (
                    item
                    for item in source_function.outputs
                    if item.port_id == edge.source_port
                ),
                None,
            )
            if source_output is None:
                raise ValueError(f"Unknown source port: {edge.source_port}")
            identities = self.states[edge.source_node].artifact_identities.get(
                edge.source_port,
                [],
            )
            if not identities:
                raise ValueError(
                    f"{edge.source_node}.{edge.source_port} produced no artifact"
                )
            parameter = function.parameter(port.parameter_id)
            values = [
                _verified_artifact_path(
                    identity,
                    expected_role=edge.source_port,
                    source_node=edge.source_node,
                )
                for identity in identities
            ]
            value = _artifact_parameter_value(
                values,
                parameter_kind=parameter.kind,
                source_multiple=source_output.multiple,
                source_label=f"{edge.source_node}.{edge.source_port}",
            )
            parameters[port.parameter_id] = value
        output = self.runtime.run_output_dir(
            self._project_id,
            self._run_id,
            node.node_id,
        )
        module, arguments, _normalized = function.build_arguments(
            parameters,
            variant_id=node.variant_id,
            default_output=str(output),
            allowed_roots=self.allowed_roots,
        )
        record = lane.enqueue_python_module(
            title=function.title,
            module_id=(
                function.page_templates[0] if function.page_templates else "workbench"
            ),
            python_module=module,
            arguments=arguments,
            output_dir=str(output),
        )
        state = self.states[node.node_id]
        state.status = "running"
        state.task_id = record.task_id
        state.output_dir = str(output)
        self._lane_nodes[lane] = node.node_id
        self._task_nodes[record.task_id] = node.node_id
        self.node_changed.emit(node.node_id)

    def _task_changed(self, lane: TaskQueueController, task_id: str) -> None:
        node_id = self._task_nodes.get(task_id)
        if node_id is None:
            return
        record = lane.task(task_id)
        if record.status not in {"succeeded", "failed", "cancelled"}:
            return
        state = self.states[node_id]
        state.status = record.status
        state.artifacts = list(record.artifacts)
        if record.status == "succeeded":
            function_id = next(
                item.function_id
                for item in self.flow.nodes
                if item.node_id == node_id
            )
            function = self.catalog.get(function_id)
            try:
                state.artifact_identities = _bind_artifacts_to_ports(
                    function.outputs,
                    record.artifact_records,
                    legacy_artifacts=state.artifacts,
                )
            except (OSError, TypeError, ValueError) as exc:
                state.status = "failed"
                state.error = str(exc)
                self.log_line.emit(node_id, str(exc))
        if record.status != "succeeded":
            state.error = "\n".join(record.logs[-3:]) or record.status
        self._lane_nodes.pop(lane, None)
        self.node_changed.emit(node_id)
        QTimer.singleShot(0, self._schedule)

    def _task_log(self, task_id: str, line: str) -> None:
        self.log_line.emit(self._task_nodes.get(task_id, task_id), line)

    def _task_artifact(self, task_id: str, path: str) -> None:
        node_id = self._task_nodes.get(task_id, task_id)
        state = self.states.get(node_id)
        if state is not None and path not in state.artifacts:
            state.artifacts.append(path)
        self.artifact_ready.emit(node_id, path)

    def _finish_if_complete(self) -> None:
        if self.states and all(
            state.status in _TERMINAL for state in self.states.values()
        ):
            self.flow_finished.emit(self.summary())


__all__ = ["FlowExecutionController", "FlowNodeExecution"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_artifacts_to_ports(
    outputs,
    artifact_records: list[dict[str, object]],
    *,
    legacy_artifacts: list[str] | None = None,
):
    """Bind products only from explicit worker roles, with one-port fallback."""

    output_by_id = {item.port_id: item for item in outputs}
    result: dict[str, list[dict[str, object]]] = {
        item.port_id: [] for item in outputs
    }
    explicit: list[tuple[str, dict[str, object]]] = []
    for artifact in artifact_records:
        source_port = artifact.get("source_port")
        if not isinstance(source_port, str) or not source_port:
            role = artifact.get("role")
            source_port = role if isinstance(role, str) and role in output_by_id else None
        if source_port is not None:
            explicit.append((source_port, artifact))

    unbound = len(artifact_records) - len(explicit)
    if len(outputs) > 1 and unbound:
        raise ValueError(
            f"Multi-output worker emitted {unbound} artifact(s) without explicit "
            "source_port"
        )
    if len(outputs) > 1 and not explicit and (artifact_records or legacy_artifacts):
        raise ValueError(
            "Multi-output worker emitted no explicit source_port; refusing "
            "artifact-order inference"
        )
    if len(outputs) == 1 and not explicit:
        sole_port = outputs[0].port_id
        candidates = artifact_records or [
            {"path": item} for item in (legacy_artifacts or [])
        ]
        explicit = [(sole_port, item) for item in candidates]

    for source_port, artifact in explicit:
        if source_port not in output_by_id:
            raise ValueError(f"Worker emitted unknown source_port: {source_port}")
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"Artifact path missing for source_port: {source_port}")
        path = Path(path_value)
        if not path.is_file():
            raise ValueError(f"Artifact missing for source_port {source_port}: {path}")
        actual_sha256 = _sha256(path)
        declared_sha256 = artifact.get("sha256")
        if declared_sha256 is not None and declared_sha256 != actual_sha256:
            raise ValueError(f"Artifact SHA-256 mismatch for source_port: {source_port}")
        result[source_port].append(
            {
                "role": source_port,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )

    for source_port, identities in result.items():
        if output_by_id[source_port].required and not identities:
            raise ValueError(f"Required source_port {source_port} emitted no artifact")
        if not output_by_id[source_port].multiple and len(identities) > 1:
            raise ValueError(
                f"Scalar source_port {source_port} emitted {len(identities)} artifacts"
            )
    return result


def _artifact_parameter_value(
    paths: list[Path],
    *,
    parameter_kind: str,
    source_multiple: bool,
    source_label: str,
):
    """Preserve a multiple port as a list, or a verified common directory."""

    if not paths:
        raise ValueError(f"{source_label} produced no artifact")
    if not source_multiple and len(paths) != 1:
        raise ValueError(f"Scalar source port {source_label} produced multiple artifacts")
    if parameter_kind == "list":
        return [str(path) for path in paths]
    if parameter_kind == "directory":
        directories = {path if path.is_dir() else path.parent for path in paths}
        if len(directories) != 1:
            raise ValueError(
                f"{source_label} artifacts do not share one input directory"
            )
        return str(next(iter(directories)))
    if len(paths) != 1:
        raise ValueError(
            f"{source_label} is multiple but target parameter {parameter_kind} is scalar"
        )
    return str(paths[0])


def _verified_artifact_path(
    identity: dict[str, object],
    *,
    expected_role: str,
    source_node: str,
) -> Path:
    """Resolve one routed artifact only after its role and bytes are verified."""

    role = identity.get("role")
    path_value = identity.get("path")
    expected_sha256 = identity.get("sha256")
    if role != expected_role:
        raise ValueError(
            f"Artifact role mismatch: {source_node}.{expected_role} received {role!r}"
        )
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Artifact path missing: {source_node}.{expected_role}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"Artifact SHA-256 missing: {source_node}.{expected_role}")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Artifact missing: {source_node}.{expected_role}")
    if _sha256(path) != expected_sha256:
        raise ValueError(
            f"Artifact changed after production: {source_node}.{expected_role}"
        )
    return path
