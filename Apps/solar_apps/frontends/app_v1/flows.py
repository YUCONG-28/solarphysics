# SPDX-License-Identifier: GPL-3.0-only
"""Version-one typed workflow graph contracts and validation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .contracts import JsonContract, validate_identifier
from .function_specs import FunctionSpec

FLOW_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FlowNodeV1(JsonContract):
    node_id: str
    function_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    variant_id: str | None = None
    disabled: bool = False
    x: float = 0.0
    y: float = 0.0
    group: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_id", validate_identifier(self.node_id, label="node_id")
        )
        object.__setattr__(
            self,
            "function_id",
            validate_identifier(self.function_id, label="function_id"),
        )
        if self.variant_id:
            object.__setattr__(
                self,
                "variant_id",
                validate_identifier(self.variant_id, label="variant_id"),
            )
        if not isinstance(self.parameters, dict):
            raise TypeError("Flow node parameters must be an object")


@dataclass(frozen=True, slots=True)
class FlowEdgeV1(JsonContract):
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __post_init__(self) -> None:
        for name in ("source_node", "target_node"):
            object.__setattr__(
                self,
                name,
                validate_identifier(getattr(self, name), label=name),
            )
        if self.source_node == self.target_node:
            raise ValueError("A flow edge cannot connect a node to itself")
        if not self.source_port.strip() or not self.target_port.strip():
            raise ValueError("Flow edge ports are required")


@dataclass(frozen=True, slots=True)
class AppV1FlowV1(JsonContract):
    flow_id: str
    name: str
    nodes: tuple[FlowNodeV1, ...] = ()
    edges: tuple[FlowEdgeV1, ...] = ()
    concurrency: int = 1
    saved_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = FLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "flow_id", validate_identifier(self.flow_id, label="flow_id")
        )
        if not self.name.strip():
            raise ValueError("Flow name is required")
        if self.schema_version != FLOW_SCHEMA_VERSION:
            raise ValueError("Unsupported .spflow.json schema")
        if not 1 <= int(self.concurrency) <= 4:
            raise ValueError("Flow concurrency must be between 1 and 4")
        node_ids = [item.node_id for item in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Flow node IDs must be unique")
        edge_keys = [
            (
                item.source_node,
                item.source_port,
                item.target_node,
                item.target_port,
            )
            for item in self.edges
        ]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("Flow edges must be unique")
        available = set(node_ids)
        for edge in self.edges:
            if edge.source_node not in available or edge.target_node not in available:
                raise ValueError("Flow edge references an unknown node")
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        indegree = {item.node_id: 0 for item in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            children[edge.source_node].append(edge.target_node)
            indegree[edge.target_node] += 1
        ready = deque(
            item.node_id for item in self.nodes if indegree[item.node_id] == 0
        )
        ordered: list[str] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(node_id)
            for child in children[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.nodes):
            raise ValueError("Flow graph contains a cycle")
        return tuple(ordered)

    def dependencies(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, list[str]] = {item.node_id: [] for item in self.nodes}
        for edge in self.edges:
            if edge.source_node not in result[edge.target_node]:
                result[edge.target_node].append(edge.source_node)
        return {key: tuple(value) for key, value in result.items()}

    def descendants(self, node_id: str) -> tuple[str, ...]:
        children: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            children[edge.source_node].append(edge.target_node)
        found: list[str] = []
        queue = deque(children[node_id])
        while queue:
            current = queue.popleft()
            if current in found:
                continue
            found.append(current)
            queue.extend(children[current])
        return tuple(found)


class FunctionCatalog:
    """One authoritative catalog shared by pages, forms, and the DAG editor."""

    def __init__(self, functions: tuple[FunctionSpec, ...]) -> None:
        self._functions = {item.function_id: item for item in functions}
        if len(self._functions) != len(functions):
            raise ValueError("Function IDs must be unique")
        self._aliases: dict[str, str] = {}
        for item in functions:
            for alias in item.aliases:
                if alias in self._functions or alias in self._aliases:
                    raise ValueError(f"Duplicate function alias: {alias}")
                self._aliases[alias] = item.function_id

    @property
    def functions(self) -> tuple[FunctionSpec, ...]:
        return tuple(self._functions.values())

    def resolve_id(self, function_id: str) -> str:
        return self._aliases.get(function_id, function_id)

    def get(self, function_id: str) -> FunctionSpec:
        resolved = self.resolve_id(function_id)
        try:
            return self._functions[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown function: {function_id}") from exc

    def validate_flow(self, flow: AppV1FlowV1) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        nodes = {item.node_id: item for item in flow.nodes}
        connected_inputs: dict[str, set[str]] = defaultdict(set)
        for edge in flow.edges:
            source = self.get(nodes[edge.source_node].function_id)
            target = self.get(nodes[edge.target_node].function_id)
            try:
                output = next(
                    item for item in source.outputs if item.port_id == edge.source_port
                )
                input_port = next(
                    item for item in target.inputs if item.port_id == edge.target_port
                )
            except StopIteration as exc:
                raise ValueError(
                    "Flow edge references an unknown function port"
                ) from exc
            if not input_port.accepts(output.artifact_types):
                raise ValueError(
                    f"Incompatible artifact edge: {source.function_id}.{output.port_id} "
                    f"→ {target.function_id}.{input_port.port_id}"
                )
            if (
                not input_port.multiple
                and edge.target_port in connected_inputs[edge.target_node]
            ):
                raise ValueError(
                    f"Input {edge.target_node}.{edge.target_port} accepts one connection"
                )
            connected_inputs[edge.target_node].add(edge.target_port)
        for node in flow.nodes:
            function = self.get(node.function_id)
            supplied = dict(node.parameters)
            for input_port in function.inputs:
                if (
                    input_port.parameter_id
                    and input_port.port_id in connected_inputs[node.node_id]
                    and supplied.get(input_port.parameter_id) in (None, "")
                ):
                    supplied[input_port.parameter_id] = "__connected_artifact__"
            normalized[node.node_id] = function.normalize_parameters(supplied)
            function.selected_module(node.variant_id)
            missing_ports = [
                item.port_id
                for item in function.inputs
                if item.required and item.port_id not in connected_inputs[node.node_id]
            ]
            if missing_ports and not node.disabled:
                raise ValueError(
                    f"Node {node.node_id} is missing inputs: {', '.join(missing_ports)}"
                )
        return normalized


__all__ = [
    "AppV1FlowV1",
    "FLOW_SCHEMA_VERSION",
    "FlowEdgeV1",
    "FlowNodeV1",
    "FunctionCatalog",
]
