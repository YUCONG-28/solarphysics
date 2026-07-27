# SPDX-License-Identifier: GPL-3.0-only
"""Atomic private persistence for App 1.0 workflow graphs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .flows import AppV1FlowV1, FlowEdgeV1, FlowNodeV1
from .runtime import AppV1RuntimePaths
from .timeline import normalize_utc


class AppV1FlowStore:
    def __init__(self, runtime: AppV1RuntimePaths) -> None:
        self.runtime = runtime

    def save(self, flow: AppV1FlowV1) -> Path:
        target = self.runtime.flow_file(flow.flow_id)
        _atomic_json(target, flow.to_dict())
        return target

    def load(self, flow_id: str) -> AppV1FlowV1:
        path = self.runtime.flow_file(flow_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("Flow payload must be an object")
        return self.from_dict(payload)

    def list_flows(self) -> tuple[Path, ...]:
        directory = self.runtime.workspaces_dir / "flows"
        if not directory.is_dir():
            return ()
        return tuple(sorted(directory.glob("*.spflow.json")))

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> AppV1FlowV1:
        nodes = tuple(
            FlowNodeV1(
                node_id=str(item["node_id"]),
                function_id=str(item["function_id"]),
                parameters=dict(item.get("parameters", {})),
                variant_id=(
                    None
                    if item.get("variant_id") in (None, "")
                    else str(item["variant_id"])
                ),
                disabled=bool(item.get("disabled", False)),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                group=str(item.get("group", "")),
            )
            for item in payload.get("nodes", ())
        )
        edges = tuple(
            FlowEdgeV1(
                source_node=str(item["source_node"]),
                source_port=str(item["source_port"]),
                target_node=str(item["target_node"]),
                target_port=str(item["target_port"]),
            )
            for item in payload.get("edges", ())
        )
        saved = normalize_utc(payload.get("saved_at_utc") or datetime.now().astimezone())
        return AppV1FlowV1(
            flow_id=str(payload["flow_id"]),
            name=str(payload["name"]),
            nodes=nodes,
            edges=edges,
            concurrency=int(payload.get("concurrency", 1)),
            saved_at_utc=saved,
            schema_version=int(payload.get("schema_version", 0)),
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


__all__ = ["AppV1FlowStore"]
