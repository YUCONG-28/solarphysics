"""Small, versioned and atomic local UI state files."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .layout import RuntimeLayout

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "history",
        "operation_history",
        "operation_log",
        "log",
        "logs",
        "timestamp",
        "timestamps",
        "task_id",
        "task_ids",
        "upload_content",
        "scientific_data",
        "result",
        "results",
    }
)


class StateStore:
    """Persist only the latest allow-listed state, never an operation history."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        namespace: str,
        schema_version: int = 1,
        *,
        allowed_keys: Iterable[str],
    ) -> None:
        if not _NAMESPACE_RE.fullmatch(namespace):
            raise ValueError(f"Invalid state namespace: {namespace!r}")
        if not isinstance(schema_version, int) or schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        self.path = Path(path).expanduser().resolve(strict=False)
        self.namespace = namespace
        self.schema_version = schema_version
        self.allowed_keys = frozenset(str(key) for key in allowed_keys)
        if not self.allowed_keys:
            raise ValueError("StateStore requires a non-empty top-level allow-list")
        self._lock = threading.RLock()

    @classmethod
    def for_frontend(
        cls,
        frontend: str,
        *,
        layout: RuntimeLayout | None = None,
        schema_version: int = 1,
        allowed_keys: Iterable[str],
    ) -> "StateStore":
        selected_layout = layout or RuntimeLayout.discover()
        return cls(
            selected_layout.state_dir / f"{frontend}.json",
            frontend,
            schema_version,
            allowed_keys=allowed_keys,
        )

    def load(self, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return validated data or a defensive copy of ``default``."""

        fallback = deepcopy(dict(default or {}))
        with self._lock:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
                return fallback
            if not isinstance(payload, dict):
                return fallback
            if payload.get("schema_version") != self.schema_version:
                return fallback
            if payload.get("namespace") != self.namespace:
                return fallback
            data = payload.get("data")
            if not isinstance(data, dict):
                return fallback
            try:
                self._validate_data(data)
            except (TypeError, ValueError):
                return fallback
            return deepcopy(data)

    def save(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically replace the state after proving it is JSON serializable."""

        if not isinstance(data, Mapping):
            raise TypeError("State data must be a mapping")
        snapshot = deepcopy(dict(data))
        self._validate_data(snapshot)
        payload = {
            "schema_version": self.schema_version,
            "namespace": self.namespace,
            "data": snapshot,
        }
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    dir=self.path.parent,
                    delete=False,
                ) as handle:
                    temporary_name = handle.name
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.path)
            finally:
                if temporary_name:
                    try:
                        Path(temporary_name).unlink(missing_ok=True)
                    except OSError:
                        pass
        return deepcopy(snapshot)

    def update(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Merge top-level keys into the latest state and persist it."""

        if not isinstance(values, Mapping):
            raise TypeError("State update must be a mapping")
        with self._lock:
            current = self.load()
            current.update(deepcopy(dict(values)))
            return self.save(current)

    def reset(self) -> None:
        """Remove this namespace's local state file."""

        with self._lock:
            self.path.unlink(missing_ok=True)

    def _validate_data(self, data: Mapping[str, Any]) -> None:
        unexpected = set(map(str, data)) - self.allowed_keys
        if unexpected:
            raise ValueError(
                "State contains keys outside the allow-list: "
                + ", ".join(sorted(unexpected))
            )

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                for raw_key, child in value.items():
                    key = str(raw_key).strip().casefold().replace("-", "_")
                    if key in _FORBIDDEN_STATE_KEYS:
                        raise ValueError(
                            f"State key {raw_key!r} is reserved for non-persistent data"
                        )
                    visit(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)

        visit(data)


__all__ = ["StateStore"]
