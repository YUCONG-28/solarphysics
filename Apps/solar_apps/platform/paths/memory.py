"""Fail-closed recent-directory memory shared by all path dialogs."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..layout import RuntimeLayout
from ..state import StateStore
from .semantics import path_is_within

_CONTEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_DIALOG_MODES = frozenset({"open_file", "open_files", "select_directory", "save_file"})


@dataclass(frozen=True, slots=True)
class PathMemoryContext:
    """Stable identity for one path field without recording user actions."""

    frontend: str
    operation: str
    field: str

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any] | None
    ) -> "PathMemoryContext | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("memory_context must be an object")
        values = tuple(
            str(payload.get(name, "")).strip()
            for name in ("frontend", "operation", "field")
        )
        if not all(_CONTEXT_RE.fullmatch(value) for value in values):
            raise ValueError(
                "memory_context frontend, operation, and field must be stable identifiers"
            )
        return cls(*values)


class RecentPathMemory:
    """Remember last usable directories with a six-level safe fallback."""

    def __init__(
        self,
        store: StateStore,
        allowed_roots: Iterable[str | os.PathLike[str]],
    ) -> None:
        self.store = store
        self.allowed_roots = tuple(
            Path(root).expanduser().resolve(strict=False) for root in allowed_roots
        )

    @classmethod
    def default(
        cls,
        allowed_roots: Iterable[str | os.PathLike[str]],
        *,
        layout: RuntimeLayout | None = None,
    ) -> "RecentPathMemory":
        selected_layout = layout or RuntimeLayout.discover()
        return cls(
            StateStore(
                selected_layout.state_dir / "recent_paths.json",
                "recent_paths",
                allowed_keys=("field", "operation", "frontend", "global"),
            ),
            allowed_roots,
        )

    def resolve_initial(
        self,
        *,
        context: PathMemoryContext | Mapping[str, Any] | None,
        dialog_mode: str,
        current_value: str = "",
    ) -> str:
        """Resolve current, field, operation, frontend, global, then root."""

        mode = self._validated_mode(dialog_mode)
        selected_context = (
            context
            if isinstance(context, PathMemoryContext)
            else PathMemoryContext.from_payload(context)
        )
        state = self.store.load(self._empty_state())
        candidates: list[str] = []
        if current_value:
            current_candidates = list(
                reversed(
                    [
                        line.strip()
                        for line in str(current_value).splitlines()
                        if line.strip()
                    ]
                )
            )
            for candidate in current_candidates:
                directory = self._usable_directory(candidate, mode)
                if directory is not None:
                    return str(directory)
        if selected_context is not None:
            candidates.extend(
                [
                    self._entry(
                        state, "field", self._field_key(selected_context, mode)
                    ),
                    self._entry(
                        state, "operation", self._operation_key(selected_context, mode)
                    ),
                    self._entry(
                        state, "frontend", self._frontend_key(selected_context, mode)
                    ),
                ]
            )
        candidates.append(self._entry(state, "global", mode))
        for candidate in candidates:
            # Persisted entries are always directories, even for file dialogs.
            directory = self._usable_directory(candidate, "select_directory")
            if directory is not None:
                return str(directory)
        for root in self.allowed_roots:
            directory = self._usable_directory(str(root), "select_directory")
            if directory is not None:
                return str(directory)
        return ""

    def remember(
        self,
        *,
        context: PathMemoryContext | Mapping[str, Any] | None,
        dialog_mode: str,
        paths: Sequence[str | os.PathLike[str]],
    ) -> None:
        """Store the latest selected directory at every applicable level."""

        mode = self._validated_mode(dialog_mode)
        selected_context = (
            context
            if isinstance(context, PathMemoryContext)
            else PathMemoryContext.from_payload(context)
        )
        if not paths:
            return
        directory = self._usable_directory(str(paths[-1]), mode)
        if directory is None:
            return
        state = self.store.load(self._empty_state())
        state = self._coerce_state(state)
        value = str(directory)
        state["global"][mode] = value
        if selected_context is not None:
            state["field"][self._field_key(selected_context, mode)] = value
            state["operation"][self._operation_key(selected_context, mode)] = value
            state["frontend"][self._frontend_key(selected_context, mode)] = value
        self.store.save(state)

    def reset(self, *, frontend: str | None = None) -> None:
        """Reset all remembered paths, or only entries for one frontend."""

        if frontend is None:
            self.store.reset()
            return
        if not _CONTEXT_RE.fullmatch(frontend):
            raise ValueError(f"Invalid frontend identifier: {frontend!r}")
        state = self._coerce_state(self.store.load(self._empty_state()))
        prefix = f"{frontend}|"
        for group in ("field", "operation", "frontend"):
            state[group] = {
                key: value
                for key, value in state[group].items()
                if not key.startswith(prefix)
            }
        self.store.save(state)

    @staticmethod
    def _empty_state() -> dict[str, dict[str, str]]:
        return {"field": {}, "operation": {}, "frontend": {}, "global": {}}

    def _coerce_state(self, value: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        result = self._empty_state()
        for group in result:
            raw = value.get(group, {})
            if isinstance(raw, Mapping):
                result[group] = {
                    str(key): str(item)
                    for key, item in raw.items()
                    if isinstance(key, str) and isinstance(item, str)
                }
        return result

    @staticmethod
    def _entry(state: Mapping[str, Any], group: str, key: str) -> str:
        values = state.get(group)
        if not isinstance(values, Mapping):
            return ""
        value = values.get(key)
        return value if isinstance(value, str) else ""

    @staticmethod
    def _validated_mode(mode: str) -> str:
        if mode not in _DIALOG_MODES:
            raise ValueError(f"Unsupported dialog mode: {mode!r}")
        return mode

    @staticmethod
    def _field_key(context: PathMemoryContext, mode: str) -> str:
        return f"{context.frontend}|{context.operation}|{context.field}|{mode}"

    @staticmethod
    def _operation_key(context: PathMemoryContext, mode: str) -> str:
        return f"{context.frontend}|{context.operation}|{mode}"

    @staticmethod
    def _frontend_key(context: PathMemoryContext, mode: str) -> str:
        return f"{context.frontend}|{mode}"

    def _usable_directory(self, raw: str, mode: str) -> Path | None:
        if not raw:
            return None
        candidate = Path(str(raw).strip().strip('"').strip("'")).expanduser()
        if not candidate.is_absolute():
            return None
        candidate = candidate.resolve(strict=False)
        if mode in {"open_file", "open_files", "save_file"}:
            if candidate.exists() and candidate.is_dir():
                directory = candidate
            else:
                directory = candidate.parent
        else:
            directory = candidate
        try:
            directory = directory.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None
        if not directory.is_dir() or not self._within_roots(directory):
            return None
        return directory

    def _within_roots(self, candidate: Path) -> bool:
        for root in self.allowed_roots:
            try:
                resolved_root = root.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if path_is_within(candidate, resolved_root):
                return True
        return False


__all__ = ["PathMemoryContext", "RecentPathMemory"]
