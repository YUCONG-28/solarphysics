"""Shared latest-state and recent-path bindings for user-facing frontends."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.memory import RecentPathMemory
from solar_apps.platform.state import StateStore

FRONTEND_STATE_KEYS = ("theme", "fields", "legacy_imported", "working_directory")


def frontend_state_store(
    frontend_id: str, *, layout: RuntimeLayout | None = None
) -> StateStore:
    """Return the versioned, atomic latest-state store for one frontend."""

    return StateStore.for_frontend(
        frontend_id,
        layout=layout,
        allowed_keys=FRONTEND_STATE_KEYS,
    )


def frontend_path_memory(
    allowed_roots: Iterable[str | Path], *, layout: RuntimeLayout | None = None
) -> RecentPathMemory:
    """Return shared recent-directory memory constrained to current roots."""

    return RecentPathMemory.default(allowed_roots, layout=layout)


def restore_streamlit_fields(
    st: Any,
    state_store: StateStore,
    field_keys: Iterable[str],
) -> dict[str, Any]:
    """Restore a frontend-declared primitive widget whitelist before rendering."""

    allowed = tuple(dict.fromkeys(str(key) for key in field_keys))
    saved = state_store.load(default={})
    raw_fields = saved.get("fields", {}) if isinstance(saved, dict) else {}
    fields = raw_fields if isinstance(raw_fields, dict) else {}
    restored: dict[str, Any] = {}
    for key in allowed:
        if key in st.session_state or key not in fields:
            continue
        value = _streamlit_primitive(fields[key])
        if value is not None or fields[key] is None:
            st.session_state[key] = value
            restored[key] = value
    return restored


def save_streamlit_fields(
    st: Any,
    state_store: StateStore,
    field_keys: Iterable[str],
) -> dict[str, Any]:
    """Atomically persist only declared widget primitives from this rerun."""

    saved = state_store.load(default={})
    raw_fields = saved.get("fields", {}) if isinstance(saved, dict) else {}
    fields = dict(raw_fields) if isinstance(raw_fields, dict) else {}
    for key in dict.fromkeys(str(item) for item in field_keys):
        if key not in st.session_state:
            continue
        value = _streamlit_primitive(st.session_state[key])
        if value is not None or st.session_state[key] is None:
            fields[key] = value
    state_store.update({"fields": fields})
    return fields


def bind_streamlit_fields(
    st: Any,
    state_store: StateStore,
    *,
    frontend_id: str,
    field_keys: Iterable[str],
) -> None:
    """Restore once, then persist the whitelist at the start of each rerun."""

    allowed = tuple(dict.fromkeys(str(key) for key in field_keys))
    marker = f"{frontend_id}_ui_state_bound"
    if st.session_state.get(marker) is True:
        save_streamlit_fields(st, state_store, allowed)
    else:
        restore_streamlit_fields(st, state_store, allowed)
        st.session_state[marker] = True


def _streamlit_primitive(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 128:
        items: list[Any] = []
        for item in value:
            if not isinstance(item, bool | int | float | str):
                return None
            items.append(item)
        return items
    return None


__all__ = [
    "FRONTEND_STATE_KEYS",
    "bind_streamlit_fields",
    "frontend_path_memory",
    "frontend_state_store",
    "restore_streamlit_fields",
    "save_streamlit_fields",
]
