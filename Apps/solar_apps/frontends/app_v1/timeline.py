# SPDX-License-Identifier: GPL-3.0-only
"""Rebuildable UTC index and deterministic App 1.0 time coordinator."""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
)

from .contracts import SyncSelection, validate_identifier

_SCHEMA_VERSION = 2
_STATUSES = frozenset({"available", "missing", "duplicate", "unreadable"})
_MATCHABLE_STATUSES = frozenset({"available", "duplicate"})


def normalize_utc(value: datetime | str) -> datetime:
    """Return an aware UTC datetime and reject ambiguous local timestamps."""

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("UTC timestamp must not be blank")
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timeline timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime | str) -> str:
    """Serialize a UTC timestamp in stable ISO-8601 Z form."""

    normalize_utc(value)
    return (
        normalize_time(value if isinstance(value, str) else normalize_utc(value)).utc
        + "Z"
    )


@dataclass(frozen=True, slots=True)
class TimelineSample:
    """One local file or table-row observation in the rebuildable index."""

    locator: str
    observed_at_utc: datetime | None
    status: str = "available"
    position: str | None = None
    metadata: dict[str, Any] | None = None
    raw_time_utc: str | None = None

    def __post_init__(self) -> None:
        if not str(self.locator).strip():
            raise ValueError("Timeline sample locator is required")
        normalized_status = str(self.status).strip().lower()
        if normalized_status not in _STATUSES:
            raise ValueError(f"Unsupported timeline sample status: {self.status}")
        object.__setattr__(self, "status", normalized_status)
        if self.observed_at_utc is not None:
            raw = self.raw_time_utc or self.observed_at_utc
            normalize_utc(raw)
            object.__setattr__(self, "raw_time_utc", normalize_time(raw).utc + "Z")
            object.__setattr__(
                self,
                "observed_at_utc",
                normalize_utc(self.observed_at_utc),
            )
        if normalized_status in _MATCHABLE_STATUSES and self.observed_at_utc is None:
            raise ValueError("Available timeline samples require a UTC timestamp")
        metadata = dict(self.metadata or {})
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Timeline sample metadata must contain JSON values"
            ) from exc
        object.__setattr__(self, "metadata", metadata)


@dataclass(frozen=True, slots=True)
class RegisteredTimeline:
    """Coordinator settings for one indexed source."""

    source_id: str
    module_id: str
    offset_seconds: float = 0.0
    tolerance_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_id",
            validate_identifier(self.source_id, label="source_id"),
        )
        object.__setattr__(
            self,
            "module_id",
            validate_identifier(self.module_id, label="module_id"),
        )
        for name in ("offset_seconds", "tolerance_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "tolerance_seconds" and value < 0:
                raise ValueError("tolerance_seconds cannot be negative")
            object.__setattr__(self, name, value)


class SQLiteTimelineIndex:
    """Small, invalidatable SQLite cache below the private runtime tree."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            previous_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if previous_version not in {0, 1, _SCHEMA_VERSION}:
                raise ValueError("Unsupported future timeline cache schema")
            if previous_version == 1:
                # This is a rebuildable runtime cache, never an observation archive.
                connection.executescript(
                    "DROP TABLE IF EXISTS timeline_samples; DROP TABLE IF EXISTS timeline_sources;"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS timeline_sources (
                    source_id TEXT PRIMARY KEY,
                    module_id TEXT NOT NULL,
                    local_source TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    record_count INTEGER NOT NULL CHECK(record_count >= 0)
                );
                CREATE TABLE IF NOT EXISTS timeline_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL
                        REFERENCES timeline_sources(source_id) ON DELETE CASCADE,
                    locator TEXT NOT NULL,
                    observed_at_utc TEXT,
                    status TEXT NOT NULL,
                    position TEXT,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_timeline_samples_source_time
                    ON timeline_samples(source_id, observed_at_utc, locator);
                """
            )
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def rebuild_source(
        self,
        *,
        source_id: str,
        module_id: str,
        local_source: str | Path,
        samples: Iterable[TimelineSample | Mapping[str, Any]],
    ) -> int:
        """Atomically replace one source; duplicate times stay addressable."""

        source = validate_identifier(source_id, label="source_id")
        module = validate_identifier(module_id, label="module_id")
        local = str(Path(local_source).expanduser().resolve(strict=False))
        normalized = [
            item if isinstance(item, TimelineSample) else TimelineSample(**dict(item))
            for item in samples
        ]
        seen_times: set[str] = set()
        rows: list[tuple[str, str | None, str, str | None, str]] = []
        for item in normalized:
            observed = item.raw_time_utc if item.observed_at_utc is not None else None
            status = item.status
            if observed is not None and status == "available":
                if observed in seen_times:
                    status = "duplicate"
                else:
                    seen_times.add(observed)
            rows.append(
                (
                    str(item.locator),
                    observed,
                    status,
                    None if item.position is None else str(item.position),
                    json.dumps(item.metadata or {}, sort_keys=True, ensure_ascii=False),
                )
            )
        now = utc_text(datetime.now(timezone.utc))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO timeline_sources(
                    source_id, module_id, local_source, updated_at_utc, record_count
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    module_id=excluded.module_id,
                    local_source=excluded.local_source,
                    updated_at_utc=excluded.updated_at_utc,
                    record_count=excluded.record_count
                """,
                (source, module, local, now, len(rows)),
            )
            connection.execute(
                "DELETE FROM timeline_samples WHERE source_id = ?",
                (source,),
            )
            connection.executemany(
                """
                INSERT INTO timeline_samples(
                    source_id, locator, observed_at_utc, status, position, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [(source, *row) for row in rows],
            )
        return len(rows)

    def invalidate_source(self, source_id: str) -> bool:
        """Remove an index source; original observations are never touched."""

        source = validate_identifier(source_id, label="source_id")
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM timeline_sources WHERE source_id = ?",
                (source,),
            )
        return cursor.rowcount > 0

    def source(self, source_id: str) -> dict[str, Any]:
        source = validate_identifier(source_id, label="source_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM timeline_sources WHERE source_id = ?",
                (source,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown timeline source: {source}")
        return dict(row)

    def list_sources(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM timeline_sources ORDER BY source_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def samples(
        self,
        source_id: str,
        *,
        matchable_only: bool = False,
    ) -> list[TimelineSample]:
        source = validate_identifier(source_id, label="source_id")
        where = "source_id = ?"
        values: list[Any] = [source]
        if matchable_only:
            where += " AND status IN ('available', 'duplicate')"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT locator, observed_at_utc, status, position, metadata_json
                FROM timeline_samples
                WHERE {where}
                ORDER BY
                    observed_at_utc IS NULL,
                    observed_at_utc,
                    locator,
                    sample_id
                """,
                values,
            ).fetchall()
        return [
            TimelineSample(
                locator=row["locator"],
                observed_at_utc=(
                    normalize_utc(row["observed_at_utc"])
                    if row["observed_at_utc"]
                    else None
                ),
                status=row["status"],
                position=row["position"],
                metadata=json.loads(row["metadata_json"]),
                raw_time_utc=row["observed_at_utc"],
            )
            for row in rows
        ]

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])


class TimeCoordinator:
    """Browse immutable chronological plans freely in either direction."""

    def __init__(
        self,
        index: SQLiteTimelineIndex,
        *,
        time_start: datetime | str | None = None,
        time_end: datetime | str | None = None,
    ) -> None:
        self.index = index
        self._sources: dict[str, RegisteredTimeline] = {}
        self.base_source_id: str | None = None
        self.output_source_id: str | None = None
        self.migration_note = ""
        self.current_time_utc: datetime | None = None
        self.current_time_text: str | None = None
        self._listeners: list[Callable[[SyncSelection], None]] = []
        self._state_listeners: list[Callable[[], None]] = []
        self._cursor_index: int | None = None
        self._plan_fingerprint: str | None = None
        self.current_plan = None
        self._source_snapshot = None
        self.time_start, self.time_end = self._normalized_window(time_start, time_end)

    @staticmethod
    def _normalized_window(time_start, time_end) -> tuple[str | None, str | None]:
        # These are observed UTC bounds, independent of offsets and the cursor.
        start = normalize_time(time_start) if time_start is not None else None
        end = normalize_time(time_end) if time_end is not None else None
        if start is not None and end is not None and start.tai_ns >= end.tai_ns:
            raise ValueError("Time window must satisfy time_start < time_end")
        return start.utc + "Z" if start else None, end.utc + "Z" if end else None

    def set_time_window(
        self,
        time_start: datetime | str | None = None,
        time_end: datetime | str | None = None,
    ) -> None:
        """Rebuild using original observations inside [start UTC, end UTC)."""

        start, end = self._normalized_window(time_start, time_end)
        previous = self.current_time_text or self.current_time_utc
        self.time_start, self.time_end = start, end
        plan = self._sequence_plan()
        if previous is not None and plan.rows:
            self.select(previous)

    @property
    def sources(self) -> tuple[RegisteredTimeline, ...]:
        return tuple(self._sources[key] for key in sorted(self._sources))

    def register_source(
        self,
        source_id: str,
        *,
        module_id: str | None = None,
        offset_seconds: float = 0.0,
        tolerance_seconds: float = 0.0,
    ) -> RegisteredTimeline:
        indexed = self.index.source(source_id)
        registration = RegisteredTimeline(
            source_id,
            module_id or str(indexed["module_id"]),
            offset_seconds,
            tolerance_seconds,
        )
        self._sources[registration.source_id] = registration
        self._sequence_plan()
        return registration

    def unregister_source(self, source_id: str) -> bool:
        source = validate_identifier(source_id, label="source_id")
        removed = self._sources.pop(source, None) is not None
        if removed:
            self._sequence_plan()
        return removed

    def configure_source(
        self,
        source_id: str,
        *,
        offset_seconds: float | None = None,
        tolerance_seconds: float | None = None,
    ) -> RegisteredTimeline:
        source = validate_identifier(source_id, label="source_id")
        current = self._sources[source]
        updated = RegisteredTimeline(
            source,
            current.module_id,
            current.offset_seconds if offset_seconds is None else offset_seconds,
            (
                current.tolerance_seconds
                if tolerance_seconds is None
                else tolerance_seconds
            ),
        )
        self._sources[source] = updated
        self._sequence_plan()
        return updated

    def set_base_source(self, source_id: str) -> None:
        source = validate_identifier(source_id, label="source_id")
        if source not in self._sources:
            raise KeyError(f"Timeline source is not registered: {source}")
        # Compatibility hook: the fewest-frame root is selected automatically.
        self._sequence_plan()

    def subscribe(self, listener: Callable[[SyncSelection], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[SyncSelection], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def subscribe_state(self, listener: Callable[[], None]) -> None:
        """Observe rebuilt plans, including a window containing no images."""

        if listener not in self._state_listeners:
            self._state_listeners.append(listener)

    def unsubscribe_state(self, listener: Callable[[], None]) -> None:
        if listener in self._state_listeners:
            self._state_listeners.remove(listener)

    def _sequence_plan(self):
        source_snapshot = tuple(
            (
                item.source_id,
                item.offset_seconds,
                item.tolerance_seconds,
                self.index.source(item.source_id)["updated_at_utc"],
                self.index.source(item.source_id)["record_count"],
            )
            for item in self.sources
        )
        snapshot = (self.time_start, self.time_end, source_snapshot)
        if snapshot == self._source_snapshot and self.current_plan is not None:
            return self.current_plan
        streams = []
        for registration in self.sources:
            samples = self.index.samples(registration.source_id, matchable_only=True)
            frames = tuple(
                TimeFrame(
                    str(item.locator),
                    normalize_time(
                        item.raw_time_utc or item.observed_at_utc,
                        source=registration.source_id,
                    ),
                    payload=item,
                )
                for item in samples
                if item.observed_at_utc is not None
            )
            streams.append(
                TimeStream(
                    registration.source_id,
                    frames,
                    registration.tolerance_seconds,
                    registration.offset_seconds,
                )
            )
        plan = build_sequence_match_plan(
            streams, time_start=self.time_start, time_end=self.time_end
        )
        if self._plan_fingerprint != plan.fingerprint:
            self._cursor_index = None
        self._plan_fingerprint = plan.fingerprint
        self.current_plan = plan
        self._source_snapshot = snapshot
        self.base_source_id = plan.master_source_id
        self.output_source_id = plan.output_source_id
        if not plan.rows:
            self.current_time_utc = None
            self.current_time_text = None
            self._cursor_index = None
        for listener in tuple(self._state_listeners):
            listener()
        return plan

    def _row_key(self, row):
        # Scientific matching epochs never replace displayed observation UTC.
        return row.time

    def _select_row(self, plan, row_index: int) -> SyncSelection:
        row = plan.rows[row_index]
        selection = SyncSelection(
            base_source_id=plan.master_source_id,
            output_source_id=plan.output_source_id,
            current_time_utc=normalize_utc(self._row_key(row).utc + "Z"),
            matched_locators={
                key: item.frame.frame_id if item.frame else None
                for key, item in row.selections.items()
            },
        )
        self.current_time_utc = selection.current_time_utc
        self.current_time_text = self._row_key(row).utc + "Z"
        self._cursor_index = row_index
        for listener in tuple(self._listeners):
            listener(selection)
        return selection

    def select(self, current_time_utc: datetime | str) -> SyncSelection:
        normalize_utc(current_time_utc)
        target = normalize_time(current_time_utc)
        plan = self._sequence_plan()
        if not plan.rows:
            raise ValueError("No available images inside the selected UTC window")
        row_index = min(
            range(len(plan.rows)),
            key=lambda index: (
                abs(self._row_key(plan.rows[index]).tai_ns - target.tai_ns),
                index,
            ),
        )
        return self._select_row(plan, row_index)

    def step(self, delta: int) -> SyncSelection:
        plan = self._sequence_plan()
        if not plan.rows:
            raise ValueError("No available images inside the selected UTC window")
        if delta == 0 and self.current_time_utc is None:
            raise ValueError("No current timeline time is selected")
        if self._cursor_index is None:
            if self.current_time_utc is not None:
                target = normalize_time(self.current_time_text or self.current_time_utc)
                index = min(
                    range(len(plan.rows)),
                    key=lambda i: (
                        abs(self._row_key(plan.rows[i]).tai_ns - target.tai_ns),
                        i,
                    ),
                )
                index += delta
            else:
                index = 0 if delta >= 0 else len(plan.rows) - 1
        else:
            index = self._cursor_index + delta
        index = min(len(plan.rows) - 1, max(0, index))
        return self._select_row(plan, index)

    def _base_times(self) -> list[datetime]:
        plan = self._sequence_plan()
        return [normalize_utc(self._row_key(row).utc + "Z") for row in plan.rows]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "master_policy": "auto_min_frames_hierarchical",
            "overflow_policy": "keep_flagged",
            "time_start": self.time_start,
            "time_end": self.time_end,
            "window_boundary": "[start,end)",
            "window_clock": "observed_utc",
            "algorithm_version": 3,
            "output_source_id": self.output_source_id,
            "source_order": (
                list(self.current_plan.source_order)
                if self.current_plan is not None
                else []
            ),
            "migration_note": self.migration_note,
            "plan_fingerprint": self._plan_fingerprint,
            "cursor_index": self._cursor_index,
            "base_source_id": self.base_source_id,
            "current_time_utc": (
                self.current_time_text or utc_text(self.current_time_utc)
                if self.current_time_utc is not None
                else None
            ),
            "sources": [
                {
                    "source_id": item.source_id,
                    "module_id": item.module_id,
                    "offset_seconds": item.offset_seconds,
                    "tolerance_seconds": item.tolerance_seconds,
                }
                for item in self.sources
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve(strict=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.tmp")
        staging.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
        return target

    def load_dict(self, payload: dict[str, Any]) -> None:
        """Restore versioned configuration already loaded from a project."""

        if payload.get("schema_version") not in {1, 2}:
            raise ValueError("Unsupported timeline coordinator schema")
        strategy = int(payload.get("algorithm_version", 1))
        if strategy not in {1, 2, 3}:
            raise ValueError("Unsupported timeline matching strategy")
        self.migration_note = (
            "Migrated strategy 1/2: rebuilt hierarchical full-coverage maps from the fewest-frame root; saved cursor references were invalidated."
            if strategy in {1, 2} or payload.get("master_policy") == "auto_max_frames"
            else str(payload.get("migration_note", ""))
        )
        registrations: dict[str, RegisteredTimeline] = {}
        for item in payload.get("sources", []):
            registration = RegisteredTimeline(**item)
            self.index.source(registration.source_id)
            registrations[registration.source_id] = registration
        base = payload.get("base_source_id")
        if base is not None and base not in registrations:
            raise ValueError("Saved base source is not registered")
        current = payload.get("current_time_utc")
        start, end = self._normalized_window(
            payload.get("time_start"), payload.get("time_end")
        )
        self._sources = registrations
        self.time_start, self.time_end = start, end
        self.base_source_id = base
        self.current_time_utc = normalize_utc(current) if current else None
        self.current_time_text = utc_text(current) if current else None
        self._source_snapshot = None
        self._cursor_index = None
        self._sequence_plan()
        cursor = payload.get("cursor_index")
        if (
            strategy == 3
            and payload.get("plan_fingerprint") == self._plan_fingerprint
            and isinstance(cursor, int)
            and 0 <= cursor < len(self.current_plan.rows)
        ):
            self._select_row(self.current_plan, cursor)
        elif current and self.current_plan.rows:
            self.select(current)

    def load(self, path: str | Path) -> bool:
        target = Path(path).expanduser().resolve(strict=False)
        if not target.is_file():
            return False
        payload = json.loads(target.read_text(encoding="utf-8"))
        self.load_dict(payload)
        return True


__all__ = [
    "RegisteredTimeline",
    "SQLiteTimelineIndex",
    "TimeCoordinator",
    "TimelineSample",
    "normalize_utc",
    "utc_text",
]
