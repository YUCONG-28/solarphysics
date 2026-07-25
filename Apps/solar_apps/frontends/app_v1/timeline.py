# SPDX-License-Identifier: GPL-3.0-only
"""Rebuildable UTC index and deterministic App 1.0 time coordinator."""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import SyncSelection, validate_identifier

_SCHEMA_VERSION = 1
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

    return (
        normalize_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class TimelineSample:
    """One local file or table-row observation in the rebuildable index."""

    locator: str
    observed_at_utc: datetime | None
    status: str = "available"
    position: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not str(self.locator).strip():
            raise ValueError("Timeline sample locator is required")
        normalized_status = str(self.status).strip().lower()
        if normalized_status not in _STATUSES:
            raise ValueError(f"Unsupported timeline sample status: {self.status}")
        object.__setattr__(self, "status", normalized_status)
        if self.observed_at_utc is not None:
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
            connection.executescript("""
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
                """)
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
            observed = (
                utc_text(item.observed_at_utc)
                if item.observed_at_utc is not None
                else None
            )
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
            )
            for row in rows
        ]

    @property
    def schema_version(self) -> int:
        with self._connect() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0])


class TimeCoordinator:
    """Nearest-time matcher with offsets, tolerances, stepping, and broadcast."""

    def __init__(self, index: SQLiteTimelineIndex) -> None:
        self.index = index
        self._sources: dict[str, RegisteredTimeline] = {}
        self.base_source_id: str | None = None
        self.current_time_utc: datetime | None = None
        self._listeners: list[Callable[[SyncSelection], None]] = []

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
            source_id=source_id,
            module_id=module_id or str(indexed["module_id"]),
            offset_seconds=offset_seconds,
            tolerance_seconds=tolerance_seconds,
        )
        self._sources[registration.source_id] = registration
        if self.base_source_id is None:
            self.base_source_id = registration.source_id
        return registration

    def unregister_source(self, source_id: str) -> bool:
        source = validate_identifier(source_id, label="source_id")
        removed = self._sources.pop(source, None) is not None
        if self.base_source_id == source:
            self.base_source_id = next(iter(sorted(self._sources)), None)
            self.current_time_utc = None
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
            source_id=current.source_id,
            module_id=current.module_id,
            offset_seconds=(
                current.offset_seconds
                if offset_seconds is None
                else float(offset_seconds)
            ),
            tolerance_seconds=(
                current.tolerance_seconds
                if tolerance_seconds is None
                else float(tolerance_seconds)
            ),
        )
        self._sources[source] = updated
        return updated

    def set_base_source(self, source_id: str) -> None:
        source = validate_identifier(source_id, label="source_id")
        if source not in self._sources:
            raise KeyError(f"Timeline source is not registered: {source}")
        self.base_source_id = source

    def subscribe(self, listener: Callable[[SyncSelection], None]) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[SyncSelection], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def select(self, current_time_utc: datetime | str) -> SyncSelection:
        if self.base_source_id is None:
            raise ValueError("No base timeline source is registered")
        current = normalize_utc(current_time_utc)
        selection = SyncSelection(
            base_source_id=self.base_source_id,
            current_time_utc=current,
            matched_locators={
                source_id: self._nearest_locator(registration, current)
                for source_id, registration in sorted(self._sources.items())
            },
        )
        self.current_time_utc = current
        for listener in tuple(self._listeners):
            listener(selection)
        return selection

    def step(self, delta: int) -> SyncSelection:
        if delta == 0:
            if self.current_time_utc is None:
                raise ValueError("No current timeline time is selected")
            return self.select(self.current_time_utc)
        timeline = self._base_times()
        if not timeline:
            raise ValueError("The base timeline has no available UTC samples")
        if self.current_time_utc is None:
            index = 0 if delta > 0 else len(timeline) - 1
        elif delta > 0:
            index = min(
                bisect.bisect_right(timeline, self.current_time_utc) + delta - 1,
                len(timeline) - 1,
            )
        else:
            index = max(
                bisect.bisect_left(timeline, self.current_time_utc) + delta,
                0,
            )
        return self.select(timeline[index])

    def _base_times(self) -> list[datetime]:
        if self.base_source_id is None:
            return []
        registration = self._sources[self.base_source_id]
        offset = timedelta(seconds=registration.offset_seconds)
        return sorted(
            {
                sample.observed_at_utc + offset
                for sample in self.index.samples(
                    registration.source_id,
                    matchable_only=True,
                )
                if sample.observed_at_utc is not None
            }
        )

    def _nearest_locator(
        self,
        registration: RegisteredTimeline,
        current: datetime,
    ) -> str | None:
        offset = timedelta(seconds=registration.offset_seconds)
        candidates = []
        for sample in self.index.samples(
            registration.source_id,
            matchable_only=True,
        ):
            if sample.observed_at_utc is None:
                continue
            effective = sample.observed_at_utc + offset
            difference = abs((effective - current).total_seconds())
            candidates.append(
                (
                    difference,
                    effective,
                    str(sample.locator).casefold(),
                    str(sample.locator),
                )
            )
        if not candidates:
            return None
        best = min(candidates)
        if best[0] > registration.tolerance_seconds:
            return None
        return best[3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_source_id": self.base_source_id,
            "current_time_utc": (
                utc_text(self.current_time_utc)
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

        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported timeline coordinator schema")
        registrations: dict[str, RegisteredTimeline] = {}
        for item in payload.get("sources", []):
            registration = RegisteredTimeline(**item)
            self.index.source(registration.source_id)
            registrations[registration.source_id] = registration
        base = payload.get("base_source_id")
        if base is not None and base not in registrations:
            raise ValueError("Saved base source is not registered")
        current = payload.get("current_time_utc")
        self._sources = registrations
        self.base_source_id = base
        self.current_time_utc = normalize_utc(current) if current else None

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
