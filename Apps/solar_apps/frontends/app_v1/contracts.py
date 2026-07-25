# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe, JSON-compatible contracts shared by App 1.0 modules."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PHASES = frozenset({"1", "2A", "2B", "2C", "3", "4", "5"})


class RunStatus(StrEnum):
    """Lifecycle states emitted by the future process task runner."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    """Validate stable IDs used in commands, manifests, and runtime paths."""

    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{label} must be a lowercase kebab-case identifier")
    return normalized


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return value.astimezone(timezone.utc)


def _json_object(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    try:
        encoded = json.dumps(dict(value), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be an object")
    return decoded


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


class JsonContract:
    """Small serialization surface for persisted version-1 contracts."""

    def to_dict(self) -> dict[str, Any]:
        value = _json_value(self)
        if not isinstance(value, dict):
            raise TypeError("Contract did not serialize to an object")
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ModuleDescriptor(JsonContract):
    """One stable App 1.0 page or dashboard."""

    module_id: str
    title: str
    category: str
    target_phase: str
    legacy_interface: str | None = None
    time_aware: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "module_id", validate_identifier(self.module_id, label="module_id")
        )
        if not self.title.strip() or not self.category.strip():
            raise ValueError("Module title and category are required")
        if self.target_phase not in _PHASES:
            raise ValueError(f"Unsupported target phase: {self.target_phase}")


@dataclass(frozen=True, slots=True)
class InputReference(JsonContract):
    """A local input reference; raw observation bytes are never embedded."""

    reference_id: str
    kind: str
    locator: str
    sha256: str | None = None
    observed_at_utc: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            validate_identifier(self.reference_id, label="reference_id"),
        )
        if not self.kind.strip() or not self.locator.strip():
            raise ValueError("Input kind and locator are required")
        if self.observed_at_utc is not None:
            object.__setattr__(
                self,
                "observed_at_utc",
                _utc(self.observed_at_utc, label="observed_at_utc"),
            )
        object.__setattr__(
            self, "metadata", _json_object(self.metadata, label="metadata")
        )


@dataclass(frozen=True, slots=True)
class ArtifactProduct(JsonContract):
    """One product stored below a module run directory."""

    kind: str
    relative_path: str
    media_type: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Artifact paths must be safe run-relative paths")
        if not self.kind.strip() or not self.media_type.strip():
            raise ValueError("Artifact kind and media type are required")


@dataclass(frozen=True, slots=True)
class RunRequest(JsonContract):
    """Confirmed request passed from the future Qt shell to a worker process."""

    run_id: str
    project_id: str
    module_id: str
    inputs: tuple[InputReference, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    requested_at_utc: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "project_id", "module_id"):
            object.__setattr__(
                self, name, validate_identifier(getattr(self, name), label=name)
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported run-request schema")
        object.__setattr__(
            self,
            "requested_at_utc",
            _utc(self.requested_at_utc, label="requested_at_utc"),
        )
        object.__setattr__(
            self, "parameters", _json_object(self.parameters, label="parameters")
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifestV1(JsonContract):
    """Reproducibility manifest written after atomic product promotion."""

    project_id: str
    run_id: str
    module_id: str
    status: RunStatus
    inputs: tuple[InputReference, ...]
    parameters: dict[str, Any]
    products: tuple[ArtifactProduct, ...]
    software: dict[str, Any]
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    time_start_utc: datetime | None = None
    time_end_utc: datetime | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("run_id", "project_id", "module_id"):
            object.__setattr__(
                self, name, validate_identifier(getattr(self, name), label=name)
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported artifact-manifest schema")
        if self.status not in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            raise ValueError("A persisted manifest requires a terminal run status")
        object.__setattr__(
            self,
            "created_at_utc",
            _utc(self.created_at_utc, label="created_at_utc"),
        )
        for name in ("time_start_utc", "time_end_utc"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _utc(value, label=name))
        if (
            self.time_start_utc is not None
            and self.time_end_utc is not None
            and self.time_end_utc < self.time_start_utc
        ):
            raise ValueError("time_end_utc cannot precede time_start_utc")
        object.__setattr__(
            self, "parameters", _json_object(self.parameters, label="parameters")
        )
        object.__setattr__(
            self, "software", _json_object(self.software, label="software")
        )


@dataclass(frozen=True, slots=True)
class RunResult(JsonContract):
    """Terminal or current worker state reported to the application shell."""

    run_id: str
    status: RunStatus
    message: str = ""
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "run_id", validate_identifier(self.run_id, label="run_id")
        )


@dataclass(frozen=True, slots=True)
class TimelineSource(JsonContract):
    """UTC sample index for one time-aware module source."""

    source_id: str
    module_id: str
    timestamps_utc: tuple[datetime, ...]
    offset_seconds: float = 0.0
    tolerance_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("source_id", "module_id"):
            object.__setattr__(
                self, name, validate_identifier(getattr(self, name), label=name)
            )
        timestamps = tuple(
            _utc(value, label="timestamps_utc") for value in self.timestamps_utc
        )
        if tuple(sorted(timestamps)) != timestamps:
            raise ValueError("Timeline timestamps must be sorted")
        if self.tolerance_seconds < 0:
            raise ValueError("Timeline tolerance cannot be negative")
        object.__setattr__(self, "timestamps_utc", timestamps)


@dataclass(frozen=True, slots=True)
class SyncSelection(JsonContract):
    """One deterministic current-time selection across module sources."""

    base_source_id: str
    current_time_utc: datetime
    matched_locators: dict[str, str | None]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_source_id",
            validate_identifier(self.base_source_id, label="base_source_id"),
        )
        object.__setattr__(
            self,
            "current_time_utc",
            _utc(self.current_time_utc, label="current_time_utc"),
        )
        matches = {
            validate_identifier(str(key), label="matched source"): (
                None if value is None else str(value)
            )
            for key, value in self.matched_locators.items()
        }
        object.__setattr__(self, "matched_locators", matches)


@dataclass(frozen=True, slots=True)
class AppV1ProjectV1(JsonContract):
    """Saved App 1.0 project without embedded observation data."""

    project_id: str
    name: str
    modules: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    timeline: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] = field(default_factory=dict)
    artifact_manifests: tuple[str, ...] = ()
    saved_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "project_id",
            validate_identifier(self.project_id, label="project_id"),
        )
        if not self.name.strip():
            raise ValueError("Project name is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unsupported project schema")
        object.__setattr__(
            self,
            "modules",
            tuple(validate_identifier(item, label="module") for item in self.modules),
        )
        for name in ("parameters", "timeline", "layout"):
            object.__setattr__(
                self,
                name,
                _json_object(getattr(self, name), label=name),
            )
        object.__setattr__(
            self,
            "saved_at_utc",
            _utc(self.saved_at_utc, label="saved_at_utc"),
        )


__all__ = [
    "SCHEMA_VERSION",
    "AppV1ProjectV1",
    "ArtifactManifestV1",
    "ArtifactProduct",
    "InputReference",
    "JsonContract",
    "ModuleDescriptor",
    "RunRequest",
    "RunResult",
    "RunStatus",
    "SyncSelection",
    "TimelineSource",
    "validate_identifier",
]
