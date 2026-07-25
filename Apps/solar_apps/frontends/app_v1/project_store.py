# SPDX-License-Identifier: GPL-3.0-only
"""Versioned, atomic App 1.0 project and parameter-preset persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .contracts import AppV1ProjectV1, validate_identifier
from .runtime import AppV1RuntimePaths
from .timeline import normalize_utc

PROJECT_SUFFIX = ".spapp.json"
PRESET_SCHEMA_VERSION = 1


class AppV1ProjectStore:
    """Persist private project metadata below the existing Local runtime."""

    def __init__(self, runtime: AppV1RuntimePaths) -> None:
        self.runtime = runtime

    def save(self, project: AppV1ProjectV1) -> Path:
        normalized = AppV1ProjectV1(
            project_id=project.project_id,
            name=project.name,
            modules=project.modules,
            parameters=project.parameters,
            timeline=project.timeline,
            layout=project.layout,
            artifact_manifests=tuple(
                self.normalize_manifest_reference(item)
                for item in project.artifact_manifests
            ),
            saved_at_utc=project.saved_at_utc,
            schema_version=project.schema_version,
        )
        target = self.runtime.project_file(normalized.project_id)
        _atomic_json(target, normalized.to_dict())
        return target

    def load(self, project_id: str) -> AppV1ProjectV1:
        target = self.runtime.project_file(project_id)
        payload = _load_object(target)
        return self.from_dict(payload)

    def from_dict(self, payload: Mapping[str, Any]) -> AppV1ProjectV1:
        if not isinstance(payload, Mapping):
            raise TypeError("Project payload must be an object")
        saved = normalize_utc(payload["saved_at_utc"])
        manifests = tuple(
            self.normalize_manifest_reference(str(item))
            for item in payload.get("artifact_manifests", ())
        )
        return AppV1ProjectV1(
            project_id=str(payload["project_id"]),
            name=str(payload["name"]),
            modules=tuple(str(item) for item in payload.get("modules", ())),
            parameters=dict(payload.get("parameters", {})),
            timeline=dict(payload.get("timeline", {})),
            layout=dict(payload.get("layout", {})),
            artifact_manifests=manifests,
            saved_at_utc=saved,
            schema_version=int(payload.get("schema_version", 0)),
        )

    def normalize_manifest_reference(self, reference: str | Path) -> str:
        """Return a safe path relative to ``Local/outputs/app_v1``."""

        root = self.runtime.outputs_dir.resolve(strict=False)
        candidate = Path(reference)
        if candidate.is_absolute():
            resolved = candidate.expanduser().resolve(strict=False)
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "Artifact manifests must stay under Local/outputs/app_v1"
                ) from exc
        else:
            pure = PurePosixPath(str(reference).replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(
                    "Artifact manifest reference is not a safe relative path"
                )
            relative = Path(*pure.parts)
        if not relative.parts or relative.name != "manifest.json":
            raise ValueError("Artifact references must point to manifest.json")
        return PurePosixPath(*relative.parts).as_posix()

    def save_preset(
        self,
        module_id: str,
        preset_id: str,
        parameters: Mapping[str, Any],
    ) -> Path:
        module = validate_identifier(module_id, label="module_id")
        preset = validate_identifier(preset_id, label="preset_id")
        payload = {
            "schema_version": PRESET_SCHEMA_VERSION,
            "module_id": module,
            "preset_id": preset,
            "parameters": _json_object(parameters, label="parameters"),
        }
        target = self.runtime.workspaces_dir / "presets" / module / f"{preset}.json"
        _atomic_json(target, payload)
        return target

    def load_preset(self, module_id: str, preset_id: str) -> dict[str, Any]:
        module = validate_identifier(module_id, label="module_id")
        preset = validate_identifier(preset_id, label="preset_id")
        target = self.runtime.workspaces_dir / "presets" / module / f"{preset}.json"
        payload = _load_object(target)
        if payload.get("schema_version") != PRESET_SCHEMA_VERSION:
            raise ValueError("Unsupported parameter preset schema")
        if payload.get("module_id") != module or payload.get("preset_id") != preset:
            raise ValueError("Parameter preset identity does not match its path")
        return _json_object(payload.get("parameters", {}), label="parameters")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
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
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "AppV1ProjectStore",
    "PRESET_SCHEMA_VERSION",
    "PROJECT_SUFFIX",
]
