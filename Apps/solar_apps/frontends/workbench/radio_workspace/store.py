"""Atomic file-backed storage and safe local-file access for Radio Workspace."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO

from solar_apps.workflows.common.image_naming import build_scientific_image_filename

from .catalog import MODULES, PRESETS
from .contracts import (
    FIGURE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    UI_LAYOUT_VERSION,
    RadioArtifact,
    RadioFigureDraft,
    RadioFigureExport,
    RadioFigurePreflight,
    RadioFigureTemporalBinding,
    RadioRunManifest,
    RadioWorkspace,
)
from .figure_media import (
    MAX_ANIMATION_FRAMES,
    MAX_FIGURE_LAYERS,
    canonical_suffix,
    copy_limited_stream,
    create_thumbnail,
    max_bytes_for_mime,
    sha256_file,
    validate_dimensions,
    validate_media_magic,
    validate_png,
    validate_raster_image,
)

_MAX_USER_ROOTS = 32
_FIGURE_PREVIEW_METADATA_FIELDS = frozenset(
    {
        "action_id",
        "adapter",
        "annotation",
        "axis_mapping",
        "candidate_count",
        "candidate_limit",
        "coverage_segments",
        "coverage_gaps",
        "discovered_file_count",
        "f_max_mhz",
        "f_min_mhz",
        "first_frame_time",
        "frame_count",
        "frame_time",
        "frequency_mhz",
        "hdu_index",
        "image_shape",
        "kind",
        "last_frame_time",
        "lr_comparison_count",
        "mode",
        "module_id",
        "observation_time",
        "pattern",
        "playback_frame_count",
        "playback_sampled",
        "plot_area",
        "polarization",
        "recursive",
        "row_count",
        "scientific_metadata",
        "selection",
        "source_name",
        "spectrogram_axis",
        "tail_n",
        "temporal_binding",
        "title",
        "total_count",
        "truncated",
        "visible_row_count",
        "x_axis",
        "x_axis_mapping",
        "x_end_iso",
        "x_start_iso",
        "y_axis",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value)
        if item not in result:
            result.append(item)
    return result


class SafePathBrowser:
    """Resolve and list paths without escaping explicitly allowed local roots."""

    def __init__(self, allowed_roots: Iterable[str | Path]) -> None:
        roots: list[Path] = []
        for value in allowed_roots:
            root = Path(value).expanduser().resolve(strict=False)
            if root not in roots:
                roots.append(root)
        if not roots:
            raise ValueError("At least one allowed root is required")
        self.allowed_roots = tuple(roots)

    def resolve(
        self,
        value: str | Path,
        *,
        must_exist: bool = True,
        file_only: bool = False,
        directory_only: bool = False,
    ) -> Path:
        raw = str(value).strip()
        if not raw:
            raise ValueError("path is required")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise ValueError("Browser paths must be absolute")
        resolved = candidate.resolve(strict=must_exist)
        if not _inside(resolved, self.allowed_roots):
            raise PermissionError(f"Path is outside allowed roots: {resolved}")
        if file_only and not resolved.is_file():
            raise FileNotFoundError(f"File does not exist: {resolved}")
        if directory_only and not resolved.is_dir():
            raise NotADirectoryError(f"Directory does not exist: {resolved}")
        return resolved

    def roots_payload(self) -> dict[str, Any]:
        return {
            "path": None,
            "roots": [str(root) for root in self.allowed_roots],
            "entries": [
                self._entry(root, root.name or str(root)) for root in self.allowed_roots
            ],
        }

    def list_directory(self, value: str | Path | None = None) -> dict[str, Any]:
        if value in (None, ""):
            return self.roots_payload()
        folder = self.resolve(value, directory_only=True)
        entries: list[dict[str, Any]] = []
        for child in folder.iterdir():
            try:
                resolved = child.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if not _inside(resolved, self.allowed_roots):
                continue
            try:
                entries.append(self._entry(child, child.name))
            except OSError:
                continue
        entries.sort(key=lambda item: (not item["is_dir"], item["name"].casefold()))
        return {
            "path": str(folder),
            "roots": [str(root) for root in self.allowed_roots],
            "entries": entries,
        }

    @staticmethod
    def _entry(path: Path, name: str) -> dict[str, Any]:
        stat = path.stat()
        return {
            "name": name,
            "path": str(path.resolve(strict=True)),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
            "size": stat.st_size if path.is_file() else None,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat(),
            "symlink": path.is_symlink(),
        }


class RadioWorkspaceStore:
    """Persist workspaces and runs below ``<output-root>/radio_workbench``."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        allowed_roots: Iterable[str | Path] = (),
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        self.root = self.output_root / "radio_workbench"
        self.root.mkdir(parents=True, exist_ok=True)
        resolved_root = self.root.resolve(strict=True)
        if (
            resolved_root.parent != self.output_root
            or resolved_root.name != "radio_workbench"
        ):
            raise PermissionError("Radio workbench storage escaped the output root")
        self.root = resolved_root
        self._lock = threading.RLock()
        startup_roots: list[Path] = []
        for value in allowed_roots:
            root = Path(value).expanduser().resolve(strict=False)
            if root not in startup_roots:
                startup_roots.append(root)
        self._startup_roots = tuple(startup_roots)
        self._user_roots = self._startup_roots
        self._protected_roots = (self.output_root,)
        self._browser = SafePathBrowser([*self._user_roots, *self._protected_roots])
        self._index_path = self.root / "workspace_index.json"
        self._module_ids = [module.id for module in MODULES]
        with self._lock:
            self._refresh_protected_roots_locked()

    @property
    def browser(self) -> SafePathBrowser:
        """Return the current immutable browser snapshot."""

        with self._lock:
            return self._browser

    @property
    def startup_roots(self) -> tuple[Path, ...]:
        """Return the immutable roots declared when the service started."""

        return self._startup_roots

    @property
    def user_roots(self) -> tuple[Path, ...]:
        """Return the roots currently selected by the local user."""

        with self._lock:
            return self._user_roots

    @property
    def protected_roots(self) -> tuple[Path, ...]:
        """Return output roots retained for persisted workspaces and results."""

        with self._lock:
            return self._protected_roots

    def replace_user_roots(self, roots: Iterable[str | Path]) -> tuple[Path, ...]:
        """Atomically replace user-selected roots after strict validation."""

        if isinstance(roots, (str, bytes, Path)):
            raise TypeError("roots must be an array of absolute directory paths")
        values = list(roots)
        if not values:
            raise ValueError("At least one user root is required")
        if len(values) > _MAX_USER_ROOTS:
            raise ValueError(f"No more than {_MAX_USER_ROOTS} user roots are allowed")

        normalized: list[Path] = []
        for value in values:
            if not isinstance(value, (str, Path)):
                raise TypeError("Every user root must be a path string")
            raw = str(value).strip()
            if not raw:
                raise ValueError("User root paths may not be empty")
            candidate = Path(raw)
            if not candidate.is_absolute():
                raise ValueError(f"User roots must be absolute paths: {raw}")
            resolved = candidate.expanduser().resolve(strict=True)
            if not resolved.is_dir():
                raise NotADirectoryError(f"User root is not a directory: {resolved}")
            if resolved not in normalized:
                normalized.append(resolved)

        if not normalized:
            raise ValueError("At least one user root is required")
        with self._lock:
            self._refresh_protected_roots_locked()
            browser = SafePathBrowser([*normalized, *self._protected_roots])
            self._user_roots = tuple(normalized)
            self._browser = browser
            return self._user_roots

    def allowed_roots_payload(self) -> dict[str, Any]:
        """Describe startup, user-selected, and always-retained output roots."""

        with self._lock:
            return {
                "user_roots": [str(root) for root in self._user_roots],
                "startup_roots": [str(root) for root in self._startup_roots],
                "output_root": str(self.output_root),
                "protected_roots": [str(root) for root in self._protected_roots],
                "effective_roots": [str(root) for root in self._browser.allowed_roots],
                "update_effect": (
                    "Changes apply to future browsing, previews, and runs; active "
                    "and queued runs keep their resolved inputs. Workspace output "
                    "roots remain protected until their workspaces are deleted."
                ),
            }

    def create_workspace(
        self,
        *,
        name: str = "Radio Workspace",
        event_preset: dict[str, Any] | None = None,
        shared_paths: dict[str, str] | None = None,
        advanced_config: dict[str, Any] | None = None,
        concurrency: int = 1,
        workspace_id: str | None = None,
        output_root: str | Path | None = None,
    ) -> RadioWorkspace:
        workspace_id = workspace_id or uuid.uuid4().hex
        with self._lock:
            selected_output_root = (
                self.output_root
                if output_root in (None, "")
                else self._browser.resolve(output_root, must_exist=False)
            )
            now = utc_now()
            enabled = [module.id for module in MODULES if module.always_available]
            collapsed = list(self._module_ids)
            workspace = RadioWorkspace(
                schema_version=SCHEMA_VERSION,
                id=workspace_id,
                name=name,
                output_root=str(selected_output_root),
                created_at=now,
                updated_at=now,
                event_preset=event_preset or {},
                shared_paths=shared_paths or {},
                advanced_config=advanced_config or {},
                enabled_modules=enabled,
                module_order=list(self._module_ids),
                collapsed_modules=collapsed,
                pinned_modules=[],
                concurrency=concurrency,
                ui_layout_version=UI_LAYOUT_VERSION,
            )
            self._validate_workspace_modules(workspace)
            self._validate_shared_paths(workspace.shared_paths)
            path = self._workspace_path(selected_output_root, workspace.id)
            if path.exists():
                raise FileExistsError(f"Radio workspace already exists: {workspace.id}")
            (path / "runs").mkdir(parents=True)
            self._atomic_json(path / "workspace.json", workspace.to_dict())
            figure_root = path / "figure_studio"
            self._ensure_figure_dirs(figure_root)
            now_draft = RadioFigureDraft.empty(workspace.id)
            now_draft.created_at = now
            now_draft.updated_at = now
            self._atomic_json(figure_root / "draft.json", now_draft.to_dict())
            index = self._read_index()
            index[workspace.id] = str(selected_output_root)
            self._write_index(index)
            self._refresh_protected_roots_locked()
        return workspace

    def list_workspaces(self) -> list[RadioWorkspace]:
        workspaces: list[RadioWorkspace] = []
        with self._lock:
            paths: dict[str, Path] = {
                workspace_id: self._workspace_path(output_root, workspace_id)
                for workspace_id, output_root in self._read_index().items()
            }
            primary_root = self._validated_primary_root()
            for item in primary_root.iterdir():
                try:
                    resolved_item = item.resolve(strict=True)
                except (FileNotFoundError, OSError):
                    continue
                if (
                    not resolved_item.is_dir()
                    or resolved_item.parent != primary_root
                    or resolved_item.name != item.name
                ):
                    continue
                paths.setdefault(item.name, resolved_item)
            for item in paths.values():
                config = item / "workspace.json"
                if not config.is_file():
                    continue
                try:
                    workspaces.append(self._read_workspace(config))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return sorted(workspaces, key=lambda item: item.updated_at, reverse=True)

    def load_workspace(self, workspace_id: str) -> RadioWorkspace:
        path = self.workspace_dir(workspace_id) / "workspace.json"
        with self._lock:
            if not path.is_file():
                raise KeyError(f"Unknown radio workspace: {workspace_id}")
            return self._read_workspace(path)

    def update_workspace(
        self, workspace_id: str, updates: dict[str, Any]
    ) -> RadioWorkspace:
        allowed = {
            "name",
            "event_preset",
            "shared_paths",
            "advanced_config",
            "enabled_modules",
            "module_order",
            "collapsed_modules",
            "pinned_modules",
            "concurrency",
            "ui_layout_version",
        }
        if "output_root" in updates:
            raise ValueError(
                "output_root cannot be changed after workspace creation; "
                "create a new workspace instead"
            )
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unknown workspace fields: {', '.join(sorted(unknown))}")
        if "ui_layout_version" in updates and int(updates["ui_layout_version"]) != (
            UI_LAYOUT_VERSION
        ):
            raise ValueError(f"ui_layout_version must be {UI_LAYOUT_VERSION}")
        with self._lock:
            workspace = self.load_workspace(workspace_id)
            payload = workspace.to_dict()
            payload.update(updates)
            payload["updated_at"] = utc_now()
            updated = RadioWorkspace.from_dict(payload)
            self._validate_workspace_modules(updated)
            self._validate_shared_paths(updated.shared_paths)
            self._atomic_json(
                self.workspace_dir(workspace_id) / "workspace.json",
                updated.to_dict(),
            )
        return updated

    def update_layout(
        self, workspace_id: str, payload: dict[str, Any]
    ) -> RadioWorkspace:
        updates = dict(payload)
        preset_id = updates.pop("preset_id", None)
        allowed = {
            "enabled_modules",
            "module_order",
            "collapsed_modules",
            "pinned_modules",
            "ui_layout_version",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unknown layout fields: {', '.join(sorted(unknown))}")
        if preset_id:
            try:
                preset_modules = list(PRESETS[str(preset_id)]["module_ids"])
            except KeyError as exc:
                raise KeyError(f"Unknown radio preset: {preset_id}") from exc
            updates.setdefault("enabled_modules", preset_modules)
            updates.setdefault(
                "module_order",
                [
                    *preset_modules,
                    *[item for item in self._module_ids if item not in preset_modules],
                ],
            )
            updates.setdefault(
                "collapsed_modules",
                [item for item in self._module_ids if item not in preset_modules],
            )
        return self.update_workspace(workspace_id, updates)

    def delete_workspace(self, workspace_id: str) -> None:
        path = self.workspace_dir(workspace_id)
        with self._lock:
            if not path.is_dir():
                raise KeyError(f"Unknown radio workspace: {workspace_id}")
            resolved = path.resolve(strict=True)
            if resolved.parent != self.root.resolve(strict=True) or path.is_symlink():
                expected_parent = (
                    Path(self.load_workspace(workspace_id).output_root)
                    / "radio_workbench"
                ).resolve(strict=True)
                if resolved.parent != expected_parent or path.is_symlink():
                    raise PermissionError("Refusing to delete an unsafe workspace path")
            shutil.rmtree(resolved)
            index = self._read_index()
            index.pop(workspace_id, None)
            self._write_index(index)
            self._refresh_protected_roots_locked()

    def workspace_dir(self, workspace_id: str) -> Path:
        if not workspace_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for char in workspace_id
        ):
            raise ValueError(f"Invalid workspace id: {workspace_id!r}")
        with self._lock:
            output_root = self._read_index().get(workspace_id, str(self.output_root))
        return self._workspace_path(output_root, workspace_id)

    def figure_studio_dir(self, workspace_id: str) -> Path:
        """Return Figure Studio storage only when every directory stays in scope."""

        workspace = self.workspace_dir(workspace_id)
        if not (workspace / "workspace.json").is_file():
            raise KeyError(f"Unknown radio workspace: {workspace_id}")
        root = workspace / "figure_studio"
        resolved = root.resolve(strict=False)
        if (
            resolved.parent != workspace.resolve(strict=True)
            or resolved.name != "figure_studio"
        ):
            raise PermissionError("Figure Studio storage escaped the workspace")
        if resolved.exists() and not resolved.is_dir():
            raise NotADirectoryError("Figure Studio storage is not a directory")
        for name in ("snapshots", "exports", "sources"):
            child = resolved / name
            if not child.exists():
                continue
            resolved_child = child.resolve(strict=True)
            if (
                not resolved_child.is_dir()
                or resolved_child.parent != resolved
                or resolved_child.name != name
            ):
                raise PermissionError(f"Figure Studio {name} escaped its storage root")
        return resolved

    def load_figure_draft(self, workspace_id: str) -> RadioFigureDraft:
        root = self.figure_studio_dir(workspace_id)
        path = root / "draft.json"
        with self._lock:
            if not path.is_file():
                draft = RadioFigureDraft.empty(workspace_id)
                now = utc_now()
                draft.created_at = now
                draft.updated_at = now
                return draft
            if path.is_symlink():
                raise PermissionError("Figure draft may not be a symlink")
            draft = RadioFigureDraft.from_dict(self._read_json(path))
            if draft.workspace_id != workspace_id:
                raise ValueError("Figure draft workspace id does not match its folder")
            return draft

    def save_figure_draft(
        self, workspace_id: str, draft: RadioFigureDraft | dict[str, Any]
    ) -> RadioFigureDraft:
        candidate = (
            draft
            if isinstance(draft, RadioFigureDraft)
            else RadioFigureDraft.from_dict(draft)
        )
        if candidate.workspace_id != workspace_id:
            raise ValueError("Figure draft workspace id does not match the request")
        self.figure_source_fingerprints(workspace_id, candidate)
        root = self.figure_studio_dir(workspace_id)
        with self._lock:
            self._ensure_figure_dirs(root)
            current_path = root / "draft.json"
            if current_path.is_file() and not current_path.is_symlink():
                current = RadioFigureDraft.from_dict(self._read_json(current_path))
                candidate.created_at = current.created_at or utc_now()
            else:
                candidate.created_at = candidate.created_at or utc_now()
            candidate.id = "active"
            candidate.updated_at = utc_now()
            self._atomic_json(current_path, candidate.to_dict())
        return candidate

    def create_figure_snapshot(
        self,
        workspace_id: str,
        draft: RadioFigureDraft | dict[str, Any],
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        candidate = (
            draft
            if isinstance(draft, RadioFigureDraft)
            else RadioFigureDraft.from_dict(draft)
        )
        if candidate.workspace_id != workspace_id:
            raise ValueError("Figure snapshot workspace id does not match the request")
        self.figure_source_fingerprints(workspace_id, candidate)
        snapshot_id = uuid.uuid4().hex
        record = {
            "figure_schema_version": FIGURE_SCHEMA_VERSION,
            "id": snapshot_id,
            "workspace_id": workspace_id,
            "name": str(name or candidate.name or "Figure Snapshot").strip()
            or "Figure Snapshot",
            "created_at": utc_now(),
            "draft": candidate.to_dict(),
        }
        root = self.figure_studio_dir(workspace_id)
        with self._lock:
            self._ensure_figure_dirs(root)
            path = root / "snapshots" / f"{snapshot_id}.json"
            if path.exists():
                raise FileExistsError(f"Figure snapshot already exists: {snapshot_id}")
            self._atomic_json(path, record)
        return record

    def register_figure_preview(
        self,
        workspace_id: str,
        metadata: dict[str, Any],
        stream: BinaryIO,
    ) -> dict[str, Any]:
        """Persist one immutable PNG preview behind a controlled identifier."""

        self.load_workspace(workspace_id)
        if not isinstance(metadata, dict):
            raise TypeError("preview metadata must be a JSON object")
        unknown_metadata = set(metadata) - _FIGURE_PREVIEW_METADATA_FIELDS
        if unknown_metadata:
            raise ValueError(
                "Unknown figure preview metadata fields: "
                + ", ".join(sorted(str(item) for item in unknown_metadata))
            )
        self._reject_unsafe_source_metadata(metadata)
        temporal = RadioFigureTemporalBinding.from_dict(
            metadata.get("temporal_binding")
        )
        preview_id = uuid.uuid4().hex
        root = self.figure_studio_dir(workspace_id)
        final_dir = root / "sources" / preview_id
        staging = root / "sources" / f".{preview_id}.{uuid.uuid4().hex}.tmp"
        with self._lock:
            self._ensure_figure_dirs(root)
            staging.mkdir()
            try:
                preview_path = staging / "preview.png"
                copy_limited_stream(
                    stream, preview_path, limit=max_bytes_for_mime("image/png")
                )
                validate_media_magic(preview_path, "image/png")
                width, height = validate_png(preview_path)
                fingerprint = sha256_file(preview_path)
                source = {
                    "figure_schema_version": FIGURE_SCHEMA_VERSION,
                    "type": "preview",
                    "preview_id": preview_id,
                    "fingerprint": fingerprint,
                    "mime_type": "image/png",
                    "width": width,
                    "height": height,
                    "created_at": utc_now(),
                    "temporal_binding": temporal.to_dict(),
                    "metadata": {
                        key: value
                        for key, value in metadata.items()
                        if key != "temporal_binding"
                    },
                }
                self._atomic_json(staging / "source.json", source)
                if final_dir.exists():
                    raise FileExistsError(
                        f"Figure preview already exists: {preview_id}"
                    )
                staging.rename(final_dir)
                return source
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def figure_preview_path(
        self, workspace_id: str, preview_id: str
    ) -> tuple[dict[str, Any], Path]:
        preview_id = self._safe_identifier(preview_id, label="preview id")
        studio_root = self.figure_studio_dir(workspace_id)
        root = studio_root / "sources"
        if root.is_symlink():
            raise PermissionError("Figure preview root may not be a symlink")
        if not root.is_dir():
            raise KeyError(f"Unknown figure preview: {preview_id}")
        folder = root / preview_id
        if folder.is_symlink():
            raise PermissionError("Figure preview storage may not be a symlink")
        resolved_folder = folder.resolve(strict=False)
        if resolved_folder.parent != root.resolve(strict=False):
            raise PermissionError("Figure preview path escaped its workspace")
        source_path = folder / "source.json"
        preview_path = folder / "preview.png"
        with self._lock:
            if not source_path.is_file() or not preview_path.is_file():
                raise KeyError(f"Unknown figure preview: {preview_id}")
            if source_path.is_symlink() or preview_path.is_symlink():
                raise PermissionError("Figure preview files may not be symlinks")
            source = self._read_json(source_path)
            if (
                source.get("type") != "preview"
                or source.get("preview_id") != preview_id
                or int(source.get("figure_schema_version", 0)) != FIGURE_SCHEMA_VERSION
                or source.get("mime_type") != "image/png"
                or source.get("fingerprint") != sha256_file(preview_path)
            ):
                raise PermissionError("Figure preview metadata does not match its file")
            try:
                self._reject_unsafe_source_metadata(source.get("metadata", {}))
                RadioFigureTemporalBinding.from_dict(source.get("temporal_binding"))
                validate_dimensions(source.get("width", 0), source.get("height", 0))
            except (TypeError, ValueError) as exc:
                raise PermissionError("Figure preview metadata is unsafe") from exc
        return source, preview_path.resolve(strict=True)

    def figure_source_fingerprints(
        self, workspace_id: str, draft: RadioFigureDraft | dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve every controlled source and return stable revision material."""

        candidate = (
            draft
            if isinstance(draft, RadioFigureDraft)
            else RadioFigureDraft.from_dict(draft)
        )
        if candidate.workspace_id != workspace_id:
            raise ValueError("Figure source workspace id does not match the request")
        if len(candidate.layers) > MAX_FIGURE_LAYERS:
            raise ValueError(
                f"A figure may contain no more than {MAX_FIGURE_LAYERS} layers"
            )
        result: dict[str, Any] = {}
        for layer in candidate.layers:
            references = self._source_references(layer.source)
            resolved_references: list[dict[str, Any]] = []
            for reference in references:
                source_type = str(
                    reference.get("type", reference.get("kind", ""))
                ).casefold()
                if source_type == "artifact":
                    artifact, path = self.artifact_path(
                        workspace_id,
                        str(reference["run_id"]),
                        str(reference["artifact_id"]),
                    )
                    artifact_mime = artifact.mime_type.casefold()
                    if artifact.kind != "image" or artifact_mime not in {
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                    }:
                        raise ValueError(
                            "Figure artifact sources must be completed PNG, JPEG, or "
                            "WebP raster images"
                        )
                    validate_raster_image(path, artifact_mime)
                    digest = sha256_file(path)
                    supplied_fingerprint = str(reference.get("fingerprint", "")).strip()
                    if supplied_fingerprint and supplied_fingerprint != digest:
                        raise RuntimeError("Figure artifact source has changed")
                    resolved_references.append(
                        {
                            "type": "artifact",
                            "run_id": str(reference["run_id"]),
                            "artifact_id": artifact.id,
                            "sha256": digest,
                            "size": path.stat().st_size,
                            "modified_ns": path.stat().st_mtime_ns,
                            "time_iso": reference.get("time_iso"),
                        }
                    )
                    continue
                source, _path = self.figure_preview_path(
                    workspace_id, str(reference["preview_id"])
                )
                supplied_fingerprint = str(reference.get("fingerprint", "")).strip()
                if (
                    supplied_fingerprint
                    and supplied_fingerprint != source["fingerprint"]
                ):
                    raise RuntimeError("Figure preview source has changed")
                resolved_references.append(
                    {
                        "type": "preview",
                        "preview_id": source["preview_id"],
                        "fingerprint": source["fingerprint"],
                        "size": int(_path.stat().st_size),
                        "time_iso": reference.get("time_iso"),
                    }
                )
            encoded = json.dumps(
                resolved_references,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
            result[layer.id] = {
                "fingerprint": sha256(encoded).hexdigest(),
                "references": resolved_references,
            }
        return result

    def create_figure_export(
        self,
        workspace_id: str,
        figure: RadioFigureDraft,
        preflight: RadioFigurePreflight,
        manifest: dict[str, Any],
        stream: BinaryIO,
        *,
        mime_type: str,
    ) -> RadioFigureExport:
        """Persist an immutable, validated browser-rendered figure export."""

        if (
            figure.workspace_id != workspace_id
            or preflight.workspace_id != workspace_id
        ):
            raise ValueError("Figure export workspace id does not match the request")
        if preflight.status != "ready":
            raise RuntimeError("Blocked figure preflight cannot be exported")
        if not isinstance(manifest, dict):
            raise TypeError("figure manifest must be a JSON object")
        self._reject_unsafe_source_metadata(manifest)
        if "frame_times" in manifest:
            submitted_frame_times = manifest["frame_times"]
            if not isinstance(submitted_frame_times, list):
                raise TypeError("Export frame_times must be a JSON array")
            if [str(item) for item in submitted_frame_times] != list(
                preflight.sample_times_iso
            ):
                raise ValueError("Export frame_times do not match the preflight")
        suffix = canonical_suffix(mime_type)
        timeline_mode = figure.timeline.mode
        if timeline_mode == "still" and suffix != ".png":
            raise ValueError("Still figures must be exported as PNG")
        if timeline_mode == "sequence" and suffix == ".png":
            raise ValueError("Figure animations must be exported as MP4 or WebM")
        if timeline_mode == "sequence" and suffix != (
            f".{figure.timeline.animation_format}"
        ):
            raise ValueError(
                "Uploaded animation MIME type does not match animation_format"
            )

        scale = float(figure.canvas.get("export_scale", 1.0))
        expected_width = round(figure.canvas["width"] * scale)
        expected_height = round(figure.canvas["height"] * scale)
        width = expected_width
        height = expected_height
        width, height = validate_dimensions(width, height)
        expected_frames = len(preflight.sample_times_iso)
        if not expected_frames:
            raise ValueError("Figure preflight contains no export samples")
        frame_count = expected_frames
        if timeline_mode == "still":
            if frame_count != 1:
                raise ValueError("Still figure preflight must contain one sample")
        elif not 1 <= frame_count <= MAX_ANIMATION_FRAMES:
            raise ValueError(
                f"Figure animation may contain at most {MAX_ANIMATION_FRAMES} frames"
            )
        expected_duration_s = (
            0.0
            if timeline_mode == "still"
            else frame_count / figure.timeline.playback_fps
        )
        duration_s = expected_duration_s

        export_id = uuid.uuid4().hex
        root = self.figure_studio_dir(workspace_id)
        final_dir = root / "exports" / export_id
        staging = root / "exports" / f".{export_id}.{uuid.uuid4().hex}.tmp"
        if suffix == ".png":
            generated_at = datetime.now(timezone.utc)
            output_name = build_scientific_image_filename(
                sequence=1,
                start_time=(
                    preflight.sample_times_iso[0]
                    if preflight.sample_times_iso
                    else None
                ),
                end_time=(
                    preflight.sample_times_iso[-1]
                    if len(preflight.sample_times_iso) > 1
                    else None
                ),
                instrument="radio",
                product="composed_figure",
                generated_at=generated_at,
            )
        else:
            output_name = f"figure{suffix}"
        with self._lock:
            self._ensure_figure_dirs(root)
            staging.mkdir()
            try:
                output_path = staging / output_name
                size = copy_limited_stream(
                    stream, output_path, limit=max_bytes_for_mime(mime_type)
                )
                validate_media_magic(output_path, mime_type)
                if suffix == ".png":
                    actual_width, actual_height = validate_png(output_path)
                    if (actual_width, actual_height) != (width, height):
                        raise ValueError(
                            "PNG dimensions do not match the export manifest"
                        )
                digest = sha256_file(output_path)
                thumbnail_name = "thumbnail.png"
                create_thumbnail(
                    output_path, staging / thumbnail_name, mime_type=mime_type
                )
                exported = RadioFigureExport(
                    figure_schema_version=FIGURE_SCHEMA_VERSION,
                    id=export_id,
                    workspace_id=workspace_id,
                    mime_type=mime_type,
                    output_path=output_name,
                    thumbnail_path=thumbnail_name,
                    preflight_revision=preflight.preflight_revision,
                    sha256=digest,
                    size=size,
                    width=width,
                    height=height,
                    frame_count=frame_count,
                    duration_s=duration_s,
                    mode=timeline_mode,
                    created_at=utc_now(),
                )
                preflight_payload = preflight.to_dict()
                layer_reports = {
                    str(item.get("layer_id", "")): dict(item)
                    for item in preflight.layers
                    if isinstance(item, dict)
                }
                layer_decisions: list[dict[str, Any]] = []
                for layer in figure.layers:
                    report = layer_reports.get(layer.id, {})
                    layer_decisions.append(
                        {
                            "layer_id": layer.id,
                            "title": layer.title,
                            "visible": layer.visible,
                            "z_index": layer.z_index,
                            "source": dict(layer.source),
                            "temporal_binding": layer.temporal_binding.to_dict(),
                            "fallback_policy": layer.temporal_binding.fallback,
                            "source_fingerprint": preflight.source_fingerprints.get(
                                layer.id
                            ),
                            "matches": list(report.get("matches", [])),
                            "coverage_segments": list(
                                report.get("coverage_segments", [])
                            ),
                            "coverage_gaps": list(report.get("coverage_gaps", [])),
                        }
                    )
                sample_times = list(preflight.sample_times_iso)
                applied_decisions = {
                    key: [dict(item) for item in value if isinstance(item, dict)]
                    for key, value in {
                        "timeline": figure.metadata.get("timeline_decisions", []),
                        "layers": figure.metadata.get("layer_decisions", []),
                    }.items()
                    if isinstance(value, list)
                }
                detailed_manifest = {
                    "figure_schema_version": FIGURE_SCHEMA_VERSION,
                    "export": exported.to_dict(),
                    "figure_mode": figure.mode,
                    "timeline": figure.timeline.to_dict(),
                    "preflight_revision": preflight.preflight_revision,
                    "sample_times_iso": sample_times,
                    "frame_times": sample_times,
                    "width": width,
                    "height": height,
                    "frame_count": frame_count,
                    "duration_s": duration_s,
                    "mime_type": mime_type,
                    "sha256": digest,
                    "size": size,
                    "layer_decisions": layer_decisions,
                    "applied_decisions": applied_decisions,
                    "preflight_layer_matches": list(preflight.layers),
                    "preflight_warnings": list(preflight.warnings),
                    "preflight_issues": list(preflight.issues),
                    "source_fingerprints": dict(preflight.source_fingerprints),
                    "common_valid_intervals": list(preflight.common_valid_intervals),
                    "common_valid_time": preflight.common_valid_time,
                    "longest_common_interval": preflight.longest_common_interval,
                    "missing_intervals": list(preflight.missing_intervals),
                    "preflight": preflight_payload,
                }
                self._atomic_json(staging / "figure.json", figure.to_dict())
                self._atomic_json(staging / "preflight.json", preflight.to_dict())
                self._atomic_json(staging / "manifest.json", detailed_manifest)
                if final_dir.exists():
                    raise FileExistsError(f"Figure export already exists: {export_id}")
                staging.rename(final_dir)
                return exported
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

    def list_figure_exports(self, workspace_id: str) -> list[RadioFigureExport]:
        root = self.figure_studio_dir(workspace_id)
        exports_dir = root / "exports"
        exports: list[RadioFigureExport] = []
        with self._lock:
            if exports_dir.is_symlink():
                raise PermissionError("Figure export root may not be a symlink")
            if not exports_dir.is_dir():
                return []
            for folder in exports_dir.iterdir():
                if folder.is_symlink() or not folder.is_dir():
                    continue
                manifest_path = folder / "manifest.json"
                if not manifest_path.is_file() or manifest_path.is_symlink():
                    continue
                try:
                    manifest = self._read_json(manifest_path)
                    exported = RadioFigureExport.from_dict(manifest.get("export", {}))
                    if (
                        exported.id != folder.name
                        or exported.workspace_id != workspace_id
                    ):
                        continue
                    exports.append(exported)
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return sorted(exports, key=lambda item: item.created_at, reverse=True)

    def load_figure_export(
        self, workspace_id: str, export_id: str
    ) -> tuple[RadioFigureExport, dict[str, Any]]:
        export_id = self._safe_identifier(export_id, label="export id")
        studio_root = self.figure_studio_dir(workspace_id)
        root = studio_root / "exports"
        if root.is_symlink():
            raise PermissionError("Figure export root may not be a symlink")
        if not root.is_dir():
            raise KeyError(f"Unknown figure export: {export_id}")
        folder = root / export_id
        if folder.is_symlink() or folder.resolve(strict=False).parent != root.resolve(
            strict=False
        ):
            raise PermissionError("Figure export path escaped its workspace")
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise KeyError(f"Unknown figure export: {export_id}")
        with self._lock:
            manifest = self._read_json(manifest_path)
            exported = RadioFigureExport.from_dict(manifest.get("export", {}))
        if exported.id != export_id or exported.workspace_id != workspace_id:
            raise PermissionError("Figure export metadata does not match its folder")
        return exported, manifest

    def figure_export_file(
        self,
        workspace_id: str,
        export_id: str,
        *,
        thumbnail: bool = False,
    ) -> tuple[RadioFigureExport, Path]:
        exported, _manifest = self.load_figure_export(workspace_id, export_id)
        folder = self.figure_studio_dir(workspace_id) / "exports" / export_id
        relative_path = exported.thumbnail_path if thumbnail else exported.output_path
        candidate = folder / relative_path
        if candidate.is_symlink():
            raise PermissionError("Figure export files may not be symlinks")
        path = candidate.resolve(strict=True)
        if folder.resolve(strict=True) not in path.parents or not path.is_file():
            raise PermissionError("Figure export file escaped its export directory")
        if thumbnail:
            try:
                validate_media_magic(path, "image/png")
                validate_png(path)
            except (TypeError, ValueError) as exc:
                raise PermissionError("Figure thumbnail is invalid") from exc
        elif sha256_file(path) != exported.sha256:
            raise PermissionError("Figure export bytes do not match the manifest")
        return exported, path

    def delete_figure_export(self, workspace_id: str, export_id: str) -> None:
        exported, _manifest = self.load_figure_export(workspace_id, export_id)
        folder = self.figure_studio_dir(workspace_id) / "exports" / exported.id
        resolved = folder.resolve(strict=True)
        expected_parent = (self.figure_studio_dir(workspace_id) / "exports").resolve(
            strict=True
        )
        if folder.is_symlink() or resolved.parent != expected_parent:
            raise PermissionError("Refusing to delete an unsafe figure export")
        with self._lock:
            shutil.rmtree(resolved)

    def run_dir(self, workspace_id: str, run_id: str) -> Path:
        if not run_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in run_id
        ):
            raise ValueError(f"Invalid run id: {run_id!r}")
        base = self.workspace_dir(workspace_id) / "runs"
        path = (base / run_id).resolve(strict=False)
        if path.parent != base.resolve(strict=False):
            raise PermissionError("Run path escaped the workspace root")
        return path

    def create_run(self, manifest: RadioRunManifest) -> RadioRunManifest:
        self.load_workspace(manifest.workspace_id)
        path = self.run_dir(manifest.workspace_id, manifest.id)
        with self._lock:
            if path.exists():
                raise FileExistsError(f"Radio run already exists: {manifest.id}")
            (path / "artifacts").mkdir(parents=True)
            self._atomic_json(path / "request.json", manifest.request)
            self._atomic_json(path / "resolved_config.json", manifest.resolved_config)
            self._atomic_json(path / "run.json", manifest.to_dict())
            (path / "run.log").write_text("", encoding="utf-8")
        return manifest

    def create_runs_atomic(
        self, manifests: Iterable[RadioRunManifest]
    ) -> list[RadioRunManifest]:
        """Create a prepared batch without leaving partial run directories."""

        runs = list(manifests)
        if not runs:
            return []
        paths: list[Path] = []
        with self._lock:
            for manifest in runs:
                self.load_workspace(manifest.workspace_id)
                path = self.run_dir(manifest.workspace_id, manifest.id)
                if path in paths:
                    raise ValueError(f"Duplicate radio run id: {manifest.id}")
                if path.exists():
                    raise FileExistsError(f"Radio run already exists: {manifest.id}")
                paths.append(path)
            try:
                for manifest in runs:
                    self.create_run(manifest)
            except BaseException as exc:
                rollback_errors: list[OSError] = []
                for path in reversed(paths):
                    try:
                        if path.is_symlink() or path.is_file():
                            path.unlink(missing_ok=True)
                        elif path.is_dir():
                            shutil.rmtree(path)
                    except OSError as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    raise RuntimeError(
                        "Radio batch creation failed and rollback was incomplete"
                    ) from exc
                raise
        return runs

    def load_run(self, workspace_id: str, run_id: str) -> RadioRunManifest:
        path = self.run_dir(workspace_id, run_id) / "run.json"
        with self._lock:
            if not path.is_file():
                raise KeyError(f"Unknown radio run: {run_id}")
            return RadioRunManifest.from_dict(self._read_json(path))

    def list_runs(self, workspace_id: str) -> list[RadioRunManifest]:
        workspace = self.load_workspace(workspace_id)
        del workspace
        runs_dir = self.workspace_dir(workspace_id) / "runs"
        runs: list[RadioRunManifest] = []
        with self._lock:
            if not runs_dir.is_dir():
                return []
            for child in runs_dir.iterdir():
                if child.is_symlink() or not child.is_dir():
                    continue
                path = child / "run.json"
                if not path.is_file():
                    continue
                try:
                    runs.append(RadioRunManifest.from_dict(self._read_json(path)))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return sorted(runs, key=lambda item: item.created_at, reverse=True)

    def save_run(self, manifest: RadioRunManifest) -> RadioRunManifest:
        run_dir = self.run_dir(manifest.workspace_id, manifest.id)
        path = run_dir / "run.json"
        with self._lock:
            if not path.is_file():
                raise KeyError(f"Unknown radio run: {manifest.id}")
            self._atomic_json(
                run_dir / "resolved_config.json", manifest.resolved_config
            )
            self._atomic_json(path, manifest.to_dict())
        return manifest

    def append_log(self, workspace_id: str, run_id: str, line: str) -> None:
        path = self.run_dir(workspace_id, run_id) / "run.log"
        with self._lock, path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(str(line).rstrip("\r\n") + "\n")

    def read_log(
        self, workspace_id: str, run_id: str, *, offset: int = 0
    ) -> tuple[list[str], int]:
        if offset < 0:
            raise ValueError("offset must be zero or greater")
        path = self.run_dir(workspace_id, run_id) / "run.log"
        if not path.is_file():
            raise KeyError(f"Unknown radio run: {run_id}")
        with self._lock:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[offset:], len(lines)

    def artifact_path(
        self, workspace_id: str, run_id: str, artifact_id: str
    ) -> tuple[RadioArtifact, Path]:
        manifest = self.load_run(workspace_id, run_id)
        try:
            artifact = next(
                item for item in manifest.artifacts if item.id == artifact_id
            )
        except StopIteration as exc:
            raise KeyError(f"Unknown radio artifact: {artifact_id}") from exc
        run_root = self.run_dir(workspace_id, run_id)
        declared_root = run_root / "artifacts"
        if declared_root.is_symlink():
            raise PermissionError("Artifact storage may not be a symlink")
        root = declared_root.resolve(strict=True)
        if root.parent != run_root.resolve(strict=True):
            raise PermissionError("Artifact storage escaped its run directory")
        candidate = declared_root / artifact.relative_path
        path = candidate.resolve(strict=True)
        if not _inside(path, (root,)) or not path.is_file():
            raise PermissionError("Artifact path escaped its run directory")
        return artifact, path

    def recover_interrupted_runs(self) -> list[RadioRunManifest]:
        recovered: list[RadioRunManifest] = []
        for workspace in self.list_workspaces():
            for manifest in self.list_runs(workspace.id):
                if manifest.status not in {"queued", "running"}:
                    continue
                manifest.status = "interrupted"
                manifest.progress = 1.0
                manifest.finished_at = utc_now()
                manifest.error = "The local service stopped before this run completed."
                self.save_run(manifest)
                recovered.append(manifest)
        return recovered

    @staticmethod
    def _safe_identifier(value: str, *, label: str) -> str:
        normalized = str(value).strip()
        if not normalized or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in normalized
        ):
            raise ValueError(f"Invalid {label}: {value!r}")
        if len(normalized) > 128:
            raise ValueError(f"Invalid {label}: {value!r}")
        return normalized

    @staticmethod
    def _ensure_figure_dirs(root: Path) -> None:
        expected_parent = root.parent.resolve(strict=True)
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
        if (
            resolved_root.parent != expected_parent
            or resolved_root.name != "figure_studio"
        ):
            raise PermissionError("Figure Studio storage escaped the workspace")
        for name in ("snapshots", "exports", "sources"):
            child = resolved_root / name
            child.mkdir(exist_ok=True)
            resolved_child = child.resolve(strict=True)
            if (
                not resolved_child.is_dir()
                or resolved_child.parent != resolved_root
                or resolved_child.name != name
            ):
                raise PermissionError(f"Figure Studio {name} escaped its storage root")

    @classmethod
    def _reject_unsafe_source_metadata(cls, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized_key = str(key).strip().casefold().replace("-", "_")
                if (
                    normalized_key
                    in {
                        "path",
                        "url",
                        "uri",
                        "src",
                        "href",
                        "data",
                        "data_url",
                        "file_path",
                    }
                    or "path" in normalized_key
                    or normalized_key.endswith(("_url", "_uri"))
                ):
                    raise ValueError(f"Figure preview metadata may not contain {key!r}")
                cls._reject_unsafe_source_metadata(item)
        elif isinstance(value, list):
            for item in value:
                cls._reject_unsafe_source_metadata(item)

    @staticmethod
    def _source_references(source: dict[str, Any]) -> list[dict[str, Any]]:
        source_type = str(source.get("type", source.get("kind", ""))).casefold()
        if source_type == "series":
            return [dict(item) for item in source.get("frames", [])]
        return [dict(source)]

    def _read_workspace(self, path: Path) -> RadioWorkspace:
        workspace = RadioWorkspace.from_dict(self._read_json(path))
        if workspace.ui_layout_version < UI_LAYOUT_VERSION:
            workspace.ui_layout_version = UI_LAYOUT_VERSION
            workspace.enabled_modules = [
                module.id for module in MODULES if module.always_available
            ]
            workspace.collapsed_modules = list(self._module_ids)
            workspace.pinned_modules = []
            workspace.updated_at = utc_now()
            self._atomic_json(path, workspace.to_dict())
        self._validate_workspace_modules(workspace)
        return workspace

    def _validate_workspace_modules(self, workspace: RadioWorkspace) -> None:
        allowed = set(self._module_ids)
        for module in MODULES:
            if module.always_available and module.id not in workspace.enabled_modules:
                workspace.enabled_modules.append(module.id)
        for field_name in (
            "enabled_modules",
            "module_order",
            "collapsed_modules",
            "pinned_modules",
        ):
            values = getattr(workspace, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} contains duplicate module ids")
            unknown = set(values) - allowed
            if unknown:
                raise ValueError(
                    f"{field_name} contains unknown modules: {', '.join(sorted(unknown))}"
                )
        workspace.module_order = _deduplicate(
            [*workspace.module_order, *self._module_ids]
        )

    def _validate_shared_paths(self, shared_paths: dict[str, str]) -> None:
        for value in shared_paths.values():
            if value:
                self.browser.resolve(value, must_exist=False)

    def _workspace_path(self, output_root: str | Path, workspace_id: str) -> Path:
        selected_output = Path(output_root).expanduser().resolve(strict=False)
        base = (selected_output / "radio_workbench").resolve(strict=False)
        if base.parent != selected_output or base.name != "radio_workbench":
            raise PermissionError("Radio workbench storage escaped the output root")
        path = (base / workspace_id).resolve(strict=False)
        if path.parent != base or path.name != workspace_id:
            raise PermissionError("Workspace path escaped the selected output root")
        return path

    def _validated_primary_root(self) -> Path:
        resolved = self.root.resolve(strict=False)
        if resolved.parent != self.output_root or resolved.name != "radio_workbench":
            raise PermissionError("Radio workbench storage escaped the output root")
        return resolved

    def _read_index(self) -> dict[str, str]:
        root = self._validated_primary_root()
        index_path = root / "workspace_index.json"
        if not index_path.is_file():
            return {}
        resolved_index = index_path.resolve(strict=True)
        if resolved_index.parent != root or resolved_index.name != index_path.name:
            raise PermissionError("Radio workspace index escaped the output root")
        payload = self._read_json(resolved_index)
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError("Unsupported radio workspace index schema version")
        raw = payload.get("workspaces", {})
        if not isinstance(raw, dict):
            raise TypeError("Radio workspace index must contain a workspaces object")
        result: dict[str, str] = {}
        for workspace_id, output_root in raw.items():
            if not isinstance(output_root, str):
                continue
            try:
                resolved = self._validated_index_output_root(
                    str(workspace_id), output_root
                )
            except (
                OSError,
                TypeError,
                ValueError,
                PermissionError,
                json.JSONDecodeError,
            ):
                continue
            result[str(workspace_id)] = str(resolved)
        return result

    def _write_index(self, index: dict[str, str]) -> None:
        root = self._validated_primary_root()
        self._atomic_json(
            root / "workspace_index.json",
            {"schema_version": SCHEMA_VERSION, "workspaces": dict(index)},
        )

    def _refresh_protected_roots_locked(self) -> None:
        protected = [self.output_root]
        for output_root in self._read_index().values():
            resolved = Path(output_root).expanduser().resolve(strict=False)
            if resolved not in protected:
                protected.append(resolved)
        self._protected_roots = tuple(protected)
        self._browser = SafePathBrowser([*self._user_roots, *protected])

    def _validated_index_output_root(self, workspace_id: str, output_root: str) -> Path:
        if not workspace_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for char in workspace_id
        ):
            raise ValueError(f"Invalid workspace id: {workspace_id!r}")
        candidate = Path(output_root).expanduser()
        if not candidate.is_absolute():
            raise ValueError("Indexed workspace output roots must be absolute")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"Indexed workspace output root is not a directory: {resolved}"
            )

        workbench = resolved / "radio_workbench"
        workspace_dir = workbench / workspace_id
        config = workspace_dir / "workspace.json"
        if workbench.is_symlink() or workspace_dir.is_symlink() or config.is_symlink():
            raise PermissionError("Indexed workspace paths may not be symlinks")
        resolved_workbench = workbench.resolve(strict=True)
        resolved_workspace = workspace_dir.resolve(strict=True)
        if (
            not resolved_workbench.is_dir()
            or not resolved_workspace.is_dir()
            or resolved_workbench.parent != resolved
            or resolved_workbench.name != "radio_workbench"
            or resolved_workspace.parent != resolved_workbench
            or resolved_workspace.name != workspace_id
            or not config.is_file()
        ):
            raise PermissionError("Indexed workspace path is invalid")

        workspace = RadioWorkspace.from_dict(self._read_json(config))
        if workspace.id != workspace_id:
            raise ValueError("Indexed workspace id does not match workspace.json")
        declared_output = Path(workspace.output_root).expanduser()
        if not declared_output.is_absolute():
            raise ValueError("Workspace output_root must be absolute")
        if declared_output.resolve(strict=True) != resolved:
            raise ValueError("Indexed output root does not match workspace.json")
        return resolved

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a JSON object in {path}")
        return payload

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp.write_text(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["RadioWorkspaceStore", "SafePathBrowser", "utc_now"]
