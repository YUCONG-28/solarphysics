"""Persistent queue and safe worker adapters for modular radio actions."""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from solar_apps.frontends.workbench.runner import (
    default_python_executable,
    prepend_conda_dll_paths_to_env,
)
from solar_apps.platform.layout import RuntimeLayout

from .catalog import EVENT_PRESETS, get_action, get_module
from .contracts import SCHEMA_VERSION, RadioArtifact, RadioRunManifest
from .store import RadioWorkspaceStore, utc_now

_SHELL_TOKENS = frozenset({"&&", "||", "|", ";", ">", ">>", "<", "`"})
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled", "interrupted"})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _action_layer(
    value: dict[str, Any], module_id: str, action_id: str
) -> dict[str, Any]:
    """Select generic, module, and action values from one configuration layer."""

    result = {
        key: item
        for key, item in value.items()
        if key not in {"modules", "actions", module_id, action_id}
    }
    module_value = value.get(module_id, {})
    modules = value.get("modules", {})
    if isinstance(modules, dict) and isinstance(modules.get(module_id), dict):
        module_value = _deep_merge(
            module_value if isinstance(module_value, dict) else {},
            modules[module_id],
        )
    if isinstance(module_value, dict):
        result = _deep_merge(
            result,
            {
                key: item
                for key, item in module_value.items()
                if key not in {"actions", action_id}
            },
        )
        module_actions = module_value.get("actions", {})
        if isinstance(module_actions, dict) and isinstance(
            module_actions.get(action_id), dict
        ):
            result = _deep_merge(result, module_actions[action_id])
        if isinstance(module_value.get(action_id), dict):
            result = _deep_merge(result, module_value[action_id])
    actions = value.get("actions", {})
    if isinstance(actions, dict) and isinstance(actions.get(action_id), dict):
        result = _deep_merge(result, actions[action_id])
    if isinstance(value.get(action_id), dict):
        result = _deep_merge(result, value[action_id])
    return result


def _normalize_arguments(value: Any) -> list[str]:
    if value in (None, []):
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError("arguments must be a JSON array of argument tokens")
    arguments: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            raise TypeError("every argument token must be a string")
        if "\n" in raw or "\r" in raw:
            raise ValueError("argument tokens may not contain newlines")
        token = raw.strip()
        if not token:
            continue
        if token in _SHELL_TOKENS:
            raise ValueError(f"Shell operator token is not allowed: {token}")
        arguments.append(token)
    return arguments


class RadioRunManager:
    """Schedule explicit radio actions and persist every state transition."""

    def __init__(
        self,
        store: RadioWorkspaceStore,
        *,
        repo_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        popen_factory=None,
        global_concurrency: int = 4,
    ) -> None:
        if not 1 <= int(global_concurrency) <= 4:
            raise ValueError("global_concurrency must be between 1 and 4")
        self.store = store
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else RuntimeLayout.discover().repo_root
        )
        selected = Path(default_python_executable()).resolve(strict=False)
        requested = (
            Path(python_executable or selected).expanduser().resolve(strict=False)
        )
        if os.path.normcase(str(requested)) != os.path.normcase(str(selected)):
            raise ValueError(
                "Radio workspace jobs must use the interpreter selected by the public Apps launcher."
            )
        self.python_executable = str(selected)
        self.popen_factory = popen_factory or subprocess.Popen
        self.global_concurrency = int(global_concurrency)
        self._pending: deque[tuple[str, str]] = deque()
        self._active_by_workspace: dict[str, int] = {}
        self._active_total = 0
        self._processes: dict[tuple[str, str], Any] = {}
        self._closed = False
        self._condition = threading.Condition(threading.RLock())
        self.store.recover_interrupted_runs()
        self._scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="radio-workspace-scheduler",
            daemon=True,
        )
        self._scheduler.start()

    def preview(
        self,
        workspace_id: str,
        module_id: str,
        action_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action = get_action(module_id, action_id)
        resolved = self.resolve_request(
            workspace_id, module_id, action_id, payload or {}
        )
        native = action.preview_adapter is not None
        preview: dict[str, Any] = {}
        if native:
            preview = self._build_native_preview(
                workspace_id,
                action.preview_adapter,
                resolved["config"],
            )
        ready = preview.get("status") == "ready" if native else False
        result = {
            "available": ready if native else False,
            "validation_only": not native,
            "adapter": action.preview_adapter,
            "message": (
                preview.get("reason", "The same-page preview is ready.")
                if native
                else "No native preview adapter is available; the request was validated only."
            ),
            "module_id": module_id,
            "action_id": action_id,
            "resolved_config": resolved["config"],
            "command": resolved["command"] if action.runnable else None,
            "provenance": resolved["provenance"],
        }
        result.update(preview)
        return result

    def resolve_request(
        self,
        workspace_id: str,
        module_id: str,
        action_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return self._resolve_request(
            workspace_id,
            module_id,
            action_id,
            payload,
            deferred_fields=frozenset(),
        )

    def _resolve_request(
        self,
        workspace_id: str,
        module_id: str,
        action_id: str,
        payload: dict[str, Any],
        *,
        deferred_fields: frozenset[str],
    ) -> dict[str, Any]:
        workspace = self.store.load_workspace(workspace_id)
        action = get_action(module_id, action_id)
        get_module(module_id)
        if not isinstance(payload, dict):
            raise TypeError("action payload must be a JSON object")
        form = payload.get("form") or {}
        advanced = payload.get("advanced_config") or {}
        if not isinstance(form, dict):
            raise TypeError("form must be a JSON object")
        if not isinstance(advanced, dict):
            raise TypeError("advanced_config must be a JSON object")
        explicit_arguments = _normalize_arguments(payload.get("arguments"))
        normalized_sources, source_bindings = self._resolve_input_sources(
            workspace_id, payload.get("input_sources") or []
        )
        binding_schema = {
            str(field["name"]): field
            for field in action.input_schema
            if field.get("path")
        }
        for source in normalized_sources:
            field = str(source.get("field", "")).strip()
            if not field:
                raise ValueError("Every input source must bind to a path field")
            if field not in binding_schema:
                raise ValueError(
                    f"Input source field {field!r} is not available for "
                    f"{module_id}/{action_id}"
                )
            field_types = set(binding_schema[field].get("artifact_types") or [])
            artifact_type = source.get("artifact_type")
            if (
                artifact_type
                and field_types
                and "*" not in field_types
                and artifact_type not in field_types
            ):
                raise ValueError(
                    f"Artifact type {artifact_type!r} is not accepted by input "
                    f"field {field!r}"
                )
        accepted_types = set(action.accepts_artifacts)
        for source in normalized_sources:
            artifact_type = source.get("artifact_type")
            if (
                artifact_type
                and "*" not in accepted_types
                and artifact_type not in accepted_types
            ):
                raise ValueError(
                    f"Artifact type {artifact_type!r} is not accepted by "
                    f"{module_id}/{action_id}"
                )
        defaults = dict(action.default_config)
        for field in action.input_schema:
            if "default" in field:
                defaults.setdefault(str(field["name"]), field["default"])
        selected_event = EVENT_PRESETS.get(str(workspace.event_preset.get("id", "")))
        event_values = dict(selected_event.get("config", {})) if selected_event else {}
        event_values = _deep_merge(
            event_values,
            {
                key: value
                for key, value in workspace.event_preset.items()
                if key not in {"id", "title"}
            },
        )
        event_preset = _action_layer(event_values, module_id, action_id)
        shared_paths = {
            key: value
            for key, value in workspace.shared_paths.items()
            if value not in (None, "")
        }
        workspace_advanced = _action_layer(
            workspace.advanced_config, module_id, action_id
        )
        request_advanced = _action_layer(advanced, module_id, action_id)
        advanced_layer = _deep_merge(workspace_advanced, request_advanced)
        # Artifact bindings are an explicit action-form input choice.  Keep the
        # rest of the form values, but ensure a stale/manual value for the same
        # field cannot make provenance claim one source while the worker reads
        # another.
        form_layer = _deep_merge(form, source_bindings)
        config: dict[str, Any] = {}
        for layer in (
            defaults,
            event_preset,
            shared_paths,
            advanced_layer,
            form_layer,
        ):
            config = _deep_merge(config, layer)
        if deferred_fields:
            config = self._without_deferred_fields(
                action, config, deferred_fields=deferred_fields
            )
        if any(str(field.get("name")) == "config" for field in action.input_schema):
            self._validate_event_config_name(config.get("config"))
        self._validate_required_fields(
            action.input_schema,
            config,
            satisfied_fields=deferred_fields,
        )
        self._validate_pattern_values(
            config,
            source=f"{module_id}/{action_id} action configuration",
        )
        self._validate_config_paths(action.input_schema, config)
        self._validate_structured_config_paths(action, config)
        artifact_dir = (
            self.store.run_dir(workspace_id, payload.get("_run_id", "preview"))
            / "artifacts"
        )
        command = self._build_command(
            action,
            config,
            explicit_arguments,
            artifact_dir=artifact_dir,
        )
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "configuration_precedence": [
                "package_defaults",
                "event_preset",
                "workspace_shared_paths",
                "advanced_config",
                "action_form",
            ],
            "layers": {
                "package_defaults": defaults,
                "event_preset": event_preset,
                "workspace_shared_paths": shared_paths,
                "workspace_advanced_config": workspace_advanced,
                "request_advanced_config": request_advanced,
                "action_form": form,
                "artifact_bindings": source_bindings,
            },
            "input_sources": normalized_sources,
            "declared_output_types": list(action.produces_artifacts),
            "dependencies_auto_run": False,
        }
        return {
            "config": config,
            "command": command,
            "input_sources": normalized_sources,
            "provenance": provenance,
            "arguments": explicit_arguments,
        }

    def start(
        self,
        workspace_id: str,
        module_id: str,
        action_id: str,
        payload: dict[str, Any] | None = None,
        *,
        depends_on: tuple[str, ...] = (),
        batch_id: str | None = None,
        batch_order: int | None = None,
    ) -> RadioRunManifest:
        with self._condition:
            if self._closed:
                raise RuntimeError("Radio run manager is closed")
        action = get_action(module_id, action_id)
        if not action.runnable:
            raise ValueError(
                f"Action {module_id}/{action_id} is interactive and has no worker adapter"
            )
        request = dict(payload or {})
        run_id = uuid.uuid4().hex
        request["_run_id"] = run_id
        resolved = self.resolve_request(workspace_id, module_id, action_id, request)
        self._validate_run_required_fields(action, resolved["config"])
        request.pop("_run_id", None)
        provenance = dict(resolved["provenance"])
        if batch_id is not None:
            provenance["batch_id"] = batch_id
            provenance["batch_order"] = batch_order
            provenance["depends_on_run_ids"] = list(depends_on)
        manifest = RadioRunManifest(
            schema_version=SCHEMA_VERSION,
            id=run_id,
            workspace_id=workspace_id,
            module_id=module_id,
            action_id=action_id,
            status="queued",
            command=resolved["command"],
            cwd=str(self.repo_root),
            request=request,
            resolved_config=resolved["config"],
            input_sources=resolved["input_sources"],
            provenance=provenance,
            created_at=utc_now(),
        )
        self.store.create_run(manifest)
        with self._condition:
            if self._closed:
                manifest.status = "canceled"
                manifest.progress = 1.0
                manifest.finished_at = utc_now()
                manifest.error = "The run manager closed before this run was queued."
                self.store.save_run(manifest)
                raise RuntimeError("Radio run manager is closed")
            self._pending.append((workspace_id, run_id))
            self._condition.notify_all()
        return manifest

    def start_batch(
        self, workspace_id: str, requests: Iterable[dict[str, Any]]
    ) -> list[RadioRunManifest]:
        with self._condition:
            if self._closed:
                raise RuntimeError("Radio run manager is closed")
        items = list(requests)
        batch_id = uuid.uuid4().hex
        prepared_items: list[tuple[str, str, Any, dict[str, Any]]] = []
        for item in items:
            if not isinstance(item, dict):
                raise TypeError("Every batch action must be a JSON object")
            module_id = str(item.get("module_id", "")).strip()
            action_id = str(item.get("action_id", "")).strip()
            if not module_id or not action_id:
                raise ValueError("Every batch action requires module_id and action_id")
            action = get_action(module_id, action_id)
            if not action.runnable:
                raise ValueError(
                    f"Action {module_id}/{action_id} is interactive and has no worker adapter"
                )
            payload = {
                key: value
                for key, value in item.items()
                if key not in {"module_id", "action_id"}
            }
            prepared_items.append((module_id, action_id, action, payload))

        run_ids = [uuid.uuid4().hex for _item in prepared_items]
        runs: list[RadioRunManifest] = []
        for index, (module_id, action_id, action, payload) in enumerate(prepared_items):
            ordinary_sources, deferred_sources = self._split_batch_input_sources(
                payload.get("input_sources") or []
            )
            planned_sources = self._normalize_batch_artifact_sources(
                index=index,
                action=action,
                prepared_items=prepared_items,
                run_ids=run_ids,
                ordinary_sources=ordinary_sources,
                deferred_sources=deferred_sources,
            )
            deferred_fields = frozenset(
                str(source["field"]) for source in planned_sources
            )
            resolution_payload = dict(payload)
            resolution_payload["input_sources"] = ordinary_sources
            run_id = run_ids[index]
            resolution_payload["_run_id"] = run_id
            resolved = self._resolve_request(
                workspace_id,
                module_id,
                action_id,
                resolution_payload,
                deferred_fields=deferred_fields,
            )
            self._validate_run_required_fields(
                action,
                resolved["config"],
                satisfied_fields=deferred_fields,
            )
            provenance = dict(resolved["provenance"])
            provenance["batch_id"] = batch_id
            provenance["batch_order"] = index
            provenance["depends_on_run_ids"] = run_ids[index - 1 : index]
            provenance["planned_input_sources"] = planned_sources
            runs.append(
                RadioRunManifest(
                    schema_version=SCHEMA_VERSION,
                    id=run_id,
                    workspace_id=workspace_id,
                    module_id=module_id,
                    action_id=action_id,
                    status="queued",
                    command=resolved["command"],
                    cwd=str(self.repo_root),
                    request=payload,
                    resolved_config=resolved["config"],
                    input_sources=[*resolved["input_sources"], *planned_sources],
                    provenance=provenance,
                    created_at=utc_now(),
                )
            )
        with self._condition:
            if self._closed:
                raise RuntimeError("Radio run manager is closed")
            self.store.create_runs_atomic(runs)
            self._pending.extend((workspace_id, run.id) for run in runs)
            self._condition.notify_all()
        return runs

    def status(self, workspace_id: str, run_id: str) -> RadioRunManifest:
        return self.store.load_run(workspace_id, run_id)

    def list_runs(self, workspace_id: str) -> list[RadioRunManifest]:
        return self.store.list_runs(workspace_id)

    def wait(
        self, workspace_id: str, run_id: str, *, timeout: float = 30.0
    ) -> RadioRunManifest:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            manifest = self.status(workspace_id, run_id)
            if manifest.status in _TERMINAL_STATUSES:
                return manifest
            time.sleep(0.01)
        raise TimeoutError(
            f"Radio run did not finish within {timeout} seconds: {run_id}"
        )

    def cancel(self, workspace_id: str, run_id: str) -> RadioRunManifest:
        key = (workspace_id, run_id)
        with self._condition:
            manifest = self.store.load_run(workspace_id, run_id)
            if manifest.status in _TERMINAL_STATUSES:
                return manifest
            if manifest.status == "queued":
                try:
                    self._pending.remove(key)
                except ValueError:
                    pass
            process = self._processes.get(key)
            manifest.status = "canceled"
            manifest.progress = 1.0
            manifest.finished_at = utc_now()
            manifest.error = "Canceled by the user."
            self.store.save_run(manifest)
            self._condition.notify_all()
        if process is not None:
            self._terminate_process_tree(process)
        return self.store.load_run(workspace_id, run_id)

    def close(self, *, cancel_running: bool = False) -> None:
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            running = list(self._processes)
            self._condition.notify_all()
        for workspace_id, run_id in pending:
            self.cancel(workspace_id, run_id)
        if cancel_running:
            for workspace_id, run_id in running:
                self.cancel(workspace_id, run_id)
        self._scheduler.join(timeout=2.0)

    def _scheduler_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or (
                        bool(self._pending)
                        and self._active_total < self.global_concurrency
                    )
                )
                if self._closed:
                    return
                selected_index = self._next_eligible_index()
                if selected_index is None:
                    self._condition.wait(timeout=0.05)
                    continue
                workspace_id, run_id = self._pending[selected_index]
                del self._pending[selected_index]
                self._active_total += 1
                self._active_by_workspace[workspace_id] = (
                    self._active_by_workspace.get(workspace_id, 0) + 1
                )
            threading.Thread(
                target=self._execute,
                args=(workspace_id, run_id),
                name=f"radio-run-{run_id[:8]}",
                daemon=True,
            ).start()

    def _next_eligible_index(self) -> int | None:
        for index, (workspace_id, run_id) in enumerate(self._pending):
            try:
                concurrency = self.store.load_workspace(workspace_id).concurrency
                manifest = self.store.load_run(workspace_id, run_id)
            except (KeyError, OSError, ValueError):
                return index
            dependencies = manifest.provenance.get("depends_on_run_ids", [])
            dependency_states = [
                self.store.load_run(workspace_id, str(item)).status
                for item in dependencies
            ]
            if any(state not in _TERMINAL_STATUSES for state in dependency_states):
                continue
            if self._active_by_workspace.get(workspace_id, 0) < concurrency:
                return index
        return None

    def _execute(self, workspace_id: str, run_id: str) -> None:
        key = (workspace_id, run_id)
        try:
            manifest = self.store.load_run(workspace_id, run_id)
            if manifest.status == "canceled":
                return
            dependency_ids = manifest.provenance.get("depends_on_run_ids", [])
            failed_dependencies = [
                str(item)
                for item in dependency_ids
                if self.store.load_run(workspace_id, str(item)).status != "succeeded"
            ]
            if failed_dependencies:
                manifest.status = "canceled"
                manifest.progress = 1.0
                manifest.finished_at = utc_now()
                manifest.error = (
                    "A confirmed batch dependency did not succeed: "
                    + ", ".join(failed_dependencies)
                )
                self.store.save_run(manifest)
                return
            if manifest.provenance.get("planned_input_sources"):
                manifest = self._resolve_planned_batch_artifacts(manifest)
            manifest.status = "running"
            manifest.progress = 0.1
            manifest.started_at = utc_now()
            self.store.save_run(manifest)
            process = self.popen_factory(
                manifest.command,
                cwd=manifest.cwd,
                env=prepend_conda_dll_paths_to_env(
                    python_executable=self.python_executable
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            with self._condition:
                self._processes[key] = process
            if self.store.load_run(workspace_id, run_id).status == "canceled":
                self._terminate_process_tree(process)
            self._read_output(workspace_id, run_id, process)
            returncode = process.wait()
            manifest = self.store.load_run(workspace_id, run_id)
            manifest.returncode = returncode
            manifest.artifacts = self._index_artifacts(
                workspace_id,
                run_id,
                module_id=manifest.module_id,
                action_id=manifest.action_id,
            )
            if manifest.status != "canceled":
                manifest.status = "succeeded" if returncode == 0 else "failed"
                manifest.progress = 1.0
                if returncode != 0:
                    manifest.error = f"Worker exited with status {returncode}."
                manifest.finished_at = utc_now()
            self.store.save_run(manifest)
        except Exception as exc:
            try:
                self.store.append_log(
                    workspace_id, run_id, f"Worker failed to start: {exc}"
                )
                manifest = self.store.load_run(workspace_id, run_id)
                if manifest.status != "canceled":
                    manifest.status = "failed"
                    manifest.progress = 1.0
                    manifest.error = str(exc)
                    manifest.returncode = -1
                    manifest.finished_at = utc_now()
                    self.store.save_run(manifest)
            except Exception:
                pass
        finally:
            with self._condition:
                self._processes.pop(key, None)
                self._active_total = max(0, self._active_total - 1)
                self._active_by_workspace[workspace_id] = max(
                    0, self._active_by_workspace.get(workspace_id, 1) - 1
                )
                self._condition.notify_all()

    def _read_output(self, workspace_id: str, run_id: str, process: Any) -> None:
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            return
        while True:
            line = stdout.readline()
            if line:
                self.store.append_log(workspace_id, run_id, line)
                continue
            if process.poll() is not None:
                break
            time.sleep(0.02)
        close = getattr(stdout, "close", None)
        if callable(close):
            close()

    def _resolve_input_sources(
        self, workspace_id: str, value: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("input_sources must be a JSON array")
        sources: list[dict[str, Any]] = []
        bindings: dict[str, Any] = {}
        for raw in value:
            if not isinstance(raw, dict):
                raise TypeError("Every input source must be a JSON object")
            source_type = str(raw.get("type", "")).strip()
            field = str(raw.get("field", "")).strip()
            if source_type == "path":
                path = self.store.browser.resolve(raw.get("path", ""), must_exist=True)
                normalized = {"type": "path", "path": str(path)}
            elif source_type == "artifact":
                source_run_id = str(raw.get("run_id", "")).strip()
                artifact_id = str(raw.get("artifact_id", "")).strip()
                artifact, path = self.store.artifact_path(
                    workspace_id, source_run_id, artifact_id
                )
                normalized = {
                    "type": "artifact",
                    "run_id": source_run_id,
                    "artifact_id": artifact.id,
                    "artifact_type": artifact.artifact_type,
                    "path": str(path),
                }
            elif source_type == "batch_artifact":
                raise ValueError(
                    "batch_artifact input sources are only allowed in a confirmed batch"
                )
            else:
                raise ValueError("input source type must be 'path' or 'artifact'")
            if field:
                if field in bindings:
                    raise ValueError(f"Input field {field!r} was bound more than once")
                normalized["field"] = field
                bindings[field] = normalized["path"]
            sources.append(normalized)
        return sources, bindings

    @staticmethod
    def _split_batch_input_sources(
        value: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("input_sources must be a JSON array")
        ordinary: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for raw in value:
            if not isinstance(raw, dict):
                raise TypeError("Every input source must be a JSON object")
            target = (
                deferred
                if str(raw.get("type", "")).strip() == "batch_artifact"
                else ordinary
            )
            target.append(dict(raw))
        return ordinary, deferred

    @staticmethod
    def _normalize_batch_artifact_sources(
        *,
        index: int,
        action: Any,
        prepared_items: list[tuple[str, str, Any, dict[str, Any]]],
        run_ids: list[str],
        ordinary_sources: list[dict[str, Any]],
        deferred_sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        binding_schema = {
            str(field["name"]): field
            for field in action.input_schema
            if field.get("path")
        }
        bound_fields = {
            str(source.get("field", "")).strip()
            for source in ordinary_sources
            if str(source.get("field", "")).strip()
        }
        planned: list[dict[str, Any]] = []
        for raw in deferred_sources:
            producer_index = raw.get("producer_index")
            if isinstance(producer_index, bool) or not isinstance(producer_index, int):
                raise TypeError("batch_artifact producer_index must be an integer")
            if producer_index < 0 or producer_index >= index:
                raise ValueError(
                    "batch_artifact producer_index must reference an earlier "
                    "selected action"
                )
            artifact_type = str(raw.get("artifact_type", "")).strip()
            if not artifact_type:
                raise ValueError("batch_artifact requires artifact_type")
            producer_action = prepared_items[producer_index][2]
            if artifact_type not in producer_action.produces_artifacts:
                raise ValueError(
                    f"Selected producer action does not declare artifact type "
                    f"{artifact_type!r}"
                )
            accepted_types = set(action.accepts_artifacts)
            if "*" not in accepted_types and artifact_type not in accepted_types:
                raise ValueError(
                    f"Artifact type {artifact_type!r} is not accepted by the "
                    "selected consumer action"
                )
            field = str(raw.get("field", "")).strip()
            if field not in binding_schema:
                raise ValueError(
                    f"batch_artifact field {field!r} must name a consumer path field"
                )
            field_types = set(binding_schema[field].get("artifact_types") or [])
            if (
                field_types
                and "*" not in field_types
                and artifact_type not in field_types
            ):
                raise ValueError(
                    f"Artifact type {artifact_type!r} is not accepted by input "
                    f"field {field!r}"
                )
            if field in bound_fields:
                raise ValueError(f"Input field {field!r} was bound more than once")
            bound_fields.add(field)
            planned.append(
                {
                    "type": "batch_artifact",
                    "run_id": run_ids[producer_index],
                    "artifact_type": artifact_type,
                    "field": field,
                }
            )
        return planned

    def _resolve_planned_batch_artifacts(
        self, manifest: RadioRunManifest
    ) -> RadioRunManifest:
        planned_sources = list(manifest.provenance.get("planned_input_sources") or [])
        planned_iterator = iter(planned_sources)
        actual_sources: list[dict[str, Any]] = []
        for raw in manifest.request.get("input_sources") or []:
            if str(raw.get("type", "")).strip() != "batch_artifact":
                actual_sources.append(dict(raw))
                continue
            try:
                planned = next(planned_iterator)
            except StopIteration as exc:
                raise ValueError(
                    "Batch artifact plan does not match the persisted request"
                ) from exc
            producer_run_id = str(planned.get("run_id", "")).strip()
            artifact_type = str(planned.get("artifact_type", "")).strip()
            producer = self.store.load_run(manifest.workspace_id, producer_run_id)
            if producer.status != "succeeded":
                raise ValueError(
                    f"Planned producer run {producer_run_id} did not succeed"
                )
            matches = [
                artifact
                for artifact in producer.artifacts
                if artifact.artifact_type == artifact_type
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Planned artifact type {artifact_type!r} from producer run "
                    f"{producer_run_id} resolved to {len(matches)} artifacts; "
                    "exactly one is required"
                )
            actual_sources.append(
                {
                    "type": "artifact",
                    "run_id": producer_run_id,
                    "artifact_id": matches[0].id,
                    "field": str(planned["field"]),
                }
            )
        try:
            next(planned_iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("Batch artifact plan does not match the persisted request")

        resolution_payload = dict(manifest.request)
        resolution_payload["input_sources"] = actual_sources
        resolution_payload["_run_id"] = manifest.id
        resolved = self.resolve_request(
            manifest.workspace_id,
            manifest.module_id,
            manifest.action_id,
            resolution_payload,
        )
        action = get_action(manifest.module_id, manifest.action_id)
        self._validate_run_required_fields(action, resolved["config"])
        batch_provenance = {
            key: manifest.provenance[key]
            for key in ("batch_id", "batch_order", "depends_on_run_ids")
            if key in manifest.provenance
        }
        provenance = dict(resolved["provenance"])
        provenance.update(batch_provenance)
        provenance["planned_input_sources"] = planned_sources
        provenance["actual_input_sources"] = resolved["input_sources"]
        manifest.command = resolved["command"]
        manifest.resolved_config = resolved["config"]
        manifest.input_sources = resolved["input_sources"]
        manifest.provenance = provenance
        self.store.save_run(manifest)
        return manifest

    @staticmethod
    def _without_deferred_fields(
        action: Any,
        config: dict[str, Any],
        *,
        deferred_fields: frozenset[str],
    ) -> dict[str, Any]:
        result = deepcopy(config)
        schema = {str(field.get("name", "")): field for field in action.input_schema}
        for name in deferred_fields:
            result.pop(name, None)
            config_path = str(schema.get(name, {}).get("config_path") or "").strip()
            if not config_path:
                continue
            target: Any = result
            parts = config_path.split(".")
            for part in parts[:-1]:
                if not isinstance(target, dict):
                    break
                target = target.get(part)
            else:
                if isinstance(target, dict):
                    target.pop(parts[-1], None)
        return result

    def _validate_config_paths(
        self, input_schema: tuple[dict[str, Any], ...], config: dict[str, Any]
    ) -> None:
        for field in input_schema:
            if not field.get("path"):
                continue
            name = str(field["name"])
            value = config.get(name)
            if value not in (None, ""):
                normalized = name.casefold().replace("-", "_")
                directory_only = normalized in {
                    "root",
                    "directory",
                    "folder",
                } or normalized.endswith(("_dir", "_root", "_directory", "_folder"))
                file_only = not directory_only and normalized != "path"
                path = self.store.browser.resolve(
                    value,
                    must_exist=True,
                    file_only=file_only,
                    directory_only=directory_only,
                )
                if name.endswith("config_file") and path.suffix.casefold() == ".json":
                    import json

                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"Configuration file must contain an object: {path}"
                        )
                    self._validate_pattern_values(payload, source=str(path))
                    self._validate_path_values(payload, source=str(path))

    def _validate_pattern_values(
        self, value: Any, *, source: str, key: str = ""
    ) -> None:
        if self._looks_like_pattern_key(key):
            if isinstance(value, (list, tuple)):
                for item in value:
                    self._validate_pattern_values(item, source=source, key=key)
                return
            if not isinstance(value, str):
                raise TypeError(f"Pattern {key!r} from {source} must be a string")
            pattern = value.strip()
            if not pattern:
                raise ValueError(f"Pattern {key!r} from {source} may not be empty")
            windows_path = PureWindowsPath(pattern)
            posix_path = PurePosixPath(pattern)
            normalized_parts = PurePosixPath(pattern.replace("\\", "/")).parts
            if (
                windows_path.is_absolute()
                or bool(windows_path.drive)
                or bool(windows_path.root)
                or posix_path.is_absolute()
                or ".." in normalized_parts
            ):
                raise ValueError(
                    f"Pattern {key!r} from {source} must be a relative pattern "
                    "without parent traversal or a drive"
                )
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                self._validate_pattern_values(
                    child_value,
                    source=source,
                    key=str(child_key),
                )
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._validate_pattern_values(item, source=source, key=key)
            return

    @staticmethod
    def _looks_like_pattern_key(key: str) -> bool:
        normalized = key.casefold().replace("-", "_")
        return normalized in {"pattern", "glob"} or normalized.endswith(
            ("_pattern", "_glob")
        )

    def _validate_structured_config_paths(self, action, config: dict[str, Any]) -> None:
        if action.config_json_flag:
            self._validate_path_values(
                self._adapter_config(action, config),
                source=f"{action.id} workspace configuration",
            )
        module = action.command_module
        if module not in {
            "solar_apps.workflows.radio.source_map_cli",
            "solar_apps.workflows.radio.pipeline_cli",
            "solar_apps.workflows.radio.overlay_cli",
            "solar_apps.workflows.radio.quicklook",
            "solar_apps.workflows.radio.raw_quality_cli",
        }:
            return
        from solar_toolkit.radio.config import (
            DEFAULT_CONFIG_NAME,
            load_radio_event_config,
        )

        config_name = str(config.get("config") or DEFAULT_CONFIG_NAME)
        event = load_radio_event_config(config_name)
        adapter = self._adapter_config(action, config)
        if module == "solar_apps.workflows.radio.source_map_cli":
            data_adapter = adapter.get("data")
            raw_selection = (
                data_adapter.get("selected_source_map_json")
                if isinstance(data_adapter, dict)
                else None
            )
            if raw_selection not in (None, ""):
                if isinstance(raw_selection, str):
                    try:
                        selection = json.loads(raw_selection)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "selected_source_map_json must be valid JSON"
                        ) from exc
                else:
                    selection = raw_selection
                self._validate_path_values(
                    selection,
                    source="source-map preview selection",
                )
            effective = _deep_merge(event.section("user"), adapter)
            effective.pop("output", None)
            data = effective.get("data")
            if isinstance(data, dict):
                if str(effective.get("mode") or "multi_band") == "multi_band":
                    data.pop("single_file_path", None)
                    data.pop("input_dir", None)
                    data.pop("data_dir", None)
                else:
                    data.pop("multi_band_root", None)
                    if data.get("single_file_path"):
                        # A valid explicit single-file request never falls back to
                        # the event's batch directory, so that unrelated default
                        # path must not enlarge or block the selected file scope.
                        data.pop("input_dir", None)
                        data.pop("data_dir", None)
            features = effective.get("features")
            if isinstance(features, dict) and not features.get(
                "spectrogram_panel", False
            ):
                effective.pop("spectrogram", None)
        elif module == "solar_apps.workflows.radio.pipeline_cli":
            effective = {name: dict(values) for name, values in event.sections.items()}
            effective["user"] = _deep_merge(event.section("user"), adapter)
            effective.pop("output", None)
            effective["user"].pop("output", None)
        elif module == "solar_apps.workflows.radio.overlay_cli":
            section = str(config.get("overlay_section") or "aia_radio_hmi")
            effective = _deep_merge(event.section(section), adapter)
            effective.pop("output", None)
            if isinstance(effective.get("paths"), dict):
                effective["paths"].pop("output_dir", None)
        elif module == "solar_apps.workflows.radio.quicklook":
            if config.get("gaussian_csv"):
                return
            effective = event.section("output")
        else:
            if config.get("root"):
                return
            effective = event.section("user").get("data", {})
        self._validate_path_values(
            effective,
            source=f"effective event configuration {config_name!r}",
        )

    def _validate_path_values(self, value: Any, *, source: str, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                self._validate_path_values(
                    child_value,
                    source=source,
                    key=str(child_key),
                )
            return
        if isinstance(value, (list, tuple)):
            if self._looks_like_path_key(key):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        self._validate_declared_path(item, source=source, key=key)
            else:
                for item in value:
                    if isinstance(item, (dict, list, tuple)):
                        self._validate_path_values(item, source=source, key=key)
            return
        if isinstance(value, str) and value.strip() and self._looks_like_path_key(key):
            self._validate_declared_path(value, source=source, key=key)

    @staticmethod
    def _looks_like_path_key(key: str) -> bool:
        normalized = key.casefold().replace("-", "_")
        return normalized in {
            "path",
            "paths",
            "file",
            "files",
            "dir",
            "dirs",
            "root",
            "roots",
            "directory",
            "directories",
        } or normalized.endswith(
            (
                "_path",
                "_paths",
                "_file",
                "_files",
                "_dir",
                "_dirs",
                "_root",
                "_roots",
                "_directory",
                "_directories",
            )
        )

    def _validate_declared_path(self, value: str, *, source: str, key: str) -> None:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        resolved = candidate.resolve(strict=False)
        allowed = self.store.browser.allowed_roots
        if not any(resolved == root or root in resolved.parents for root in allowed):
            raise PermissionError(
                f"Configured path {key!r} from {source} is outside allowed roots: "
                f"{resolved}"
            )

    @staticmethod
    def _validate_event_config_name(value: Any) -> None:
        """Restrict web actions to package-owned event configuration modules."""

        if value in (None, ""):
            return
        name = str(value).strip()
        if name.endswith(".py"):
            name = name[:-3]
        for prefix in (
            "solar_apps.workflows.radio.configs.",
            "scripts.radio.configs.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if (
            not name
            or "." in name
            or "/" in name
            or "\\" in name
            or not name.replace("_", "").isalnum()
        ):
            raise ValueError("Event config must name a package-owned radio config")
        from solar_apps.workflows.radio import configs as config_package

        config_dir = Path(config_package.__file__).resolve().parent
        if not (config_dir / f"{name}.py").is_file():
            raise ValueError(f"Unknown package-owned radio event config: {name}")

    @staticmethod
    def _validate_required_fields(
        input_schema: tuple[dict[str, Any], ...],
        config: dict[str, Any],
        *,
        satisfied_fields: frozenset[str] = frozenset(),
    ) -> None:
        missing = [
            str(field["name"])
            for field in input_schema
            if field.get("required")
            and str(field["name"]) not in satisfied_fields
            and config.get(str(field["name"])) in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

    @staticmethod
    def _validate_run_required_fields(
        action: Any,
        config: dict[str, Any],
        *,
        satisfied_fields: frozenset[str] = frozenset(),
    ) -> None:
        missing = [
            name
            for name in action.run_required_fields
            if name not in satisfied_fields
            if config.get(name) in (None, "")
        ]
        if missing:
            raise ValueError(
                "Run requires explicit action output from Preview: "
                + ", ".join(missing)
            )
        if action.run_required_any_fields and not any(
            name in satisfied_fields or config.get(name) not in (None, "")
            for name in action.run_required_any_fields
        ):
            raise ValueError(
                "Run requires at least one of: "
                + ", ".join(action.run_required_any_fields)
            )

    def _build_command(
        self,
        action,
        config: dict[str, Any],
        explicit_arguments: list[str],
        *,
        artifact_dir: Path,
    ) -> list[str]:
        if not action.command_module:
            return []
        managed_flags = {
            str(field["cli_flag"])
            for field in action.input_schema
            if field.get("cli_flag")
        }
        managed_flags.update(
            flag for flag in (action.output_flag, action.config_json_flag) if flag
        )
        blocked_flags = set(action.blocked_arguments)
        for token in explicit_arguments:
            flag = token.partition("=")[0]
            if flag in blocked_flags:
                raise ValueError(f"{flag} is blocked by the Radio Workspace")
            if flag in managed_flags:
                raise ValueError(f"{flag} is managed by the Radio Workspace")
        arguments = list(action.fixed_arguments)
        if action.config_json_flag:
            adapter_config = self._adapter_config(action, config)
            arguments.extend(
                [
                    action.config_json_flag,
                    json.dumps(
                        adapter_config,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        path_flags: set[str] = set()
        for field in action.input_schema:
            name = str(field["name"])
            if name == "arguments":
                continue
            cli_flag = field.get("cli_flag")
            if not cli_flag or name not in config or config[name] in (None, ""):
                continue
            if field.get("path"):
                path_flags.add(str(cli_flag))
            value = config[name]
            if field.get("type") == "checkbox":
                if bool(value):
                    arguments.append(str(cli_flag))
                continue
            if isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            arguments.extend([str(cli_flag), str(value)])
        self._validate_explicit_argument_paths(explicit_arguments, path_flags)
        arguments.extend(explicit_arguments)
        if action.output_flag:
            output_path = (
                artifact_dir / action.output_filename
                if action.output_filename
                else artifact_dir
            )
            arguments.extend([action.output_flag, str(output_path)])
        return [
            self.python_executable,
            "-m",
            action.command_module,
            *arguments,
        ]

    @staticmethod
    def _adapter_config(action, config: dict[str, Any]) -> dict[str, Any]:
        result = dict(config)
        for field in action.input_schema:
            config_path = str(field.get("config_path") or "").strip()
            name = str(field.get("name") or "")
            if not config_path or name not in config:
                continue
            target = result
            parts = config_path.split(".")
            for part in parts[:-1]:
                nested = target.get(part)
                if not isinstance(nested, dict):
                    nested = {}
                    target[part] = nested
                target = nested
            target[parts[-1]] = config[name]
            if name != parts[0]:
                result.pop(name, None)
        return result

    def _validate_explicit_argument_paths(
        self, arguments: list[str], path_flags: set[str]
    ) -> None:
        previous = ""
        for token in arguments:
            normalized_previous = previous.casefold()
            likely_path_flag = (
                previous in path_flags
                or (
                    any(
                        part in normalized_previous
                        for part in ("path", "file", "dir", "root")
                    )
                    and "prefix" not in normalized_previous
                )
                or normalized_previous
                in {
                    "--out",
                    "--output",
                    "--output-dir",
                }
            )
            candidate = Path(token.strip('"').strip("'"))
            looks_like_path = (
                candidate.is_absolute()
                or "/" in token
                or "\\" in token
                or candidate.suffix.casefold()
                in {
                    ".csv",
                    ".fits",
                    ".fit",
                    ".fts",
                    ".html",
                    ".json",
                    ".png",
                    ".xlsx",
                }
            )
            if likely_path_flag:
                if not candidate.is_absolute():
                    raise ValueError(
                        f"Explicit path arguments must be absolute: {token}"
                    )
                self.store.browser.resolve(token, must_exist=False)
                previous = ""
                continue
            if looks_like_path:
                if not candidate.is_absolute():
                    raise ValueError(
                        f"Explicit path arguments must be absolute: {token}"
                    )
                self.store.browser.resolve(token, must_exist=False)
            previous = token if token.startswith("-") else ""

    def _index_artifacts(
        self,
        workspace_id: str,
        run_id: str,
        *,
        module_id: str,
        action_id: str,
    ) -> list[RadioArtifact]:
        root = (self.store.run_dir(workspace_id, run_id) / "artifacts").resolve(
            strict=True
        )
        declared_types = get_action(module_id, action_id).produces_artifacts
        artifacts: list[RadioArtifact] = []
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve(strict=True)
            if resolved != root and root not in resolved.parents:
                continue
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            kind, previewable = self._artifact_kind(path, mime_type)
            artifact_type = self._semantic_artifact_type(
                path,
                kind=kind,
                declared_types=declared_types,
            )
            artifacts.append(
                RadioArtifact(
                    id=uuid.uuid4().hex,
                    relative_path=path.relative_to(root).as_posix(),
                    kind=kind,
                    mime_type=mime_type,
                    artifact_type=artifact_type,
                    source_run_id=run_id,
                    size=path.stat().st_size,
                    previewable=previewable,
                    created_at=utc_now(),
                )
            )
        return artifacts

    def _build_native_preview(
        self,
        workspace_id: str,
        adapter: str | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        if adapter == "run-index":
            return {
                "adapter": adapter,
                "status": "ready",
                "kind": adapter,
                "items": [
                    item.to_dict() for item in self.store.list_runs(workspace_id)
                ],
            }
        if adapter == "artifact-index":
            items: list[dict[str, Any]] = []
            for run in self.store.list_runs(workspace_id):
                for artifact in run.artifacts:
                    payload = artifact.to_dict()
                    payload["run_id"] = run.id
                    payload["module_id"] = run.module_id
                    payload["action_id"] = run.action_id
                    items.append(payload)
            return {
                "adapter": adapter,
                "status": "ready",
                "kind": adapter,
                "items": items,
            }
        from .native_previews import build_native_preview

        preview = build_native_preview(
            str(adapter),
            config,
            validate_path=lambda value: self.store.browser.resolve(
                value, must_exist=True
            ),
        )
        if "selection" in preview:
            preview["selection_target"] = preview["selection"]
        return preview

    @staticmethod
    def _artifact_kind(path: Path, mime_type: str) -> tuple[str, bool]:
        suffix = path.suffix.casefold()
        if mime_type.startswith("image/"):
            return "image", True
        if suffix in {".csv", ".tsv", ".xlsx", ".xls"}:
            return "table", True
        if suffix == ".json":
            return "json", True
        if suffix in {".html", ".htm"}:
            return "html", True
        if mime_type.startswith("video/") or suffix in {".mp4", ".webm", ".gif"}:
            return "video", True
        return "file", False

    @staticmethod
    def _semantic_artifact_type(
        path: Path,
        *,
        kind: str,
        declared_types: tuple[str, ...],
    ) -> str:
        if not declared_types:
            return "file"
        if len(declared_types) == 1:
            return declared_types[0]
        name = path.name.casefold().replace("_", "-")
        if "metadata" in name:
            metadata_types = [
                artifact_type
                for artifact_type in declared_types
                if "metadata" in artifact_type.casefold()
            ]
            if len(metadata_types) == 1:
                return metadata_types[0]
        name_matches: list[tuple[int, int, str]] = []
        for declaration_index, artifact_type in enumerate(declared_types):
            tokens = [
                token
                for token in artifact_type.casefold().split("-")
                if token not in {"table", "image", "json", "html", "video"}
            ]
            if tokens and all(token in name for token in tokens):
                name_matches.append((len(tokens), -declaration_index, artifact_type))
        if name_matches:
            # Prefer the most specific semantic name.  For example,
            # ``spectrogram_drift_selection_metadata.json`` must be indexed as
            # ``spectrogram-metadata`` rather than the broader ``spectrogram``.
            return max(name_matches)[2]
        compatible_tokens = {
            "table": ("table",),
            "image": ("image", "map", "spectrogram"),
            "json": ("json", "selection", "provenance"),
            "html": ("html", "dashboard"),
            "video": ("video", "animation"),
        }
        compatible = [
            item
            for item in declared_types
            if any(token in item for token in compatible_tokens.get(kind, ()))
        ]
        return compatible[0] if len(compatible) == 1 else "file"

    @staticmethod
    def _terminate_process_tree(process: Any) -> None:
        pid = getattr(process, "pid", None)
        if pid is not None:
            try:
                import psutil

                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                _gone, alive = psutil.wait_procs([*children, parent], timeout=2.0)
                for item in alive:
                    item.kill()
                return
            except Exception:
                pass
        try:
            process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


__all__ = ["RadioRunManager"]
