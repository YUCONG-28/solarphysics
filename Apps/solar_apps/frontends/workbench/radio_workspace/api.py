"""Flask Blueprint for the modular Radio Workspace API."""

from __future__ import annotations

import ipaddress
import json
import re
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from solar_apps.platform.layout import RuntimeLayout

from .catalog import catalog_payload, get_action, presets_payload
from .contracts import FIGURE_SCHEMA_VERSION, RadioFigureDraft, RadioFigurePreflight
from .figure_media import (
    MAX_ANIMATION_BYTES,
    MAX_PNG_BYTES,
    FigureMediaTooLarge,
)
from .figure_time import preflight_figure
from .runner import RadioRunManager
from .store import RadioWorkspaceStore

_MAX_PREVIEW_METADATA_JSON_BYTES = 256 * 1024
_MAX_FIGURE_JSON_BYTES = 8 * 1024 * 1024
_MAX_FIGURE_JSON_REQUEST_BYTES = _MAX_FIGURE_JSON_BYTES + 256 * 1024
_MAX_PREFLIGHT_JSON_BYTES = 128 * 1024 * 1024
_MAX_CLIENT_MANIFEST_JSON_BYTES = 1024 * 1024
_MAX_PREFLIGHT_REVISION_BYTES = 512
_MAX_MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024
_MAX_PREVIEW_MULTIPART_BYTES = (
    MAX_PNG_BYTES + _MAX_PREVIEW_METADATA_JSON_BYTES + _MAX_MULTIPART_OVERHEAD_BYTES
)
_MAX_EXPORT_MULTIPART_BYTES = (
    MAX_ANIMATION_BYTES
    + _MAX_FIGURE_JSON_BYTES
    + _MAX_PREFLIGHT_JSON_BYTES
    + _MAX_CLIENT_MANIFEST_JSON_BYTES
    + _MAX_MULTIPART_OVERHEAD_BYTES
)


def create_radio_blueprint(
    output_root: str | Path | None = None,
    *,
    allowed_roots: Iterable[str | Path] = (),
    repo_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    store: RadioWorkspaceStore | None = None,
    run_manager: RadioRunManager | None = None,
    root_update_token: str | None = None,
    name: str = "radio_workspace",
    url_prefix: str = "/api/radio",
):
    """Create a self-contained Blueprint without modifying the parent Flask app."""

    from flask import Blueprint, Response, jsonify, request, send_file
    from werkzeug.exceptions import RequestEntityTooLarge

    if store is None:
        if output_root is None:
            resolved_repo = (
                Path(repo_root).resolve()
                if repo_root is not None
                else RuntimeLayout.discover().repo_root
            )
            output_root = resolved_repo
        store = RadioWorkspaceStore(output_root, allowed_roots=allowed_roots)
    manager = run_manager or RadioRunManager(
        store,
        repo_root=repo_root,
        python_executable=python_executable,
    )
    blueprint = Blueprint(name, __name__, url_prefix=url_prefix)
    blueprint.radio_workspace_store = store
    blueprint.radio_run_manager = manager

    def require_figure_local_access() -> None:
        if not request_is_local(request):
            raise PermissionError("Figure Studio may only be accessed locally")

    def require_figure_write_access() -> None:
        require_figure_local_access()
        supplied_token = request.headers.get("X-Radio-Root-Token", "")
        if (
            not root_update_token
            or not supplied_token
            or not secrets.compare_digest(supplied_token, root_update_token)
        ):
            raise PermissionError("A valid local Radio Workspace token is required")

    def run_figure_preflight(
        workspace_id: str, draft: RadioFigureDraft
    ) -> RadioFigurePreflight:
        fingerprints = store.figure_source_fingerprints(workspace_id, draft)
        result = preflight_figure(draft.to_dict(), source_fingerprints=fingerprints)
        result.update(
            {
                "figure_schema_version": FIGURE_SCHEMA_VERSION,
                "workspace_id": workspace_id,
                "source_fingerprints": fingerprints,
            }
        )
        return RadioFigurePreflight.from_dict(result)

    @blueprint.errorhandler(KeyError)
    def _key_error(exc):
        return jsonify({"ok": False, "error": _error_text(exc)}), 404

    @blueprint.errorhandler(FileNotFoundError)
    @blueprint.errorhandler(NotADirectoryError)
    def _not_found(exc):
        return jsonify({"ok": False, "error": _error_text(exc)}), 404

    @blueprint.errorhandler(PermissionError)
    def _permission_error(exc):
        return jsonify({"ok": False, "error": _error_text(exc)}), 403

    @blueprint.errorhandler(FileExistsError)
    @blueprint.errorhandler(RuntimeError)
    def _conflict(exc):
        return jsonify({"ok": False, "error": _error_text(exc)}), 409

    @blueprint.errorhandler(TypeError)
    @blueprint.errorhandler(ValueError)
    def _bad_request(exc):
        return jsonify({"ok": False, "error": _error_text(exc)}), 400

    @blueprint.errorhandler(RequestEntityTooLarge)
    @blueprint.errorhandler(FigureMediaTooLarge)
    def _figure_payload_too_large(exc):
        message = getattr(exc, "description", None) or _error_text(exc)
        return jsonify({"ok": False, "error": str(message)}), 413

    @blueprint.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "schema_version": 1,
                "figure_schema_version": FIGURE_SCHEMA_VERSION,
            }
        )

    @blueprint.get("/assets/plotly.js")
    def plotly_asset():
        try:
            from plotly.offline import get_plotlyjs
        except ImportError:
            return jsonify({"ok": False, "error": "Plotly is unavailable"}), 503
        response = Response(get_plotlyjs(), content_type="application/javascript")
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/assets/<asset_name>")
    def media_asset(asset_name: str):
        from solar_toolkit.visualization import _media_assets as media_assets

        allowed = {
            "mediabunny-1.50.8.cjs",
            "browser_media.js",
            "NOTICE.txt",
            "mediabunny-MPL-2.0.txt",
        }
        if asset_name not in allowed:
            return jsonify({"ok": False, "error": "Asset not found"}), 404
        content = media_assets.read_asset_bytes(asset_name)
        response = Response(
            content,
            content_type=media_assets.asset_mimetype(asset_name),
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/modules")
    def modules():
        return jsonify({"ok": True, **catalog_payload()})

    @blueprint.get("/presets")
    def presets():
        return jsonify({"ok": True, **presets_payload()})

    @blueprint.get("/allowed-roots")
    def allowed_roots():
        return jsonify({"ok": True, **store.allowed_roots_payload()})

    @blueprint.put("/allowed-roots")
    def update_allowed_roots():
        if not request_is_local(request):
            raise PermissionError("Allowed roots may only be changed locally")
        supplied_token = request.headers.get("X-Radio-Root-Token", "")
        if (
            not root_update_token
            or not supplied_token
            or not secrets.compare_digest(supplied_token, root_update_token)
        ):
            raise PermissionError("A valid local root-update token is required")
        payload = _json_object(request)
        roots = payload.get("roots")
        if not isinstance(roots, list):
            raise TypeError("roots must be a JSON array")
        store.replace_user_roots(roots)
        return jsonify({"ok": True, **store.allowed_roots_payload()})

    @blueprint.get("/workspaces")
    def workspaces():
        return jsonify(
            {
                "ok": True,
                "workspaces": [item.to_dict() for item in store.list_workspaces()],
            }
        )

    @blueprint.post("/workspaces")
    def create_workspace():
        payload = _json_object(request)
        workspace = store.create_workspace(
            name=str(payload.get("name", "Radio Workspace")),
            event_preset=_object_field(payload, "event_preset"),
            shared_paths=_object_field(payload, "shared_paths"),
            advanced_config=_object_field(payload, "advanced_config"),
            concurrency=payload.get("concurrency", 1),
            output_root=payload.get("output_root"),
        )
        return jsonify({"ok": True, "workspace": workspace.to_dict()}), 201

    @blueprint.get("/workspaces/<workspace_id>")
    def workspace_detail(workspace_id: str):
        workspace = store.load_workspace(workspace_id)
        return jsonify({"ok": True, "workspace": workspace.to_dict()})

    @blueprint.patch("/workspaces/<workspace_id>")
    def update_workspace(workspace_id: str):
        workspace = store.update_workspace(workspace_id, _json_object(request))
        return jsonify({"ok": True, "workspace": workspace.to_dict()})

    @blueprint.delete("/workspaces/<workspace_id>")
    def delete_workspace(workspace_id: str):
        active = [
            item
            for item in store.list_runs(workspace_id)
            if item.status in {"queued", "running"}
        ]
        if active:
            raise RuntimeError("Cancel active runs before deleting this workspace")
        store.delete_workspace(workspace_id)
        return jsonify({"ok": True, "deleted": workspace_id})

    @blueprint.patch("/workspaces/<workspace_id>/layout")
    def update_layout(workspace_id: str):
        workspace = store.update_layout(workspace_id, _json_object(request))
        return jsonify({"ok": True, "workspace": workspace.to_dict()})

    @blueprint.get("/workspaces/<workspace_id>/figures/draft")
    def figure_draft(workspace_id: str):
        draft = store.load_figure_draft(workspace_id)
        return jsonify({"ok": True, "draft": draft.to_dict()})

    @blueprint.put("/workspaces/<workspace_id>/figures/draft")
    def save_figure_draft(workspace_id: str):
        require_figure_write_access()
        payload = _bounded_json_object(
            request,
            max_bytes=_MAX_FIGURE_JSON_REQUEST_BYTES,
            label="Figure draft",
        )
        raw_draft = payload.get("draft", payload)
        if not isinstance(raw_draft, dict):
            raise TypeError("draft must be a JSON object")
        draft = store.save_figure_draft(workspace_id, raw_draft)
        return jsonify({"ok": True, "draft": draft.to_dict()})

    @blueprint.post("/workspaces/<workspace_id>/figures/snapshots")
    def create_figure_snapshot(workspace_id: str):
        require_figure_write_access()
        payload = _bounded_json_object(
            request,
            max_bytes=_MAX_FIGURE_JSON_REQUEST_BYTES,
            label="Figure snapshot",
        )
        raw_draft = payload.get("draft")
        if not isinstance(raw_draft, dict):
            raise TypeError("draft must be a JSON object")
        snapshot = store.create_figure_snapshot(
            workspace_id,
            raw_draft,
            name=payload.get("name"),
        )
        return jsonify({"ok": True, "snapshot": snapshot}), 201

    @blueprint.post("/workspaces/<workspace_id>/figures/preflight")
    def figure_preflight(workspace_id: str):
        require_figure_local_access()
        store.load_workspace(workspace_id)
        payload = _bounded_json_object(
            request,
            max_bytes=_MAX_FIGURE_JSON_REQUEST_BYTES,
            label="Figure preflight",
        )
        raw_draft = payload.get("draft")
        if raw_draft is None:
            draft = store.load_figure_draft(workspace_id)
        elif isinstance(raw_draft, dict):
            draft = RadioFigureDraft.from_dict(raw_draft)
        else:
            raise TypeError("draft must be a JSON object")
        if draft.workspace_id != workspace_id:
            raise ValueError("Figure draft workspace id does not match the request")
        preflight = run_figure_preflight(workspace_id, draft)
        return jsonify({"ok": True, "preflight": preflight.to_dict()})

    @blueprint.post("/workspaces/<workspace_id>/figures/sources/previews")
    def register_figure_preview(workspace_id: str):
        require_figure_write_access()
        _require_bounded_multipart(
            request,
            max_bytes=_MAX_PREVIEW_MULTIPART_BYTES,
            max_form_memory_bytes=(
                _MAX_PREVIEW_METADATA_JSON_BYTES + _MAX_MULTIPART_OVERHEAD_BYTES
            ),
            label="Figure preview upload",
        )
        metadata = _multipart_json_field(
            request, "metadata", max_bytes=_MAX_PREVIEW_METADATA_JSON_BYTES
        )
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        if str(upload.mimetype).casefold() != "image/png":
            raise ValueError("Figure previews must use image/png")
        source = store.register_figure_preview(workspace_id, metadata, upload.stream)
        return jsonify({"ok": True, "source": source}), 201

    @blueprint.get("/workspaces/<workspace_id>/figures/sources/previews/<preview_id>")
    def figure_preview_source(workspace_id: str, preview_id: str):
        source, path = store.figure_preview_path(workspace_id, preview_id)
        response = send_file(
            path,
            mimetype="image/png",
            as_attachment=False,
            conditional=True,
            download_name=f"{source['preview_id']}.png",
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/workspaces/<workspace_id>/figures/exports")
    def list_figure_exports(workspace_id: str):
        exports = store.list_figure_exports(workspace_id)
        return jsonify({"ok": True, "exports": [item.to_dict() for item in exports]})

    @blueprint.post("/workspaces/<workspace_id>/figures/exports")
    def create_figure_export(workspace_id: str):
        require_figure_write_access()
        _require_bounded_multipart(
            request,
            max_bytes=_MAX_EXPORT_MULTIPART_BYTES,
            max_form_memory_bytes=(
                _MAX_FIGURE_JSON_BYTES
                + _MAX_PREFLIGHT_JSON_BYTES
                + _MAX_CLIENT_MANIFEST_JSON_BYTES
                + _MAX_MULTIPART_OVERHEAD_BYTES
            ),
            label="Figure export upload",
        )
        figure = RadioFigureDraft.from_dict(
            _multipart_json_field(request, "figure", max_bytes=_MAX_FIGURE_JSON_BYTES)
        )
        submitted_preflight = RadioFigurePreflight.from_dict(
            _multipart_json_field(
                request, "preflight", max_bytes=_MAX_PREFLIGHT_JSON_BYTES
            )
        )
        manifest = _multipart_json_field(
            request,
            "manifest",
            max_bytes=_MAX_CLIENT_MANIFEST_JSON_BYTES,
        )
        revision = _multipart_text_field(
            request,
            "preflight_revision",
            max_bytes=_MAX_PREFLIGHT_REVISION_BYTES,
        )
        if not revision:
            raise ValueError("preflight_revision is required")
        if figure.workspace_id != workspace_id:
            raise ValueError("Figure draft workspace id does not match the request")
        if submitted_preflight.workspace_id != workspace_id:
            raise ValueError("Figure preflight workspace id does not match the request")
        if submitted_preflight.preflight_revision != revision:
            raise RuntimeError("Submitted preflight revision does not match")

        current_preflight = run_figure_preflight(workspace_id, figure)
        if current_preflight.preflight_revision != revision:
            raise RuntimeError(
                "Figure draft or source changed after preflight; run preflight again"
            )
        if current_preflight.status != "ready":
            raise RuntimeError("Figure time coverage is incomplete")

        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        mime_type = str(upload.mimetype).strip().casefold()
        exported = store.create_figure_export(
            workspace_id,
            figure,
            current_preflight,
            manifest,
            upload.stream,
            mime_type=mime_type,
        )
        return jsonify({"ok": True, "export": exported.to_dict()}), 201

    @blueprint.get("/workspaces/<workspace_id>/figures/exports/<export_id>")
    def figure_export_detail(workspace_id: str, export_id: str):
        exported, manifest = store.load_figure_export(workspace_id, export_id)
        return jsonify({"ok": True, "export": exported.to_dict(), "manifest": manifest})

    @blueprint.get("/workspaces/<workspace_id>/figures/exports/<export_id>/preview")
    def preview_figure_export(workspace_id: str, export_id: str):
        exported, path = store.figure_export_file(workspace_id, export_id)
        response = send_file(
            path,
            mimetype=exported.mime_type,
            as_attachment=False,
            conditional=True,
            download_name=path.name,
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/workspaces/<workspace_id>/figures/exports/<export_id>/thumbnail")
    def figure_export_thumbnail(workspace_id: str, export_id: str):
        _exported, path = store.figure_export_file(
            workspace_id, export_id, thumbnail=True
        )
        response = send_file(
            path,
            mimetype="image/png",
            as_attachment=False,
            conditional=True,
            download_name="thumbnail.png",
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/workspaces/<workspace_id>/figures/exports/<export_id>/download")
    def download_figure_export(workspace_id: str, export_id: str):
        exported, path = store.figure_export_file(workspace_id, export_id)
        response = send_file(
            path,
            mimetype=exported.mime_type,
            as_attachment=True,
            conditional=True,
            download_name=path.name,
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.delete("/workspaces/<workspace_id>/figures/exports/<export_id>")
    def delete_figure_export(workspace_id: str, export_id: str):
        require_figure_write_access()
        store.delete_figure_export(workspace_id, export_id)
        return jsonify({"ok": True, "deleted": export_id})

    @blueprint.get("/files")
    def files():
        payload = store.browser.list_directory(request.args.get("path"))
        return jsonify({"ok": True, **payload})

    @blueprint.post(
        "/workspaces/<workspace_id>/modules/<module_id>/actions/<action_id>/preview"
    )
    def preview_action(workspace_id: str, module_id: str, action_id: str):
        preview = manager.preview(
            workspace_id,
            module_id,
            action_id,
            _json_object(request),
        )
        return jsonify({"ok": True, "preview": preview})

    @blueprint.get("/workspaces/<workspace_id>/runs")
    def list_runs(workspace_id: str):
        return jsonify(
            {
                "ok": True,
                "runs": [item.to_dict() for item in manager.list_runs(workspace_id)],
            }
        )

    @blueprint.post("/workspaces/<workspace_id>/runs")
    def start_run(workspace_id: str):
        payload = _json_object(request)
        module_id = str(payload.pop("module_id", "")).strip()
        action_id = str(payload.pop("action_id", "")).strip()
        if not module_id or not action_id:
            raise ValueError("module_id and action_id are required")
        run = manager.start(workspace_id, module_id, action_id, payload)
        return jsonify({"ok": True, "run": run.to_dict()}), 202

    @blueprint.post("/workspaces/<workspace_id>/runs/batch")
    def start_batch(workspace_id: str):
        payload = _json_object(request)
        if payload.get("confirmed") is not True:
            raise ValueError("confirmed must be true before running selected actions")
        actions = payload.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty JSON array")
        runs = manager.start_batch(workspace_id, actions)
        return jsonify({"ok": True, "runs": [item.to_dict() for item in runs]}), 202

    @blueprint.get("/workspaces/<workspace_id>/runs/<run_id>")
    @blueprint.get("/workspaces/<workspace_id>/runs/<run_id>/status")
    def run_status(workspace_id: str, run_id: str):
        run = manager.status(workspace_id, run_id)
        return jsonify({"ok": True, "run": run.to_dict()})

    @blueprint.get("/workspaces/<workspace_id>/runs/<run_id>/log")
    def run_log(workspace_id: str, run_id: str):
        raw_offset = request.args.get("offset", "0")
        try:
            offset = int(raw_offset)
        except ValueError as exc:
            raise ValueError("offset must be an integer") from exc
        lines, next_offset = store.read_log(workspace_id, run_id, offset=offset)
        return jsonify({"ok": True, "lines": lines, "next_offset": next_offset})

    @blueprint.post("/workspaces/<workspace_id>/runs/<run_id>/cancel")
    def cancel_run(workspace_id: str, run_id: str):
        run = manager.cancel(workspace_id, run_id)
        return jsonify({"ok": True, "run": run.to_dict()})

    @blueprint.get("/workspaces/<workspace_id>/artifacts")
    def workspace_artifacts(workspace_id: str):
        artifacts: list[dict[str, Any]] = []
        for run in manager.list_runs(workspace_id):
            for artifact in run.artifacts:
                artifacts.append(_artifact_api_payload(run, artifact))
        return jsonify({"ok": True, "artifacts": artifacts})

    @blueprint.get("/workspaces/<workspace_id>/runs/<run_id>/artifacts")
    def run_artifacts(workspace_id: str, run_id: str):
        run = manager.status(workspace_id, run_id)
        return jsonify(
            {
                "ok": True,
                "artifacts": [
                    _artifact_api_payload(run, item) for item in run.artifacts
                ],
            }
        )

    @blueprint.get("/workspaces/<workspace_id>/runs/<run_id>/artifacts/<artifact_id>")
    def artifact_file(workspace_id: str, run_id: str, artifact_id: str):
        artifact, path = store.artifact_path(workspace_id, run_id, artifact_id)
        response = send_file(
            path,
            mimetype=artifact.mime_type,
            as_attachment=request.args.get("download") in {"1", "true", "yes"},
            conditional=True,
            download_name=path.name,
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        if artifact.kind == "html":
            response.headers["Content-Security-Policy"] = "sandbox"
        return response

    return blueprint


def _json_object(request) -> dict[str, Any]:
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("request body must be a JSON object")
    return dict(payload)


def _request_entity_too_large(description: str) -> Exception:
    """Build Werkzeug's 413 exception only when the optional web stack is used."""

    from werkzeug.exceptions import RequestEntityTooLarge

    return RequestEntityTooLarge(description=description)


def _bounded_json_object(request, *, max_bytes: int, label: str) -> dict[str, Any]:
    """Reject an unbounded or oversized JSON body before parsing it."""

    request.max_content_length = int(max_bytes)
    content_length = request.content_length
    if content_length is None:
        raise ValueError(f"{label} requires a Content-Length header")
    if int(content_length) < 0 or int(content_length) > int(max_bytes):
        raise _request_entity_too_large(
            f"{label} exceeds the {int(max_bytes)}-byte request limit"
        )
    return _json_object(request)


def _require_bounded_multipart(
    request, *, max_bytes: int, max_form_memory_bytes: int | None = None, label: str
) -> None:
    """Reject an unbounded/oversized body before Werkzeug parses multipart data."""

    if str(request.mimetype or "").casefold() != "multipart/form-data":
        raise ValueError(f"{label} must use multipart/form-data")
    request.max_content_length = int(max_bytes)
    request.max_form_memory_size = int(max_form_memory_bytes or max_bytes)
    request.max_form_parts = 8
    content_length = request.content_length
    if content_length is None:
        raise ValueError(f"{label} requires a Content-Length header")
    if int(content_length) > int(max_bytes):
        raise _request_entity_too_large(
            f"{label} exceeds the {int(max_bytes)}-byte request limit"
        )


def _multipart_text_field(request, name: str, *, max_bytes: int) -> str:
    raw = request.form.get(name)
    if raw in (None, ""):
        raise ValueError(f"{name} is required")
    encoded_size = len(str(raw).encode("utf-8"))
    if encoded_size > int(max_bytes):
        raise _request_entity_too_large(
            f"{name} exceeds the {int(max_bytes)}-byte field limit"
        )
    return str(raw).strip()


def _multipart_json_field(request, name: str, *, max_bytes: int) -> dict[str, Any]:
    raw = _multipart_text_field(request, name, max_bytes=max_bytes)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return dict(payload)


def _object_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a JSON object")
    return dict(value)


def _error_text(exc: Exception) -> str:
    return str(exc.args[0]) if exc.args else str(exc)


_FULL_ARTIFACT_TIME_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}T\d{2}:?\d{2}:?\d{2}Z?)"
    r"|(?:\d{8}T\d{6}Z?)"
    r"|(?:\d{8}[_-]\d{6}(?:_TAI)?)"
    r"|(?:(?<!\d)\d{14}(?!\d))",
    re.IGNORECASE,
)


def _artifact_api_payload(run: Any, artifact: Any) -> dict[str, Any]:
    payload = artifact.to_dict()
    payload["run_id"] = run.id
    payload["module_id"] = run.module_id
    payload["action_id"] = run.action_id
    payload["declared_types"] = list(
        get_action(run.module_id, run.action_id).produces_artifacts
    )
    if artifact.kind == "image":
        observed_at, series_key = _artifact_time_info(run, artifact)
        if observed_at is not None:
            payload["observed_at"] = observed_at
        if series_key is not None:
            payload["series_key"] = series_key
    return payload


def _utc_text(value: Any) -> str:
    from solar_toolkit.time import parse_time

    parsed = parse_time(str(value))
    return f"{parsed.isoformat(timespec='microseconds').rstrip('0').rstrip('.')}Z"


def _artifact_time_info(run: Any, artifact: Any) -> tuple[str | None, str | None]:
    from solar_toolkit.time import extract_time_from_filename, parse_time

    relative_path = str(artifact.relative_path).replace("\\", "/")
    if _FULL_ARTIFACT_TIME_RE.search(relative_path):
        try:
            parsed = extract_time_from_filename(relative_path)
        except ValueError:
            pass
        else:
            observed_at = _utc_text(parsed)
            product_pattern = _FULL_ARTIFACT_TIME_RE.sub("{time}", relative_path)
            series_key = ":".join(
                (
                    run.id,
                    run.module_id,
                    run.action_id,
                    artifact.artifact_type,
                    product_pattern,
                )
            )
            return observed_at, series_key

    keys = (
        "observed_at",
        "observation_time_iso",
        "frame_time",
        "frame_time_iso",
        "time_iso",
    )
    for mapping in (
        getattr(run, "resolved_config", {}),
        getattr(run, "provenance", {}),
        getattr(run, "request", {}),
    ):
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            value = mapping.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = parse_time(str(value))
            except (TypeError, ValueError):
                continue
            return _utc_text(parsed), None
    return None, None


def request_is_local(request) -> bool:
    """Return whether both the peer and Host header identify this machine."""

    remote_addr = str(request.remote_addr or "").strip()
    if not remote_addr:
        return False
    try:
        if not ipaddress.ip_address(remote_addr).is_loopback:
            return False
    except ValueError:
        return False

    raw_host = str(request.host or "").strip()
    if not raw_host:
        return False
    try:
        hostname = (urlsplit(f"//{raw_host}").hostname or "").rstrip(".").casefold()
    except ValueError:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


__all__ = ["create_radio_blueprint", "request_is_local"]
