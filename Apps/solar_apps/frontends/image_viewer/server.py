"""Flask app factory and filesystem helpers for the image web viewer."""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from solar_toolkit.visualization import media
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.native_dialog import validate_allowed_path
from solar_apps.ui import media as media_assets
from solar_apps.ui.flask_dialog import register_native_path_dialog
from solar_apps.ui.flask_state import register_ui_state
from solar_apps.ui.theme import register_theme_assets

from . import export as export_mod
from .export import ExportConfig

__all__ = [
    "ClientLifecycle",
    "IMAGE_EXTENSIONS",
    "configured_roots",
    "create_app",
    "is_under_allowed_root",
    "natural_key",
    "normalize_allowed_roots",
    "scan_images",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}


class ClientLifecycle:
    """Track browser clients and request local server shutdown when they close."""

    def __init__(
        self,
        *,
        stop_on_client_close: bool,
        shutdown_callback: Callable[[], None] | None,
        close_grace_seconds: float = 2.0,
        heartbeat_timeout_seconds: float = 20.0,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self.stop_on_client_close = stop_on_client_close
        self.shutdown_callback = shutdown_callback
        self.close_grace_seconds = close_grace_seconds
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._clients: dict[str, float] = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self.shutdown_requested = False

    def config(self) -> dict[str, Any]:
        return {
            "ok": True,
            "stop_on_close": self.stop_on_client_close,
            "heartbeat_interval_ms": int(self.heartbeat_interval_seconds * 1000),
            "close_grace_ms": int(self.close_grace_seconds * 1000),
        }

    def heartbeat(self, client_id: str) -> dict[str, Any]:
        if not client_id:
            return {"ok": False, "error": "client_id is required"}
        with self._lock:
            self._clients[client_id] = time.monotonic()
            if (
                self.stop_on_client_close
                and self.shutdown_callback is not None
                and not self.shutdown_requested
                and not (self._timer and self._timer.is_alive())
            ):
                self._start_timer_locked(
                    self.heartbeat_timeout_seconds + self.close_grace_seconds
                )
        return {"ok": True}

    def close(
        self, client_id: str, *, client_requests_stop: bool = True
    ) -> dict[str, Any]:
        if client_id:
            with self._lock:
                self._clients.pop(client_id, None)
        if client_requests_stop and self.stop_on_client_close:
            self.schedule_shutdown_check()
        return {
            "ok": True,
            "shutdown_scheduled": client_requests_stop and self.stop_on_client_close,
        }

    def schedule_shutdown_check(self) -> None:
        if self.shutdown_callback is None:
            return
        with self._lock:
            if self.shutdown_requested:
                return
            if self._timer and self._timer.is_alive():
                self._timer.cancel()
            self._start_timer_locked(self.close_grace_seconds)

    def _start_timer_locked(self, delay_seconds: float) -> None:
        self._timer = threading.Timer(
            max(float(delay_seconds), 0.001), self._maybe_shutdown
        )
        self._timer.daemon = True
        self._timer.start()

    def _maybe_shutdown(self) -> None:
        callback: Callable[[], None] | None = None
        now = time.monotonic()
        with self._lock:
            self._timer = None
            self._clients = {
                client_id: seen_at
                for client_id, seen_at in self._clients.items()
                if now - seen_at <= self.heartbeat_timeout_seconds
            }
            if self.shutdown_requested:
                return
            if self._clients:
                remaining = max(
                    self.heartbeat_timeout_seconds - (now - seen_at)
                    for seen_at in self._clients.values()
                )
                self._start_timer_locked(remaining + self.close_grace_seconds)
                return
            self.shutdown_requested = True
            callback = self.shutdown_callback
        if callback is not None:
            callback()


def natural_key(path: Path) -> list[Any]:
    """Sort image names naturally, e.g. frame2 before frame10."""

    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.casefold() for part in parts]


def configured_roots() -> list[Path]:
    """Read optional allowed roots from IMAGE_VIEWER_ALLOWED_ROOTS."""

    raw = os.getenv("IMAGE_VIEWER_ALLOWED_ROOTS", "").strip()
    if not raw:
        return []
    return [
        Path(item.strip()).expanduser().resolve()
        for item in raw.split(os.pathsep)
        if item.strip()
    ]


def normalize_allowed_roots(allowed_roots: list[str | Path] | None) -> list[Path]:
    """Normalize explicit roots or fall back to the environment setting."""

    if allowed_roots is None:
        return configured_roots()
    return [
        Path(root).expanduser().resolve() for root in allowed_roots if str(root).strip()
    ]


def is_under_allowed_root(
    path: Path, allowed_roots: list[str | Path] | None = None
) -> bool:
    """Return whether path is inside the optional local access boundary."""

    roots = normalize_allowed_roots(allowed_roots)
    if not roots:
        return True
    resolved = Path(path).expanduser().resolve()
    return any(resolved == root or root in resolved.parents for root in roots)


def scan_images(
    folder: str | Path,
    recursive: bool = False,
    *,
    allowed_roots: list[str | Path] | None = None,
) -> tuple[Path, list[Path]]:
    """Scan a local folder for supported image files."""

    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    if not is_under_allowed_root(root, allowed_roots):
        raise PermissionError(f"Path is outside IMAGE_VIEWER_ALLOWED_ROOTS: {root}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return root, sorted(files, key=natural_key)


def create_app(
    allowed_roots: list[str | Path] | None = None,
    *,
    stop_on_client_close: bool = True,
    shutdown_callback: Callable[[], None] | None = None,
    close_grace_seconds: float = 2.0,
    heartbeat_timeout_seconds: float = 20.0,
    default_output_format: str = "mp4",
    native_dialog_service=None,
):
    """Create the Flask app. Flask is imported lazily because it is optional."""

    from flask import Flask, Response, jsonify, render_template, request, send_file

    package_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_dir / "templates"),
        static_folder=str(package_dir / "static"),
    )
    register_theme_assets(app)
    sessions: dict[str, list[dict[str, Any]]] = {}
    cancelled_recordings: dict[str, float] = {}
    cancellation_lock = threading.Lock()
    roots = normalize_allowed_roots(allowed_roots)
    local_root = RuntimeLayout.discover().local_root
    protected_output_root = local_root / "outputs" / "image_web_viewer"
    output_roots = [*roots, protected_output_root]
    register_ui_state(app, frontend_id="image-viewer", allowed_roots=output_roots)
    register_native_path_dialog(
        app,
        allowed_roots=output_roots,
        service=native_dialog_service,
        memory=app.extensions["ui_state"]["recent_paths"],
    )
    default_output_format = media.normalize_output_format(default_output_format)
    lifecycle = ClientLifecycle(
        stop_on_client_close=stop_on_client_close,
        shutdown_callback=shutdown_callback,
        close_grace_seconds=close_grace_seconds,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )

    def prune_recording_cancellations(now: float) -> None:
        expired = [
            recording_id
            for recording_id, canceled_at in cancelled_recordings.items()
            if now - canceled_at > 600.0
        ]
        for recording_id in expired:
            cancelled_recordings.pop(recording_id, None)

    def mark_recording_cancelled(recording_id: str) -> None:
        with cancellation_lock:
            now = time.monotonic()
            prune_recording_cancellations(now)
            cancelled_recordings[recording_id] = now

    def is_recording_cancelled(recording_id: str) -> bool:
        with cancellation_lock:
            prune_recording_cancellations(time.monotonic())
            return recording_id in cancelled_recordings

    def clear_recording_cancellation(recording_id: str) -> None:
        with cancellation_lock:
            cancelled_recordings.pop(recording_id, None)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            default_output_format=default_output_format,
        )

    @app.get("/media-assets/<name>")
    def media_asset(name: str):
        try:
            content = media_assets.read_asset_bytes(name)
            content_type = media_assets.asset_mimetype(name)
        except FileNotFoundError:
            return "media asset not found", 404
        return Response(content, content_type=content_type)

    @app.post("/api/load")
    def load_folders():
        payload = request.get_json(force=True, silent=True) or {}
        folders = payload.get("folders", [])
        recursive = bool(payload.get("recursive", False))
        if not isinstance(folders, list):
            return jsonify({"ok": False, "error": "folders must be a list"}), 400

        cleaned = [str(folder).strip() for folder in folders if str(folder).strip()]
        if not cleaned:
            return (
                jsonify({"ok": False, "error": "Please enter at least one folder."}),
                400,
            )

        groups: list[dict[str, Any]] = []
        errors: list[str] = []
        for folder in cleaned:
            try:
                root, files = scan_images(
                    folder, recursive=recursive, allowed_roots=roots
                )
                groups.append(
                    {
                        "folder": str(root),
                        "name": root.name or str(root),
                        "files": files,
                    }
                )
            except Exception as exc:
                errors.append(str(exc))

        if errors:
            return jsonify({"ok": False, "error": "\n".join(errors)}), 400
        session_id = uuid.uuid4().hex
        sessions[session_id] = groups
        public_groups = [
            _public_group(group, index) for index, group in enumerate(groups)
        ]
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "groups": public_groups,
                "max_frames": max((len(group["files"]) for group in groups), default=0),
            }
        )

    @app.get("/api/image/<session_id>/<int:group_index>/<int:frame_index>")
    def image(session_id: str, group_index: int, frame_index: int):
        groups = sessions.get(session_id)
        if groups is None:
            return "session not found or expired", 404
        if group_index < 0 or group_index >= len(groups):
            return "group_index out of range", 404

        group = groups[group_index]
        files: list[Path] = group["files"]
        if frame_index < 0 or frame_index >= len(files):
            return "frame not found in this folder", 404

        path = files[frame_index]
        folder = Path(group["folder"]).resolve()
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:
            return "image file no longer exists", 404
        if not (resolved == folder or folder in resolved.parents):
            return "illegal path", 403
        return send_file(resolved, conditional=True, max_age=0)

    @app.post("/api/export-video")
    def export_video():
        payload = request.get_json(force=True, silent=True) or {}
        session_id = str(payload.get("session_id", ""))
        groups = sessions.get(session_id)
        if groups is None:
            return jsonify({"ok": False, "error": "session not found or expired"}), 404

        mode = str(payload.get("mode", "composite")).strip().lower()
        if mode not in {"composite", "separate", "both"}:
            return (
                jsonify(
                    {"ok": False, "error": "mode must be composite, separate, or both"}
                ),
                400,
            )
        try:
            config = _export_config_from_payload(
                payload,
                default_output_format=default_output_format,
            ).normalized()
            config.output_dir = validate_allowed_path(
                config.output_dir,
                allowed_roots=output_roots,
                kind="output_directory",
                base_directory=local_root,
            )
            result: dict[str, Any] = {
                "ok": True,
                "mode": mode,
                "format": config.output_format,
            }
            if mode in {"separate", "both"}:
                result["separate"] = export_mod.export_separate_videos(groups, config)
            if mode in {"composite", "both"}:
                result["composite"] = export_mod.export_composite_video(groups, config)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/save-recording")
    def save_recording():
        recording_file = request.files.get("recording")
        if recording_file is None:
            return jsonify({"ok": False, "error": "recording file is required"}), 400

        output_dir = request.form.get("output_dir") or "outputs/image_web_viewer"
        file_prefix = request.form.get("file_prefix") or "image_viewer"
        output_format = request.form.get("format") or default_output_format
        source_format = request.form.get("source_format") or "webm"
        fps = _float_form(request.form.get("fps"), default=5.0)
        quality = request.form.get("quality") or "low"

        recording_id: str | None = None
        try:
            output_dir = str(
                validate_allowed_path(
                    output_dir,
                    allowed_roots=output_roots,
                    kind="output_directory",
                    base_directory=local_root,
                )
            )
            recording_id = _optional_recording_id(request.form.get("recording_id"))
            recording_size = _optional_size(
                request.form.get("recording_width"),
                request.form.get("recording_height"),
            )
            save_kwargs = {
                "output_dir": output_dir,
                "file_prefix": file_prefix,
                "output_format": output_format,
                "source_format": source_format,
                "fps": fps,
                "quality": quality,
                "recording_size": recording_size,
                "expected_frame_count": _optional_positive_int(
                    request.form.get("frame_count")
                ),
            }
            if recording_id:
                save_kwargs["cancel_check"] = lambda: is_recording_cancelled(
                    recording_id
                )
            result = media.save_browser_recording(recording_file, **save_kwargs)
            return jsonify({"ok": True, **result})
        except Exception as exc:
            return jsonify(_media_error_payload(exc)), 400
        finally:
            if recording_id:
                clear_recording_cancellation(recording_id)

    @app.post("/api/save-recording-stream")
    def save_recording_stream():
        output_dir = request.args.get("output_dir") or "outputs/image_web_viewer"
        file_prefix = request.args.get("file_prefix") or "image_viewer"
        output_format = request.args.get("format") or default_output_format
        source_format = request.args.get("source_format") or "webm"
        fps = _float_form(request.args.get("fps"), default=5.0)
        quality = request.args.get("quality") or "low"

        recording_id: str | None = None
        try:
            output_dir = str(
                validate_allowed_path(
                    output_dir,
                    allowed_roots=output_roots,
                    kind="output_directory",
                    base_directory=local_root,
                )
            )
            recording_id = _optional_recording_id(request.args.get("recording_id"))
            recording_size = _optional_size(
                request.args.get("recording_width"),
                request.args.get("recording_height"),
            )
            save_kwargs = {
                "output_dir": output_dir,
                "file_prefix": file_prefix,
                "output_format": output_format,
                "source_format": source_format,
                "fps": fps,
                "quality": quality,
                "recording_size": recording_size,
                "expected_frame_count": _optional_positive_int(
                    request.args.get("frame_count")
                ),
            }
            if recording_id:
                save_kwargs["cancel_check"] = lambda: is_recording_cancelled(
                    recording_id
                )
            result = media.save_browser_recording_stream(request.stream, **save_kwargs)
            return jsonify({"ok": True, **result})
        except Exception as exc:
            return jsonify(_media_error_payload(exc)), 400
        finally:
            if recording_id:
                clear_recording_cancellation(recording_id)

    @app.post("/api/cancel-recording")
    def cancel_recording():
        payload = request.get_json(force=True, silent=True) or {}
        try:
            recording_id = _optional_recording_id(payload.get("recording_id"))
            if not recording_id:
                raise ValueError("recording_id is required")
            mark_recording_cancelled(recording_id)
            return jsonify({"ok": True, "recording_id": recording_id})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/client-config")
    def client_config():
        config = lifecycle.config()
        config["default_output_format"] = default_output_format
        return jsonify(config)

    @app.post("/api/client-heartbeat")
    def client_heartbeat():
        payload = request.get_json(force=True, silent=True) or {}
        result = lifecycle.heartbeat(str(payload.get("client_id", "")))
        return jsonify(result), (200 if result.get("ok") else 400)

    @app.post("/api/client-close")
    def client_close():
        payload = request.get_json(force=True, silent=True) or {}
        result = lifecycle.close(
            str(payload.get("client_id", "")),
            client_requests_stop=bool(payload.get("stop_on_close", True)),
        )
        return jsonify(result)

    return app


def _public_group(group: dict[str, Any], index: int) -> dict[str, Any]:
    files: list[Path] = group["files"]
    return {
        "index": index,
        "name": group["name"],
        "folder": group["folder"],
        "count": len(files),
        "first": files[0].name if files else None,
        "last": files[-1].name if files else None,
    }


def _export_config_from_payload(
    payload: dict[str, Any],
    *,
    default_output_format: str = "mp4",
) -> ExportConfig:
    output_dir = payload.get("output_dir") or "outputs/image_web_viewer"
    target_size = _optional_size(
        payload.get("target_width"), payload.get("target_height")
    )
    panel_size = _optional_size(payload.get("panel_width"), payload.get("panel_height"))
    return ExportConfig(
        output_dir=output_dir,
        file_prefix=str(payload.get("file_prefix") or "image_viewer"),
        output_format=str(payload.get("format") or default_output_format),
        fps=float(payload.get("fps") or 5.0),
        quality=str(payload.get("quality") or "low"),
        start_frame=int(payload.get("start_frame") or 0),
        end_frame=(
            None
            if payload.get("end_frame") in (None, "")
            else int(payload.get("end_frame"))
        ),
        target_size=target_size,
        composite_panel_size=panel_size or target_size,
        roi=payload.get("roi") if payload.get("use_roi", False) else None,
    )


def _optional_size(width, height) -> tuple[int, int] | None:
    width_missing = width in (None, "")
    height_missing = height in (None, "")
    if width_missing and height_missing:
        return None
    if width_missing or height_missing:
        raise ValueError("Width and height must be set together.")
    parsed = int(width), int(height)
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ValueError("Width and height must be positive integers.")
    return parsed


def _float_form(value: str | None, *, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_recording_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", text):
        raise ValueError("recording_id is invalid")
    return text


def _media_error_payload(exc: Exception) -> dict[str, Any]:
    error = getattr(exc, "user_message", None) or str(exc)
    detail = getattr(exc, "detail", "")
    payload: dict[str, Any] = {"ok": False, "error": str(error)}
    if detail:
        payload["detail"] = str(detail)
    return payload
