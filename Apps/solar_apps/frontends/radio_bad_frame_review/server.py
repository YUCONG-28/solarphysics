"""Flask application factory for the standalone bad-frame reviewer."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from solar_apps.platform.paths.allowed_roots import normalize_allowed_roots
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.ui.flask_dialog import register_native_path_dialog
from solar_apps.ui.flask_state import register_ui_state
from solar_apps.ui.theme import register_theme_assets

from .lifecycle import ClientLifecycle
from .model_registry import QualityModelRegistry
from .review import (
    PREVIEW_COLORMAPS,
    PREVIEW_PERCENTILE_DEFAULTS,
    PREVIEW_RANGE_MODES,
    PREVIEW_TRANSFORMS,
    BadFrameReviewStore,
    PreviewDisplaySettings,
    StaleReviewError,
)
from .training import load_published_quality_model, predict_review_candidate

__all__ = ["create_app"]


def create_app(
    allowed_roots: list[str | Path] | tuple[str | Path, ...],
    *,
    output_root: str | Path | None = None,
    stop_on_client_close: bool = True,
    shutdown_callback=None,
    close_grace_seconds: float = 2.0,
    heartbeat_timeout_seconds: float = 20.0,
    store: BadFrameReviewStore | None = None,
    native_dialog_service=None,
    model_registry: QualityModelRegistry | None = None,
):
    """Create the local-only bad-frame review Flask application."""

    from flask import Flask, jsonify, render_template, request, send_file

    package_dir = Path(__file__).resolve().parent
    local_root = RuntimeLayout.discover().local_root
    roots = normalize_allowed_roots(list(allowed_roots))
    selected_output = (
        Path(output_root).expanduser().resolve(strict=False)
        if output_root is not None
        else local_root / "outputs" / "bad_frame_reviews"
    )
    default_model_root = (
        selected_output.parent / "bad_frame_models"
        if output_root is not None
        else local_root / "outputs" / "bad_frame_models"
    )
    registry = model_registry or QualityModelRegistry(default_model_root)
    shadow_model_id = None
    model_warning = None
    shadow_predictor = None
    if store is None:
        registry_state = registry.list_models()
        active_model_id = registry_state.get("active_model_id")
        if active_model_id:
            try:
                model_bundle = load_published_quality_model(registry)
                model_entry = registry_state["models"][active_model_id]
                shadow_model_id = str(active_model_id)

                def shadow_predictor(candidate):
                    return predict_review_candidate(
                        model_bundle,
                        candidate,
                        model_id=shadow_model_id,
                        bundle_sha256=str(model_entry["manifest_sha256"]),
                    )

            except Exception as exc:  # noqa: BLE001 - fail safely to rules/review
                model_warning = (
                    f"Published model unavailable: {type(exc).__name__}: {exc}"
                )
        review_store = BadFrameReviewStore(
            selected_output,
            roots,
            shadow_predictor=shadow_predictor,
            shadow_model_id=shadow_model_id,
        )
    else:
        review_store = store
    lifecycle = ClientLifecycle(
        stop_on_client_close=stop_on_client_close,
        shutdown_callback=shutdown_callback,
        close_grace_seconds=close_grace_seconds,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
    )
    app = Flask(
        __name__,
        template_folder=str(package_dir / "templates"),
        static_folder=str(package_dir / "static"),
    )
    register_theme_assets(app)
    register_ui_state(
        app,
        frontend_id="bad-frame-review",
        allowed_roots=review_store.allowed_roots,
    )
    register_native_path_dialog(
        app,
        allowed_roots=review_store.allowed_roots,
        service=native_dialog_service,
        memory=app.extensions["ui_state"]["recent_paths"],
    )
    app.extensions["bad_frame_review"] = {
        "store": review_store,
        "lifecycle": lifecycle,
        "model_registry": registry,
        "model_warning": model_warning,
    }

    @app.after_request
    def no_store(response):
        if (
            request.path == "/"
            or request.path.startswith("/api/")
            or request.path.startswith("/static/")
        ):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(StaleReviewError)
    def stale_review(exc):
        return jsonify({"ok": False, "error": str(exc), "stale": True}), 409

    @app.errorhandler(KeyError)
    def missing_item(exc):
        return jsonify({"ok": False, "error": str(exc.args[0])}), 404

    @app.errorhandler(PermissionError)
    def forbidden_path(exc):
        return jsonify({"ok": False, "error": str(exc)}), 403

    @app.errorhandler(FileNotFoundError)
    @app.errorhandler(NotADirectoryError)
    @app.errorhandler(TypeError)
    @app.errorhandler(ValueError)
    def invalid_request(exc):
        return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/config")
    def config():
        return jsonify(
            {
                "ok": True,
                "allowed_roots": [str(root) for root in review_store.allowed_roots],
                "output_root": str(review_store.output_root),
                "active_model_id": shadow_model_id,
                "model_warning": model_warning,
                "preview_display": {
                    "colormaps": list(PREVIEW_COLORMAPS),
                    "transforms": list(PREVIEW_TRANSFORMS),
                    "range_modes": list(PREVIEW_RANGE_MODES),
                    "percentile_bounds": [0.0, 100.0],
                    "percentile_defaults": list(PREVIEW_PERCENTILE_DEFAULTS),
                    "defaults": PreviewDisplaySettings().to_dict(),
                },
            }
        )

    @app.get("/api/files")
    def files():
        payload = review_store.list_directories(request.args.get("path"))
        return jsonify({"ok": True, **payload})

    @app.post("/api/discover")
    def discover():
        payload = _json_object(request)
        result = review_store.discover(str(payload.get("root", "")))
        return jsonify({"ok": True, **result})

    @app.get("/api/reviews")
    def list_reviews():
        return jsonify({"ok": True, "reviews": review_store.list_reviews()})

    @app.get("/api/models")
    def list_models():
        return jsonify({"ok": True, **registry.list_models()})

    @app.post("/api/reviews")
    def create_review():
        manifest = review_store.create_review(_json_object(request))
        return (
            jsonify(
                {
                    "ok": True,
                    "review": review_store.public_payload(manifest),
                }
            ),
            201,
        )

    @app.get("/api/reviews/<review_id>")
    def review_detail(review_id: str):
        manifest = review_store.load_review(review_id)
        return jsonify({"ok": True, "review": review_store.public_payload(manifest)})

    @app.get("/api/reviews/<review_id>/frames")
    def review_frames(review_id: str):
        payload = review_store.list_frames(
            review_id,
            offset=int(request.args.get("offset", 0)),
            limit=int(request.args.get("limit", 100)),
        )
        return jsonify({"ok": True, **payload})

    @app.patch("/api/reviews/<review_id>")
    def update_review(review_id: str):
        payload = _json_object(request)
        decisions = payload.get("decisions")
        labels = payload.get("labels")
        has_decisions = isinstance(decisions, dict) and bool(decisions)
        has_labels = isinstance(labels, dict) and bool(labels)
        if has_decisions == has_labels:
            raise ValueError(
                "provide exactly one non-empty labels or decisions JSON object"
            )
        manifest = (
            review_store.update_labels(review_id, labels)
            if has_labels
            else review_store.update_decisions(review_id, decisions)
        )
        return jsonify({"ok": True, "review": review_store.public_payload(manifest)})

    @app.post("/api/reviews/<review_id>/finalize")
    def finalize_review(review_id: str):
        payload = _json_object(request)
        manifest = review_store.finalize(review_id, str(payload.get("mode", "")))
        return jsonify({"ok": True, "review": review_store.public_payload(manifest)})

    @app.get("/api/reviews/<review_id>/candidates/<candidate_id>/preview")
    def candidate_preview(review_id: str, candidate_id: str):
        payload = review_store.render_candidate_preview(
            review_id,
            candidate_id,
            display=_preview_display_query(request),
        )
        return send_file(
            io.BytesIO(payload),
            mimetype="image/png",
            download_name=f"{candidate_id}.png",
            conditional=False,
            max_age=0,
        )

    @app.get("/api/reviews/<review_id>/frames/<file_id>/preview")
    def frame_preview(review_id: str, file_id: str):
        payload = review_store.render_frame_preview(
            review_id,
            file_id,
            display=_preview_display_query(request),
        )
        return send_file(
            io.BytesIO(payload),
            mimetype="image/png",
            download_name=f"{file_id}.png",
            conditional=False,
            max_age=0,
        )

    @app.post("/api/reviews/<review_id>/frames/<file_id>/viewed")
    def frame_viewed(review_id: str, file_id: str):
        manifest = review_store.mark_frame_viewed(review_id, file_id)
        return jsonify({"ok": True, "review": review_store.public_payload(manifest)})

    @app.patch("/api/reviews/<review_id>/frames/<file_id>")
    def frame_label(review_id: str, file_id: str):
        payload = _json_object(request)
        label = payload.get("label")
        if not isinstance(label, dict):
            raise TypeError("label must be a JSON object")
        manifest = review_store.update_frame_label(review_id, file_id, label)
        return jsonify({"ok": True, "review": review_store.public_payload(manifest)})

    @app.get("/api/reviews/<review_id>/manifest.json")
    def manifest_download(review_id: str):
        manifest = review_store.load_review(review_id)
        payload = review_store.public_payload(manifest, include_files=True)
        return send_file(
            io.BytesIO(
                (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode(
                    "utf-8"
                )
            ),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"bad-frame-review-{review_id}.json",
            conditional=False,
        )

    @app.get("/api/reviews/<review_id>/table.csv")
    def table_download(review_id: str):
        path = review_store.download_path(review_id, "candidates.csv")
        return send_file(
            path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"bad-frame-review-{review_id}.csv",
            conditional=True,
        )

    @app.get("/api/reviews/<review_id>/audit.csv")
    def audit_download(review_id: str):
        path = review_store.download_path(review_id, "viewed_frames.csv")
        return send_file(
            path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"bad-frame-audit-{review_id}.csv",
            conditional=True,
        )

    @app.get("/api/client-config")
    def client_config():
        return jsonify(lifecycle.config())

    @app.post("/api/client-heartbeat")
    def client_heartbeat():
        payload = _json_object(request)
        result = lifecycle.heartbeat(str(payload.get("client_id", "")))
        return jsonify(result), (200 if result.get("ok") else 400)

    @app.post("/api/client-close")
    def client_close():
        payload = _json_object(request)
        result = lifecycle.close(
            str(payload.get("client_id", "")),
            client_requests_stop=bool(payload.get("stop_on_close", True)),
        )
        return jsonify(result)

    return app


def _json_object(request) -> dict[str, Any]:
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError("Request body must be a JSON object")
    return payload


def _preview_display_query(request) -> PreviewDisplaySettings:
    return PreviewDisplaySettings.from_mapping(
        {
            key: request.args.get(key)
            for key in (
                "cmap",
                "transform",
                "range_mode",
                "pmin",
                "pmax",
                "vmin",
                "vmax",
            )
        }
    )
