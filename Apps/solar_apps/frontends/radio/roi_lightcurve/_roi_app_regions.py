"""ROI selection events, region-document import and staging.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    radio_roi_from_json,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
)
from ._roi_app_helpers import _roi_app_facade


from ._roi_app_state import (
    ANALYSIS_KEYS,
    EXPORT_KEYS,
    _RoiImportChoice,
    _RoiImportDocument,
)


def selection_to_radio_roi(
    selection_event: dict[str, Any] | None,
    *,
    mode: str = "box",
    label: str = "",
) -> RadioRoi | None:
    """Convert a Plotly selection event into a radio ROI."""

    if not selection_event:
        return None
    selection = _event_get(selection_event, "selection", selection_event)
    mode_norm = str(mode).lower()
    if mode_norm == "lasso":
        coords = _selection_xy(selection, "lasso")
        if coords is None:
            coords = _selection_xy(selection, "lassoPoints")
        if coords is not None:
            xs, ys = coords
            vertices = list(zip(xs, ys, strict=False))
            if len(vertices) >= 3:
                return RadioRoi.from_polygon(vertices, label=label)
    if mode_norm == "box":
        coords = _selection_xy(selection, "box")
        if coords is not None:
            roi = _box_roi_from_xy(*coords, label=label)
            if roi is not None:
                return roi

    points = _event_get(selection, "points", []) or []
    xs = [
        float(_event_get(point, "x"))
        for point in points
        if _event_get(point, "x") is not None
    ]
    ys = [
        float(_event_get(point, "y"))
        for point in points
        if _event_get(point, "y") is not None
    ]
    if mode_norm == "lasso" and len(xs) >= 3 and len(ys) >= 3:
        return RadioRoi.from_polygon(list(zip(xs, ys, strict=False)), label=label)
    if mode_norm == "box" and xs and ys:
        return _box_roi_from_xy(xs, ys, label=label)
    return None


def _store_roi_import_document(
    st: Any,
    document: _RoiImportDocument,
    *,
    source_kind: str,
    source_label: str,
    upload_signature: str | None = None,
) -> None:
    st.session_state["roi_import_document"] = document
    st.session_state["roi_import_source_kind"] = source_kind
    st.session_state["roi_import_source_label"] = source_label
    if upload_signature is not None:
        st.session_state["roi_import_upload_signature"] = upload_signature
    else:
        st.session_state.pop("roi_import_upload_signature", None)
    st.session_state["roi_import_selected_key"] = document.default_choice_key
    if len(document.choices) == 1:
        _stage_imported_roi(st, document.choices[0])


def _stage_imported_roi(st: Any, choice: _RoiImportChoice) -> bool:
    active = _session_roi(st, "candidate_roi") or _session_roi(st, "confirmed_roi")
    if active is not None and active.to_json_dict() == choice.roi.to_json_dict():
        return False
    st.session_state["candidate_roi"] = choice.roi.to_json_dict()
    _roi_app_facade()._clear_keys(st, ANALYSIS_KEYS + EXPORT_KEYS)
    return True


def _session_roi(st: Any, key: str) -> RadioRoi | None:
    payload = st.session_state.get(key)
    if payload is None:
        return None
    try:
        return radio_roi_from_json(payload)
    except Exception:
        return None


def _roi_from_uploaded_or_path(
    *,
    uploaded_payload: bytes | None,
    path_text: str,
    path_policy: PathAccessPolicy,
) -> RadioRoi:
    """Load upload bytes first, otherwise validate and read a local ROI JSON path."""

    document = _roi_import_document_from_uploaded_or_path(
        uploaded_payload=uploaded_payload,
        path_text=path_text,
        path_policy=path_policy,
    )
    if len(document.choices) != 1:
        raise ValueError(
            "ROI JSON contains multiple regions; select one region in the frontend."
        )
    return document.choices[0].roi


def _roi_import_document_from_uploaded_or_path(
    *,
    uploaded_payload: bytes | None,
    path_text: str,
    path_policy: PathAccessPolicy,
) -> _RoiImportDocument:
    """Load and validate an uploaded or allowed-root ROI JSON document."""

    if uploaded_payload is not None:
        text = uploaded_payload.decode("utf-8-sig")
    else:
        path = path_policy.input_file(path_text)
        text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ROI JSON is invalid: {exc.msg}") from exc
    return _parse_roi_import_document(payload)


def _parse_roi_import_document(payload: Any) -> _RoiImportDocument:
    if not isinstance(payload, Mapping):
        raise ValueError("ROI JSON must contain a JSON object.")
    if "rois" not in payload:
        roi = radio_roi_from_json(dict(payload))
        name = roi.label.strip() or "ROI 1"
        return _RoiImportDocument(
            source_format="radio_roi",
            choices=(
                _RoiImportChoice(
                    key=f"0:{roi.roi_id}",
                    source_id=roi.roi_id,
                    name=name,
                    source_type=roi.kind,
                    visible=True,
                    color="#00d4ff",
                    roi=roi,
                ),
            ),
        )

    from solar_apps.frontends.radio.source_map.exporting import validate_roi_template

    normalized = validate_roi_template(payload, template_mode=True)
    raw_rois = normalized["rois"]
    if not raw_rois:
        raise ValueError("Source Map ROI JSON contains no regions.")
    choices: list[_RoiImportChoice] = []
    for index, item in enumerate(raw_rois):
        name = str(item["name"])
        source_type = str(item["type"])
        geometry = item["geometry"]
        if source_type == "rectangle":
            roi = RadioRoi.from_box(
                float(geometry["left"]),
                float(geometry["bottom"]),
                float(geometry["right"]),
                float(geometry["top"]),
                label=name,
            )
        else:
            roi = RadioRoi.from_polygon(
                [
                    tuple(float(value) for value in point)
                    for point in geometry["points"]
                ],
                label=name,
            )
        source_id = str(item["id"])
        choices.append(
            _RoiImportChoice(
                key=f"{index}:{source_id}",
                source_id=source_id,
                name=name,
                source_type=source_type,
                visible=bool(item["visible"]),
                color=str(item["style"]["color"]),
                roi=roi,
            )
        )
    provenance = normalized.get("provenance")
    return _RoiImportDocument(
        source_format="source_map",
        choices=tuple(choices),
        source_image_sha256=str(payload.get("image_sha256") or ""),
        provenance=dict(provenance) if isinstance(provenance, Mapping) else None,
    )


def _selection_xy(selection: Any, key: str) -> tuple[list[float], list[float]] | None:
    payload = _event_get(selection, key, None)
    if isinstance(payload, (list, tuple)):
        payload = payload[-1] if payload else None
    if payload is None:
        return None
    xs = _float_list(_event_get(payload, "x", []))
    ys = _float_list(_event_get(payload, "y", []))
    if not xs or not ys:
        return None
    return xs, ys


def _float_list(values: Any) -> list[float]:
    if isinstance(values, (int, float, np.integer, np.floating)):
        return [float(values)]
    if values is None:
        return []
    return [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]


def _box_roi_from_xy(
    xs: list[float], ys: list[float], *, label: str
) -> RadioRoi | None:
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    if left == right or bottom == top:
        return None
    return RadioRoi.from_box(left, bottom, right, top, label=label)


def _event_get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


__all__ = [
    "selection_to_radio_roi",
    "_store_roi_import_document",
    "_stage_imported_roi",
    "_session_roi",
    "_roi_from_uploaded_or_path",
    "_roi_import_document_from_uploaded_or_path",
    "_parse_roi_import_document",
    "_selection_xy",
    "_float_list",
    "_box_roi_from_xy",
    "_event_get",
]
