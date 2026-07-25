from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from solar_apps.frontends.radio.roi_lightcurve import roi_lightcurve_app as app
from solar_apps.frontends.radio.composite_figure import (
    composite_figure_app as composite_app,
)
from solar_apps.ui.streamlit_paths import PathAccessPolicy
from solar_toolkit.radio.roi_lightcurve import RadioRoi


def _source_map_payload(*, rois: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "coordinate_system": "HPLN/HPLT arcsec",
        "image_sha256": "a" * 64,
        "rois": rois,
        **extra,
    }


def _rectangle(
    name: str = "Burst",
    *,
    visible: bool = True,
    source_id: str = "box-1",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "name": name,
        "type": "rectangle",
        "geometry": {"left": 20, "right": -10, "bottom": 5, "top": -5},
        "visible": visible,
        "style": {"color": "#12abef", "line_width": 3, "show_label": True},
    }


def _lasso(
    name: str = "Loop",
    *,
    visible: bool = True,
    source_id: str = "lasso-1",
) -> dict[str, Any]:
    return {
        "id": source_id,
        "name": name,
        "type": "lasso",
        "geometry": {"points": [[0, 0], [10, 0], [5, 8]]},
        "visible": visible,
        "style": {"color": "#fedcba", "line_width": 2, "show_label": False},
    }


def test_legacy_direct_and_wrapped_roi_json_remain_supported() -> None:
    direct = {
        "kind": "box",
        "label": "legacy",
        "bounds_arcsec": {"left": 1, "right": 2, "bottom": 3, "top": 4},
    }
    wrapped = {"roi": direct, "settings": {"metric": "raw_sum"}}

    direct_document = app._parse_roi_import_document(direct)
    wrapped_document = app._parse_roi_import_document(wrapped)

    assert direct_document.source_format == "radio_roi"
    assert wrapped_document.choices[0].roi.to_json_dict() == (
        direct_document.choices[0].roi.to_json_dict()
    )
    assert direct_document.choices[0].name == "legacy"


def test_source_map_rectangle_and_lasso_preserve_order_names_and_coordinates() -> None:
    document = app._parse_roi_import_document(
        _source_map_payload(
            rois=[_rectangle(visible=False), _lasso(visible=True)],
            provenance={"template_source_image_sha256": "b" * 64},
        )
    )

    assert document.source_format == "source_map"
    assert document.source_image_sha256 == "a" * 64
    assert document.default_choice_key == document.choices[1].key
    assert [choice.name for choice in document.choices] == ["Burst", "Loop"]
    assert [choice.source_type for choice in document.choices] == [
        "rectangle",
        "lasso",
    ]
    assert document.choices[0].roi.kind == "box"
    assert document.choices[0].roi.bounds_arcsec == {
        "left": -10.0,
        "bottom": -5.0,
        "right": 20.0,
        "top": 5.0,
    }
    assert document.choices[1].roi.kind == "polygon"
    assert document.choices[1].roi.vertices_arcsec == (
        (0.0, 0.0),
        (10.0, 0.0),
        (5.0, 8.0),
    )
    assert document.provenance == {
        "template_source_image_sha256": "a" * 64,
        "template_mode": True,
    }


def test_source_map_default_falls_back_to_first_hidden_region() -> None:
    document = app._parse_roi_import_document(
        _source_map_payload(
            rois=[
                _rectangle(visible=False),
                _lasso(visible=False),
            ]
        )
    )

    assert document.default_choice_key == document.choices[0].key
    assert "hidden" in document.choices[0].display_label


def test_single_source_map_region_loads_from_allowed_path(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    roi_path = root / "selection.roi-set.json"
    roi_path.write_text(
        json.dumps(_source_map_payload(rois=[_rectangle()])), encoding="utf-8"
    )
    policy = PathAccessPolicy.create((root,), base_directory=tmp_path)

    roi = app._roi_from_uploaded_or_path(
        uploaded_payload=None,
        path_text=str(roi_path),
        path_policy=policy,
    )

    assert roi.label == "Burst"
    assert roi.bounds_arcsec["left"] == -10.0


def test_upload_bytes_take_priority_over_an_outside_path(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    policy = PathAccessPolicy.create((root,), base_directory=tmp_path)

    document = app._roi_import_document_from_uploaded_or_path(
        uploaded_payload=json.dumps(
            _source_map_payload(rois=[_lasso("Uploaded")])
        ).encode("utf-8"),
        path_text=str(tmp_path / "outside.json"),
        path_policy=policy,
    )

    assert document.choices[0].name == "Uploaded"


def test_legacy_single_roi_helper_refuses_to_silently_choose_multiple_regions(
    tmp_path: Path,
) -> None:
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    payload = json.dumps(_source_map_payload(rois=[_rectangle(), _lasso()])).encode(
        "utf-8"
    )

    with pytest.raises(ValueError, match="multiple regions"):
        app._roi_from_uploaded_or_path(
            uploaded_payload=payload,
            path_text="",
            path_policy=policy,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_source_map_payload(rois=[]), "contains no regions"),
        (
            {
                **_source_map_payload(rois=[_rectangle()]),
                "coordinate_system": "pixel",
            },
            "coordinate system",
        ),
        (
            {
                key: value
                for key, value in _source_map_payload(rois=[_rectangle()]).items()
                if key != "image_sha256"
            },
            "image_sha256",
        ),
        (
            _source_map_payload(rois=[_rectangle("Same"), _lasso("same")]),
            "unique",
        ),
        (
            _source_map_payload(
                rois=[
                    {
                        **_rectangle(),
                        "geometry": {"left": 1, "right": 1, "bottom": 0, "top": 2},
                    }
                ]
            ),
            "positive area",
        ),
        (
            _source_map_payload(
                rois=[
                    {
                        **_lasso(),
                        "geometry": {"points": [[0, 0], [1, 1], [0, 0]]},
                    }
                ]
            ),
            "three unique points",
        ),
    ],
)
def test_invalid_source_map_roi_documents_fail_closed(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        app._parse_roi_import_document(payload)


def test_invalid_json_does_not_mutate_existing_roi_state(tmp_path: Path) -> None:
    old = RadioRoi.from_box(1, 2, 3, 4, label="old")
    state = {
        "candidate_roi": old.to_json_dict(),
        "analysis_df": "cached",
        "export_artifacts": {"json": b"cached"},
    }
    before = dict(state)
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)

    with pytest.raises(ValueError, match="ROI JSON is invalid"):
        app._roi_import_document_from_uploaded_or_path(
            uploaded_payload=b"{not-json",
            path_text="",
            path_policy=policy,
        )

    assert state == before


def test_staging_a_changed_region_preserves_confirmation_and_clears_results() -> None:
    old = RadioRoi.from_box(1, 2, 3, 4, label="old")
    choice = app._parse_roi_import_document(
        _source_map_payload(rois=[_rectangle()])
    ).choices[0]
    st = SimpleNamespace(
        session_state={
            "confirmed_roi": old.to_json_dict(),
            "analysis_df": "cached",
            "analysis_signature": "old",
            "export_artifacts": {"json": b"cached"},
        }
    )

    assert app._stage_imported_roi(st, choice) is True
    assert app._session_roi(st, "candidate_roi").label == "Burst"
    assert app._session_roi(st, "confirmed_roi").label == "old"
    assert "analysis_df" not in st.session_state
    assert "analysis_signature" not in st.session_state
    assert "export_artifacts" not in st.session_state

    st.session_state["analysis_df"] = "new-cache"
    assert app._stage_imported_roi(st, choice) is False
    assert st.session_state["analysis_df"] == "new-cache"


def test_storing_multiple_choices_does_not_replace_roi_or_clear_results() -> None:
    old = RadioRoi.from_box(1, 2, 3, 4, label="old")
    document = app._parse_roi_import_document(
        _source_map_payload(rois=[_rectangle(visible=False), _lasso(visible=True)])
    )
    st = SimpleNamespace(
        session_state={
            "confirmed_roi": old.to_json_dict(),
            "analysis_df": "cached",
        }
    )

    app._store_roi_import_document(
        st,
        document,
        source_kind="upload",
        source_label="selection.roi-set.json",
        upload_signature="signature",
    )

    assert "candidate_roi" not in st.session_state
    assert app._session_roi(st, "confirmed_roi").label == "old"
    assert st.session_state["analysis_df"] == "cached"
    assert st.session_state["roi_import_selected_key"] == document.choices[1].key


class _UploadedJson:
    def __init__(self, payload: bytes, name: str = "selection.roi-set.json") -> None:
        self._payload = payload
        self.name = name

    def getvalue(self) -> bytes:
        return self._payload


class _ImportControlsStreamlit:
    def __init__(
        self,
        uploaded: _UploadedJson | None,
        *,
        state: dict[str, Any] | None = None,
        clicked: set[str] | None = None,
    ) -> None:
        self.uploaded = uploaded
        self.session_state = state or {}
        self.clicked = clicked or set()
        self.errors: list[str] = []

    def markdown(self, _value: str) -> None:
        return None

    def file_uploader(self, *_args: Any, **_kwargs: Any) -> _UploadedJson | None:
        return self.uploaded

    def button(self, label: str, **_kwargs: Any) -> bool:
        return label in self.clicked

    def caption(self, _value: str) -> None:
        return None

    def error(self, value: str) -> None:
        self.errors.append(value)

    def success(self, _value: str) -> None:
        return None

    def info(self, _value: str) -> None:
        return None

    def selectbox(
        self,
        _label: str,
        options: list[str],
        *,
        key: str,
        **_kwargs: Any,
    ) -> str:
        selected = str(self.session_state.get(key) or options[0])
        assert selected in options
        self.session_state[key] = selected
        return selected


def test_import_controls_stage_only_the_explicitly_selected_multiple_region(
    tmp_path: Path, monkeypatch
) -> None:
    old = RadioRoi.from_box(1, 2, 3, 4, label="old")
    payload = json.dumps(
        _source_map_payload(rois=[_rectangle(visible=False), _lasso(visible=True)])
    ).encode("utf-8")
    st = _ImportControlsStreamlit(
        _UploadedJson(payload),
        state={"confirmed_roi": old.to_json_dict(), "analysis_df": "cached"},
    )
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    monkeypatch.setattr(app, "render_native_path_input", lambda *_args, **_kwargs: "")

    app._render_roi_import_controls(st, policy, SimpleNamespace())

    document = st.session_state["roi_import_document"]
    assert st.session_state["roi_import_selected_key"] == document.choices[1].key
    assert "candidate_roi" not in st.session_state
    assert st.session_state["analysis_df"] == "cached"

    st.session_state["roi_import_selected_key"] = document.choices[0].key
    st.clicked.add("Use Selected Imported Region")
    app._render_roi_import_controls(st, policy, SimpleNamespace())

    assert app._session_roi(st, "candidate_roi").label == "Burst"
    assert "analysis_df" not in st.session_state


def test_import_controls_stage_one_region_and_leave_state_on_invalid_json(
    tmp_path: Path, monkeypatch
) -> None:
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    monkeypatch.setattr(app, "render_native_path_input", lambda *_args, **_kwargs: "")
    valid = _ImportControlsStreamlit(
        _UploadedJson(json.dumps(_source_map_payload(rois=[_lasso()])).encode("utf-8")),
        state={"analysis_df": "cached"},
    )

    app._render_roi_import_controls(valid, policy, SimpleNamespace())

    assert app._session_roi(valid, "candidate_roi").label == "Loop"
    assert "analysis_df" not in valid.session_state

    invalid_state = {
        "candidate_roi": valid.session_state["candidate_roi"],
        "analysis_df": "keep",
    }
    invalid = _ImportControlsStreamlit(_UploadedJson(b"{broken"), state=invalid_state)
    before = dict(invalid.session_state)

    app._render_roi_import_controls(invalid, policy, SimpleNamespace())

    assert invalid.errors and "ROI JSON is invalid" in invalid.errors[0]
    assert invalid.session_state == before


def test_composite_import_stages_single_roi_and_invalidates_sequence_results(
    tmp_path: Path, monkeypatch
) -> None:
    old = RadioRoi.from_box(1, 2, 3, 4, label="old")
    uploaded = _UploadedJson(
        json.dumps(_source_map_payload(rois=[_lasso("Saved loop")])).encode("utf-8")
    )
    st = _ImportControlsStreamlit(
        uploaded,
        state={
            "confirmed_roi": old.to_json_dict(),
            "composite_bundle": "cached-composite",
            "sequence_bundle": "cached-sequence",
        },
    )
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    monkeypatch.setattr(
        composite_app,
        "render_native_path_input",
        lambda *_args, **_kwargs: "",
    )

    composite_app._render_roi_import_controls(st, policy, SimpleNamespace())

    assert composite_app._session_roi(st, "candidate_roi").label == "Saved loop"
    assert "confirmed_roi" not in st.session_state
    assert "composite_bundle" not in st.session_state
    assert "sequence_bundle" not in st.session_state


def test_composite_import_requires_explicit_choice_for_multiple_regions(
    tmp_path: Path, monkeypatch
) -> None:
    payload = json.dumps(
        _source_map_payload(rois=[_rectangle(visible=False), _lasso(visible=True)])
    ).encode("utf-8")
    st = _ImportControlsStreamlit(_UploadedJson(payload))
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    monkeypatch.setattr(
        composite_app,
        "render_native_path_input",
        lambda *_args, **_kwargs: "",
    )

    composite_app._render_roi_import_controls(st, policy, SimpleNamespace())

    document = st.session_state["roi_import_document"]
    assert st.session_state["roi_import_selected_key"] == document.choices[1].key
    assert "candidate_roi" not in st.session_state

    st.session_state["roi_import_selected_key"] = document.choices[0].key
    st.clicked.add("Use Selected Imported Region")
    composite_app._render_roi_import_controls(st, policy, SimpleNamespace())

    assert composite_app._session_roi(st, "candidate_roi").label == "Burst"


def test_composite_invalid_import_preserves_current_valid_state(
    tmp_path: Path, monkeypatch
) -> None:
    current = RadioRoi.from_box(1, 2, 3, 4, label="keep")
    st = _ImportControlsStreamlit(
        _UploadedJson(b"{broken"),
        state={
            "candidate_roi": current.to_json_dict(),
            "composite_bundle": "keep-composite",
            "sequence_bundle": "keep-sequence",
        },
    )
    before = dict(st.session_state)
    policy = PathAccessPolicy.create((tmp_path,), base_directory=tmp_path)
    monkeypatch.setattr(
        composite_app,
        "render_native_path_input",
        lambda *_args, **_kwargs: "",
    )

    composite_app._render_roi_import_controls(st, policy, SimpleNamespace())

    assert st.errors and "ROI JSON is invalid" in st.errors[0]
    assert st.session_state == before


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _RoiRenderStreamlit:
    def __init__(self, roi: RadioRoi) -> None:
        self.session_state: dict[str, Any] = {"candidate_roi": roi.to_json_dict()}
        self.figures: list[Any] = []

    def subheader(self, _value: str) -> None:
        return None

    def columns(self, count: int | list[int]) -> list[_Context]:
        size = count if isinstance(count, int) else len(count)
        return [_Context() for _ in range(size)]

    def radio(self, *_args: Any, **_kwargs: Any) -> str:
        return "box"

    def caption(self, _value: str) -> None:
        return None

    def plotly_chart(self, figure: Any, **_kwargs: Any) -> None:
        self.figures.append(figure)
        return None

    def button(self, _label: str, **_kwargs: Any) -> bool:
        return False

    def info(self, _value: str) -> None:
        return None

    def success(self, _value: str) -> None:
        return None

    def json(self, _value: Any, **_kwargs: Any) -> None:
        return None


def test_imported_roi_is_projected_to_every_reference_panel(monkeypatch) -> None:
    imported = (
        app._parse_roi_import_document(_source_map_payload(rois=[_rectangle()]))
        .choices[0]
        .roi
    )
    st = _RoiRenderStreamlit(imported)
    captured: list[RadioRoi] = []
    monkeypatch.setattr(app, "_render_roi_import_controls", lambda *_args: None)

    def fake_figure(_reference: Any, _preview: Any, *, roi: RadioRoi, **_kwargs: Any):
        captured.append(roi)
        return object()

    monkeypatch.setattr(app, "_build_reference_figure_from_preview", fake_figure)
    monkeypatch.setattr(app, "selection_to_radio_roi", lambda *_args, **_kwargs: None)

    result = app._render_roi_step(
        st,
        references=[object(), object(), object()],
        previews=[object(), object(), object()],
        display_config={},
        path_policy=SimpleNamespace(),
        ui_store=SimpleNamespace(),
    )

    assert result is None
    assert len(captured) == 3
    assert all(roi is captured[0] for roi in captured)
    assert captured[0].label == "Burst"
