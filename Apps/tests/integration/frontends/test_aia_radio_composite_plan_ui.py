"""The visible preview and export buttons retain one image match plan."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image
from solar_toolkit.time import (
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
)
from streamlit.testing.v1 import AppTest

from solar_apps.frontends.radio.aia_radio_composite import app as composite_app
from solar_apps.frontends.radio.aia_radio_composite import application, rendering
from solar_apps.frontends.radio.aia_radio_composite.models import SpectrumWindow
from solar_apps.frontends.radio.aia_radio_composite.rendering import TopPanelArtifact


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_preview_and_video_share_plan_until_explicit_refresh(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(data_root))
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    paths = {
        key: data_root / key
        for key in ("aia_directory", "radio_directory", "spectrum_path")
    }
    for path in paths.values():
        path.mkdir()
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "black").save(image, format="PNG")
    plans, previews, composites, exports = [], [], [], []
    changed = False

    def plan_factory(request, **kwargs):
        frame = TimeFrame(str(len(plans)), normalize_time(kwargs["time_start"]))
        plan = build_sequence_match_plan([TimeStream("aia:171", [frame], 12)])
        plans.append((plan, kwargs))
        return plan

    def preview(request, **kwargs):
        previews.append(kwargs)
        return TopPanelArtifact(
            image.getvalue(),
            {"reference_radio_time_utc": kwargs["match_plan"].rows[0].time.utc},
        )

    def export(result, request, **kwargs):
        exports.append(kwargs)
        if changed:
            raise RuntimeError(
                "Image inputs changed; rebuild the preview before export"
            )
        return SimpleNamespace(video_mp4=b"video", metadata_json=b"{}")

    def composite(request, roi, **kwargs):
        composites.append(kwargs)
        return SimpleNamespace(to_metadata_dict=lambda: {}), None

    monkeypatch.setattr(application, "build_composite_match_plan", plan_factory)
    monkeypatch.setattr(application, "build_top_panel", preview)
    monkeypatch.setattr(application, "build_dynamic_composite_video", export)
    monkeypatch.setattr(application, "build_composite", composite)
    monkeypatch.setattr(
        rendering,
        "build_top_panel_selection_figure",
        lambda *_args, **_kwargs: go.Figure(),
    )
    monkeypatch.setattr(
        rendering,
        "build_spectrum_selection_figure",
        lambda *_args, **_kwargs: go.Figure(),
    )
    app = AppTest.from_file(str(Path(composite_app.__file__)), default_timeout=40).run()
    for key, path in paths.items():
        app.text_input(key=key).set_value(str(path))
    start, end = "2025-01-24T04:48:30.000000001Z", "2025-01-24T04:48:30.000000004Z"
    app.text_input(key="aia_time").set_value(start)
    for prefix in ("image", "flux", "spectrum"):
        app.text_input(key=f"{prefix}_time_start").set_value(start)
        app.text_input(key=f"{prefix}_time_end").set_value(end)
    app.run()
    _button(app, "Build top panel").click().run()
    assert not app.exception
    assert any("No radio image is available" in notice.value for notice in app.info)
    plan, settings = plans[0]
    assert settings["time_start"].nanosecond == 1
    assert settings["time_end"].nanosecond == 4
    assert previews[0]["match_plan"] is plan
    assert (
        previews[0]["aia_max_dt_seconds"]
        == settings["aia_match_tolerance_seconds"]
        == 12
    )
    assert (
        previews[0]["radio_max_dt_seconds"]
        == settings["radio_match_tolerance_seconds"]
        == 0.1
    )
    app.session_state["aia_radio_confirmed_roi"] = (
        composite_app._placeholder_roi().to_json_dict()
    )
    app.session_state["aia_radio_spectrum"] = SpectrumWindow(
        data=np.ones((8, 2)),
        # Both edge bands need their full +/-1 MHz extent inside the spectrum.
        frequency_mhz=np.array([148, 149, 164, 190, 205, 223, 238, 239]),
        time_utc=(settings["time_start"], settings["time_end"]),
        polarization="RR",
        unit="sfu",
        source="CSO",
    )
    app.run()
    assert not _button(app, "Generate synchronized composite").disabled, {
        "warnings": [notice.value for notice in app.warning],
        "errors": [notice.value for notice in app.error],
        "infos": [notice.value for notice in app.info],
        "roi": app.session_state["aia_radio_confirmed_roi"],
        "curve_frequencies": app.multiselect(key="curve_frequencies").value,
    }
    _button(app, "Generate synchronized composite").click().run()
    assert not app.exception
    assert composites[0]["match_plan"] is plan
    assert composites[0]["match_time_range_utc"] == (
        settings["time_start"],
        settings["time_end"],
    )
    _button(app, "Generate MP4 video").click().run()
    assert not app.exception
    assert exports[0]["match_plan"] is plan
    assert exports[0]["time_start"] == settings["time_start"]
    assert exports[0]["time_end"] == settings["time_end"]
    assert exports[0]["display_time_range_utc"] == (
        settings["time_start"],
        settings["time_end"],
    )
    assert len(plans) == 1
    changed = True
    _button(app, "Generate MP4 video").click().run()
    assert not app.exception
    assert exports[1]["match_plan"] is plan
    assert any("rebuild the preview" in error.value for error in app.error)
    assert "aia_radio_video_artifact" not in app.session_state
    app.text_input(key="flux_time_end").set_value(
        "2025-01-24T04:48:30.000000005Z"
    ).run()
    app.text_input(key="spectrum_time_end").set_value(
        "2025-01-24T04:48:30.000000009Z"
    ).run()
    assert app.session_state["aia_radio_match_plan"] is plan
    assert app.session_state["aia_radio_top"] is not None
    assert len(plans) == 1
    assert app.text_input(key="image_time_end").value == end
    assert "aia_radio_result" not in app.session_state
    app.text_input(key="image_time_end").set_value(
        "2025-01-24T04:48:30.000000005Z"
    ).run()
    assert "aia_radio_match_plan" not in app.session_state
    assert "aia_radio_top" not in app.session_state
    assert "aia_radio_video_artifact" not in app.session_state
    _button(app, "Build top panel").click().run()
    assert not app.exception
    assert len(plans) == 2
    assert plans[1][1]["time_end"].nanosecond == 5
    assert app.session_state["aia_radio_match_plan"] is plans[1][0]


def _empty_event_app(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(data_root))
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    paths = {
        key: data_root / key
        for key in ("aia_directory", "radio_directory", "spectrum_path")
    }
    for path in paths.values():
        path.mkdir()
    app = AppTest.from_file(str(Path(composite_app.__file__)), default_timeout=40).run()
    for key, path in paths.items():
        app.text_input(key=key).set_value(str(path))
    app.text_input(key="aia_time").set_value("2025-01-24T04:48:30Z")
    return app


def test_image_window_defaults_once_from_saved_event_window(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    app = AppTest.from_file(str(Path(composite_app.__file__)), default_timeout=40)
    app.session_state["flux_time_start"] = "2025-01-24T04:47:00.000000001Z"
    app.session_state["flux_time_end"] = "2025-01-24T04:50:00.000000002Z"
    app.run()
    assert (
        app.text_input(key="image_time_start").value == "2025-01-24T04:47:00.000000001Z"
    )
    assert (
        app.text_input(key="image_time_end").value == "2025-01-24T04:50:00.000000002Z"
    )
    app.text_input(key="spectrum_time_start").set_value("2025-01-24T04:46:00Z")
    app.text_input(key="flux_time_end").set_value("2025-01-24T04:51:00Z").run()
    assert (
        app.text_input(key="image_time_start").value == "2025-01-24T04:47:00.000000001Z"
    )
    assert (
        app.text_input(key="image_time_end").value == "2025-01-24T04:50:00.000000002Z"
    )


@pytest.mark.parametrize(
    "end",
    ["2025-01-24T04:48:30.000000001Z", "2025-01-24T04:48:30Z", "invalid", "", "NaT"],
)
def test_invalid_image_window_disables_matching_before_loading(
    tmp_path, monkeypatch, end
):
    app = _empty_event_app(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        application,
        "build_composite_match_plan",
        lambda *_args, **_kwargs: calls.append(True),
    )
    app.text_input(key="image_time_start").set_value("2025-01-24T04:48:30.000000001Z")
    app.text_input(key="image_time_end").set_value(end).run()
    assert not app.exception
    assert _button(app, "Build top panel").disabled
    assert any("Correct the image UTC window" in notice.value for notice in app.warning)
    assert calls == []
    assert "aia_radio_match_plan" not in app.session_state


def test_valid_image_window_with_no_observations_has_no_export_plan(
    tmp_path, monkeypatch
):
    app = _empty_event_app(tmp_path, monkeypatch)
    app.text_input(key="image_time_start").set_value("2025-01-24T04:48:30.000000001Z")
    app.text_input(key="image_time_end").set_value(
        "2025-01-24T04:48:30.000000002Z"
    ).run()
    assert not _button(app, "Build top panel").disabled, [
        notice.value for notice in app.warning
    ]
    _button(app, "Build top panel").click().run()
    assert not app.exception
    assert any("No valid image observations" in error.value for error in app.error)
    assert "aia_radio_match_plan" not in app.session_state
    assert "aia_radio_top" not in app.session_state
    assert "aia_radio_video_artifact" not in app.session_state
    assert all(button.label != "Generate MP4 video" for button in app.button)


def test_control_window_keeps_offset_and_nanosecond_boundary():
    start, end = composite_app._validated_utc_range(
        "2025-01-24T12:48:30.000000001+08:00",
        "2025-01-24T04:48:30.000000002Z",
        label="Image window",
    )
    assert start.hour == 4
    assert end.value - start.value == 1
    with pytest.raises(ValueError, match="start must be before end"):
        composite_app._validated_utc_range(end, start, label="Image window")
