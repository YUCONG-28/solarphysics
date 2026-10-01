"""Configuration boundaries exercised through the retained callable entrypoints."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from solar_apps.workflows.radio import pipeline_workflow as pipeline
from solar_apps.workflows.radio import source_map_workflow as workflow
from solar_apps.workflows.radio import _source_map_workflow_core as core


class _ReachedSourceMap(RuntimeError):
    pass


def test_pipeline_passes_event_config_without_rebinding_shared_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_core = core.CONFIG
    original_facade = workflow.CONFIG
    cfg = copy.deepcopy(workflow.DEFAULT_CONFIG)
    cfg.update({"output_dir": str(tmp_path), "event_token": "requested-event"})
    captured: list[dict] = []

    def capture(config: dict) -> None:
        captured.append(copy.deepcopy(config))
        raise _ReachedSourceMap

    monkeypatch.setattr(core, "_workspace_source_map_selection", capture)
    monkeypatch.setattr(pipeline, "_pd", lambda: None)
    monkeypatch.setattr(pipeline, "load_radio_user_config", lambda name: ({}, {}))
    for loader in (
        "load_radio_output_config",
        "load_newkirk_height_comparison_config",
        "load_drift_selection_product_config",
        "load_radio_diagnostic_presentation_config",
    ):
        monkeypatch.setattr(pipeline, loader, lambda name: {})
    monkeypatch.setattr(pipeline, "build_legacy_config", lambda *args: cfg)
    # Restore the old facade automatically even while reproducing the bug.
    monkeypatch.setattr(workflow, "CONFIG", original_facade)

    with pytest.raises(_ReachedSourceMap):
        pipeline.run_pipeline(["--config", "requested-event"])

    assert captured[0].get("event_token") == "requested-event"
    assert captured[0]["enable_gaussian_overlay"] is True
    assert workflow.CONFIG is original_facade
    assert core.CONFIG is original_core




@pytest.mark.parametrize("explicit", [False, True])
def test_request_options_leave_legacy_defaults_and_nested_values_unchanged(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    config = copy.deepcopy(workflow.DEFAULT_CONFIG)
    config["drift_rate_interactive"] = {"port": 9999, "auto_open_browser": True}
    original = copy.deepcopy(config)
    observed = []

    def capture(cfg):
        observed.append(copy.deepcopy(cfg))
        raise _ReachedSourceMap

    monkeypatch.setattr(workflow, "CONFIG", config)
    monkeypatch.setattr(core, "_workspace_source_map_selection", capture)
    argv = ["--drift-port", "9998", "--no-drift-browser"]
    with pytest.raises(_ReachedSourceMap):
        if explicit:
            workflow._run_source_map_config(config, argv=argv)
        else:
            workflow.main(argv=argv)

    assert observed[0]["drift_rate_interactive"] == {
        "port": 9998,
        "auto_open_browser": False,
    }
    assert config == original
    assert workflow.CONFIG is config


def test_event_requests_do_not_rebind_legacy_config_or_user_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_core = core.CONFIG
    original_facade = workflow.CONFIG
    original_user = core.USER_CONFIG
    snapshots = (copy.deepcopy(original_core), copy.deepcopy(original_user))
    seen = []

    def capture(cfg):
        seen.append(cfg["output_dir"])
        raise _ReachedSourceMap

    monkeypatch.setattr(core, "_workspace_source_map_selection", capture)
    monkeypatch.setattr(
        core, "load_script_config", lambda *args: copy.deepcopy(workflow.DEFAULT_CONFIG)
    )
    for name in ("first-event", "second-event"):
        with pytest.raises(_ReachedSourceMap):
            workflow.run_source_map({"output": {"output_dir": name}}, argv=[])

    assert seen == ["first-event", "second-event"]
    assert core.CONFIG is original_core
    assert workflow.CONFIG is original_facade
    assert core.USER_CONFIG is original_user
    assert (core.CONFIG, core.USER_CONFIG) == snapshots
