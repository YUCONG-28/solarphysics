"""Configuration boundaries exercised through the retained callable entrypoints."""

from __future__ import annotations

import copy
import sys
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


@pytest.mark.parametrize("process_argv", [False, True], ids=["explicit", "process"])
@pytest.mark.parametrize("drift_option", ["disable", "enable", "manual", "select"])
def test_pipeline_forwards_drift_options_without_mutating_shared_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process_argv: bool,
    drift_option: str,
) -> None:
    shared = (core.CONFIG, workflow.CONFIG, core.USER_CONFIG, workflow.USER_CONFIG)
    snapshots = copy.deepcopy(shared)
    cfg = copy.deepcopy(workflow.DEFAULT_CONFIG)
    cfg.update(
        {
            "output_dir": str(tmp_path),
            "enable_drift_rate_overlay": drift_option in {"disable", "select"},
            "drift_rate_mode": (
                "interactive_manual" if drift_option in {"disable", "select"} else "off"
            ),
            "drift_rate_interactive": {
                "port": 9999,
                "auto_open_browser": True,
                "launch_policy": "cli_only",
            },
        }
    )
    original_interactive = copy.deepcopy(cfg["drift_rate_interactive"])
    expected_request = copy.deepcopy(cfg)
    expected_request.update(
        {"enable_gaussian_overlay": True, "save_gaussian_diagnostics": True}
    )
    observed: list[tuple[str, dict]] = []

    def capture(config: dict, *, route: str = "source-map") -> None:
        observed.append((route, copy.deepcopy(config)))
        raise _ReachedSourceMap

    monkeypatch.setattr(core, "_workspace_source_map_selection", capture)
    monkeypatch.setattr(
        core,
        "_run_select_drift_workflow",
        lambda config: capture(config, route="select"),
    )
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

    selection_path = str(tmp_path / "requested-drift.json")
    drift_args = {
        "disable": ["--disable-drift"],
        "enable": ["--enable-drift"],
        "manual": ["--use-drift-selection", selection_path],
        "select": ["--select-drift"],
    }[drift_option]
    argv = [
        "--config",
        "requested-event",
        *drift_args,
        "--drift-port",
        "9998",
        "--no-drift-browser",
        "--drift-launch-policy",
        "always",
    ]
    # Explicit calls must use their own options even when the process differs.
    monkeypatch.setattr(
        sys, "argv", ["pipeline", *(argv if process_argv else ["--disable-drift"])]
    )
    with pytest.raises(_ReachedSourceMap):
        pipeline.run_pipeline(None if process_argv else argv)

    route, resolved = observed[0]
    assert route == ("select" if drift_option == "select" else "source-map")
    assert resolved["enable_drift_rate_overlay"] is (drift_option != "disable")
    assert (
        resolved["drift_rate_mode"]
        == {
            "disable": "off",
            "enable": "interactive_manual",
            "manual": "manual_json",
            "select": "interactive_manual",
        }[drift_option]
    )
    if drift_option == "manual":
        assert resolved["_drift_selection_cli_path"] == selection_path
    assert resolved["drift_rate_interactive"] == {
        "port": 9998,
        "auto_open_browser": False,
        "launch_policy": "always",
    }
    assert cfg["drift_rate_interactive"] == original_interactive
    assert cfg == expected_request
    current = (core.CONFIG, workflow.CONFIG, core.USER_CONFIG, workflow.USER_CONFIG)
    assert all(actual is original for actual, original in zip(current, shared))
    assert current == snapshots
