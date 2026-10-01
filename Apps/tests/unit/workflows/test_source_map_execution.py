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


class _ReachedPipelineDrift(RuntimeError):
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
    original_request = copy.deepcopy(cfg)
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
    assert cfg == original_request
    current = (core.CONFIG, workflow.CONFIG, core.USER_CONFIG, workflow.USER_CONFIG)
    assert all(actual is original for actual, original in zip(current, shared))
    assert current == snapshots


@pytest.mark.parametrize("process_argv", [False, True], ids=["explicit", "process"])
@pytest.mark.parametrize("drift_option", ["disable", "enable", "manual"])
@pytest.mark.parametrize("no_browser", [False, True], ids=["browser", "no-browser"])
def test_pipeline_stages_share_cli_config_and_leave_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process_argv: bool,
    drift_option: str,
    no_browser: bool,
) -> None:
    import pandas as pd

    from solar_toolkit.radio import spectrogram

    shared = (
        core.CONFIG,
        workflow.CONFIG,
        core.USER_CONFIG,
        workflow.USER_CONFIG,
        core.DEFAULT_CONFIG,
        workflow.DEFAULT_CONFIG,
    )
    snapshots = copy.deepcopy(shared)
    cfg = copy.deepcopy(workflow.DEFAULT_CONFIG)
    cfg.update(
        {
            "output_dir": str(tmp_path),
            "analysis_subdir": "products",
            "enable_drift_rate_overlay": drift_option == "disable",
            "drift_rate_mode": (
                "interactive_manual" if drift_option == "disable" else "off"
            ),
            "drift_rate_interactive": {
                "port": 9999,
                "auto_open_browser": True,
                "launch_policy": "cli_only",
            },
        }
    )
    original_request = copy.deepcopy(cfg)
    source_config: list[dict] = []
    provenance_config: list[dict] = []
    downstream_config: list[dict] = []
    run_source_map = workflow._run_source_map_config
    load_drift = pipeline._load_or_create_drift_diagnostics

    def capture_source(config: dict) -> None:
        source_config.append(copy.deepcopy(config))
        raise _ReachedSourceMap

    def stop_source_before_science(config: dict, *, argv) -> None:
        try:
            run_source_map(config, argv=argv)
        except _ReachedSourceMap:
            pass

    def capture_provenance(output_dir, config, **kwargs):
        provenance_config.append(copy.deepcopy(config))
        return output_dir / "radio_run_provenance.json"

    def reject_spectrogram_build(config):
        pytest.fail("--disable-drift must not build a spectrogram cache")

    def capture_drift(config, products, *, return_cache):
        downstream_config.append(copy.deepcopy(config))
        if drift_option == "disable":
            drift_df, cache = load_drift(config, products, return_cache=return_cache)
            assert drift_df.empty
            assert cache is None
        raise _ReachedPipelineDrift

    monkeypatch.setattr(core, "_workspace_source_map_selection", capture_source)
    monkeypatch.setattr(workflow, "_run_source_map_config", stop_source_before_science)
    monkeypatch.setattr(pipeline, "write_radio_provenance", capture_provenance)
    monkeypatch.setattr(pipeline, "_load_or_create_drift_diagnostics", capture_drift)
    monkeypatch.setattr(
        spectrogram, "build_spectrogram_cache", reject_spectrogram_build
    )
    monkeypatch.setattr(
        pipeline, "load_radio_user_config", lambda name: ({}, {"enabled": False})
    )
    for loader in (
        "load_radio_output_config",
        "load_newkirk_height_comparison_config",
        "load_drift_selection_product_config",
        "load_radio_diagnostic_presentation_config",
    ):
        monkeypatch.setattr(pipeline, loader, lambda name: {})
    monkeypatch.setattr(pipeline, "build_legacy_config", lambda *args: cfg)

    analysis_dir = tmp_path / "products"
    analysis_dir.mkdir()
    pd.DataFrame(
        {
            "quality_flag": ["ok"],
            "overlay_valid": [True],
            "trajectory_valid": [True],
        }
    ).to_csv(analysis_dir / cfg["gaussian_diagnostics_csv"], index=False)
    selection_path = str(tmp_path / "requested-drift.json")
    drift_args = {
        "disable": ["--disable-drift"],
        "enable": ["--enable-drift"],
        "manual": ["--use-drift-selection", selection_path],
    }[drift_option]
    argv = [
        "--config",
        "requested-event",
        *drift_args,
        "--drift-port",
        "9998",
        "--drift-launch-policy",
        "always",
    ]
    if no_browser:
        argv.append("--no-drift-browser")
    monkeypatch.setattr(
        sys, "argv", ["pipeline", *(argv if process_argv else ["--disable-drift"])]
    )
    with pytest.raises(_ReachedPipelineDrift):
        pipeline.run_pipeline(None if process_argv else argv)

    for stage, captured in (
        ("source-map", source_config),
        ("provenance", provenance_config),
        ("downstream", downstream_config),
    ):
        assert len(captured) == 1, stage
        resolved = captured[0]
        assert resolved["enable_drift_rate_overlay"] is (
            drift_option != "disable"
        ), stage
        assert (
            resolved["drift_rate_mode"]
            == {
                "disable": "off",
                "enable": "interactive_manual",
                "manual": "manual_json",
            }[drift_option]
        ), stage
        if drift_option == "manual":
            assert resolved["_drift_selection_cli_path"] == selection_path, stage
        assert resolved["drift_rate_interactive"] == {
            "port": 9998,
            "auto_open_browser": not no_browser,
            "launch_policy": "always",
        }, stage
    assert source_config[0] == provenance_config[0] == downstream_config[0]
    assert cfg == original_request
    current = (
        core.CONFIG,
        workflow.CONFIG,
        core.USER_CONFIG,
        workflow.USER_CONFIG,
        core.DEFAULT_CONFIG,
        workflow.DEFAULT_CONFIG,
    )
    assert all(actual is original for actual, original in zip(current, shared))
    assert current == snapshots


def test_source_map_cli_application_is_idempotent_and_request_owned(
    tmp_path: Path,
) -> None:
    shared = (
        core.CONFIG,
        workflow.CONFIG,
        core.DEFAULT_CONFIG,
        workflow.DEFAULT_CONFIG,
    )
    snapshots = copy.deepcopy(shared)
    config = copy.deepcopy(workflow.DEFAULT_CONFIG)
    selection_path = str(tmp_path / "requested-drift.json")
    args = workflow._parse_source_map_args(
        [
            "--disable-drift",
            "--enable-drift",
            "--use-drift-selection",
            selection_path,
            "--drift-port",
            "9998",
            "--no-drift-browser",
            "--drift-launch-policy",
            "always",
        ]
    )
    workflow._apply_source_map_cli_args(config, args)
    first_application = copy.deepcopy(config)
    workflow._apply_source_map_cli_args(config, args)

    assert config == first_application
    assert config["enable_drift_rate_overlay"] is True
    assert config["drift_rate_mode"] == "manual_json"
    assert config["_drift_selection_cli_path"] == selection_path
    assert config["drift_rate_interactive"]["port"] == 9998
    assert config["drift_rate_interactive"]["auto_open_browser"] is False
    assert config["drift_rate_interactive"]["launch_policy"] == "always"
    current = (
        core.CONFIG,
        workflow.CONFIG,
        core.DEFAULT_CONFIG,
        workflow.DEFAULT_CONFIG,
    )
    assert all(actual is original for actual, original in zip(current, shared))
    assert current == snapshots
