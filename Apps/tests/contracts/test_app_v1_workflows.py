"""Typed function, workflow, migration, and unique-base contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from solar_apps.frontends.app_v1.basic_services import (
    ArtifactExportService,
    RoiController,
    validate_allowed_path,
    validate_roi,
)
from solar_apps.frontends.app_v1.flow_store import AppV1FlowStore
from solar_apps.frontends.app_v1.flows import (
    AppV1FlowV1,
    FlowEdgeV1,
    FlowNodeV1,
)
from solar_apps.frontends.app_v1.function_catalog import (
    DEFAULT_FUNCTION_CATALOG,
    PAGE_TEMPLATE_FUNCTIONS,
    page_template,
)
from solar_apps.frontends.app_v1.function_specs import (
    ArtifactPortSpec,
    FunctionSpec,
    INFRASTRUCTURE_FLAGS,
    ParameterSpec,
)
from solar_apps.frontends.app_v1.migration import migrate_argv
from solar_apps.frontends.app_v1.plot_specs import PlotSpec
from solar_apps.frontends.app_v1.runtime import AppV1RuntimePaths

APP_V1 = Path(__file__).resolve().parents[2] / "solar_apps/frontends/app_v1"


def test_parameter_schema_builds_deterministic_argv_and_rejects_unknowns() -> None:
    function = FunctionSpec(
        "example-function",
        "Example",
        "Data",
        "Typed example.",
        "example.worker",
        parameters=(
            ParameterSpec(
                "count",
                "Count",
                "integer",
                default=2,
                minimum=1,
                maximum=4,
                cli_flag="--count",
            ),
            ParameterSpec(
                "enabled",
                "Enabled",
                "boolean",
                default=True,
                cli_flag="--enabled",
                negative_cli_flag="--no-enabled",
            ),
        ),
    )

    module, argv, values = function.build_arguments({"count": 4, "enabled": False})

    assert module == "example.worker"
    assert argv == ("--count", "4", "--no-enabled")
    assert values == {"count": 4, "enabled": False}
    with pytest.raises(ValueError, match="Unknown parameters"):
        function.build_arguments({"raw_arguments": "--unsafe"})
    with pytest.raises(ValueError, match="at most"):
        function.build_arguments({"count": 5})


def test_infrastructure_flags_can_never_be_exposed_as_business_parameters() -> None:
    for flag in INFRASTRUCTURE_FLAGS:
        with pytest.raises(ValueError, match="Infrastructure"):
            ParameterSpec("unsafe", "Unsafe", "string", cli_flag=flag)
    exposed = {
        parameter.cli_flag
        for function in DEFAULT_FUNCTION_CATALOG.functions
        for parameter in function.parameters
        if parameter.cli_flag
    }
    assert not exposed.intersection(INFRASTRUCTURE_FLAGS)


def test_all_eleven_pages_have_schema_one_editable_flow_templates() -> None:
    assert len(PAGE_TEMPLATE_FUNCTIONS) == 11
    for module_id in PAGE_TEMPLATE_FUNCTIONS:
        flow = page_template(module_id)
        assert flow.schema_version == 1
        assert flow.nodes
        assert all(
            DEFAULT_FUNCTION_CATALOG.get(node.function_id) for node in flow.nodes
        )


def test_data_download_template_connects_search_to_download() -> None:
    flow = page_template("data-download")

    assert [node.function_id for node in flow.nodes] == [
        "observation-search",
        "observation-download",
    ]
    assert flow.edges == (
        FlowEdgeV1("node-1", "records", "node-2", "records"),
    )


def test_radio_workspace_actions_migrate_without_freeform_arguments() -> None:
    from solar_apps.frontends.workbench.radio_workspace.catalog import MODULES

    runnable = {
        action.id for module in MODULES for action in module.actions if action.runnable
    }
    for action_id in runnable:
        function = DEFAULT_FUNCTION_CATALOG.get(action_id)
        assert all(item.parameter_id != "arguments" for item in function.parameters)
    assert DEFAULT_FUNCTION_CATALOG.resolve_id("cso-legacy-mode") == (
        "dynamic-spectrum-drift"
    )


def test_dag_rejects_cycles_and_incompatible_artifact_types() -> None:
    with pytest.raises(ValueError, match="cycle"):
        AppV1FlowV1(
            "cycle-flow",
            "Cycle",
            nodes=(
                FlowNodeV1("one", "artifact-input"),
                FlowNodeV1("two", "image-discover"),
            ),
            edges=(
                FlowEdgeV1("one", "artifact", "two", "input"),
                FlowEdgeV1("two", "images", "one", "input"),
            ),
        )
    flow = page_template("radio-composite")
    bad = replace(
        flow,
        edges=(FlowEdgeV1("node-1", "metadata", "node-3", "maps"),),
    )
    with pytest.raises(ValueError, match="Incompatible"):
        DEFAULT_FUNCTION_CATALOG.validate_flow(bad)


def test_flow_store_round_trips_schema_one(tmp_path: Path) -> None:
    runtime = AppV1RuntimePaths(
        tmp_path / "state",
        tmp_path / "workspaces",
        tmp_path / "outputs",
        tmp_path / "logs",
        tmp_path / "tmp",
    )
    store = AppV1FlowStore(runtime)
    flow = AppV1FlowV1(
        "round-trip",
        "Round Trip",
        nodes=(
            FlowNodeV1(
                "input",
                "artifact-input",
                {"path": "/data/example.png", "artifact_type": "image"},
            ),
        ),
        concurrency=4,
    )

    target = store.save(flow)
    restored = store.load("round-trip")

    assert target.name == "round-trip.spflow.json"
    assert restored.to_dict() == flow.to_dict()


def test_legacy_argv_migration_blocks_unknown_arguments() -> None:
    function = DEFAULT_FUNCTION_CATALOG.get("image-discover")
    migrated = migrate_argv(
        function,
        ["--input-dir", "/data/images", "--recursive"],
    )
    blocked = migrate_argv(
        function,
        ["--input-dir", "/data/images", "--mystery", "value"],
    )

    assert migrated.runnable
    assert migrated.parameters["recursive"] is True
    assert not blocked.runnable
    assert blocked.unknown_arguments == ("--mystery", "value")


def test_scientific_variant_changes_algorithm_and_round_trips_selection() -> None:
    function = DEFAULT_FUNCTION_CATALOG.get("center-extraction")
    family = function.variant_family
    assert family is not None
    assert family.primary.variant_id == "weighted"

    primary_module, primary_args, _values = function.build_arguments(
        {"radio_dir": "/data/radio"},
        default_output="/outputs/run",
        allowed_roots=("/data", "/outputs"),
    )
    alternate_module, alternate_args, _values = function.build_arguments(
        {"radio_dir": "/data/radio"},
        variant_id="geometric",
        default_output="/outputs/run",
        allowed_roots=("/data", "/outputs"),
    )
    node = FlowNodeV1(
        "centers",
        function.function_id,
        {"radio_dir": "/data/radio"},
        variant_id="geometric",
    )

    assert primary_module == alternate_module
    assert ("--centroid", "weighted") == primary_args[:2]
    assert ("--centroid", "geometric") == alternate_args[:2]
    assert node.to_dict()["variant_id"] == "geometric"


def test_aia_business_parser_flags_are_covered_by_parameter_schema() -> None:
    from solar_apps.workflows.aia.application import build_parser

    parser_flags = {
        flag
        for action in build_parser()._actions
        for flag in action.option_strings
        if flag.startswith("--") and flag != "--help"
    }
    function = DEFAULT_FUNCTION_CATALOG.get("aia-process")
    schema_flags = {
        flag
        for parameter in function.parameters
        for flag in (
            parameter.cli_flag,
            parameter.negative_cli_flag,
            *parameter.cli_aliases,
        )
        if flag
    }
    assert parser_flags == schema_flags


def test_unique_foundation_contracts_and_plot_parameters(tmp_path: Path) -> None:
    class_names: dict[str, list[str]] = {}
    for path in APP_V1.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_names.setdefault(node.name, []).append(path.name)
    for name in (
        "AllowedPathField",
        "ArtifactBrowser",
        "ArtifactExportService",
        "PlaybackController",
        "RoiController",
        "SchemaForm",
        "ScientificImageCanvas",
        "ScientificPlotRenderer",
    ):
        assert len(class_names.get(name, ())) == 1

    plot = PlotSpec(
        "spectrum",
        cmap="magma",
        vmin=-5,
        vmax=20,
        width_inches=10,
        height_inches=4,
        dpi=300,
        grid=True,
    )
    assert plot.to_dict()["dpi"] == 300
    assert ArtifactExportService.SUPPORTED_FORMATS == {
        "png",
        "csv",
        "json",
        "gif",
        "mp4",
        "webm",
        "zip",
    }
    root = tmp_path / "allowed"
    root.mkdir()
    assert validate_allowed_path(root / "result.png", (root,)).is_relative_to(root)
    with pytest.raises(ValueError, match="outside"):
        validate_allowed_path(tmp_path / "outside.png", (root,))
    roi = validate_roi(
        {
            "type": "rectangle",
            "geometry": {"left": 0, "right": 2, "top": 0, "bottom": 3},
        }
    )
    assert roi["schema_version"] == 1
    assert roi["name"] == "Rectangle ROI"
    assert roi["style"] == {
        "color": "#00d4ff",
        "line_width": 3.0,
        "show_label": True,
    }
    controller = RoiController()
    first = controller.add(
        {
            "type": "rectangle",
            "geometry": {"left": 0, "right": 2, "top": 0, "bottom": 3},
        }
    )
    second = controller.add(
        {
            "type": "lasso",
            "geometry": {"points": [[0, 0], [2, 0], [1, 2]]},
        }
    )
    assert [first["name"], second["name"]] == ["ROI 1", "ROI 2"]


def test_ui_process_modules_do_not_import_heavy_plot_or_legacy_frontends() -> None:
    forbidden = {"matplotlib", "flask", "streamlit", "PySide6"}
    offenders: list[tuple[str, str]] = []
    for name in ("window.py", "workflow_builder.py", "schema_form.py"):
        tree = ast.parse((APP_V1 / name).read_text(encoding="utf-8"))
        imports = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        offenders.extend((name, item) for item in sorted(imports & forbidden))
    assert offenders == []


def test_artifact_port_parameter_binding_round_trip() -> None:
    port = ArtifactPortSpec(
        "images",
        "Images",
        ("image",),
        parameter_id="input_dir",
    )
    assert port.to_dict()["parameter_id"] == "input_dir"
