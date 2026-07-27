# SPDX-License-Identifier: GPL-3.0-only
"""Convert retained Radio Workspace action schemas into shared FunctionSpecs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from solar_apps.frontends.workbench.radio_workspace.catalog import (
    MODULES as RADIO_MODULES,
)

from .function_specs import ArtifactPortSpec, FunctionSpec, ParameterSpec

_FIELD_KINDS = {
    "checkbox": "boolean",
    "json": "object",
    "multiselect": "list",
    "number": "number",
    "select": "enum",
    "text": "string",
}


def radio_workspace_functions(
    existing: Iterable[FunctionSpec],
) -> tuple[FunctionSpec, ...]:
    """Return runnable legacy actions not already represented by a function/alias."""

    existing_items = tuple(existing)
    claimed = {
        identifier
        for function in existing_items
        for identifier in (function.function_id, *function.aliases)
    }
    result: list[FunctionSpec] = []
    for module in RADIO_MODULES:
        for action in module.actions:
            if not action.runnable or action.id in claimed:
                continue
            if action.id == "cso-legacy-mode":
                continue
            aliases = (
                ("cso-legacy-mode",)
                if action.id == "dynamic-spectrum-drift"
                else ()
            )
            parameters = tuple(
                _parameter(field, action)
                for field in action.input_schema
                if field.get("type") != "argv"
            )
            output_parameter = next(
                (
                    item.parameter_id
                    for item in parameters
                    if item.cli_flag == action.output_flag
                ),
                None,
            )
            inputs = (
                (
                    ArtifactPortSpec(
                        "input",
                        "Input artifacts",
                        tuple(action.accepts_artifacts),
                        required=False,
                        multiple=True,
                    ),
                )
                if action.accepts_artifacts
                else ()
            )
            outputs = (
                (
                    ArtifactPortSpec(
                        "output",
                        "Output artifacts",
                        tuple(action.produces_artifacts),
                        required=False,
                        multiple=True,
                    ),
                )
                if action.produces_artifacts
                else ()
            )
            result.append(
                FunctionSpec(
                    function_id=action.id,
                    title=action.title,
                    category=f"Radio · {module.title}",
                    description=action.description,
                    python_module=str(action.command_module),
                    parameters=parameters,
                    inputs=inputs,
                    outputs=outputs,
                    fixed_arguments=tuple(action.fixed_arguments),
                    output_flag=action.output_flag,
                    output_parameter=output_parameter,
                    default_output_name=action.output_filename,
                    config_json_flag=action.config_json_flag,
                    aliases=aliases,
                    page_templates=("radio-workspace", "workbench"),
                    base_config=dict(action.default_config),
                    required_any=tuple(action.run_required_any_fields),
                )
            )
            claimed.add(action.id)
            claimed.update(aliases)
    return tuple(result)


def _parameter(field: dict[str, Any], action: object) -> ParameterSpec:
    field_type = str(field.get("type", "text"))
    if field_type == "path":
        kind = str(field.get("path_kind") or "file")
    else:
        kind = _FIELD_KINDS.get(field_type, "string")
    name = str(field["name"])
    required_fields = set(getattr(action, "run_required_fields", ()))
    default = field.get("default")
    return ParameterSpec(
        parameter_id=name,
        label=str(field.get("label") or name.replace("_", " ").title()),
        kind=kind,  # type: ignore[arg-type]
        default=default,
        required=bool(field.get("required")) or name in required_fields,
        group=(
            "advanced"
            if bool(field.get("hidden"))
            or getattr(action, "risk_level", "standard") == "advanced"
            else "common"
        ),
        help_text=str(field.get("help") or ""),
        choices=tuple(field.get("choices") or ()),
        cli_flag=(
            None if field.get("cli_flag") in (None, "") else str(field["cli_flag"])
        ),
        config_path=(
            None
            if field.get("config_path") in (None, "")
            else str(field["config_path"])
        ),
        path_extensions=tuple(field.get("extensions") or ()),
        allow_empty=not (
            bool(field.get("required")) or name in required_fields
        ),
        item_kind="string" if field_type == "multiselect" else None,
    )


__all__ = ["radio_workspace_functions"]
