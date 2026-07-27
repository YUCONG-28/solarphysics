# SPDX-License-Identifier: GPL-3.0-only
"""Fail-closed migration of legacy argv/action records into typed functions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .contracts import JsonContract
from .function_specs import FunctionSpec, ParameterSpec


@dataclass(frozen=True, slots=True)
class ArgvMigrationReportV1(JsonContract):
    function_id: str
    parameters: dict[str, Any]
    unknown_arguments: tuple[str, ...] = ()
    schema_version: int = 1

    @property
    def runnable(self) -> bool:
        return not self.unknown_arguments


def migrate_argv(
    function: FunctionSpec,
    argv: tuple[str, ...] | list[str],
) -> ArgvMigrationReportV1:
    """Convert only declared flags; preserve unknown tokens in a blocking report."""

    by_positive = {
        flag: item
        for item in function.parameters
        for flag in ((item.cli_flag,) if item.cli_flag else ()) + item.cli_aliases
    }
    by_negative = {
        item.negative_cli_flag: item
        for item in function.parameters
        if item.negative_cli_flag is not None
    }
    positional = [item for item in function.parameters if item.positional]
    parameters: dict[str, Any] = {}
    unknown: list[str] = []
    items = list(map(str, argv))
    index = 0
    position = 0
    fixed = list(function.fixed_arguments)
    if items[: len(fixed)] == fixed:
        index = len(fixed)
    while index < len(items):
        token = items[index]
        if token in by_negative:
            parameters[by_negative[token].parameter_id] = False
            index += 1
            continue
        spec = by_positive.get(token)
        if spec is not None:
            if spec.kind == "boolean":
                parameters[spec.parameter_id] = True
                index += 1
                continue
            if index + 1 >= len(items):
                unknown.append(token)
                break
            raw = items[index + 1]
            try:
                parameters[spec.parameter_id] = _parse_value(spec, raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                unknown.extend((token, raw))
            index += 2
            continue
        if not token.startswith("-") and position < len(positional):
            spec = positional[position]
            try:
                parameters[spec.parameter_id] = _parse_value(spec, token)
            except (TypeError, ValueError, json.JSONDecodeError):
                unknown.append(token)
            position += 1
            index += 1
            continue
        unknown.append(token)
        index += 1
    if not unknown:
        parameters = function.normalize_parameters(parameters)
    return ArgvMigrationReportV1(
        function_id=function.function_id,
        parameters=parameters,
        unknown_arguments=tuple(unknown),
    )


def _parse_value(spec: ParameterSpec, raw: str) -> Any:
    if spec.kind == "integer":
        value: Any = int(raw)
    elif spec.kind == "number":
        value = float(raw)
    elif spec.kind == "list":
        value = (
            json.loads(raw)
            if spec.list_style == "json"
            else [item.strip() for item in raw.split(",") if item.strip()]
        )
    elif spec.kind in {"object", "roi"}:
        value = json.loads(raw)
    else:
        value = raw
    return spec.validate(value)


__all__ = ["ArgvMigrationReportV1", "migrate_argv"]
