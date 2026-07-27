# SPDX-License-Identifier: GPL-3.0-only
"""Typed atomic-function contracts for App 1.0 workflows."""

from __future__ import annotations

import json
import math
import os
import re
import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .contracts import JsonContract, validate_identifier

ParameterKind = Literal[
    "boolean",
    "integer",
    "number",
    "string",
    "enum",
    "time",
    "directory",
    "file",
    "list",
    "object",
    "roi",
    "color",
]
ParameterGroup = Literal["common", "advanced"]
_PARAMETER_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_PARAMETER_KINDS = frozenset(
    {
        "boolean",
        "integer",
        "number",
        "string",
        "enum",
        "time",
        "directory",
        "file",
        "list",
        "object",
        "roi",
        "color",
    }
)
_PORT_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_INFRASTRUCTURE_FLAGS = frozenset(
    {
        "--allowed-roots",
        "--port",
        "--browser",
        "--no-browser",
        "--open-browser",
        "--auto-stop",
        "--no-auto-stop",
        "--auto-stop-idle-sec",
        "--dry-run",
        "--settings-file",
        "--reset-settings",
    }
)


def _finite_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


@dataclass(frozen=True, slots=True)
class ParameterSpec(JsonContract):
    """One user-adjustable business parameter and its executor mapping."""

    parameter_id: str
    label: str
    kind: ParameterKind
    default: Any = None
    required: bool = False
    group: ParameterGroup = "common"
    help_text: str = ""
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    cli_flag: str | None = None
    cli_aliases: tuple[str, ...] = ()
    negative_cli_flag: str | None = None
    positional: bool = False
    config_path: str | None = None
    path_extensions: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    allow_empty: bool = True
    item_kind: str | None = None
    list_style: Literal["comma", "repeat", "json"] = "comma"

    def __post_init__(self) -> None:
        if not _PARAMETER_ID.fullmatch(self.parameter_id):
            raise ValueError("parameter_id must be lowercase snake_case")
        if not self.label.strip():
            raise ValueError("Parameter label is required")
        if self.kind not in _PARAMETER_KINDS:
            raise ValueError(f"Unsupported parameter kind: {self.kind}")
        if self.group not in {"common", "advanced"}:
            raise ValueError(f"Unsupported parameter group: {self.group}")
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("Parameter minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("Parameter maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum < self.minimum
        ):
            raise ValueError("Parameter maximum cannot be below minimum")
        if self.kind == "enum" and not self.choices:
            raise ValueError("Enum parameters require choices")
        if self.positional and self.cli_flag is not None:
            raise ValueError("A positional parameter cannot also have cli_flag")
        if self.list_style not in {"comma", "repeat", "json"}:
            raise ValueError(f"Unsupported list style: {self.list_style}")
        if self.item_kind is not None and (
            self.kind != "list"
            or self.item_kind not in {"integer", "number", "string"}
        ):
            raise ValueError("item_kind is supported only for typed list parameters")
        for flag in (self.cli_flag, self.negative_cli_flag, *self.cli_aliases):
            if flag in _INFRASTRUCTURE_FLAGS:
                raise ValueError(f"Infrastructure flag cannot be user-adjustable: {flag}")
            if flag is not None and not flag.startswith("--"):
                raise ValueError(f"CLI flag must start with '--': {flag}")
        object.__setattr__(
            self,
            "default",
            _finite_json(self.default, label=f"{self.parameter_id}.default"),
        )
        object.__setattr__(
            self,
            "choices",
            tuple(
                _finite_json(item, label=f"{self.parameter_id}.choices")
                for item in self.choices
            ),
        )

    def validate(self, value: Any) -> Any:
        """Return one normalized parameter value or raise a precise error."""

        if value in (None, ""):
            if self.required and not self.allow_empty:
                raise ValueError(f"{self.label} is required")
            return None if value is None else ""
        if self.kind == "boolean":
            if not isinstance(value, bool):
                raise TypeError(f"{self.label} must be true or false")
            normalized: Any = value
        elif self.kind == "integer":
            if isinstance(value, bool):
                raise TypeError(f"{self.label} must be an integer")
            normalized = int(value)
        elif self.kind == "number":
            if isinstance(value, bool):
                raise TypeError(f"{self.label} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{self.label} must be finite")
        elif self.kind == "list":
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{self.label} must be a list")
            normalized = list(value)
            if self.item_kind:
                normalized = [
                    _normalize_list_item(self.item_kind, item, label=self.label)
                    for item in normalized
                ]
        elif self.kind in {"object", "roi"}:
            if not isinstance(value, Mapping):
                raise TypeError(f"{self.label} must be an object")
            normalized = dict(value)
        else:
            normalized = str(value)
        if self.choices and normalized not in self.choices:
            raise ValueError(
                f"{self.label} must be one of: "
                + ", ".join(map(str, self.choices))
            )
        if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
            if self.minimum is not None and normalized < self.minimum:
                raise ValueError(f"{self.label} must be at least {self.minimum:g}")
            if self.maximum is not None and normalized > self.maximum:
                raise ValueError(f"{self.label} must be at most {self.maximum:g}")
        return _finite_json(normalized, label=self.label)


@dataclass(frozen=True, slots=True)
class ArtifactPortSpec(JsonContract):
    """One typed input or output port on an atomic function."""

    port_id: str
    label: str
    artifact_types: tuple[str, ...]
    required: bool = True
    multiple: bool = False
    parameter_id: str | None = None

    def __post_init__(self) -> None:
        if not _PORT_ID.fullmatch(self.port_id):
            raise ValueError("port_id must be lowercase kebab/snake compatible")
        if not self.label.strip() or not self.artifact_types:
            raise ValueError("Artifact port label and types are required")
        if self.parameter_id is not None and not _PARAMETER_ID.fullmatch(
            self.parameter_id
        ):
            raise ValueError("Artifact port parameter_id must be lowercase snake_case")
        object.__setattr__(
            self,
            "artifact_types",
            tuple(
                item
                if item == "*"
                else validate_identifier(item, label="artifact_type")
                for item in self.artifact_types
            ),
        )

    def accepts(self, artifact_types: Sequence[str]) -> bool:
        own = set(self.artifact_types)
        other = set(artifact_types)
        return "*" in own or "*" in other or bool(own & other)


@dataclass(frozen=True, slots=True)
class ScientificVariantSpec(JsonContract):
    """A scientifically meaningful algorithm choice for one function."""

    variant_id: str
    title: str
    description: str
    python_module: str
    fixed_arguments: tuple[str, ...] = ()
    is_primary: bool = False
    compatibility: str = "native"
    test_status: str = "verified"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variant_id",
            validate_identifier(self.variant_id, label="variant_id"),
        )
        if not self.title.strip() or not self.description.strip():
            raise ValueError("Scientific variant title and description are required")
        if not self.python_module.strip():
            raise ValueError("Scientific variant python_module is required")


@dataclass(frozen=True, slots=True)
class VariantFamilySpec(JsonContract):
    """User-visible family of algorithms that change scientific results."""

    family_id: str
    title: str
    comparison_note: str
    variants: tuple[ScientificVariantSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_id",
            validate_identifier(self.family_id, label="family_id"),
        )
        ids = [item.variant_id for item in self.variants]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("Variant families require at least two unique variants")
        if sum(item.is_primary for item in self.variants) != 1:
            raise ValueError("Variant families require exactly one primary variant")

    @property
    def primary(self) -> ScientificVariantSpec:
        return next(item for item in self.variants if item.is_primary)

    def get(self, variant_id: str | None) -> ScientificVariantSpec:
        selected = variant_id or self.primary.variant_id
        for item in self.variants:
            if item.variant_id == selected:
                return item
        raise KeyError(f"Unknown scientific variant: {selected}")


@dataclass(frozen=True, slots=True)
class FunctionSpec(JsonContract):
    """One reusable atomic capability in the App 1.0 function catalog."""

    function_id: str
    title: str
    category: str
    description: str
    python_module: str
    parameters: tuple[ParameterSpec, ...] = ()
    inputs: tuple[ArtifactPortSpec, ...] = ()
    outputs: tuple[ArtifactPortSpec, ...] = ()
    fixed_arguments: tuple[str, ...] = ()
    output_flag: str | None = None
    output_parameter: str | None = None
    default_output_name: str | None = None
    config_json_flag: str | None = None
    variant_family: VariantFamilySpec | None = None
    allowed_roots_flag: str | None = None
    repeat_allowed_roots: bool = False
    aliases: tuple[str, ...] = ()
    page_templates: tuple[str, ...] = ()
    base_config: dict[str, Any] = field(default_factory=dict)
    required_any: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "function_id",
            validate_identifier(self.function_id, label="function_id"),
        )
        if not self.title.strip() or not self.category.strip() or not self.description.strip():
            raise ValueError("Function title, category, and description are required")
        if not self.python_module.strip():
            raise ValueError("Function python_module is required")
        parameter_ids = [item.parameter_id for item in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError(f"Duplicate parameters in {self.function_id}")
        known_parameters = set(parameter_ids)
        for parameter in self.parameters:
            references = set(parameter.conflicts_with) | set(parameter.requires)
            unknown_references = references - known_parameters
            if unknown_references:
                raise ValueError(
                    f"{parameter.parameter_id} references unknown parameters: "
                    f"{', '.join(sorted(unknown_references))}"
                )
            if parameter.parameter_id in references:
                raise ValueError("A parameter cannot depend on or conflict with itself")
        for ports, label in ((self.inputs, "input"), (self.outputs, "output")):
            port_ids = [item.port_id for item in ports]
            if len(port_ids) != len(set(port_ids)):
                raise ValueError(f"Duplicate {label} ports in {self.function_id}")
        if self.output_parameter and self.output_parameter not in parameter_ids:
            raise ValueError("output_parameter must reference a declared parameter")
        unknown_required = set(self.required_any) - known_parameters
        if unknown_required:
            raise ValueError(
                "required_any references unknown parameters: "
                + ", ".join(sorted(unknown_required))
            )
        if self.default_output_name and Path(self.default_output_name).name != (
            self.default_output_name
        ):
            raise ValueError("default_output_name must be a plain filename")
        if self.allowed_roots_flag not in {None, "--allowed-root", "--allowed-roots"}:
            raise ValueError("Unsupported internal allowed-roots flag")
        clean_base_config = _finite_json(
            self.base_config,
            label=f"{self.function_id}.base_config",
        )
        if clean_base_config and not self.config_json_flag:
            raise ValueError("base_config requires config_json_flag")
        object.__setattr__(self, "base_config", clean_base_config)
        for alias in self.aliases:
            validate_identifier(alias, label="function alias")

    def parameter(self, parameter_id: str) -> ParameterSpec:
        for item in self.parameters:
            if item.parameter_id == parameter_id:
                return item
        raise KeyError(f"Unknown parameter {self.function_id}.{parameter_id}")

    def normalize_parameters(self, values: Mapping[str, Any] | None) -> dict[str, Any]:
        supplied = dict(values or {})
        known = {item.parameter_id for item in self.parameters}
        unknown = sorted(set(supplied) - known)
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.function_id}: {', '.join(unknown)}"
            )
        normalized: dict[str, Any] = {}
        for spec in self.parameters:
            value = supplied.get(spec.parameter_id, spec.default)
            clean = spec.validate(value)
            if clean not in (None, "") or spec.required:
                normalized[spec.parameter_id] = clean
        for spec in self.parameters:
            value = normalized.get(spec.parameter_id)
            if value in (None, "", False, []):
                continue
            conflicts = sorted(
                item
                for item in spec.conflicts_with
                if normalized.get(item) not in (None, "", False, [])
            )
            if conflicts:
                raise ValueError(
                    f"{spec.label} conflicts with: {', '.join(conflicts)}"
                )
            missing = [
                item
                for item in spec.requires
                if normalized.get(item) in (None, "", False, [])
            ]
            if missing:
                raise ValueError(f"{spec.label} requires: {', '.join(missing)}")
        if self.required_any and not any(
            normalized.get(item) not in (None, "", False, [])
            for item in self.required_any
        ):
            raise ValueError(
                "Provide at least one of: " + ", ".join(self.required_any)
            )
        return normalized

    def selected_module(self, variant_id: str | None = None) -> tuple[str, tuple[str, ...]]:
        if self.variant_family is None:
            if variant_id not in (None, "", "primary"):
                raise ValueError(f"{self.function_id} has no scientific variants")
            return self.python_module, self.fixed_arguments
        variant = self.variant_family.get(variant_id)
        return variant.python_module, (*self.fixed_arguments, *variant.fixed_arguments)

    def build_arguments(
        self,
        values: Mapping[str, Any] | None,
        *,
        variant_id: str | None = None,
        default_output: str | None = None,
        allowed_roots: Sequence[str | Path] = (),
    ) -> tuple[str, tuple[str, ...], dict[str, Any]]:
        """Validate parameters and build deterministic non-shell argv."""

        normalized = self.normalize_parameters(values)
        path_parameters = [
            spec
            for spec in self.parameters
            if spec.kind in {"directory", "file"}
            and normalized.get(spec.parameter_id) not in (None, "")
        ]
        if path_parameters:
            roots = tuple(
                Path(item).expanduser().resolve(strict=False)
                for item in allowed_roots
            )
            if not roots:
                raise ValueError(
                    f"{self.function_id} path parameters require allowed roots"
                )
            for spec in path_parameters:
                path = Path(str(normalized[spec.parameter_id])).expanduser().resolve(
                    strict=False
                )
                if not any(path == root or path.is_relative_to(root) for root in roots):
                    raise ValueError(
                        f"{spec.label} is outside the configured allowed roots"
                    )
        module, arguments = self.selected_module(variant_id)
        result = list(arguments)
        config: dict[str, Any] = copy.deepcopy(self.base_config)
        for spec in self.parameters:
            if spec.parameter_id not in normalized:
                continue
            value = normalized[spec.parameter_id]
            if spec.config_path:
                target = config
                parts = spec.config_path.split(".")
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
                continue
            if spec.positional:
                result.extend(_argument_values(spec, value))
                continue
            if spec.kind == "boolean":
                flag = spec.cli_flag if value else spec.negative_cli_flag
                if flag:
                    result.append(flag)
                continue
            if spec.cli_flag:
                result.append(spec.cli_flag)
                result.extend(_argument_values(spec, value))
        if config:
            if not self.config_json_flag:
                raise ValueError(
                    f"{self.function_id} has config_path parameters but no JSON flag"
                )
            result.extend(
                [
                    self.config_json_flag,
                    json.dumps(
                        config,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                ]
            )
        if default_output and self.output_flag and not (
            self.output_parameter and self.output_parameter in normalized
        ):
            target = (
                str(Path(default_output) / self.default_output_name)
                if self.default_output_name
                else default_output
            )
            result.extend([self.output_flag, target])
        if self.allowed_roots_flag:
            roots = tuple(
                str(Path(item).expanduser().resolve(strict=False))
                for item in allowed_roots
            )
            if not roots:
                raise ValueError(
                    f"{self.function_id} requires configured allowed roots"
                )
            if self.repeat_allowed_roots:
                for root in roots:
                    result.extend([self.allowed_roots_flag, root])
            else:
                result.extend([self.allowed_roots_flag, os.pathsep.join(roots)])
        return module, tuple(result), normalized


def _argument_values(spec: ParameterSpec, value: Any) -> list[str]:
    if spec.kind == "list":
        items = list(value)
        if spec.list_style == "repeat":
            return [str(item) for item in items]
        if spec.list_style == "json":
            return [json.dumps(items, separators=(",", ":"), allow_nan=False)]
        return [",".join(map(str, items))]
    if spec.kind in {"object", "roi"}:
        return [
            json.dumps(
                value,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        ]
    return [str(value)]


def _normalize_list_item(kind: str, value: Any, *, label: str) -> Any:
    if kind == "integer":
        if isinstance(value, bool):
            raise TypeError(f"{label} items must be integers")
        return int(value)
    if kind == "number":
        if isinstance(value, bool):
            raise TypeError(f"{label} items must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{label} items must be finite")
        return result
    if kind == "string":
        return str(value)
    raise ValueError(f"Unsupported list item kind: {kind}")


INFRASTRUCTURE_FLAGS = _INFRASTRUCTURE_FLAGS

__all__ = [
    "ArtifactPortSpec",
    "FunctionSpec",
    "INFRASTRUCTURE_FLAGS",
    "ParameterGroup",
    "ParameterKind",
    "ParameterSpec",
    "ScientificVariantSpec",
    "VariantFamilySpec",
]
