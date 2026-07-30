# SPDX-License-Identifier: GPL-3.0-only
"""Versioned contracts and runtime adapters for Solar Physics App 1.0."""

from __future__ import annotations

from solar_toolkit.net.observations import (
    ObservationCollectionV1,
    ObservationQueryV1,
    RemoteObservationV1,
)

from .catalog import MODULES, module_by_id
from .contracts import (
    AppV1ProjectV1,
    ArtifactManifestV1,
    ArtifactProduct,
    InputReference,
    ModuleDescriptor,
    RunRequest,
    RunResult,
    RunStatus,
    SyncSelection,
    TimelineSource,
    WorkerEventV1,
)
from .runtime import AppV1RuntimePaths
from .flows import AppV1FlowV1, FlowEdgeV1, FlowNodeV1, FunctionCatalog
from .function_specs import (
    ArtifactPortSpec,
    FunctionSpec,
    ParameterSpec,
    VariantFamilySpec,
)
from .plot_specs import PlotSpec

__all__ = [
    "MODULES",
    "AppV1FlowV1",
    "AppV1ProjectV1",
    "AppV1RuntimePaths",
    "ArtifactManifestV1",
    "ArtifactPortSpec",
    "ArtifactProduct",
    "FlowEdgeV1",
    "FlowNodeV1",
    "FunctionCatalog",
    "FunctionSpec",
    "InputReference",
    "ModuleDescriptor",
    "ObservationCollectionV1",
    "ObservationQueryV1",
    "ParameterSpec",
    "PlotSpec",
    "RunRequest",
    "RunResult",
    "RunStatus",
    "RemoteObservationV1",
    "SyncSelection",
    "TimelineSource",
    "WorkerEventV1",
    "VariantFamilySpec",
    "module_by_id",
]
