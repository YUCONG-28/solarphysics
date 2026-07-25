# SPDX-License-Identifier: GPL-3.0-only
"""Versioned contracts and runtime adapters for Solar Physics App 1.0."""

from __future__ import annotations

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
)
from .runtime import AppV1RuntimePaths

__all__ = [
    "MODULES",
    "AppV1ProjectV1",
    "AppV1RuntimePaths",
    "ArtifactManifestV1",
    "ArtifactProduct",
    "InputReference",
    "ModuleDescriptor",
    "RunRequest",
    "RunResult",
    "RunStatus",
    "SyncSelection",
    "TimelineSource",
    "module_by_id",
]
