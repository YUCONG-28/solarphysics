# SPDX-License-Identifier: GPL-3.0-only
"""The ten visible interfaces assigned to App 1.0 implementation phases."""

from __future__ import annotations

from .contracts import ModuleDescriptor

MODULES: tuple[ModuleDescriptor, ...] = (
    ModuleDescriptor("workbench", "Workbench", "General", "1", "workbench"),
    ModuleDescriptor(
        "radio-workspace",
        "Radio Workspace",
        "Radio",
        "1",
        "radio-workspace",
        True,
    ),
    ModuleDescriptor(
        "image-viewer", "Image Viewer", "Visualization", "2A", "image-viewer", True
    ),
    ModuleDescriptor(
        "image-composer",
        "Image Composer",
        "Visualization",
        "4",
        "image-composer",
        True,
    ),
    ModuleDescriptor(
        "bad-frame-review",
        "Bad Frame Review",
        "Radio",
        "2B",
        "bad-frame-review",
        True,
    ),
    ModuleDescriptor("source-map", "Source Map", "Radio", "2B", "source-map", True),
    ModuleDescriptor(
        "dart-spectrogram",
        "DART Spectrogram",
        "Radio",
        "2C",
        "dart-spectrogram",
        True,
    ),
    ModuleDescriptor(
        "roi-lightcurve",
        "ROI Light Curve",
        "Radio",
        "2B",
        "roi-lightcurve",
        True,
    ),
    ModuleDescriptor(
        "radio-composite",
        "Radio Composite Figure",
        "Radio",
        "2B",
        "radio-composite",
        True,
    ),
    ModuleDescriptor(
        "source-trajectory",
        "Source Trajectory",
        "Radio",
        "2C",
        "source-trajectory",
        True,
    ),
)

_BY_ID = {module.module_id: module for module in MODULES}


def module_by_id(module_id: str) -> ModuleDescriptor:
    """Return one registered App 1.0 interface by stable ID."""

    try:
        return _BY_ID[module_id]
    except KeyError as exc:
        raise KeyError(f"Unknown App 1.0 module: {module_id}") from exc


__all__ = ["MODULES", "module_by_id"]
