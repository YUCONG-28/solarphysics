# SPDX-License-Identifier: GPL-3.0-only
"""Versioned, backend-neutral plotting contract for App 1.0 workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import JsonContract


@dataclass(frozen=True, slots=True)
class PlotSpec(JsonContract):
    """All adjustable plot styling without creating renderer variants."""

    kind: Literal["image", "time-series", "spectrum", "composite"]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    width_inches: float = 8.0
    height_inches: float = 5.0
    dpi: int = 150
    grid: bool = False
    colorbar: bool = True
    legend: bool = True
    annotations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported PlotSpec schema")
        if self.kind not in {"image", "time-series", "spectrum", "composite"}:
            raise ValueError(f"Unsupported plot kind: {self.kind}")
        if not 1 <= float(self.width_inches) <= 100:
            raise ValueError("Plot width must be between 1 and 100 inches")
        if not 1 <= float(self.height_inches) <= 100:
            raise ValueError("Plot height must be between 1 and 100 inches")
        if not 36 <= int(self.dpi) <= 2400:
            raise ValueError("Plot DPI must be between 36 and 2400")
        if self.vmin is not None and self.vmax is not None and self.vmin >= self.vmax:
            raise ValueError("Plot vmin must be below vmax")


__all__ = ["PlotSpec"]
