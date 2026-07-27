# SPDX-License-Identifier: GPL-3.0-only
"""The single Matplotlib Agg renderer, imported only by worker processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .plot_specs import PlotSpec


class ScientificPlotRenderer:
    """Render every static App 1.0 plot through one versioned entry point."""

    def render(
        self,
        spec: PlotSpec,
        data: Mapping[str, Any],
        output_path: str | Path,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure = Figure(
            figsize=(spec.width_inches, spec.height_inches),
            dpi=spec.dpi,
        )
        axis = figure.subplots()
        if spec.kind in {"image", "spectrum", "composite"}:
            image = axis.imshow(
                data["image"],
                origin=str(data.get("origin", "lower")),
                cmap=spec.cmap,
                vmin=spec.vmin,
                vmax=spec.vmax,
                aspect=data.get("aspect", "auto"),
            )
            if spec.colorbar:
                figure.colorbar(image, ax=axis)
        else:
            x_values = data["x"]
            series = data.get("series", {})
            if not isinstance(series, Mapping):
                raise TypeError("Time-series data requires a series mapping")
            for label, values in series.items():
                axis.plot(x_values, values, label=str(label))
            if spec.legend and series:
                axis.legend()
        axis.set_title(spec.title)
        axis.set_xlabel(spec.x_label)
        axis.set_ylabel(spec.y_label)
        axis.grid(spec.grid)
        for annotation in spec.annotations:
            axis.annotate(
                str(annotation.get("text", "")),
                tuple(annotation.get("xy", (0, 0))),
            )
        figure.savefig(target, dpi=spec.dpi)
        return target

    def render_frames(
        self,
        spec: PlotSpec,
        frames: Sequence[Mapping[str, Any]],
        output_dir: str | Path,
    ) -> tuple[Path, ...]:
        directory = Path(output_dir)
        return tuple(
            self.render(spec, frame, directory / f"frame-{index:06d}.png")
            for index, frame in enumerate(frames)
        )


__all__ = ["ScientificPlotRenderer"]
