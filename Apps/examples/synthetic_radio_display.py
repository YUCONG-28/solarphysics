"""Render a deterministic synthetic image with the spatial-radio contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from solar_apps.platform.layout import RuntimeLayout, validate_private_output_path
from solar_apps.workflows.radio.spatial_display import SpatialRadioDisplay

DEFAULT_FILENAME = "synthetic_radio_display.png"


def _safe_output_path(
    requested: str | Path | None,
    *,
    layout: RuntimeLayout,
) -> Path:
    """Resolve an output file and refuse writes to the public source tree."""

    if requested is None:
        candidate = (
            layout.outputs_dir / "examples" / "spatial-radio-display" / DEFAULT_FILENAME
        )
    else:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    resolved = validate_private_output_path(candidate, repo_root=layout.repo_root)
    if resolved.suffix.lower() != ".png":
        raise ValueError("--output must name a .png file")
    return resolved


def make_synthetic_map(size: int = 192) -> np.ndarray:
    """Return a deterministic two-source array with one invalid corner."""

    if size < 32:
        raise ValueError("size must be at least 32 pixels")
    coordinate = np.linspace(-1.0, 1.0, size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    primary = 120.0 * np.exp(
        -(((x_grid + 0.27) / 0.18) ** 2 + ((y_grid - 0.12) / 0.24) ** 2) / 2.0
    )
    secondary = 55.0 * np.exp(
        -(((x_grid - 0.35) / 0.12) ** 2 + ((y_grid + 0.30) / 0.16) ** 2) / 2.0
    )
    background = 0.35 + 0.08 * np.cos(4.0 * np.pi * x_grid) * np.cos(
        3.0 * np.pi * y_grid
    )
    image = primary + secondary + background
    image[:5, :5] = np.nan
    return image


def run_demo(
    *,
    output: str | Path | None = None,
    transform: str = "log10",
    size: int = 192,
    layout: RuntimeLayout | None = None,
) -> dict[str, Any]:
    """Render the synthetic map and return the generated artifact metadata."""

    selected_layout = layout or RuntimeLayout.discover()
    output_path = _safe_output_path(output, layout=selected_layout)
    sidecar_path = validate_private_output_path(
        output_path.with_suffix(".json"), repo_root=selected_layout.repo_root
    )
    selected_layout.ensure()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = SpatialRadioDisplay(
        cmap="hot",
        bad_color="#000080",
        transform=transform,
        range_mode="auto",
        range_scope="frame",
        auto_method="fixed_percentile",
        percentiles=(1.0, 99.5),
        unit="synthetic intensity",
        fov=(-960.0, 960.0, -960.0, 960.0),
        render_profile="export",
    )
    source = make_synthetic_map(size=size)
    rendered = display.transformed(source)
    vmin, vmax = display.display_limits(source)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(6.4, 5.4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(
        rendered,
        cmap=display.matplotlib_cmap(),
        origin="lower",
        extent=display.fov,
        vmin=vmin,
        vmax=vmax,
    )
    axes.set(
        title="Synthetic radio source map",
        xlabel="Solar X (arcsec)",
        ylabel="Solar Y (arcsec)",
    )
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.set_label(
        f"log10({display.unit})" if transform == "log10" else display.unit
    )
    figure.savefig(output_path, dpi=160, facecolor="white")
    figure.clear()

    sidecar = {
        "schema_version": 1,
        "synthetic": True,
        "display": display.sidecar_payload(),
        "display_limits": [vmin, vmax],
        "display_cache_signature": display.cache_signature(),
    }
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "image": output_path,
        "sidecar": sidecar_path,
        "display": display,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone synthetic example parser."""

    parser = argparse.ArgumentParser(
        description="Render a deterministic synthetic spatial radio image."
    )
    parser.add_argument(
        "--output",
        help=(
            "Destination PNG. Defaults to "
            "Local/outputs/examples/spatial-radio-display/."
        ),
    )
    parser.add_argument(
        "--transform",
        choices=("linear", "log10"),
        default="log10",
        help="Display-only intensity transform.",
    )
    parser.add_argument("--size", type=int, default=192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example without performing work at import time."""

    arguments = build_parser().parse_args(argv)
    artifacts = run_demo(
        output=arguments.output,
        transform=arguments.transform,
        size=arguments.size,
    )
    print(f"image: {artifacts['image']}")
    print(f"sidecar: {artifacts['sidecar']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
