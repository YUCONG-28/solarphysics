# SPDX-License-Identifier: GPL-3.0-only
"""Offline unit tests locking the App 1.0 scientific plot renderer contract."""

from __future__ import annotations

from pathlib import Path

import solar_apps.frontends.app_v1 as app_v1
from solar_apps.frontends.app_v1 import PlotSpec, ScientificPlotRenderer
from solar_apps.frontends.app_v1.scientific_plot_renderer import (
    ScientificPlotRenderer as ImplementationScientificPlotRenderer,
)

IMAGE_DATA = {"image": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}
TIME_SERIES_DATA = {
    "x": [0.0, 1.0, 2.0],
    "series": {"quiet": [1.0, 2.0, 3.0]},
}


def test_public_entry_exports_implementation_class() -> None:
    assert app_v1.ScientificPlotRenderer is ImplementationScientificPlotRenderer
    assert app_v1.PlotSpec is PlotSpec
    assert "ScientificPlotRenderer" in app_v1.__all__


def test_static_image_branch_render(tmp_path: Path) -> None:
    spec = PlotSpec(
        kind="image",
        title="Static image",
        x_label="Pixel X",
        y_label="Pixel Y",
        cmap="viridis",
        vmin=0,
        vmax=8,
        colorbar=True,
        grid=True,
    )

    target = tmp_path / "static.png"
    result = ScientificPlotRenderer().render(spec, IMAGE_DATA, target)

    assert result == target
    assert isinstance(result, Path)
    assert target.is_file()
    assert target.stat().st_size > 0


def test_time_series_branch_render(tmp_path: Path) -> None:
    spec = PlotSpec(
        kind="time-series",
        title="Time series",
        x_label="Time",
        y_label="Amplitude",
        legend=True,
        grid=True,
        annotations=({"text": "peak", "xy": (1.0, 2.0)},),
    )

    target = tmp_path / "series.png"
    result = ScientificPlotRenderer().render(spec, TIME_SERIES_DATA, target)

    assert result == target
    assert target.is_file()
    assert target.stat().st_size > 0


def test_render_frames_uses_continuous_names_and_returns_paths(tmp_path: Path) -> None:
    spec = PlotSpec(kind="time-series", title="Frame", legend=False)
    frames = [
        {"x": [0.0, 1.0], "series": {"signal": [value, value + 1.0]}}
        for value in range(3)
    ]

    output_dir = tmp_path / "frames"
    paths = ScientificPlotRenderer().render_frames(spec, frames, output_dir)

    assert isinstance(paths, tuple)
    assert all(isinstance(path, Path) for path in paths)
    assert [path.name for path in paths] == [
        "frame-000000.png",
        "frame-000001.png",
        "frame-000002.png",
    ]
    assert output_dir.is_dir()
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
