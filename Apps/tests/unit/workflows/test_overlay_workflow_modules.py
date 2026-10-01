"""Overlay style isolation and compatibility during responsibility extraction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

from solar_apps.workflows.radio import overlay_workflow as overlay
from solar_apps.workflows.radio import _overlay_workflow_core as core


@pytest.mark.parametrize("fail", [False, True])
def test_overlay_run_restores_callers_plotting_style(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    config = overlay.Config(output_dir=str(tmp_path))
    monkeypatch.setattr(core, "Config", lambda: config)
    seen = []

    def matched(cfg):
        seen.append(matplotlib.rcParams["axes.unicode_minus"])
        if fail:
            raise ValueError("invalid synthetic inputs")
        return []

    monkeypatch.setattr(core, "build_matched_pairs", matched)
    monkeypatch.setattr(core, "build_multi_wave_matched_pairs", matched)
    with matplotlib.rc_context(
        {"axes.unicode_minus": True, "font.family": ["DejaVu Sans"]}
    ):
        if fail:
            with pytest.raises(ValueError, match="invalid synthetic inputs"):
                overlay.run_overlay_workflow(argv=[])
        else:
            assert overlay.run_overlay_workflow(argv=[]) == []
        assert seen == [False]
        assert matplotlib.rcParams["axes.unicode_minus"] is True
        assert matplotlib.rcParams["font.family"] == ["DejaVu Sans"]


def test_overlay_run_keeps_later_colorbar_negative_scientific_exponents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from solar_apps.workflows.radio.artifacts import apply_colorbar_tick_notation

    monkeypatch.setattr(
        core, "Config", lambda: overlay.Config(output_dir=str(tmp_path))
    )
    monkeypatch.setattr(core, "build_matched_pairs", lambda cfg: [])
    monkeypatch.setattr(core, "build_multi_wave_matched_pairs", lambda cfg: [])
    with matplotlib.rc_context(
        {"axes.unicode_minus": True, "font.family": ["DejaVu Sans"]}
    ):
        overlay.run_overlay_workflow(argv=[])
        figure, axis = plt.subplots()
        try:
            image = axis.imshow(np.array([[1.0, 2.0], [3.0, 4.0]]) * 1e-7)
            colorbar = figure.colorbar(image)
            apply_colorbar_tick_notation(colorbar, transform="linear")
            figure.canvas.draw()
            assert "10^{−7}" in colorbar.ax.yaxis.get_offset_text().get_text()
        finally:
            plt.close(figure)
