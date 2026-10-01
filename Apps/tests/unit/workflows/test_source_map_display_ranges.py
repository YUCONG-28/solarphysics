"""Directory display ranges retain the requested linear units."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.radio import source_map_workflow as workflow
from solar_apps.workflows.radio import _source_map_workflow_core as core


@pytest.mark.parametrize("range_mode", ["fixed", "auto", "global"])
def test_directory_run_keeps_single_band_limits_in_linear_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, range_mode: str
) -> None:
    source_dir = tmp_path / "radio"
    source_dir.mkdir()
    fits.PrimaryHDU(np.array([[10.0, 100.0], [1000.0, 10000.0]])).writeto(
        source_dir / "frame.fits"
    )
    captured: list[dict] = []

    def capture(path, output, config, *args, **kwargs):
        captured.append(copy.deepcopy(config))
        return "unused.png"

    monkeypatch.setattr(workflow, "plot_single_band", capture)
    monkeypatch.setattr(core, "_estimate_safe_workers", lambda **kwargs: 1)
    event = {
        "mode": "single_band",
        "study_mode": "exploratory",
        "data": {
            "data_dir": str(source_dir),
            "single_file_path": None,
            "start_idx": 0,
            "end_idx": None,
            "combine_polarizations": False,
            "polarization": "RR",
        },
        "output": {"output_dir": str(tmp_path / "output")},
        "features": {"spectrogram_panel": False, "raw_quality_filter": False},
        "display": {
            "color_range_mode": range_mode,
            "fixed_vmin": 10.0,
            "fixed_vmax": 10000.0,
            "show_plot": False,
        },
    }
    monkeypatch.setattr(
        core,
        "load_script_config",
        lambda *args: copy.deepcopy(workflow.DEFAULT_CONFIG),
    )
    monkeypatch.setattr(core, "CONFIG", copy.deepcopy(core.CONFIG))
    monkeypatch.setattr(core, "USER_CONFIG", copy.deepcopy(core.USER_CONFIG))

    assert workflow.run_source_map(event, argv=[]) == 0

    assert len(captured) == 1
    assert captured[0]["fixed_vmin"] == 10.0
    assert captured[0]["fixed_vmax"] == 10000.0

