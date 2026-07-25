from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from solar_apps.workflows.radio import source_map_workflow as workflow
from solar_apps.workflows.radio.artifacts import validate_source_map_artifact
from solar_apps.frontends.radio.source_map.jobs import ArtifactRegistry, JobRegistry
from solar_apps.frontends.radio.source_map.service import PathPolicy


def _write_map(
    path: Path,
    *,
    frequency: float,
    bunit: str = "K",
    scale: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[-1:1:48j, -1:1:48j]
    data = scale * (10.0 + 200.0 * np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / 0.08))
    header = fits.Header()
    header["DATE-OBS"] = "2025-01-24T04:48:30"
    header["FREQ"] = frequency
    header["POLAR"] = "RR"
    header["BUNIT"] = bunit
    header["CRVAL1"] = 0.0
    header["CRPIX1"] = 24.5
    header["CDELT1"] = 40.0
    header["CRVAL2"] = 0.0
    header["CRPIX2"] = 24.5
    header["CDELT2"] = 40.0
    header["RSUN_OBS"] = 960.0
    fits.PrimaryHDU(data.astype(np.float32), header=header).writeto(path)


def _config() -> dict:
    cfg = dict(workflow.DEFAULT_CONFIG)
    cfg.update(
        {
            "enable_gaussian_overlay": False,
            "enable_spectrogram_panel": False,
            "enable_raw_quality_filter": False,
            "radio_background_strategy": "off",
            "radio_background_force_off": True,
            "enable_radio_background_subtraction": False,
            "save_background_diagnostics": False,
            "save_background_subtracted_image": False,
            "save_estimated_background_map": False,
            "show_plot": False,
            "save_plot": True,
            "write_source_map_sidecar": True,
            "fig_size": (5, 4),
            "dpi": 80,
            "scale_factor": 1.4,
            "title_fontsize": 10,
            "label_fontsize": 10,
            "tick_fontsize": 9,
            "legend_fontsize": 8,
            "annotation_fontsize": 8,
            "analysis_subdir": "source_maps",
            "polarization": "RR",
            "combine_polarizations": False,
        }
    )
    return cfg


def test_single_band_workflow_emits_linear_unit_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "149MHz" / "RR" / "frame.fits"
    _write_map(source, frequency=149.0)
    output = workflow.plot_single_band(
        str(source), str(tmp_path / "output"), _config(), write_sidecar=True
    )
    metadata = validate_source_map_artifact(output)
    assert metadata["mode"] == "single_band"
    assert metadata["panels"][0]["colorbar_label"] == "Intensity [K]"
    assert metadata["panels"][0]["tick_notation"] == "scientific_offset"
    assert metadata["panels"][0]["unit"]["source"] == "fits_bunit"


def test_sequence_single_band_layout_is_fixed_across_titles_and_color_scales(
    tmp_path: Path,
) -> None:
    first = tmp_path / "149MHz" / "RR" / "short.fits"
    second = (
        tmp_path
        / "149MHz"
        / "RR"
        / "a_very_long_source_map_filename_for_layout_regression.fits"
    )
    _write_map(first, frequency=149.0, scale=1.0)
    _write_map(second, frequency=149.0, scale=1e12)
    cfg = _config()
    cfg.update(
        {
            "_artifact_bbox_inches": None,
            "_artifact_pad_inches": 0.0,
            "_artifact_fixed_panel_layout": True,
        }
    )

    first_output = workflow.plot_single_band(
        str(first), str(tmp_path / "first-output"), cfg, write_sidecar=True
    )
    second_output = workflow.plot_single_band(
        str(second), str(tmp_path / "second-output"), cfg, write_sidecar=True
    )
    first_metadata = validate_source_map_artifact(first_output)
    second_metadata = validate_source_map_artifact(second_output)

    assert first_metadata["image"]["width"] == second_metadata["image"]["width"]
    assert first_metadata["image"]["height"] == second_metadata["image"]["height"]
    np.testing.assert_allclose(
        first_metadata["panels"][0]["bbox_normalized"],
        second_metadata["panels"][0]["bbox_normalized"],
        rtol=0.0,
        atol=1e-9,
    )


def test_sequence_colorbar_has_fixed_intensity_unit_and_visible_values() -> None:
    fig, axis = plt.subplots(figsize=(5, 4), dpi=80)
    image = axis.imshow(np.arange(100, dtype=float).reshape(10, 10))
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.ax.tick_params(colors="white")

    workflow._apply_fixed_single_band_artifact_layout(
        fig,
        axis,
        colorbar,
        intensity_unit="K",
        cfg={"tick_fontsize": 18, "label_fontsize": 22},
    )
    fig.canvas.draw()

    assert colorbar.ax.get_ylabel() == "Intensity [K]"
    assert [tick.get_text() for tick in colorbar.ax.get_yticklabels()]
    assert all(tick.get_color() == "black" for tick in colorbar.ax.get_yticklabels())
    assert colorbar.ax.yaxis.label.get_color() == "black"
    assert colorbar.ax.yaxis.get_offset_text().get_color() == "black"
    plt.close(fig)


def test_multi_band_workflow_emits_log_unit_for_each_panel(tmp_path: Path) -> None:
    first = tmp_path / "149MHz" / "RR" / "frame.fits"
    second = tmp_path / "164MHz" / "RR" / "frame.fits"
    _write_map(first, frequency=149.0)
    _write_map(second, frequency=164.0)
    cfg = _config()
    cfg.update({"polarization": "RR", "combine_polarizations": False})
    output = workflow.plot_multi_band_slot(
        0,
        [str(first), str(second)],
        str(tmp_path / "output"),
        cfg,
        write_sidecar=True,
    )
    metadata = validate_source_map_artifact(output)
    assert metadata["mode"] == "multi_band"
    assert len(metadata["panels"]) == 2
    assert {panel["colorbar_label"] for panel in metadata["panels"]} == {
        "Intensity [K]"
    }
    assert {panel["tick_notation"] for panel in metadata["panels"]} == {"power_of_ten"}


def test_multi_band_fixed_range_is_converted_to_log10() -> None:
    data = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
    cfg = _config()
    cfg.update({"color_range_mode": "fixed", "fixed_vmin": 10.0, "fixed_vmax": 1e5})

    lows, highs = workflow._resolve_multi_band_display_ranges(data, cfg)

    assert lows == [1.0, 1.0]
    assert highs == [5.0, 5.0]


def test_multi_band_global_range_is_shared_across_panels() -> None:
    data = [np.array([[-1.5, 2.0]]), np.array([[3.25, 4.5]])]
    cfg = _config()
    cfg["color_range_mode"] = "global"

    lows, highs = workflow._resolve_multi_band_display_ranges(data, cfg)

    assert lows == [-1.5, -1.5]
    assert highs == [4.5, 4.5]


def test_render_job_runs_in_worker_process_and_registers_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "149MHz" / "RR" / "frame.fits"
    _write_map(source, frequency=149.0)
    cfg = _config()
    cfg["output_dir"] = str(tmp_path / "output")
    policy = PathPolicy([tmp_path])
    artifacts = ArtifactRegistry(policy)
    jobs = JobRegistry(policy=policy, artifacts=artifacts)
    started = jobs.start(
        cfg,
        {
            "id": "file-0000",
            "mode": "single_band",
            "run_path": str(source),
        },
    )
    deadline = time.monotonic() + 30
    result = started
    while result["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        result = jobs.public(started["id"])
    assert result["status"] == "completed", result
    artifact = artifacts.get(result["artifact_id"])
    assert artifact["metadata"]["panels"][0]["colorbar_label"] == "Intensity [K]"
