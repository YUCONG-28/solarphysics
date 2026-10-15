from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from solar_apps.frontends.radio.source_map.jobs import ArtifactRegistry, JobRegistry
from solar_apps.frontends.radio.source_map.service import PathPolicy
from solar_apps.workflows.radio import source_map_workflow as workflow
from solar_apps.workflows.radio.artifacts import validate_source_map_artifact


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
    header["DATE-OBS"] = "2000-01-01T04:48:30"
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


def test_source_map_json_worker_materializes_each_exact_hdu(tmp_path, monkeypatch):
    import json

    from solar_apps.frontends.radio.source_map import worker
    from solar_apps.frontends.radio.source_map.service import discover_candidates
    from solar_apps.workflows.radio import _source_map_workflow_core as inputs

    source = tmp_path / "RR" / "multi.fits"
    _write_map(source, frequency=149.0)
    with fits.open(source) as hdus:
        primary = fits.PrimaryHDU(hdus[0].data.copy(), hdus[0].header.copy())
    second_header = primary.header.copy()
    second_header["FREQ"] = 164.0
    fits.HDUList([primary, fits.ImageHDU(primary.data * 3, second_header)]).writeto(
        source, overwrite=True
    )
    cfg = dict(
        _config(),
        mode="single_band",
        single_file_path=str(source),
        output_dir=str(tmp_path / "output"),
    )
    candidates = discover_candidates(cfg, policy=PathPolicy([tmp_path]))
    assert len(candidates) == 2
    observed_pixels = []
    original_read = inputs.read_fits

    def capture_exact_read(item):
        result = original_read(item)
        if getattr(item, "logical_frame", None) is not None:
            observed_pixels.append(
                (item.logical_frame.payload["hdu_index"], result[0].max())
            )
        return result

    monkeypatch.setattr(inputs, "read_fits", capture_exact_read)
    result = worker.run_job(
        json.loads(json.dumps({"config": cfg, "candidates": candidates}))
    )
    assert len(result["artifacts"]) == 2
    metadata = [
        validate_source_map_artifact(artifact["image_path"])
        for artifact in result["artifacts"]
    ]
    assert [value["panels"][0]["frequency_mhz"] for value in metadata] == [149.0, 164.0]
    assert [
        next(iter(value["time_match"]["selections"].values()))["locator"]["hdu_index"]
        for value in metadata
    ] == [0, 1]
    assert {index for index, _ in observed_pixels} == {0, 1}
    maxima = {index: maximum for index, maximum in observed_pixels}
    assert np.isclose(maxima[1], maxima[0] * 3)


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


def test_multi_band_artifact_uses_explicit_master_clock_and_keeps_match_flags(tmp_path):
    from solar_toolkit.time import normalize_time

    from solar_apps.workflows.radio import _source_map_selection

    first = tmp_path / "149MHz" / "RR" / "frame.fits"
    second = tmp_path / "164MHz" / "RR" / "frame.fits"
    _write_map(first, frequency=149.0)
    _write_map(second, frequency=164.0)
    fits.setval(second, "DATE-OBS", value="2025-01-24T04:48:32.123456789")
    cfg = _config()
    metadata = {
        "algorithm_version": 3,
        "output_time_utc": "2025-01-24T04:48:32.123456789Z",
        "master_time_utc": "2025-01-24T04:48:30.000000000Z",
        "master_source_id": "band:0000",
        "master_time_tai_ns": normalize_time("2025-01-24T04:48:30Z").tai_ns,
        "output_source_id": "band:0001",
        "output_time_tai_ns": normalize_time("2025-01-24T04:48:32.123456789Z").tai_ns,
        "selections": {
            "band:0000": {
                "frame_id": str(first),
                "actual_time_utc": "2025-01-24T04:48:30.000000000Z",
                "delta_ns": -2123456789,
                "reused": True,
                "within_tolerance": False,
            }
        },
    }
    slot = _source_map_selection.RadioTimeSlot(
        [str(first), str(second)], time_match=metadata
    )
    output = workflow.plot_multi_band_slot(
        0, slot, str(tmp_path / "output"), cfg, write_sidecar=True
    )
    saved = validate_source_map_artifact(output)
    assert "044832" in Path(output).name
    assert saved["time_match"] == metadata
    assert saved["panels"][0]["time_match"]["reused"] is True


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
