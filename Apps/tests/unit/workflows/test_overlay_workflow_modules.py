"""Overlay style isolation and compatibility during responsibility extraction."""

from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys

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


@pytest.mark.parametrize("module", ["context", "selection", "science", "render"])
def test_overlay_private_module_can_be_imported_first(module: str) -> None:
    code = f"""
import importlib
import pickle
private = importlib.import_module('solar_apps.workflows.radio._overlay_{module}')
from solar_apps.workflows.radio import overlay_workflow as overlay
assert pickle.loads(pickle.dumps(overlay.Config())).selected_bands == overlay.Config().selected_bands
assert pickle.loads(pickle.dumps(overlay.parse_aia_time_from_filename)) is overlay.parse_aia_time_from_filename
assert overlay.parse_aia_time_from_filename('aia.lev1_euv_12s.2000-01-01T120000Z.171.image_lev1.fits').hour == 12
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_overlay_historical_pickle_paths_remain_resolvable() -> None:
    for module in ("overlay_workflow", "_overlay_workflow_core"):
        for name in (
            "Config",
            "GaussianReprojectResult",
            "parse_aia_time_from_filename",
        ):
            old_reference = f"csolar_apps.workflows.radio.{module}\n{name}\n.".encode()
            assert pickle.loads(old_reference) is getattr(overlay, name)
        config = overlay.Config(selected_bands=["149MHz"])
        old_instance = pickle.dumps(config, protocol=0).replace(
            b"solar_apps.workflows.radio._overlay_context",
            f"solar_apps.workflows.radio.{module}".encode(),
        )
        assert pickle.loads(old_instance).selected_bands == ["149MHz"]


def test_overlay_selection_keeps_core_time_parser_replacement_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime

    expected = datetime(2000, 1, 1, 12)
    monkeypatch.setattr(core, "_parse_flexible_datetime", lambda value: expected)
    assert overlay.parse_radio_time_from_filename("20000101_120000.fits") == expected


def test_overlay_science_keeps_canonical_replacement_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    expected = np.array([31.0])
    monkeypatch.setattr(
        core, "_canonical_elliptical_gaussian_2d", lambda *args: expected
    )
    assert (
        overlay.elliptical_gaussian_2d(
            (np.array([0.0]), np.array([0.0])), 1, 0, 0, 1, 1, 0
        )
        is expected
    )


def test_overlay_render_keeps_band_color_replacement_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "get_band_color", lambda *args: ("magenta", "black"))
    config = overlay.Config(selected_bands=["149MHz"], combine_polarizations=False)
    elements = overlay._build_selected_band_legend_elements(config, [])
    assert len(elements) == 1
    assert elements[0].get_color() == "magenta"


def test_overlay_run_keeps_cache_ownership_and_frame_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = overlay.Config(output_dir=str(tmp_path), aia_panel_wavelengths=None)
    monkeypatch.setattr(core, "Config", lambda: config)
    monkeypatch.setattr(
        core, "build_matched_pairs", lambda cfg: [("a", None, [1, 2]), ("b", None, [3])]
    )
    spectrum = object()
    monkeypatch.setattr(core, "_build_aia_spectrogram_cache", lambda *args: spectrum)
    seen = []

    def render(aia, hmi, tasks, index, total, cfg, colors, **kwargs):
        seen.append(
            (
                colors,
                kwargs["spectrogram_cache"],
                kwargs["sequence_start"],
                kwargs["generated_at"],
            )
        )
        colors.append(aia)
        return [f"{aia}-{i}" for i in tasks]

    monkeypatch.setattr(core, "process_aia_group", render)
    assert overlay.run_overlay_workflow(argv=[]) == ["a-1", "a-2", "b-3"]
    assert seen[0][0] is seen[1][0]
    assert seen[0][0] == ["a", "b"]
    assert seen[0][1] is seen[1][1] is spectrum
    assert [call[2] for call in seen] == [1, 3]
    assert seen[0][3] is seen[1][3]


@pytest.mark.parametrize("origin", ["lower", "upper"])
def test_overlay_synthetic_gaussian_retains_source_coordinates(origin: str) -> None:
    import numpy as np

    y, x = np.indices((96, 128), dtype=np.float64)
    data = 2 + 80 * np.exp(-0.5 * (((x - 70.25) / 7) ** 2 + ((y - 34.75) / 5) ** 2))
    config = overlay.Config()
    config.fit_snr_threshold = 1.0
    config.gaussian_fit_use_roi = False
    config.gaussian_quality_requirements["require_quality_ok"] = False
    extent = [-63.5, 63.5, -47.5, 47.5]
    fitted = overlay.fit_elliptical_gaussian_on_radio_image(
        data, extent, dict(vars(config)), image_origin=origin
    )
    assert fitted is not None
    assert fitted.center_pixel == pytest.approx((70.25, 34.75), abs=0.1)
    assert sorted(fitted.sigma_pixel) == pytest.approx([5, 7], abs=0.1)
    # Image extents describe outer edges; pixel coordinates refer to centers.
    expected_x = -63.5 + (70.25 + 0.5) * 127 / 128
    expected_y = -47.5 + (34.75 + 0.5) * 95 / 96
    if origin == "upper":
        expected_y = -expected_y
    assert fitted.center_arcsec == pytest.approx((expected_x, expected_y), abs=0.1)
    assert fitted.coordinate_roundtrip_error_pixel < 1e-8
    assert np.isfinite(fitted.model).all()
