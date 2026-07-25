"""Phase 2B adapter tests without importing PyQt6 into pytest."""

from __future__ import annotations

import json
from pathlib import Path

from solar_apps.frontends.app_v1.phase2b import Phase2BAdapter
from solar_apps.platform.layout import RuntimeLayout


def _adapter(tmp_path: Path) -> tuple[Phase2BAdapter, Path, RuntimeLayout]:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )
    observations = tmp_path / "observations"
    observations.mkdir()
    return (
        Phase2BAdapter(layout, allowed_roots=(observations,)),
        observations,
        layout,
    )


def test_bad_frame_and_source_map_launch_in_separate_existing_frontends(
    tmp_path: Path,
) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    (observations / "frame.fits").write_bytes(b"fixture")

    bad = adapter.build_bad_frame_review(observations)
    source_map = adapter.build_source_map_app(observations)

    assert bad.python_module == "solar_apps.frontends.radio_bad_frame_review.cli"
    assert source_map.python_module == "solar_apps.frontends.radio.source_map.cli"
    assert "--open-browser" in bad.arguments
    assert "--open-browser" in source_map.arguments
    assert bad.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert source_map.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert "Workload: 1 FITS candidate(s)" in bad.summary


def test_gaussian_launch_uses_existing_source_map_workflow_for_one_frame(
    tmp_path: Path,
) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    radio = observations / "radio"
    radio.mkdir()
    fits_path = radio / "radio_149MHz_RR_20250124_044829.fits"
    fits_path.write_bytes(b"fixture")

    launch = adapter.build_gaussian_fit(radio, source_count=2)

    assert launch.python_module == "solar_apps.workflows.radio.source_map_cli"
    raw = launch.arguments[launch.arguments.index("--workspace-config-json") + 1]
    config = json.loads(raw)
    assert config["data"]["single_file_path"] == str(fits_path)
    assert config["gaussian"]["gaussian_source_mode"] == "multi"
    assert config["gaussian"]["multi_gaussian_source_count"] == 2
    assert config["features"]["spectrogram_panel"] is False
    assert Path(config["output"]["output_dir"]).is_relative_to(
        layout.outputs_dir / "app_v1"
    )


def test_roi_and_composite_launchers_receive_paths_and_private_outputs(
    tmp_path: Path,
) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    radio = observations / "radio"
    dart = observations / "dart"
    radio.mkdir()
    dart.mkdir()
    (radio / "radio.fits").write_bytes(b"radio")
    (dart / "dart.fits").write_bytes(b"dart")

    roi = adapter.build_roi_lightcurve(radio, polarization="RCP")
    composite = adapter.build_radio_composite(radio, dart)

    assert roi.python_module.endswith("roi_lightcurve_launcher")
    assert roi.arguments[roi.arguments.index("--polarization") + 1] == "RCP"
    assert composite.python_module.endswith("composite_figure_launcher")
    assert composite.arguments[composite.arguments.index("--radio-dir") + 1] == str(
        radio
    )
    assert composite.arguments[composite.arguments.index("--dart-dir") + 1] == str(dart)
    assert roi.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert composite.output_dir.is_relative_to(layout.outputs_dir / "app_v1")


def test_phase2b_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    adapter, _observations, _layout = _adapter(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    try:
        adapter.build_bad_frame_review(outside)
    except PermissionError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover - explicit assertion message is clearer here
        raise AssertionError("outside path was accepted")
