"""Phase 2B adapter tests without importing PyQt6 into pytest."""

from __future__ import annotations

import json
from pathlib import Path

from solar_apps.frontends.app_v1.phase2b import Phase2BAdapter
from solar_apps.frontends.app_v1.radio_composite_worker import (
    _normalize_candidate_contract,
)
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


def test_bad_frame_and_source_map_launch_native_workers_without_browser(
    tmp_path: Path,
) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    (observations / "frame.fits").write_bytes(b"fixture")

    bad = adapter.build_bad_frame_review(
        observations,
        frequencies="149,164",
        polarizations="RR",
        start_index=2,
        end_index=8,
        strategy="labeling",
        scope="all_scanned",
        sample_count=20,
    )
    source_map = adapter.build_source_map_app(observations)

    assert bad.python_module.endswith("app_v1.bad_frame_worker")
    assert source_map.python_module.endswith("app_v1.native_science_worker")
    assert bad.arguments[0] == "create"
    assert bad.arguments[bad.arguments.index("--strategy") + 1] == "labeling"
    assert bad.arguments[bad.arguments.index("--scope") + 1] == "all_scanned"
    assert bad.arguments[bad.arguments.index("--end-index") + 1] == "8"
    assert source_map.arguments[0] == "source-map-discover"
    assert "--open-browser" not in bad.arguments
    assert "--open-browser" not in source_map.arguments
    assert bad.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert source_map.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert "Workload: 1 FITS candidate(s)" in bad.summary

    bad.output_dir.mkdir(parents=True)
    action = adapter.build_bad_frame_action(
        bad.output_dir,
        "review-id",
        action="label",
        target_kind="candidate",
        target_id="candidate-id",
        quality="bad",
        event_tags="solar_burst",
        artifact_tags="stripe",
    )
    assert action.python_module.endswith("app_v1.bad_frame_worker")
    assert action.arguments[0] == "label"
    assert action.arguments[action.arguments.index("--quality") + 1] == "bad"


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
    composite = adapter.build_radio_composite(
        radio,
        dart,
        frequencies="149,164",
        polarization="RR",
        roi_bounds="-10,-20,30,40",
        dart_bandwidth_mhz=4,
        fps=12,
        stride=2,
        dpi=200,
        transform="log10",
        save_video=False,
        save_frames=True,
    )

    assert roi.python_module.endswith("app_v1.native_science_worker")
    assert roi.arguments[0] == "roi-run"
    assert roi.arguments[roi.arguments.index("--polarization") + 1] == "RCP"
    assert composite.python_module.endswith("app_v1.radio_composite_worker")
    assert composite.arguments[0] == "--radio-dir"
    assert composite.arguments[composite.arguments.index("--radio-dir") + 1] == str(
        radio
    )
    assert composite.arguments[composite.arguments.index("--dart-dir") + 1] == str(dart)
    assert composite.arguments[composite.arguments.index("--frequencies") + 1] == (
        "149,164"
    )
    assert "--no-save-video" in composite.arguments
    assert "--save-frames" in composite.arguments
    assert "--browser" not in roi.arguments
    assert "--browser" not in composite.arguments
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


def test_composite_candidate_contract_uses_manifest_mhz_and_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "149MHz_2025124_044829_108.fits"
    candidates = [
        {
            "id": "file-0000",
            "paths": [str(source)],
            "frequencies_mhz": [149_000_000.0],
            "observation_time": None,
        }
    ]

    _normalize_candidate_contract(
        candidates,
        149.0,
        {str(source.resolve()): "2025-01-24T04:48:29.108Z"},
    )

    assert candidates[0]["id"] == "149mhz-file-0000"
    assert candidates[0]["frequencies_mhz"] == [149.0]
    assert candidates[0]["observation_time"] == "2025-01-24T04:48:29.108Z"
