"""Phase 2C adapter tests without importing PyQt6 into pytest."""

from __future__ import annotations

import json
from pathlib import Path

from solar_apps.frontends.app_v1.phase2c import Phase2CAdapter
from solar_apps.platform.layout import RuntimeLayout


def _adapter(tmp_path: Path) -> tuple[Phase2CAdapter, Path, RuntimeLayout]:
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
        Phase2CAdapter(layout, allowed_roots=(observations,)),
        observations,
        layout,
    )


def test_dart_launch_uses_native_worker_without_browser(tmp_path: Path) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    dart = observations / "dart"
    dart.mkdir()
    for name in ("data-i.fits", "data-v.fits", "frequency.fits", "time.fits"):
        (dart / name).write_bytes(b"fixture")

    launch = adapter.build_dart_spectrogram(dart)

    assert launch.python_module.endswith("app_v1.native_science_worker")
    assert launch.arguments[0] == "dart-render"
    assert launch.arguments[launch.arguments.index("--input-dir") + 1] == str(dart)
    assert "--browser" not in launch.arguments
    assert "Workload: 4 input file(s)" in launch.summary
    assert launch.output_dir.is_relative_to(layout.outputs_dir / "app_v1")


def test_drift_and_newkirk_launch_existing_diagnostics(tmp_path: Path) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    gaussian = observations / "gaussian.csv"
    drift = observations / "drift.csv"
    gaussian.write_text("time,freq\n2025-01-24T04:48:30Z,149\n", encoding="utf-8")
    drift.write_text(
        "label,t_start,f_start_mhz,t_end,f_end_mhz\n"
        "d1,2025-01-24T04:48:30Z,300,2025-01-24T04:48:35Z,149\n",
        encoding="utf-8",
    )

    drift_launch = adapter.build_drift_rate(
        t_start="2025-01-24T04:48:30Z",
        f_start_mhz=300,
        t_end="2025-01-24T04:48:35Z",
        f_end_mhz=149,
    )
    newkirk = adapter.build_newkirk_diagnostics(
        gaussian_csv=gaussian,
        drift_csv=drift,
    )

    assert drift_launch.python_module.endswith("drift_selection_cli")
    raw = drift_launch.arguments[drift_launch.arguments.index("--drift-lines-json") + 1]
    assert json.loads(raw)[0]["f_end_mhz"] == 149.0
    assert newkirk.python_module.endswith("physical_diagnostics_cli")
    assert newkirk.arguments[newkirk.arguments.index("--gaussian-csv") + 1] == str(
        gaussian
    )
    assert newkirk.arguments[newkirk.arguments.index("--drift-csv") + 1] == str(drift)
    assert newkirk.output_dir.is_relative_to(layout.outputs_dir / "app_v1")


def test_trajectory_and_dem_launchers_keep_science_in_existing_modules(
    tmp_path: Path,
) -> None:
    adapter, observations, layout = _adapter(tmp_path)
    centers = observations / "centers.csv"
    centers.write_text(
        "time,freq,polarization,center_x_arcsec,center_y_arcsec\n"
        "2025-01-24T04:48:30Z,149,RR,1,2\n",
        encoding="utf-8",
    )
    aia_dir = observations / "aia"
    aia_dir.mkdir()
    aia = aia_dir / "aia.fits"
    tb = observations / "tb.npy"
    radio = observations / "radio.fits"
    for path in (aia, tb, radio):
        path.write_bytes(b"fixture")

    interactive = adapter.build_source_trajectory(
        centers,
        aia_dir=aia_dir,
        frame_mode="all",
        tail_n=7,
        width=800,
        height=600,
        theme="dark",
        max_frames=20,
    )
    media = adapter.build_trajectory_media(
        centers,
        aia_dir=aia_dir,
        output_format="webm",
        frame_mode="all",
        tail_n=7,
        fps=12,
        width=800,
        height=600,
        theme="dark",
    )
    export = adapter.build_trajectory_export(centers, aia_dir=aia_dir, tail_n=8)
    dem = adapter.build_dem_radio_overlay(
        aia_fits=aia,
        tb_data=tb,
        radio_file=radio,
    )

    assert interactive.python_module.endswith("trajectory_preview_worker")
    assert "--browser" not in interactive.arguments
    assert interactive.arguments[interactive.arguments.index("--frame-mode") + 1] == (
        "all"
    )
    assert interactive.arguments[interactive.arguments.index("--max-frames") + 1] == (
        "20"
    )
    assert media.python_module.endswith("trajectory_media_cli")
    assert media.arguments[media.arguments.index("--format") + 1] == "webm"
    assert "--use-aia" in media.arguments
    assert export.python_module.endswith("trajectory_cli")
    assert export.arguments[export.arguments.index("--tail-n") + 1] == "8"
    assert dem.python_module.endswith("dem_radio_cli")
    assert dem.arguments[dem.arguments.index("--radio-file") + 1] == str(radio)
    assert all(
        item.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
        for item in (interactive, media, export, dem)
    )


def test_phase2c_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    adapter, _observations, _layout = _adapter(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("time,freq\n", encoding="utf-8")

    try:
        adapter.build_source_trajectory(outside)
    except PermissionError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("outside path was accepted")
