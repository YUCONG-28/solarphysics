"""Phase 2A adapter tests without importing PyQt6 into pytest."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from solar_apps.frontends.app_v1.phase2a import Phase2AAdapter
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.workflows.aia.application import build_parser, config_from_args


def _layout(tmp_path: Path) -> RuntimeLayout:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    return RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )


def test_image_selection_reuses_existing_natural_sort_and_allowed_roots(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    images = tmp_path / "observations" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(images / "frame10.png")
    Image.new("RGB", (4, 4), "blue").save(images / "frame2.png")
    adapter = Phase2AAdapter(layout, allowed_roots=(tmp_path / "observations",))

    selection = adapter.select_images(images)

    assert [item.name for item in selection.images] == ["frame2.png", "frame10.png"]
    assert "Workload: 2 image(s)" in selection.summary
    assert "Output: none" in selection.summary


def test_image_groups_and_media_export_use_one_synchronized_worker(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    observations = tmp_path / "observations"
    first = observations / "171"
    second = observations / "193"
    first.mkdir(parents=True)
    second.mkdir()
    Image.new("RGB", (4, 4), "red").save(first / "frame1.png")
    Image.new("RGB", (4, 4), "blue").save(second / "frame1.png")
    adapter = Phase2AAdapter(layout, allowed_roots=(observations,))

    selection = adapter.select_image_groups((first, second))
    launch = adapter.build_image_export(
        selection,
        output_format="webm",
        composite=False,
        fps=12.5,
        workers=2,
    )

    assert len(selection.groups) == 2
    assert "groups=2" in selection.summary
    assert launch.python_module == ("solar_apps.frontends.app_v1.image_viewer_worker")
    assert launch.arguments.count("--folder") == 2
    assert launch.arguments[launch.arguments.index("--format") + 1] == "webm"
    assert launch.arguments[launch.arguments.index("--mode") + 1] == "separate"
    assert launch.arguments[launch.arguments.index("--fps") + 1] == "12.5"
    assert launch.arguments[launch.arguments.index("--workers") + 1] == "2"
    assert launch.output_dir.is_relative_to(layout.outputs_dir / "app_v1")


def test_aia_launch_targets_existing_workflow_and_private_output(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    observations = tmp_path / "observations"
    aia = observations / "AIA"
    aia.mkdir(parents=True)
    (aia / "aia_171.fits").write_bytes(b"fixture")
    adapter = Phase2AAdapter(layout, allowed_roots=(observations,))

    launch = adapter.build_aia(
        aia,
        mode="test",
        waves=(171,),
        test_index=0,
        workers=1,
    )

    assert launch.python_module == "solar_apps.workflows.aia.application"
    assert launch.arguments[launch.arguments.index("--data-path") + 1] == str(aia)
    assert launch.arguments[launch.arguments.index("--output-dir") + 1] == str(
        launch.output_dir
    )
    assert launch.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert not launch.output_dir.exists()
    assert "Workload: 1 FITS candidate(s)" in launch.summary
    assert launch.arguments[launch.arguments.index("--test-index") + 1] == "0"
    assert launch.arguments[launch.arguments.index("--workers") + 1] == "1"


def test_hmi_launch_validates_both_inputs_and_uses_existing_overlay(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    observations = tmp_path / "observations"
    aia = observations / "AIA"
    hmi = observations / "HMI"
    aia.mkdir(parents=True)
    hmi.mkdir()
    (aia / "aia.fits").write_bytes(b"aia")
    (hmi / "hmi.fits").write_bytes(b"hmi")
    adapter = Phase2AAdapter(layout, allowed_roots=(observations,))

    launch = adapter.build_hmi_overlay(aia, hmi, dpi=200)

    assert launch.python_module == "solar_apps.workflows.hmi.overlay_cli"
    assert "--no-show-plot" in launch.arguments
    assert launch.output_dir.is_relative_to(layout.outputs_dir / "app_v1")
    assert "1 AIA and 1 HMI" in launch.summary


def test_aia_cli_output_dir_overrides_historical_data_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    data = tmp_path / "observations" / "AIA"
    output = tmp_path / "Local" / "outputs" / "app_v1" / "aia"
    data.mkdir(parents=True)
    args = build_parser().parse_args(
        [
            "--data-path",
            str(data),
            "--output-dir",
            str(output),
            "--mode",
            "test",
            "--waves",
            "171",
        ]
    )

    config = config_from_args(args)

    assert Path(config.data_path) == data
    assert Path(config.output_dir) == output
