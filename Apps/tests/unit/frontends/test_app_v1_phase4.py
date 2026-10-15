"""Phase 4 adapter and subprocess renderer tests without importing Qt."""

from __future__ import annotations

import os
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image
import pytest

from solar_apps.frontends.app_v1.composer_worker import main as worker_main
from solar_apps.frontends.app_v1.phase4 import Phase4ComposerAdapter
from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.models import (
    CanvasSettings,
    ComposerProject,
    FolderSource,
    LayoutSlot,
    MatchSettings,
)
from solar_apps.frontends.image_composer.project import load_project, save_project
from solar_apps.frontends.image_composer.matching import StaleMatchPlanError
from solar_apps.platform.layout import RuntimeLayout


def _folder(root: Path, folder_id: str, colors: list[str]) -> FolderSource:
    root.mkdir(parents=True)
    base = datetime(2026, 7, 24, 12, 0, 0)
    for index, color in enumerate(colors):
        observed = base + timedelta(seconds=index)
        Image.new("RGB", (40, 20), color).save(
            root / f"{folder_id}_{observed:%Y%m%d_%H%M%S}.png"
        )
    records = scan_folder(root)
    return FolderSource(
        id=folder_id,
        path=root,
        name=folder_id,
        records=records,
        start_index=1,
        end_index=len(records),
    )


def _project(tmp_path: Path, *, frames: int = 1) -> ComposerProject:
    folder = _folder(tmp_path / "images", "camera", ["red", "green"][:frames])
    return ComposerProject(
        canvas=CanvasSettings(width=64, height=48, background="#000000"),
        folders=[folder],
        slots=[
            LayoutSlot.create(
                folder.id,
                1,
                x=0,
                y=0,
                width=64,
                height=48,
            )
        ],
        matching=MatchSettings(master_folder_id=folder.id),
    )


def test_adapter_persists_schema2_and_builds_private_exports(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    local = tmp_path / "Local"
    layout = RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(local)},
    )
    project = _project(tmp_path / "observations")
    adapter = Phase4ComposerAdapter(
        layout,
        allowed_roots=(tmp_path / "observations",),
    )

    static = adapter.build_static_export(project, scale=3)
    sequence = adapter.build_sequence_export(project, scale=2, fps=7.5)

    project_path = Path(static.arguments[static.arguments.index("--project") + 1])
    assert load_project(project_path).schema_version == 2
    assert static.python_module.endswith("composer_worker")
    assert static.arguments[static.arguments.index("--mode") + 1] == "static"
    assert static.arguments[static.arguments.index("--scale") + 1] == "3"
    assert static.output_dir.is_relative_to(local / "outputs" / "app_v1")
    assert sequence.arguments[sequence.arguments.index("--mode") + 1] == "sequence"
    assert "canvas=128x96" in sequence.summary


def test_preparing_and_discarding_second_export_preserves_first_snapshot(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")}
    )
    adapter = Phase4ComposerAdapter(layout, allowed_roots=(tmp_path,))
    project = _project(tmp_path / "observations", frames=2)
    first = adapter.build_static_export(project)
    first_path = Path(first.arguments[first.arguments.index("--project") + 1])
    original_bytes = first_path.read_bytes()
    adapter.confirm_prepared_export(first)
    project.slots[0].preview_ordinal = 2
    second = adapter.build_static_export(project)
    second_path = Path(second.arguments[second.arguments.index("--project") + 1])

    assert second_path != first_path
    adapter.discard_prepared_export(second)
    adapter.discard_prepared_export(first)
    assert not second_path.exists()
    assert adapter._prepared_projects == set()
    assert first_path.read_bytes() == original_bytes
    assert load_project(first_path).slots[0].preview_ordinal == 1
    assert first.arguments[first.arguments.index("--project-sha256") + 1] == (
        hashlib.sha256(original_bytes).hexdigest()
    )
    output = first.output_dir / "images" / "composition.png"
    assert worker_main(list(first.arguments)) == 0
    with Image.open(output) as image:
        assert image.getpixel((32, 24)) == (255, 0, 0)


def test_export_worker_rejects_changed_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    layout = RuntimeLayout.discover(
        repo, environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")}
    )
    adapter = Phase4ComposerAdapter(layout, allowed_roots=(tmp_path,))
    project = _project(tmp_path / "observations")
    launch = adapter.build_static_export(project)
    project_path = Path(launch.arguments[launch.arguments.index("--project") + 1])
    project.canvas.background = "#0000ff"
    save_project(project_path, project)

    with pytest.raises(ValueError, match="snapshot SHA-256 mismatch"):
        worker_main(list(launch.arguments))


def test_adapter_accepts_explicit_avi_inside_roots_and_rejects_escape(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "Apps").mkdir(parents=True)
    (repo / "Python").mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    project = _project(allowed / "observations")
    layout = RuntimeLayout.discover(
        repo,
        environ={"SOLAR_APPS_LOCAL_ROOT": str(tmp_path / "Local")},
    )
    adapter = Phase4ComposerAdapter(layout, allowed_roots=(allowed,))
    project.export.output_format = "avi"

    launch = adapter.build_sequence_export(
        project,
        output_path=allowed / "composition.avi",
    )
    assert launch.arguments[launch.arguments.index("--output") + 1].endswith(
        "composition.avi"
    )

    with pytest.raises(PermissionError, match="outside configured allowed roots"):
        adapter.build_sequence_export(
            project,
            output_path=tmp_path / "escaped.avi",
        )


def test_static_worker_imports_fic_schema1_and_exports_high_resolution_png(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "observations")
    project_path = save_project(tmp_path / "composition.fic.json", project)
    output = tmp_path / "output" / "composition.png"

    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "static",
                "--output",
                str(output),
                "--scale",
                "3",
                "--allowed-roots",
                os.pathsep.join((str(tmp_path), str(output.parent))),
            ]
        )
        == 0
    )

    with Image.open(output) as image:
        assert image.size == (192, 144)
        assert image.convert("RGB").getpixel((96, 72)) == (255, 0, 0)


def test_sequence_worker_reuses_matching_and_atomic_video_export(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "observations", frames=2)
    project_path = save_project(tmp_path / "sequence.fic.json", project)
    output = tmp_path / "output" / "composition.mp4"

    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "sequence",
                "--output",
                str(output),
                "--scale",
                "1",
                "--fps",
                "4",
                "--save-png-frames",
                "--allowed-roots",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert output.stat().st_size > 0
    assert output.with_name("composition_matches.csv").is_file()
    assert len(list(output.with_name("composition_frames").glob("frame_*.png"))) == 2


def test_worker_rejects_input_change_after_preview_snapshot(tmp_path: Path) -> None:
    project = _project(tmp_path / "observations", frames=2)
    project_path = save_project(tmp_path / "sequence.fic.json", project)
    changed = project.folders[0].records[0].path
    Image.new("RGB", (44, 24), "blue").save(changed)
    output = tmp_path / "composition.mp4"
    output.write_bytes(b"existing output")
    with pytest.raises(StaleMatchPlanError, match="stale"):
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "sequence",
                "--output",
                str(output),
                "--allowed-roots",
                str(tmp_path),
            ]
        )
    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob("*.partial*"))


def test_static_worker_uses_filtered_exact_master_with_clock_offset(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "observations", frames=2)
    project.matching.time_start = "2026-07-24T12:00:00Z"
    project.matching.time_end = "2026-07-24T12:00:02Z"
    project.folders[0].offset_seconds = 100
    project.slots[0].preview_ordinal = 2
    project_path = save_project(tmp_path / "composition.fic.json", project)
    output = tmp_path / "composition.png"
    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "static",
                "--output",
                str(output),
                "--allowed-roots",
                str(tmp_path),
            ]
        )
        == 0
    )
    with Image.open(output) as image:
        assert image.convert("RGB").getpixel((32, 24)) == (0, 128, 0)


def test_static_worker_does_not_use_preview_image_outside_window(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path / "observations", frames=2)
    project.matching.time_start = "2026-07-24T12:00:01Z"
    project.matching.time_end = "2026-07-24T12:00:02Z"
    project_path = save_project(tmp_path / "composition.fic.json", project)
    output = tmp_path / "composition.png"
    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "static",
                "--output",
                str(output),
                "--allowed-roots",
                str(tmp_path),
            ]
        )
        == 0
    )
    with Image.open(output) as image:
        assert image.convert("RGB").getpixel((32, 24)) == (0, 128, 0)


def test_hierarchical_static_preview_and_sequence_export_use_same_inherited_records(
    tmp_path: Path,
) -> None:
    folders = []
    for source_id, seconds, colors in (
        ("root", [0, 10], ["red", "blue"]),
        ("middle", [0, 6, 10], ["yellow", "cyan", "magenta"]),
        (
            "output",
            [0, 1, 2, 4, 5, 10],
            ["white", "green", "orange", "purple", "pink", "black"],
        ),
    ):
        directory = tmp_path / source_id
        directory.mkdir()
        for second, color in zip(seconds, colors, strict=True):
            Image.new("RGB", (40, 20), color).save(
                directory / f"{source_id}_20260724_1200{second:02d}.png"
            )
        records = scan_folder(directory)
        folders.append(
            FolderSource(
                id=source_id,
                path=directory,
                name=source_id,
                records=records,
                end_index=len(records),
            )
        )
    project = ComposerProject(
        canvas=CanvasSettings(192, 48, "#000000"),
        folders=folders,
        slots=[
            LayoutSlot.create(folder.id, 1, x=index * 64, y=0, width=64, height=48)
            for index, folder in enumerate(folders)
        ],
        matching=MatchSettings(
            time_start="2026-07-24T12:00:00Z", time_end="2026-07-24T12:00:11Z"
        ),
    )
    project.slots[-1].preview_ordinal = 4
    project_path = save_project(tmp_path / "hierarchy.fic.json", project)
    static = tmp_path / "static.png"
    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "static",
                "--output",
                str(static),
                "--allowed-roots",
                str(tmp_path),
            ]
        )
        == 0
    )
    movie = tmp_path / "hierarchy.avi"
    assert (
        worker_main(
            [
                "--project",
                str(project_path),
                "--mode",
                "sequence",
                "--output",
                str(movie),
                "--allowed-roots",
                str(tmp_path),
                "--save-png-frames",
            ]
        )
        == 0
    )
    with (
        Image.open(static) as preview,
        Image.open(tmp_path / "hierarchy_frames" / "frame_000004.png") as exported,
    ):
        assert preview.tobytes() == exported.tobytes()
        assert preview.convert("RGB").getpixel((32, 24)) == (0, 0, 255)
        assert preview.convert("RGB").getpixel((96, 24)) == (0, 255, 255)
        assert preview.convert("RGB").getpixel((160, 24)) == (128, 0, 128)
    assert len(list((tmp_path / "hierarchy_frames").glob("frame_*.png"))) == 6
