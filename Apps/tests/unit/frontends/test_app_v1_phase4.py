"""Phase 4 adapter and subprocess renderer tests without importing Qt."""

from __future__ import annotations

import os
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


def test_adapter_persists_schema1_and_builds_private_exports(tmp_path: Path) -> None:
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
    assert load_project(project_path).schema_version == 1
    assert static.python_module.endswith("composer_worker")
    assert static.arguments[static.arguments.index("--mode") + 1] == "static"
    assert static.arguments[static.arguments.index("--scale") + 1] == "3"
    assert static.output_dir.is_relative_to(local / "outputs" / "app_v1")
    assert sequence.arguments[sequence.arguments.index("--mode") + 1] == "sequence"
    assert "canvas=128x96" in sequence.summary


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
