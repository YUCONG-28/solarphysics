from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import cv2
from PIL import Image

from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.models import (
    CanvasSettings,
    ComposerProject,
    ExportSettings,
    FolderSource,
    LayoutSlot,
    MatchSettings,
)
from solar_apps.frontends.image_composer.rendering import (
    ExportCancelled,
    ExportError,
    compose_frame,
    export_project,
    load_oriented_rgba,
    render_slot_tile,
)


def _write(path: Path, color: tuple[int, int, int], size=(20, 20)) -> None:
    Image.new("RGB", size, color).save(path)


def _folder(
    folder_id: str, path: Path, colors: list[tuple[int, int, int]]
) -> FolderSource:
    path.mkdir(parents=True)
    base = datetime(2026, 7, 17, 12, 0, 0)
    for index, color in enumerate(colors):
        timestamp = base + timedelta(seconds=index)
        _write(path / f"{folder_id}_{timestamp:%Y%m%d_%H%M%S}.png", color)
    records = scan_folder(path)
    return FolderSource(
        id=folder_id,
        path=path,
        name=folder_id,
        records=records,
        start_index=1,
        end_index=len(records),
    )


def test_slot_fit_modes_and_z_order(tmp_path: Path) -> None:
    wide = tmp_path / "wide.png"
    _write(wide, (255, 0, 0), size=(40, 10))
    image = load_oriented_rgba(wide)
    contained = render_slot_tile(image, 20, 20, "contain")
    covered = render_slot_tile(image, 20, 20, "cover")
    stretched = render_slot_tile(image, 20, 20, "stretch")

    assert contained.getpixel((0, 0))[3] == 0
    assert contained.getpixel((10, 10))[:3] == (255, 0, 0)
    assert covered.getpixel((0, 0))[:3] == (255, 0, 0)
    assert stretched.size == (20, 20)

    red = _folder("red", tmp_path / "red", [(255, 0, 0)])
    blue = _folder("blue", tmp_path / "blue", [(0, 0, 255)])
    project = ComposerProject(
        canvas=CanvasSettings(width=40, height=40, background="#000000"),
        folders=[red, blue],
        slots=[
            LayoutSlot.create("red", 1, x=0, y=0, width=40, height=40, z_index=0),
            LayoutSlot.create("blue", 1, x=0, y=0, width=40, height=40, z_index=1),
        ],
        matching=MatchSettings(master_folder_id="red"),
    )
    frame = compose_frame(project, {"red": red.records[0], "blue": blue.records[0]})
    assert frame.getpixel((20, 20)) == (0, 0, 255)


def test_static_worker_uses_each_slot_ordinal_within_one_folder(
    tmp_path: Path,
) -> None:
    from solar_apps.frontends.app_v1.composer_worker import _static

    folder = _folder("camera", tmp_path / "camera", [(255, 0, 0), (0, 0, 255)])
    slots = [
        LayoutSlot.create("camera", 1, x=0, y=0, width=20, height=20),
        LayoutSlot.create("camera", 2, x=20, y=0, width=20, height=20),
    ]
    project = ComposerProject(
        canvas=CanvasSettings(width=40, height=20), folders=[folder], slots=slots
    )
    output = tmp_path / "composition.png"
    _static(project, output)

    with Image.open(output) as image:
        assert image.getpixel((10, 10)) == (255, 0, 0)
        assert image.getpixel((30, 10)) == (0, 0, 255)


def test_slot_records_override_folder_match_without_changing_folder_default(
    tmp_path: Path,
) -> None:
    folder = _folder("camera", tmp_path / "camera", [(255, 0, 0), (0, 0, 255)])
    slots = [
        LayoutSlot.create("camera", 1, x=0, y=0, width=20, height=20),
        LayoutSlot.create("camera", 2, x=20, y=0, width=20, height=20),
    ]
    project = ComposerProject(
        canvas=CanvasSettings(width=40, height=20), folders=[folder], slots=slots
    )
    frame = compose_frame(
        project,
        {"camera": folder.records[0]},
        slot_records={slots[1].id: folder.records[1]},
    )

    assert frame.getpixel((10, 10)) == (255, 0, 0)
    assert frame.getpixel((30, 10)) == (0, 0, 255)


def test_exif_orientation_rotation_and_opacity_match_export_geometry(
    tmp_path: Path,
) -> None:
    oriented_path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (10, 20), "white").save(oriented_path, exif=exif)
    assert load_oriented_rgba(oriented_path).size == (20, 10)

    red = _folder("red", tmp_path / "rotated", [(255, 0, 0)])
    slot = LayoutSlot.create("red", 1, x=10, y=15, width=20, height=10, z_index=0)
    slot.rotation = 90
    slot.opacity = 0.5
    slot.fit = "stretch"
    project = ComposerProject(
        canvas=CanvasSettings(width=40, height=40, background="#000000"),
        folders=[red],
        slots=[slot],
        matching=MatchSettings(master_folder_id="red"),
    )

    frame = compose_frame(project, {"red": red.records[0]})

    center = frame.getpixel((20, 20))
    assert 126 <= center[0] <= 129
    assert center[1:] == (0, 0)
    assert frame.getpixel((20, 12))[0] > 120
    assert frame.getpixel((12, 20)) == (0, 0, 0)


@pytest.mark.parametrize("output_format", ["mp4", "avi"])
def test_end_to_end_video_csv_and_png_export(
    tmp_path: Path, output_format: str
) -> None:
    folder = _folder(
        "camera",
        tmp_path / f"camera_{output_format}",
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],
    )
    output = tmp_path / f"movie.{output_format}"
    project = ComposerProject(
        canvas=CanvasSettings(width=64, height=48, background="#000000"),
        folders=[folder],
        slots=[LayoutSlot.create("camera", 1, x=0, y=0, width=64, height=48)],
        matching=MatchSettings(master_folder_id="camera"),
        export=ExportSettings(
            output_path=str(output),
            output_format=output_format,
            fps=4.0,
            save_png_frames=True,
        ),
    )

    result = export_project(project)

    assert result.status == "saved"
    assert result.emitted_frames == 3
    assert output.stat().st_size > 0
    reader = cv2.VideoCapture(str(output))
    decoded = []
    try:
        while True:
            available, frame = reader.read()
            if not available:
                break
            assert frame.shape[:2] == (48, 64)
            decoded.append(tuple(int(channel) for channel in frame[24, 32][::-1]))
    finally:
        reader.release()
    assert len(decoded) == 3
    for actual, expected in zip(
        decoded, [(255, 0, 0), (0, 255, 0), (0, 0, 255)], strict=True
    ):
        assert all(abs(value - target) < 20 for value, target in zip(actual, expected))
    assert len(list(result.frames_path.glob("frame_*.png"))) == 3
    with result.csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [row["output_frame_index"] for row in rows] == ["1", "2", "3"]


def test_sequence_same_folder_slots_keep_shared_time_matching(tmp_path: Path) -> None:
    folder = _folder("camera", tmp_path / "camera", [(255, 0, 0), (0, 0, 255)])
    project = ComposerProject(
        canvas=CanvasSettings(width=40, height=20),
        folders=[folder],
        slots=[
            LayoutSlot.create("camera", 1, x=0, y=0, width=20, height=20),
            LayoutSlot.create("camera", 2, x=20, y=0, width=20, height=20),
        ],
        matching=MatchSettings(master_folder_id="camera"),
        export=ExportSettings(
            output_path=str(tmp_path / "movie.mp4"), save_png_frames=True
        ),
    )
    result = export_project(project)
    for frame_path, color in zip(
        sorted(result.frames_path.glob("frame_*.png")),
        [(255, 0, 0), (0, 0, 255)],
        strict=True,
    ):
        with Image.open(frame_path) as image:
            assert image.getpixel((10, 10)) == color
            assert image.getpixel((30, 10)) == color


def test_all_strict_frames_skipped_publishes_csv_only(tmp_path: Path) -> None:
    master = _folder("master", tmp_path / "master", [(255, 0, 0)])
    other = _folder("other", tmp_path / "other", [(0, 0, 255)])
    other.offset_seconds = 10
    output = tmp_path / "strict.mp4"
    project = ComposerProject(
        canvas=CanvasSettings(width=64, height=48),
        folders=[master, other],
        slots=[LayoutSlot.create("other", 1, x=0, y=0, width=64, height=48)],
        matching=MatchSettings(
            master_folder_id="master", tolerance_seconds=1.0, strict=True
        ),
        export=ExportSettings(output_path=str(output), output_format="mp4"),
    )

    result = export_project(project)

    assert result.status == "no_frames"
    assert not output.exists()
    assert result.csv_path.exists()


def test_bad_image_preserves_existing_output_and_cleans_partials(
    tmp_path: Path,
) -> None:
    folder_path = tmp_path / "broken"
    folder_path.mkdir()
    image_path = folder_path / "broken_20260717_120000.png"
    image_path.write_bytes(b"not an image")
    records = scan_folder(folder_path)
    folder = FolderSource(
        id="broken",
        path=folder_path,
        name="broken",
        records=records,
        start_index=1,
        end_index=1,
    )
    output = tmp_path / "movie.mp4"
    output.write_bytes(b"existing output")
    project = ComposerProject(
        canvas=CanvasSettings(width=64, height=48),
        folders=[folder],
        slots=[LayoutSlot.create("broken", 1, x=0, y=0, width=64, height=48)],
        matching=MatchSettings(master_folder_id="broken"),
        export=ExportSettings(output_path=str(output), output_format="mp4"),
    )

    with pytest.raises(ExportError, match="Could not decode"):
        export_project(project)

    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob("*.partial*"))


def test_cancelled_export_preserves_existing_output_and_cleans_partials(
    tmp_path: Path,
) -> None:
    folder = _folder("camera", tmp_path / "camera_cancel", [(255, 0, 0)])
    output = tmp_path / "cancel.mp4"
    output.write_bytes(b"existing output")
    project = ComposerProject(
        canvas=CanvasSettings(width=64, height=48),
        folders=[folder],
        slots=[LayoutSlot.create("camera", 1, x=0, y=0, width=64, height=48)],
        matching=MatchSettings(master_folder_id="camera"),
        export=ExportSettings(output_path=str(output), output_format="mp4"),
    )

    with pytest.raises(ExportCancelled):
        export_project(project, cancelled=lambda: True)

    assert output.read_bytes() == b"existing output"
    assert not list(tmp_path.glob("*.partial*"))
