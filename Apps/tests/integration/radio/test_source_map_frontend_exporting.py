from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from PIL import Image
from solar_toolkit.visualization.media import resolve_ffmpeg, resolve_ffprobe

from solar_apps.workflows.radio.artifacts import (
    COORDINATE_SYSTEM,
    sha256_file,
    sidecar_path_for,
    validate_roi_set,
    validate_source_map_artifact,
)
from solar_apps.frontends.radio.source_map.export_jobs import ExportJobRegistry
from solar_apps.frontends.radio.source_map.exporting import (
    ExportCancelled,
    ExportConflictError,
    export_source_maps,
    freeze_artifact_record,
    preflight_export_destination,
    rasterize_roi_overlay,
    scan_external_artifact_directory,
    scientific_utc_from_filename,
    validate_roi_template,
)


def _artifact(
    root: Path,
    name: str,
    *,
    color: tuple[int, int, int] = (20, 30, 40),
    size: tuple[int, int] = (120, 80),
    panels: int = 1,
) -> dict:
    image_path = root / name
    Image.new("RGB", size, color).save(image_path, format="PNG")
    panel_records = []
    for index in range(panels):
        panel_width = 0.8 / panels
        left = 0.1 + index * panel_width
        panel_records.append(
            {
                "id": f"radio-{index + 1}",
                "frequency_mhz": 149.0 + index,
                "bbox_normalized": [left, 0.1, left + panel_width, 0.9],
                "xlim_arcsec": [-100.0, 100.0],
                "ylim_arcsec": [-50.0, 50.0],
            }
        )
    metadata = {
        "schema_version": 1,
        "coordinate_system": COORDINATE_SYSTEM,
        "created_at": "2026-07-17T00:00:00+00:00",
        "image": {
            "filename": image_path.name,
            "width": size[0],
            "height": size[1],
            "sha256": sha256_file(image_path),
        },
        "mode": "multi_band" if panels > 1 else "single_band",
        "polarization": "RR",
        "source_files": ["input.fits"],
        "warnings": [],
        "panels": panel_records,
    }
    sidecar = sidecar_path_for(image_path)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        "id": image_path.stem,
        "image_path": image_path,
        "sidecar_path": sidecar,
        "metadata": metadata,
    }


def _roi(image_sha256: str, *, outside: bool = False) -> dict:
    left, right = (200.0, 250.0) if outside else (-30.0, 30.0)
    return {
        "schema_version": 1,
        "coordinate_system": COORDINATE_SYSTEM,
        "image_sha256": image_sha256,
        "rois": [
            {
                "id": "roi-1",
                "name": "Burst",
                "type": "rectangle",
                "geometry": {
                    "left": left,
                    "right": right,
                    "bottom": -20.0,
                    "top": 20.0,
                },
                "visible": True,
                "style": {
                    "color": "#00d4ff",
                    "line_width": 3,
                    "show_label": True,
                },
            },
            {
                "id": "roi-2",
                "name": "Loop",
                "type": "lasso",
                "geometry": {"points": [[-60.0, -10.0], [0.0, 35.0], [60.0, -10.0]]},
                "visible": True,
                "style": {
                    "color": "yellow",
                    "line_width": 2,
                    "show_label": False,
                },
            },
        ],
    }


def test_external_directory_scan_validates_and_sorts(tmp_path: Path) -> None:
    later = _artifact(tmp_path, "map_20000101T044832Z.png")
    earlier = _artifact(tmp_path, "map_20000101T044831Z.png")
    natural_ten = _artifact(tmp_path, "map10.png")
    natural_two = _artifact(tmp_path, "map2.png")

    scanned = scan_external_artifact_directory(tmp_path)

    assert [item.image_path.name for item in scanned] == [
        Path(earlier["image_path"]).name,
        Path(later["image_path"]).name,
        Path(natural_two["image_path"]).name,
        Path(natural_ten["image_path"]).name,
    ]
    assert scientific_utc_from_filename("0001_20000101T044829Z_radio.png") is not None
    assert scientific_utc_from_filename("bad_20250230T120000Z.png") is None


def test_external_directory_scan_fails_closed_for_missing_sidecar(
    tmp_path: Path,
) -> None:
    _artifact(tmp_path, "valid.png")
    Image.new("RGB", (20, 20)).save(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        scan_external_artifact_directory(tmp_path)


def test_roi_template_cross_image_hash_requires_explicit_template_mode() -> None:
    payload = _roi("historical-sha")
    with pytest.raises(ValueError, match="does not match"):
        validate_roi_template(
            payload,
            expected_image_sha256="current-sha",
            template_mode=False,
        )

    normalized = validate_roi_template(
        payload,
        expected_image_sha256="current-sha",
        template_mode=True,
    )
    assert normalized["image_sha256"] == "current-sha"
    assert normalized["provenance"]["template_source_image_sha256"] == "historical-sha"


def test_pillow_rasterizer_projects_across_panels_and_preserves_source(
    tmp_path: Path,
) -> None:
    record = _artifact(tmp_path, "source.png", panels=2)
    original_sha = sha256_file(record["image_path"])
    template = _roi("historical-sha")

    rendered, warnings = rasterize_roi_overlay(record, template)

    assert rendered.size == (120, 80)
    assert sha256_file(record["image_path"]) == original_sha
    output = tmp_path / "rendered.png"
    rendered.save(output)
    assert sha256_file(output) != original_sha
    assert warnings == []

    outside, warnings = rasterize_roi_overlay(
        record, {**template, "rois": [_roi("x", outside=True)["rois"][0]]}
    )
    assert outside.size == rendered.size
    assert len(warnings) == 2
    assert all("frame retained" in warning for warning in warnings)


def test_roi_sequence_export_is_atomic_and_writes_matching_contracts(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    records = [
        _artifact(sources, f"map_20000101T04483{index}Z.png", color=(index, 20, 30))
        for index in range(3)
    ]
    output_root = tmp_path / "exports"
    output_root.mkdir()
    source_hashes = [sha256_file(record["image_path"]) for record in records]

    result = export_source_maps(
        records,
        export_kind="image_sequence",
        content="roi",
        destination=output_root,
        roi_template=_roi("historical-sha"),
        start_index=2,
        end_index=3,
    )

    output_dir = Path(result["output_dir"])
    assert output_dir.parent == output_root
    assert len(result["files"]) == 2
    assert all(Path(path).name.endswith("_annotated.png") for path in result["files"])
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["range"] == {
        "start_index": 2,
        "end_index": 3,
        "inclusive": True,
    }
    assert [frame["selection_index"] for frame in manifest["frames"]] == [2, 3]
    assert manifest["roi_source"]["template_source_image_sha256"] == "historical-sha"
    for image_text in result["files"]:
        image = Path(image_text)
        metadata = validate_source_map_artifact(image)
        roi = json.loads(image.with_suffix(".roi-set.json").read_text(encoding="utf-8"))
        validate_roi_set(roi, expected_image_sha256=metadata["image"]["sha256"])
        assert roi["provenance"]["template_source_image_sha256"] == "historical-sha"
    assert [sha256_file(record["image_path"]) for record in records] == source_hashes
    assert not list(output_root.glob(".*.partial-*"))


def test_single_image_conflict_and_cancel_leave_existing_files_unchanged(
    tmp_path: Path,
) -> None:
    record = _artifact(tmp_path, "source.png")
    destination = tmp_path / "selected.png"
    destination.write_bytes(b"existing")

    with pytest.raises(ExportConflictError) as caught:
        export_source_maps(
            [record],
            export_kind="image",
            content="original",
            destination=destination,
        )
    assert caught.value.code == "target_exists"
    assert destination.read_bytes() == b"existing"

    canceled_destination = tmp_path / "canceled.png"
    with pytest.raises(ExportCancelled):
        export_source_maps(
            [record],
            export_kind="image",
            content="original",
            destination=canceled_destination,
            cancel_check=lambda: True,
        )
    assert not canceled_destination.exists()


def test_destination_preflight_expands_companions_and_detects_conflict(
    tmp_path: Path,
) -> None:
    checked = preflight_export_destination(
        export_kind="image",
        content="roi",
        destination=tmp_path / "selected",
    )
    assert Path(checked["destination"]).name == "selected.png"
    assert {Path(path).name for path in checked["targets"]} == {
        "selected.png",
        "selected.source-map.json",
        "selected.roi-set.json",
        "selected.source-map-export.json",
    }
    companion = tmp_path / "selected.source-map-export.json"
    companion.write_text("existing", encoding="utf-8")
    with pytest.raises(ExportConflictError) as caught:
        preflight_export_destination(
            export_kind="image",
            content="roi",
            destination=tmp_path / "selected.png",
        )
    assert caught.value.paths == (str(companion),)


def test_video_export_uses_first_frame_even_size_and_validates_media(
    tmp_path: Path,
) -> None:
    records = [
        _artifact(tmp_path, "frame1.png", size=(121, 81)),
        _artifact(tmp_path, "frame2.png", size=(100, 90)),
    ]
    captured: dict = {}

    def writer(paths, output_path, fps, **kwargs):
        captured.update(paths=list(paths), fps=fps, **kwargs)
        assert all(Image.open(path).size == (120, 80) for path in paths)
        Path(output_path).write_bytes(b"fake-mp4")
        return True

    def probe(path, *, expected_size, expected_frame_count):
        assert Path(path).read_bytes() == b"fake-mp4"
        assert expected_size == (120, 80)
        assert expected_frame_count == 2
        return {
            "codec": "h264",
            "width": 120,
            "height": 80,
            "frame_count": 2,
            "frame_rate": 10.0,
            "duration": 0.2,
        }

    result = export_source_maps(
        records,
        export_kind="video",
        content="original",
        destination=tmp_path / "movie.mp4",
        media_writer=writer,
        media_probe=probe,
    )

    assert Path(result["path"]).read_bytes() == b"fake-mp4"
    assert captured["target_size_tuple"] == (120, 80)
    assert captured["quality"] == "high"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["video"]["codec"] == "h264"
    assert manifest["frame_count"] == 2


def test_video_export_writes_probeable_mp4_when_shared_tools_are_available(
    tmp_path: Path,
) -> None:
    if not resolve_ffmpeg() or not resolve_ffprobe():
        pytest.skip("Shared FFmpeg/FFprobe tools are unavailable")
    records = [
        _artifact(tmp_path, "real-frame1.png", color=(200, 20, 20)),
        _artifact(tmp_path, "real-frame2.png", color=(20, 200, 20)),
    ]

    result = export_source_maps(
        records,
        export_kind="video",
        content="original",
        destination=tmp_path / "real.mp4",
    )

    assert Path(result["path"]).stat().st_size > 0
    assert result["media"]["codec"] == "h264"
    assert result["media"]["frame_count"] == 2
    assert result["media"]["frame_rate"] == pytest.approx(10.0)


def test_export_job_registry_reports_progress_conflict_and_cancellation(
    tmp_path: Path,
) -> None:
    record = _artifact(tmp_path, "source.png")
    destination = tmp_path / "exists.png"
    destination.write_bytes(b"existing")
    registry = ExportJobRegistry()
    conflict = registry.start(
        [record],
        export_kind="image",
        content="original",
        destination=destination,
    )
    conflict = registry.wait(conflict["id"], timeout=5)
    assert conflict["status"] == "failed"
    assert conflict["error_code"] == "target_exists"
    assert conflict["conflict_paths"] == [str(destination)]

    entered = threading.Event()

    def blocking_exporter(_artifacts, *, cancel_check, progress_callback, **_kwargs):
        entered.set()
        progress_callback(1, 2, 1, ["partial warning"])
        while not cancel_check():
            time.sleep(0.005)
        raise ExportCancelled("canceled")

    cancel_registry = ExportJobRegistry(exporter=blocking_exporter)
    second_record = _artifact(tmp_path, "source-two.png")
    started = cancel_registry.start(
        [freeze_artifact_record(record), freeze_artifact_record(second_record)],
        export_kind="video",
        content="original",
        destination=tmp_path / "cancel.mp4",
    )
    assert entered.wait(timeout=2)
    cancel_registry.cancel(started["id"])
    canceled = cancel_registry.wait(started["id"], timeout=5)
    assert canceled["status"] == "canceled"
    assert canceled["completed"] == 1
    assert canceled["current_index"] == 1
    assert canceled["warnings"] == ["partial warning"]
