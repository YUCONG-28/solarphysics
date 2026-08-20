# SPDX-License-Identifier: MIT
"""Pillow composition and OpenCV video export for the image composer."""

from __future__ import annotations

import csv
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageColor, ImageOps

from .matching import (
    FrameMatch,
    MatchPlanError,
    build_match_plan,
    iter_match_rows,
    validate_project,
)
from .models import ComposerProject, ImageRecord

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]

CSV_FIELDS = (
    "attempt_index",
    "output_frame_index",
    "emitted",
    "skip_reason",
    "match_mode",
    "strict",
    "tolerance_seconds",
    "master_folder_id",
    "master_folder",
    "master_file",
    "master_ordinal",
    "master_timestamp",
    "master_time_source",
    "master_offset_seconds",
    "master_corrected_timestamp",
    "slot_id",
    "slot_z_index",
    "source_folder_id",
    "source_folder",
    "source_file",
    "source_ordinal",
    "source_timestamp",
    "source_time_source",
    "source_offset_seconds",
    "source_corrected_timestamp",
    "delta_seconds",
    "abs_delta_seconds",
    "over_tolerance",
)


class ExportError(RuntimeError):
    """Raised for a render, writer, validation, or publishing failure."""


class ExportCancelled(ExportError):
    """Raised when the user cancels a running export."""


@dataclass(slots=True, frozen=True)
class ExportResult:
    status: str
    video_path: Path | None
    csv_path: Path
    frames_path: Path | None
    attempted_frames: int
    emitted_frames: int


def load_oriented_rgba(path: str | Path) -> Image.Image:
    """Decode the first image frame and apply EXIF orientation."""

    candidate = Path(path)
    try:
        with Image.open(candidate) as source:
            source.seek(0)
            return ImageOps.exif_transpose(source).convert("RGBA").copy()
    except (OSError, ValueError) as exc:
        raise ExportError(f"Could not decode image {candidate}: {exc}") from exc


def render_slot_tile(
    image: Image.Image, width: int, height: int, fit: str
) -> Image.Image:
    """Fit one oriented RGBA image into an unrotated transparent slot tile."""

    width = max(1, int(width))
    height = max(1, int(height))
    source = image.convert("RGBA")
    if fit == "stretch":
        return source.resize((width, height), Image.Resampling.LANCZOS)
    if fit == "cover":
        source_ratio = source.width / max(1, source.height)
        target_ratio = width / max(1, height)
        if source_ratio > target_ratio:
            crop_width = max(1, round(source.height * target_ratio))
            left = (source.width - crop_width) // 2
            source = source.crop((left, 0, left + crop_width, source.height))
        else:
            crop_height = max(1, round(source.width / target_ratio))
            top = (source.height - crop_height) // 2
            source = source.crop((0, top, source.width, top + crop_height))
        return source.resize((width, height), Image.Resampling.LANCZOS)
    if fit != "contain":
        raise ExportError(f"Unsupported fit mode: {fit}")
    contained = source.copy()
    contained.thumbnail((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    tile.alpha_composite(
        contained, ((width - contained.width) // 2, (height - contained.height) // 2)
    )
    return tile


def compose_frame(
    project: ComposerProject, matched_records: dict[str, ImageRecord]
) -> Image.Image:
    """Render one RGB canvas from the folder matches for a single frame."""

    try:
        background = ImageColor.getrgb(project.canvas.background)
    except ValueError as exc:
        raise ExportError(
            f"Invalid canvas background: {project.canvas.background}"
        ) from exc
    canvas = Image.new(
        "RGBA", (project.canvas.width, project.canvas.height), (*background, 255)
    )
    for slot in sorted(project.slots, key=lambda item: item.z_index):
        record = matched_records.get(slot.folder_id)
        if record is None:
            raise ExportError(f"No matched image for slot {slot.id}")
        image = load_oriented_rgba(record.path)
        tile = render_slot_tile(
            image, max(1, round(slot.width)), max(1, round(slot.height)), slot.fit
        )
        if slot.opacity < 1:
            alpha = tile.getchannel("A").point(
                lambda value: max(0, min(255, round(value * slot.opacity)))
            )
            tile.putalpha(alpha)
        if slot.rotation % 360:
            tile = tile.rotate(
                -slot.rotation,
                resample=Image.Resampling.BICUBIC,
                expand=True,
                fillcolor=(0, 0, 0, 0),
            )
        center_x = slot.x + slot.width / 2
        center_y = slot.y + slot.height / 2
        destination = (
            round(center_x - tile.width / 2),
            round(center_y - tile.height / 2),
        )
        canvas.alpha_composite(tile, destination)
    return canvas.convert("RGB")


def export_project(
    project: ComposerProject,
    *,
    cancelled: CancelCallback | None = None,
    progress: ProgressCallback | None = None,
) -> ExportResult:
    """Write matching CSV and optional frames/video without replacing early."""

    validate_project(project, require_output=True)
    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda _current, _total, _message: None)
    output_path = Path(project.export.output_path).expanduser().resolve()
    expected_suffix = f".{project.export.output_format}"
    if output_path.suffix.casefold() != expected_suffix:
        raise ExportError(
            f"Output path must end with {expected_suffix} for the selected format."
        )
    if not output_path.parent.is_dir():
        raise ExportError(f"Output folder does not exist: {output_path.parent}")

    plan = build_match_plan(project)
    csv_path = output_path.with_name(f"{output_path.stem}_matches.csv")
    frames_path = output_path.with_name(f"{output_path.stem}_frames")
    temp_video = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    temp_csv = csv_path.with_name(f"{csv_path.stem}.partial{csv_path.suffix}")
    temp_frames = frames_path.with_name(f"{frames_path.name}.partial")
    emitted = sum(frame.emitted for frame in plan)
    writer = None
    try:
        _clean_path(temp_video)
        _clean_path(temp_csv)
        _clean_path(temp_frames)
        _write_match_csv(temp_csv, iter_match_rows(project, plan))
        if emitted == 0:
            os.replace(temp_csv, csv_path)
            progress(len(plan), len(plan), "No frames met the strict tolerance.")
            return ExportResult(
                status="no_frames",
                video_path=None,
                csv_path=csv_path,
                frames_path=None,
                attempted_frames=len(plan),
                emitted_frames=0,
            )

        if project.export.save_png_frames:
            temp_frames.mkdir(parents=False, exist_ok=False)
        writer = _open_video_writer(
            temp_video,
            project.export.output_format,
            project.export.fps,
            (project.canvas.width, project.canvas.height),
        )
        output_count = 0
        for frame in plan:
            if cancelled():
                raise ExportCancelled("Export cancelled by user.")
            if frame.emitted:
                matched = {
                    folder_id: folder_match.record
                    for folder_id, folder_match in frame.matches.items()
                }
                composed = compose_frame(project, matched)
                rgb = np.asarray(composed, dtype=np.uint8)
                writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
                output_count += 1
                if project.export.save_png_frames:
                    composed.save(
                        temp_frames / f"frame_{output_count:06d}.png", format="PNG"
                    )
            progress(frame.attempt_index, len(plan), _progress_message(frame))
        writer.release()
        writer = None
        _validate_video(
            temp_video,
            expected_frames=emitted,
            expected_size=(project.canvas.width, project.canvas.height),
        )

        replacements = [(temp_video, output_path), (temp_csv, csv_path)]
        if project.export.save_png_frames:
            replacements.append((temp_frames, frames_path))
        _publish_replacements(replacements)
        published_frames = frames_path if project.export.save_png_frames else None
        return ExportResult(
            status="saved",
            video_path=output_path,
            csv_path=csv_path,
            frames_path=published_frames,
            attempted_frames=len(plan),
            emitted_frames=emitted,
        )
    except MatchPlanError:
        raise
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(str(exc)) from exc
    finally:
        if writer is not None:
            writer.release()
        _clean_path(temp_video)
        _clean_path(temp_csv)
        _clean_path(temp_frames)


def _open_video_writer(
    path: Path, output_format: str, fps: float, size: tuple[int, int]
):
    try:
        import cv2
    except ImportError as exc:
        raise ExportError("OpenCV is required for video export.") from exc
    codec = "mp4v" if output_format == "mp4" else "MJPG"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), float(fps), size
    )
    if not writer.isOpened():
        writer.release()
        suggestion = (
            "Select AVI and try again."
            if output_format == "mp4"
            else "Select MP4 and try again."
        )
        raise ExportError(
            f"Could not open the {output_format.upper()} encoder. {suggestion}"
        )
    return writer


def _validate_video(
    path: Path, *, expected_frames: int, expected_size: tuple[int, int]
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ExportError("OpenCV is required to validate video output.") from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise ExportError("Video writer produced an empty file.")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ExportError("Generated video could not be reopened.")
        actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        actual_size = (
            int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if actual_frames != expected_frames:
            raise ExportError(
                f"Video frame count mismatch: expected {expected_frames}, got {actual_frames}."
            )
        if actual_size != expected_size:
            raise ExportError(
                f"Video size mismatch: expected {expected_size}, got {actual_size}."
            )
        ok, _frame = capture.read()
        if not ok:
            raise ExportError("Generated video contains no readable first frame.")
    finally:
        capture.release()


def _write_match_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _progress_message(frame: FrameMatch) -> str:
    if frame.emitted:
        return f"Rendered output frame {frame.output_frame_index}."
    return f"Skipped master frame {frame.attempt_index}: {frame.skip_reason}"


def _publish_replacements(replacements: list[tuple[Path, Path]]) -> None:
    """Publish a set and restore every previous final if any move fails."""

    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for _temporary, final in replacements:
            if not final.exists():
                continue
            backup = final.with_name(f".{final.name}.backup-{uuid4().hex}")
            final.rename(backup)
            backups[final] = backup
        for temporary, final in replacements:
            temporary.replace(final)
            published.append(final)
    except Exception:
        for final in reversed(published):
            _clean_path(final)
        for final, backup in backups.items():
            if backup.exists():
                backup.rename(final)
        raise
    else:
        for backup in backups.values():
            _clean_path(backup)


def _clean_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
