"""Video export helpers for the local image web viewer.

English: Export either one video per image folder or a side-by-side composite
video that matches the viewer's index-synchronized comparison mode.

中文：为本地图片查看器导出视频，支持每个文件夹单独导出，或导出与网页并列
对比一致的合成视频。
"""

from __future__ import annotations

import math
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from solar_toolkit.visualization import media

__all__ = [
    "ExportConfig",
    "build_composite_frame",
    "export_composite_video",
    "export_separate_videos",
    "normalize_roi",
    "sanitize_filename",
]


@dataclass
class ExportConfig:
    """Configuration for image-viewer video export."""

    output_dir: str | Path
    file_prefix: str = "image_viewer"
    output_format: str = "mp4"
    fps: float = 5.0
    quality: str = "low"
    start_frame: int = 0
    end_frame: int | None = None
    target_size: tuple[int, int] | None = None
    roi: dict[str, float] | None = None
    composite_panel_size: tuple[int, int] | None = None
    workers: int = 4
    batch_size: int = 8

    def normalized(self) -> ExportConfig:
        quality = self.quality if self.quality in {"low", "high"} else "low"
        return ExportConfig(
            output_dir=Path(self.output_dir).expanduser().resolve(),
            file_prefix=sanitize_filename(self.file_prefix or "image_viewer"),
            output_format=media.normalize_output_format(self.output_format),
            fps=max(0.2, float(self.fps or 5.0)),
            quality=quality,
            start_frame=max(0, int(self.start_frame or 0)),
            end_frame=(None if self.end_frame is None else max(0, int(self.end_frame))),
            target_size=_normalize_size(self.target_size),
            roi=normalize_roi(self.roi),
            composite_panel_size=_normalize_size(self.composite_panel_size),
            workers=max(1, int(self.workers or 1)),
            batch_size=max(1, int(self.batch_size or 1)),
        )


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "image_viewer"


def normalize_roi(roi: dict[str, Any] | None) -> dict[str, float] | None:
    """Normalize a browser ROI dict with x/y/w/h fractional values."""

    if not roi:
        return None
    try:
        x = float(roi["x"])
        y = float(roi["y"])
        w = float(roi["w"])
        h = float(roi["h"])
    except (KeyError, TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0 - x)
    h = min(max(h, 0.0), 1.0 - y)
    if w <= 0.0 or h <= 0.0:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def export_separate_videos(
    groups: list[dict[str, Any]], export_config: ExportConfig
) -> dict[str, Any]:
    """Export one media file per image folder."""

    config = export_config.normalized()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    failures: list[str] = []
    for index, group in enumerate(groups):
        selected = _select_paths(group, config)
        if not selected:
            failures.append(f"{_group_name(group, index)}: no frames selected")
            continue
        output_path = config.output_dir / (
            f"{config.file_prefix}_{index + 1:02d}_"
            f"{sanitize_filename(_group_name(group, index))}.{config.output_format}"
        )
        ok = _write_video_from_paths(
            selected,
            output_path,
            fps=_video_fps(config.fps),
            quality=config.quality,
            target_size_tuple=config.target_size,
            workers=config.workers,
            batch_size=config.batch_size,
            output_format=config.output_format,
        )
        if ok:
            paths.append(str(output_path))
        else:
            failures.append(f"{_group_name(group, index)}: writer failed")
    return {
        "status": (
            "saved" if paths and not failures else ("partial" if paths else "failed")
        ),
        "paths": paths,
        "failures": failures,
    }


def export_composite_video(
    groups: list[dict[str, Any]], export_config: ExportConfig
) -> dict[str, Any]:
    """Export a side-by-side composite using index-synchronized frames."""

    config = export_config.normalized()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not groups:
        return {"status": "failed", "reason": "no_groups"}
    start, stop = _frame_range(groups, config)
    if stop <= start:
        return {"status": "failed", "reason": "no_frames_selected"}
    panel_size = (
        config.composite_panel_size or config.target_size or _default_panel_size(groups)
    )
    output_path = (
        config.output_dir / f"{config.file_prefix}_composite.{config.output_format}"
    )

    frame_size = (panel_size[0] * max(1, len(groups)), panel_size[1])
    frame_count = stop - start
    ok = media.write_media_from_frames(
        lambda: _iter_composite_frames(
            groups,
            start,
            stop,
            panel_size,
            config.roi,
            workers=config.workers,
            batch_size=config.batch_size,
        ),
        output_path,
        fps=_video_fps(config.fps),
        quality=config.quality,
        frame_size=frame_size,
        output_format=config.output_format,
    )
    return {
        "status": "saved" if ok else "failed",
        "path": str(output_path),
        "frame_count": frame_count,
    }


def build_composite_frame(
    groups: list[dict[str, Any]],
    frame_index: int,
    *,
    panel_size: tuple[int, int],
    roi: dict[str, float] | None = None,
) -> np.ndarray:
    """Build one side-by-side RGB frame for the selected frame index."""

    panel_width, panel_height = _normalize_size(panel_size) or (640, 480)
    panels = []
    for index, group in enumerate(groups):
        files = _group_files(group)
        if frame_index < len(files):
            panel = _image_panel(files[frame_index], panel_width, panel_height, roi)
        else:
            panel = _missing_panel(
                panel_width,
                panel_height,
                f"{_group_name(group, index)}\nmissing frame {frame_index + 1}",
            )
        panels.append(panel)
    if not panels:
        panels.append(_missing_panel(panel_width, panel_height, "no image folders"))
    return np.concatenate(panels, axis=1)


def _iter_composite_frames(
    groups: list[dict[str, Any]],
    start: int,
    stop: int,
    panel_size: tuple[int, int],
    roi: dict[str, float] | None,
    *,
    workers: int = 1,
    batch_size: int = 8,
):
    def build(frame_index: int):
        frame = build_composite_frame(
            groups,
            frame_index,
            panel_size=panel_size,
            roi=roi,
        )
        return frame, (frame.shape[1], frame.shape[0])

    workers = max(1, int(workers))
    batch_size = max(1, int(batch_size))
    if workers == 1:
        for frame_index in range(start, stop):
            yield build(frame_index)
        return

    indexes = iter(range(start, stop))
    pending: deque[Future] = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(batch_size):
            try:
                pending.append(executor.submit(build, next(indexes)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(build, next(indexes)))
            except StopIteration:
                continue


def _write_video_from_paths(
    paths,
    output_path,
    fps,
    quality,
    target_size_tuple=None,
    workers=4,
    batch_size=8,
    output_format="mp4",
) -> bool:
    return media.write_media_from_paths(
        paths,
        output_path,
        int(fps),
        quality=quality,
        target_size_tuple=target_size_tuple,
        workers=workers,
        batch_size=batch_size,
        output_format=output_format,
    )


def _select_paths(group: dict[str, Any], config: ExportConfig) -> list[str]:
    files = _group_files(group)
    start, stop = _frame_range([group], config)
    return [str(path) for path in files[start:stop]]


def _frame_range(groups: list[dict[str, Any]], config: ExportConfig) -> tuple[int, int]:
    max_frames = max((len(_group_files(group)) for group in groups), default=0)
    start = min(max(0, int(config.start_frame)), max_frames)
    stop = (
        max_frames
        if config.end_frame is None
        else min(int(config.end_frame), max_frames)
    )
    stop = max(start, stop)
    return start, stop


def _default_panel_size(groups: list[dict[str, Any]]) -> tuple[int, int]:
    for group in groups:
        for path in _group_files(group):
            try:
                with Image.open(path) as image:
                    width, height = image.size
                return _even_size((width, height))
            except Exception:
                continue
    return 640, 480


def _image_panel(
    path: Path,
    panel_width: int,
    panel_height: int,
    roi: dict[str, float] | None,
) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = _crop_roi(image, roi)
            image.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (panel_width, panel_height), (0, 0, 0))
            x = (panel_width - image.width) // 2
            y = (panel_height - image.height) // 2
            panel.paste(image, (x, y))
            return np.asarray(panel, dtype=np.uint8)
    except Exception:
        return _missing_panel(panel_width, panel_height, f"failed\n{path.name}")


def _crop_roi(image: Image.Image, roi: dict[str, float] | None) -> Image.Image:
    roi = normalize_roi(roi)
    if roi is None:
        return image
    width, height = image.size
    left = int(math.floor(roi["x"] * width))
    top = int(math.floor(roi["y"] * height))
    right = int(math.ceil((roi["x"] + roi["w"]) * width))
    bottom = int(math.ceil((roi["y"] + roi["h"]) * height))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, min(width, right), min(height, bottom)))


def _missing_panel(width: int, height: int, text: str) -> np.ndarray:
    _ = text
    panel = Image.new("RGB", (width, height), (0, 0, 0))
    return np.asarray(panel, dtype=np.uint8)


def _video_fps(value: float) -> int:
    return max(1, int(round(float(value or 5.0))))


def _group_files(group: dict[str, Any]) -> list[Path]:
    return [Path(path) for path in group.get("files", [])]


def _group_name(group: dict[str, Any], index: int) -> str:
    return str(group.get("name") or f"group_{index + 1}")


def _normalize_size(
    value: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        width, height = int(value[0]), int(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if width <= 0 or height <= 0:
        return None
    return _even_size((width, height))


def _even_size(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    width = width if width % 2 == 0 else max(2, width - 1)
    height = height if height % 2 == 0 else max(2, height - 1)
    return width, height
