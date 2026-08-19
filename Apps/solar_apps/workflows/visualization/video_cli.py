"""Compatibility CLI for image-sequence video generation.

Reusable frame decoding and media encoding live in
``solar_toolkit.visualization``.  Importing this module is side-effect free;
local path configuration is read only by :func:`main`.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from solar_toolkit.visualization.frames import (
    FrameResult,
    FrameStats,
    align_even,
    determine_target_size_for_paths,
    make_frame_iter_from_paths,
    normalize_channels,
    process_single_frame,
    resize_image,
    sample_image_sizes,
    write_video_imageio_stream,
    write_video_opencv_stream,
)
from solar_toolkit.visualization.frames import (
    iter_processed_frames as _iter_processed_frames,
)
from solar_toolkit.visualization.media import write_frames_ffmpeg_stream

__all__ = [
    "FrameResult",
    "FrameStats",
    "align_even",
    "determine_target_size",
    "determine_target_size_for_paths",
    "iter_processed_frames",
    "main",
    "make_frame_iter",
    "make_frame_iter_from_paths",
    "normalize_channels",
    "parse_timestamp",
    "process_single_frame",
    "resize_image",
    "sample_image_sizes",
    "scan_image_files",
    "select_frame_range",
    "sort_entries",
    "write_video_ffmpeg_stream",
    "write_video_from_paths",
    "write_video_imageio_stream",
    "write_video_opencv_stream",
]

fps = 10
input_dir = "data/images"
output_dir = "outputs/video"
video_name = "AIA_spectrogram_overlay.mp4"
target_suffix = ".png"
video_quality = "low"
start_frame = 1
end_frame = None
sort_by = "filename"
target_size = None
prefetch_workers = 8
prefetch_batch_size = 8


def _dt(year, month, day, hour=0, minute=0, second=0):
    try:
        return datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except (TypeError, ValueError):
        return None


def _dt_doy(year, doy, hour=0, minute=0, second=0):
    try:
        base = datetime(int(year), 1, 1)
        value = base + timedelta(days=int(doy) - 1)
        return value.replace(hour=int(hour), minute=int(minute), second=int(second))
    except (TypeError, ValueError):
        return None


_TS_PATTERNS = [
    (
        re.compile(r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):?(\d{2}):?(\d{2})"),
        lambda groups: _dt(*groups[:6]),
    ),
    (
        re.compile(r"(\d{4})(\d{2})(\d{2})[_\-T](\d{2})(\d{2})(\d{2})"),
        lambda groups: _dt(*groups[:6]),
    ),
    (
        re.compile(r"(\d{4})(\d{3})[_\-T](\d{2})(\d{2})(\d{2})"),
        lambda groups: _dt_doy(*groups[:5]),
    ),
    (re.compile(r"(\d{4})(\d{3})(?!\d)"), lambda groups: _dt_doy(*groups[:2])),
    (re.compile(r"(\d{4})(\d{2})(\d{2})(?!\d)"), lambda groups: _dt(*groups[:3])),
    (
        re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)"),
        lambda groups: _dt(1970, 1, 1, *groups[:3]),
    ),
]


def parse_timestamp(filename: str) -> datetime | None:
    """Extract the first valid timestamp encoded in a filename."""

    stem = os.path.splitext(filename)[0]
    for pattern, builder in _TS_PATTERNS:
        for match in pattern.finditer(stem):
            value = builder(match.groups())
            if value is not None:
                return value
    return None


def determine_target_size(paths: Sequence[str]) -> tuple[int, int, int, int]:
    """Compatibility wrapper using the module-level requested target size."""

    return determine_target_size_for_paths(paths, target_size)


def iter_processed_frames(
    paths: Sequence[str],
    target_size_tuple: tuple[int, int] | None,
    stats: FrameStats | None = None,
    workers: int = 1,
    batch_size: int = 8,
    missing_frame_policy: str = "skip",
):
    """Compatibility wrapper preserving monkeypatchable frame decoding."""

    return _iter_processed_frames(
        paths,
        target_size_tuple,
        stats=stats,
        workers=workers,
        batch_size=batch_size,
        missing_frame_policy=missing_frame_policy,
        _processor=process_single_frame,
    )


def write_video_ffmpeg_stream(
    frame_iter: Iterable[tuple[np.ndarray, tuple[int, int]]],
    output_path: str,
    fps: float,
    width: int,
    height: int,
    quality: str = "high",
) -> bool:
    """Compatibility wrapper around the shared FFmpeg stream writer."""

    return write_frames_ffmpeg_stream(
        frame_iter,
        output_path,
        fps=fps,
        width=width,
        height=height,
        output_format="mp4",
        quality=quality,
    )


def scan_image_files(folder: str, suffix: str):
    """Return matching directory entries without recursively scanning."""

    suffix_lower = suffix.lower()
    with os.scandir(folder) as entries:
        return [
            entry
            for entry in entries
            if entry.is_file() and entry.name.lower().endswith(suffix_lower)
        ]


def sort_entries(entries, method: str):
    """Sort directory entries by encoded timestamp or modification time."""

    if method == "filename":
        parsed, failed = [], []
        for entry in entries:
            timestamp = parse_timestamp(entry.name)
            (parsed if timestamp is not None else failed).append((timestamp, entry))
        parsed.sort(key=lambda item: item[0])
        return [entry for _, entry in parsed] + [entry for _, entry in failed]
    if method == "mtime":
        return sorted(entries, key=lambda entry: entry.stat().st_mtime)
    raise ValueError("sort_by must be 'filename' or 'mtime'")


def select_frame_range(entries):
    """Apply the legacy one-based module-level frame range."""

    total = len(entries)
    start = max(1, min(int(start_frame), total))
    stop = total if end_frame is None else max(start, min(int(end_frame), total))
    return entries[start - 1 : stop], start, stop


def make_frame_iter(paths: Sequence[str], stats: FrameStats, size: tuple[int, int]):
    """Build a frame iterator with the legacy prefetch controls."""

    return make_frame_iter_from_paths(
        paths,
        stats,
        size,
        workers=prefetch_workers,
        batch_size=prefetch_batch_size,
    )


def write_video_from_paths(
    paths: Sequence[str],
    output_path: str,
    fps: float,
    quality: str = "high",
    target_size_tuple: tuple[int, int] | None = None,
    workers: int = 8,
    batch_size: int = 8,
) -> bool:
    """Write an MP4 while delegating all frame operations to public helpers."""

    selected_paths = list(paths)
    if not selected_paths:
        return False
    if quality not in {"high", "low"}:
        quality = "high"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height, _, _ = determine_target_size_for_paths(
        selected_paths, requested_size=target_size_tuple
    )
    for writer in ("ffmpeg", "imageio", "opencv"):
        stats = FrameStats()
        frame_iter = make_frame_iter_from_paths(
            selected_paths,
            stats,
            (width, height),
            workers=workers,
            batch_size=batch_size,
        )
        if writer == "ffmpeg" and write_video_ffmpeg_stream(
            frame_iter, str(output), fps, width, height, quality
        ):
            return True
        if writer == "imageio" and write_video_imageio_stream(
            frame_iter, str(output), fps, quality
        ):
            return True
        if writer == "opencv":
            return write_video_opencv_stream(
                frame_iter, str(output), fps, width, height
            )
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow visualization video",
        description=__doc__,
    )
    parser.add_argument("input_dir", nargs="?")
    parser.add_argument("--output-dir")
    parser.add_argument("--video-name")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--quality", choices=("high", "low"))
    parser.add_argument("--suffix", default=target_suffix)
    parser.add_argument("--sort-by", choices=("filename", "mtime"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load local path defaults, parse CLI overrides, and write one MP4."""

    from solar_apps.platform.config import load_script_config

    configured = load_script_config(
        "image_sequence_to_video",
        {
            "fps": fps,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "video_name": video_name,
        },
    )
    args = _parser().parse_args(argv)
    source = Path(args.input_dir or configured["input_dir"])
    destination = Path(args.output_dir or configured["output_dir"])
    name = args.video_name or configured["video_name"]
    selected_fps = args.fps or float(configured["fps"])
    quality = args.quality or video_quality
    method = args.sort_by or sort_by
    if not source.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {source}")

    entries = sort_entries(scan_image_files(str(source), args.suffix), method)
    if not entries:
        raise FileNotFoundError(f"No {args.suffix} files found in: {source}")
    selected, _, _ = select_frame_range(entries)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [entry.path for entry in selected]
    return (
        0
        if write_video_from_paths(
            paths,
            str(destination / name),
            selected_fps,
            quality=quality,
            target_size_tuple=target_size,
            workers=prefetch_workers,
            batch_size=prefetch_batch_size,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
