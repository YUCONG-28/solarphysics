"""Shared media-format helpers for local visualization frontends."""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .frames import (
    FrameStats,
    determine_target_size_for_paths,
    make_frame_iter_from_paths,
    write_video_imageio_stream,
    write_video_opencv_stream,
)


def _conda_bin_for_python(python_executable: str | Path) -> Path:
    """Return the Conda Library bin directory beside a Python executable."""

    return Path(python_executable).resolve().parent / "Library" / "bin"


SUPPORTED_OUTPUT_FORMATS = {"mp4", "gif", "webm"}
SUPPORTED_RECORDING_SOURCE_FORMATS = {"mp4", "webm"}
CONDA_BIN = _conda_bin_for_python(sys.executable)
CONDA_FFMPEG = CONDA_BIN / "ffmpeg.exe"
CONDA_FFPROBE = CONDA_BIN / "ffprobe.exe"
FFMPEG_FINALIZE_TIMEOUT_SECONDS = 120
WINDOWS_FILE_OPERATION_TIMEOUT_SECONDS = 3.0
WINDOWS_FILE_OPERATION_INITIAL_DELAY_SECONDS = 0.025
PROBE_CACHE_MAX_ENTRIES = 32
_PROBE_CACHE: OrderedDict[tuple[str, int, int, str], dict[str, Any]] = OrderedDict()
_PROBE_CACHE_LOCK = threading.Lock()

Frame = tuple[np.ndarray, tuple[int, int]]
FrameFactory = Callable[[], Iterable[Frame]]
FrameSource = Iterable[Frame] | FrameFactory
CancelCheck = Callable[[], bool]

__all__ = [
    "MediaProcessingError",
    "normalize_even_size",
    "normalize_output_format",
    "normalize_recording_source_format",
    "probe_video",
    "resolve_ffmpeg",
    "resolve_ffprobe",
    "sanitize_filename",
    "save_browser_recording",
    "save_browser_recording_stream",
    "transcode_recording",
    "write_media_from_frames",
    "write_media_from_paths",
    "write_frames_pyav_stream",
]


class MediaProcessingError(RuntimeError):
    """Media error with a concise user message and optional diagnostics."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(f"{message} Diagnostics: {detail}" if detail else message)
        self.user_message = message
        self.detail = detail


def normalize_output_format(value: str | None) -> str:
    """Normalize a user-facing media format label."""

    output_format = str(value or "mp4").strip().lower().lstrip(".")
    return output_format if output_format in SUPPORTED_OUTPUT_FORMATS else "mp4"


def normalize_recording_source_format(value: str | None) -> str:
    """Normalize a browser recording container."""

    source_format = str(value or "webm").strip().lower().lstrip(".")
    return (
        source_format if source_format in SUPPORTED_RECORDING_SOURCE_FORMATS else "webm"
    )


def normalize_even_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return a positive, codec-safe even width and height."""

    return _normalized_even_size(size) or (2, 2)


def _normalize_fps(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 5.0
    if not np.isfinite(parsed):
        parsed = 5.0
    return max(0.2, parsed)


def _fps_text(value: float) -> str:
    return f"{_normalize_fps(value):g}"


def sanitize_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "image_viewer"


def resolve_ffmpeg() -> str | None:
    """Resolve FFmpeg from configuration, PATH, or Miniforge."""

    return _resolve_media_executable(
        configured_names=("SOLAR_TOOLKIT_FFMPEG", "IMAGE_VIEWER_FFMPEG"),
        executable_name="ffmpeg",
        conda_path=CONDA_FFMPEG,
    )


def resolve_ffprobe() -> str | None:
    """Resolve FFprobe from configuration, PATH, or beside FFmpeg."""

    for configured_name in ("SOLAR_TOOLKIT_FFPROBE", "IMAGE_VIEWER_FFPROBE"):
        configured = os.getenv(configured_name, "").strip()
        if configured:
            configured_path = Path(configured).expanduser()
            if configured_path.exists():
                return str(configured_path)

    discovered = shutil.which("ffprobe")
    if discovered:
        return discovered

    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        sibling_name = (
            "ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"
        )
        sibling = Path(ffmpeg).with_name(sibling_name)
        if sibling.exists():
            return str(sibling)

    if CONDA_FFPROBE.exists():
        return str(CONDA_FFPROBE)
    return None


def _resolve_media_executable(
    *, configured_names: Sequence[str], executable_name: str, conda_path: Path
) -> str | None:
    for configured_name in configured_names:
        configured = os.getenv(configured_name, "").strip()
        if configured:
            configured_path = Path(configured).expanduser()
            if configured_path.exists():
                return str(configured_path)
    discovered = shutil.which(executable_name)
    if discovered:
        return discovered
    return str(conda_path) if conda_path.exists() else None


def write_media_from_paths(
    paths: Sequence[str],
    output_path: str | Path,
    fps: float,
    quality: str = "high",
    target_size_tuple: tuple[int, int] | None = None,
    workers: int = 8,
    batch_size: int = 8,
    output_format: str = "mp4",
) -> bool:
    """Write an image sequence to MP4, GIF, or WebM."""

    selected_paths = list(paths)
    if not selected_paths:
        return False

    width, height, _sample_size, _native_count = determine_target_size_for_paths(
        selected_paths,
        requested_size=target_size_tuple,
    )

    def frame_factory() -> Iterable[Frame]:
        stats = FrameStats()
        return make_frame_iter_from_paths(
            selected_paths,
            stats,
            (width, height),
            workers=workers,
            batch_size=batch_size,
            missing_frame_policy="repeat",
        )

    return write_media_from_frames(
        frame_factory,
        output_path,
        float(fps),
        quality=quality,
        frame_size=(width, height),
        output_format=output_format,
    )


def write_media_from_frames(
    frame_source: FrameSource,
    output_path: str | Path,
    fps: float,
    quality: str = "high",
    *,
    frame_size: tuple[int, int],
    output_format: str = "mp4",
) -> bool:
    """Write already-rendered RGB frames to MP4, GIF, or WebM."""

    output_format = normalize_output_format(output_format)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = int(frame_size[0]), int(frame_size[1])
    if width <= 0 or height <= 0:
        return False

    fps_value = _normalize_fps(fps)
    frame_factory, reusable = _coerce_frame_factory(frame_source)
    primary_frames = frame_factory()
    ok = False
    if output_format in {"mp4", "webm"}:
        av_module = _load_pyav()
        if av_module is not None:
            ok = write_frames_pyav_stream(
                primary_frames,
                output_path,
                fps=fps_value,
                width=width,
                height=height,
                output_format=output_format,
                quality=quality,
                av_module=av_module,
            )
    if not ok:
        ok = write_frames_ffmpeg_stream(
            primary_frames,
            output_path,
            fps=fps_value,
            width=width,
            height=height,
            output_format=output_format,
            quality=quality,
        )
    if ok:
        return True

    if not reusable:
        return False
    if output_format == "gif":
        return _write_fallback_atomic(
            output_path,
            lambda path: write_gif_imageio_stream(frame_factory(), path, fps=fps_value),
        )
    if output_format == "mp4":
        if _write_fallback_atomic(
            output_path,
            lambda path: write_video_imageio_stream(
                frame_factory(), str(path), fps_value, quality
            ),
        ):
            return True
        return _write_fallback_atomic(
            output_path,
            lambda path: write_video_opencv_stream(
                frame_factory(), str(path), fps_value, width, height
            ),
        )
    return False


def _load_pyav() -> Any | None:
    """Return the optional PyAV module without making it a base dependency."""

    try:
        return importlib.import_module("av")
    except (ImportError, OSError):
        return None


def write_frames_pyav_stream(
    frame_iter: Iterable[Frame],
    output_path: str | Path,
    *,
    fps: float,
    width: int,
    height: int,
    output_format: str,
    quality: str,
    av_module: Any | None = None,
) -> bool:
    """Stream RGB arrays through PyAV and atomically publish MP4 or WebM."""

    av_module = av_module or _load_pyav()
    output_format = normalize_output_format(output_format)
    if av_module is None or output_format not in {"mp4", "webm"}:
        return False
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _partial_output_path(output_path)
    fps_value = _normalize_fps(fps)
    rate = Fraction(str(fps_value)).limit_denominator(100_000)
    encoded_width, encoded_height = _encoded_size(output_format, width, height)
    container = None
    stream = None
    frame_count = 0
    try:
        try:
            options = {"movflags": "+faststart"} if output_format == "mp4" else None
            container = av_module.open(
                str(temporary_path),
                mode="w",
                format=output_format,
                options=options,
            )
            codec_name = "libx264" if output_format == "mp4" else "libvpx-vp9"
            stream = container.add_stream(codec_name, rate=rate)
            stream.width = int(encoded_width)
            stream.height = int(encoded_height)
            stream.pix_fmt = "yuv420p"
            if output_format == "mp4":
                stream.options = {
                    "crf": "18" if quality == "high" else "28",
                    "preset": "slow" if quality == "high" else "fast",
                }
            else:
                stream.options = {
                    "crf": "28" if quality == "high" else "38",
                    "b": "0",
                    "deadline": "good",
                    "cpu-used": "2" if quality == "high" else "4",
                    "row-mt": "1",
                }
            stream.codec_context.open()
        except Exception as exc:
            print(f"[PyAV] encoder unavailable: {exc}")
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
                container = None
            return False

        iterator = iter(frame_iter)
        while True:
            try:
                item = next(iterator)
            except StopIteration:
                break
            except (OSError, ValueError, TypeError) as exc:
                raise MediaProcessingError(
                    "A composite frame failed while the video was being encoded.",
                    str(exc),
                ) from exc
            try:
                frame, _original_size = item
                if frame.shape != (int(height), int(width), 3):
                    raise ValueError(
                        f"Frame shape {frame.shape} does not match "
                        f"{(int(height), int(width), 3)}"
                    )
                if frame.dtype != np.uint8:
                    raise TypeError(f"Frame dtype must be uint8, got {frame.dtype}")
                rgb_frame = (
                    frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
                )
                video_frame = av_module.VideoFrame.from_ndarray(
                    rgb_frame, format="rgb24"
                )
                if (encoded_width, encoded_height) != (int(width), int(height)):
                    video_frame = video_frame.reformat(
                        width=int(encoded_width),
                        height=int(encoded_height),
                        format="yuv420p",
                    )
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, 1) / rate
                for packet in stream.encode(video_frame):
                    container.mux(packet)
                frame_count += 1
            except MediaProcessingError:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise MediaProcessingError(
                    "A composite frame failed while the video was being encoded.",
                    str(exc),
                ) from exc
            except Exception as exc:
                raise MediaProcessingError(
                    "PyAV failed while encoding the video.", str(exc)
                ) from exc
        if frame_count <= 0:
            return False
        try:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            container = None
        except Exception as exc:
            raise MediaProcessingError(
                "PyAV failed while finalizing the video.", str(exc)
            ) from exc
        _raise_if_empty(temporary_path)
        ffprobe = resolve_ffprobe()
        if ffprobe:
            probe_video(
                temporary_path,
                expected_size=(encoded_width, encoded_height),
                expected_frame_count=frame_count,
                ffprobe=ffprobe,
            )
        _atomic_replace(temporary_path, output_path)
        return True
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        _safe_unlink(temporary_path)


def _coerce_frame_factory(frame_source: FrameSource) -> tuple[FrameFactory, bool]:
    if callable(frame_source):
        return frame_source, True
    if isinstance(frame_source, Sequence):
        return lambda: iter(frame_source), True
    iterator = iter(frame_source)
    used = False

    def one_shot_factory() -> Iterable[Frame]:
        nonlocal used
        if used:
            return iter(())
        used = True
        return iterator

    return one_shot_factory, False


def write_frames_ffmpeg_stream(
    frame_iter: Iterable[Frame],
    output_path: str | Path,
    *,
    fps: float,
    width: int,
    height: int,
    output_format: str,
    quality: str,
) -> bool:
    """Stream RGB frames into FFmpeg and atomically publish the result."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = normalize_output_format(output_format)
    fps = _normalize_fps(fps)
    temporary_path = _partial_output_path(output_path)
    frame_count = 0

    def counted_frames() -> Iterable[Frame]:
        nonlocal frame_count
        for item in frame_iter:
            frame_count += 1
            yield item

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-r",
        _fps_text(fps),
        "-i",
        "pipe:0",
        "-an",
    ]
    cmd.extend(_ffmpeg_output_args(output_format, quality, fps, ffmpeg=ffmpeg))
    cmd.append(str(temporary_path))

    try:
        ok = _run_ffmpeg_stream(
            cmd,
            counted_frames(),
            width=int(width),
            height=int(height),
        )
        if not ok or frame_count <= 0:
            return False
        _raise_if_empty(temporary_path)
        ffprobe = resolve_ffprobe()
        if ffprobe:
            expected_size = _encoded_size(output_format, width, height)
            probe_video(
                temporary_path,
                expected_size=expected_size,
                expected_frame_count=frame_count,
                ffprobe=ffprobe,
            )
        _atomic_replace(temporary_path, output_path)
        return True
    except MediaProcessingError:
        raise
    except RuntimeError:
        # Preserve cancellation and frame-generation exceptions from lazy sources.
        raise
    except Exception as exc:
        print(f"[FFmpeg] output validation failed: {exc}")
        return False
    finally:
        _safe_unlink(temporary_path)


def _encoded_size(output_format: str, width: int, height: int) -> tuple[int, int]:
    if output_format == "mp4":
        return _normalized_even_size((width, height)) or (2, 2)
    return width, height


def write_gif_imageio_stream(
    frame_iter: Iterable[Frame],
    output_path: str | Path,
    *,
    fps: float,
) -> bool:
    """Fallback GIF writer when FFmpeg is unavailable."""

    import imageio.v2 as imageio

    try:
        with imageio.get_writer(
            str(output_path), mode="I", duration=1 / _normalize_fps(fps)
        ) as writer:
            for frame, _original_size in frame_iter:
                writer.append_data(frame)
        return True
    except Exception as exc:
        print(f"[imageio] GIF write failed: {exc}")
        return False


def save_browser_recording(
    recording_file: Any,
    *,
    output_dir: str | Path,
    file_prefix: str,
    output_format: str,
    fps: float,
    quality: str,
    source_format: str = "webm",
    recording_size: tuple[int, int] | None = None,
    expected_frame_count: int | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Save a buffered Mediabunny recording uploaded by the frontend."""

    return _save_recording_input(
        recording_file,
        output_dir=output_dir,
        file_prefix=file_prefix,
        output_format=output_format,
        source_format=source_format,
        fps=fps,
        quality=quality,
        recording_size=recording_size,
        expected_frame_count=expected_frame_count,
        cancel_check=cancel_check,
    )


def save_browser_recording_stream(
    recording_stream: Any,
    *,
    output_dir: str | Path,
    file_prefix: str,
    output_format: str,
    fps: float,
    quality: str,
    source_format: str,
    recording_size: tuple[int, int] | None = None,
    expected_frame_count: int | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Save a request stream produced by Mediabunny."""

    return _save_recording_input(
        recording_stream,
        output_dir=output_dir,
        file_prefix=file_prefix,
        output_format=output_format,
        source_format=source_format,
        fps=fps,
        quality=quality,
        recording_size=recording_size,
        expected_frame_count=expected_frame_count,
        cancel_check=cancel_check,
    )


def _save_recording_input(
    recording_input: Any,
    *,
    output_dir: str | Path,
    file_prefix: str,
    output_format: str,
    source_format: str,
    fps: float,
    quality: str,
    recording_size: tuple[int, int] | None,
    expected_frame_count: int | None,
    cancel_check: CancelCheck | None,
) -> dict[str, Any]:
    output_format = normalize_output_format(output_format)
    source_format = normalize_recording_source_format(source_format)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_root / f"{sanitize_filename(file_prefix)}_recording.{output_format}"
    )
    source_path = _upload_path(output_root, file_prefix, source_format)
    temporary_output = _partial_output_path(output_path)

    try:
        _raise_if_recording_cancelled(cancel_check)
        _copy_recording_input(recording_input, source_path, cancel_check=cancel_check)
        _raise_if_recording_cancelled(cancel_check)
        _raise_if_empty(source_path)
        probe_video(
            source_path,
            expected_size=recording_size,
            expected_frame_count=expected_frame_count,
        )
        _raise_if_recording_cancelled(cancel_check)
        transcode_kwargs = {
            "fps": max(0.2, float(fps or 5.0)),
            "quality": quality,
            "recording_size": recording_size,
            "source_format": source_format,
        }
        if cancel_check is not None:
            transcode_kwargs["cancel_check"] = cancel_check
        transcode_recording(
            source_path,
            temporary_output,
            output_format,
            **transcode_kwargs,
        )
        _raise_if_recording_cancelled(cancel_check)
        metadata = probe_video(
            temporary_output,
            expected_size=_normalized_even_size(recording_size),
            expected_frame_count=expected_frame_count,
        )
        _raise_if_recording_cancelled(cancel_check)
        _atomic_replace(temporary_output, output_path)
        return {
            "status": "saved",
            "path": str(output_path.resolve()),
            "format": output_format,
            **metadata,
        }
    finally:
        _safe_unlink(source_path)
        _safe_unlink(temporary_output)


def _copy_recording_input(
    recording_input: Any,
    destination: Path,
    *,
    cancel_check: CancelCheck | None = None,
) -> None:
    if hasattr(recording_input, "save"):
        recording_input.save(destination)
        _raise_if_recording_cancelled(cancel_check)
        return
    with destination.open("wb") as handle:
        while True:
            _raise_if_recording_cancelled(cancel_check)
            chunk = recording_input.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def transcode_recording(
    input_path: str | Path,
    output_path: str | Path,
    output_format: str,
    fps: float,
    quality: str,
    recording_size: tuple[int, int] | None = None,
    source_format: str = "webm",
    cancel_check: CancelCheck | None = None,
) -> bool:
    """Remux Mediabunny output or transcode it to the selected format."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise MediaProcessingError("FFmpeg is required to finalize browser recordings.")

    output_format = normalize_output_format(output_format)
    source_format = normalize_recording_source_format(source_format)
    fps_value = _normalize_fps(fps or 5.0)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
    ]
    if output_format == source_format:
        cmd.extend(["-c:v", "copy"])
        if output_format == "mp4":
            cmd.extend(["-movflags", "+faststart"])
    else:
        cmd.extend(
            _ffmpeg_output_args(
                output_format,
                quality,
                fps_value,
                ffmpeg=ffmpeg,
                recording_size=recording_size,
            )
        )
    cmd.append(str(output_path))
    if cancel_check is None:
        _run_ffmpeg_file(cmd)
    else:
        _run_ffmpeg_file(cmd, cancel_check=cancel_check)
    return True


def probe_video(
    path: str | Path,
    *,
    expected_size: tuple[int, int] | None = None,
    expected_frame_count: int | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Validate a media file and return its primary video metadata."""

    media_path = Path(path)
    _raise_if_empty(media_path)
    ffprobe = ffprobe or resolve_ffprobe()
    if not ffprobe:
        raise MediaProcessingError(
            "FFprobe is required to validate browser recordings."
        )
    stat = media_path.stat()
    cache_key = (
        str(media_path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        str(Path(ffprobe).resolve()),
    )
    with _PROBE_CACHE_LOCK:
        cached = _PROBE_CACHE.get(cache_key)
        if cached is not None:
            _PROBE_CACHE.move_to_end(cache_key)
            return _validate_probe_metadata(
                dict(cached),
                expected_size=expected_size,
                expected_frame_count=expected_frame_count,
            )
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        (
            "stream=codec_name,width,height,nb_read_packets,duration,"
            "avg_frame_rate,r_frame_rate:format=duration"
        ),
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = _compact_ffmpeg_stderr(result.stderr)
    if result.returncode != 0:
        raise MediaProcessingError(
            "Recording contains no decodable video frames.", diagnostics
        )
    try:
        payload = json.loads(result.stdout or "{}")
        stream = (payload.get("streams") or [])[0]
        frame_count = int(stream.get("nb_read_packets") or 0)
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaProcessingError(
            "Recording contains no decodable video frames.", str(exc)
        ) from exc
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise MediaProcessingError(
            "Recording contains no decodable video frames.", diagnostics
        )
    reported_duration = _first_float(
        stream.get("duration"), (payload.get("format") or {}).get("duration")
    )
    frame_rate = _parse_frame_rate(
        stream.get("avg_frame_rate"), stream.get("r_frame_rate")
    )
    nominal_duration = frame_count / frame_rate if frame_rate else None
    duration = reported_duration
    if nominal_duration is not None and (
        duration is None or duration + 0.5 / frame_rate < nominal_duration
    ):
        duration = nominal_duration
    metadata = {
        "codec": str(stream.get("codec_name") or "unknown"),
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration": duration,
    }
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE[cache_key] = dict(metadata)
        _PROBE_CACHE.move_to_end(cache_key)
        while len(_PROBE_CACHE) > PROBE_CACHE_MAX_ENTRIES:
            _PROBE_CACHE.popitem(last=False)
    return _validate_probe_metadata(
        metadata,
        expected_size=expected_size,
        expected_frame_count=expected_frame_count,
    )


def _validate_probe_metadata(
    metadata: dict[str, Any],
    *,
    expected_size: tuple[int, int] | None,
    expected_frame_count: int | None,
) -> dict[str, Any]:
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])
    normalized_expected_size = _normalized_even_size(expected_size)
    if normalized_expected_size and (width, height) != normalized_expected_size:
        raise MediaProcessingError(
            "Recording dimensions do not match the requested size.",
            f"expected {normalized_expected_size[0]}x{normalized_expected_size[1]}, "
            f"got {width}x{height}",
        )
    if expected_frame_count is not None and frame_count != int(expected_frame_count):
        raise MediaProcessingError(
            "Recording frame count does not match the selected range.",
            f"expected {int(expected_frame_count)}, got {frame_count}",
        )
    return dict(metadata)


def _first_float(*values: Any) -> float | None:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            return parsed
    return None


def _parse_frame_rate(*values: Any) -> float | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        numerator, separator, denominator = text.partition("/")
        try:
            parsed = (
                float(numerator) / float(denominator)
                if separator and float(denominator) != 0
                else float(text)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if np.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def _normalized_even_size(
    size: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if not size:
        return None
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return None
    width = width if width % 2 == 0 else max(2, width - 1)
    height = height if height % 2 == 0 else max(2, height - 1)
    return width, height


def _ffmpeg_output_args(
    output_format: str,
    quality: str,
    fps: float,
    *,
    ffmpeg: str | None = None,
    recording_size: tuple[int, int] | None = None,
) -> list[str]:
    del recording_size
    output_format = normalize_output_format(output_format)
    quality = "high" if quality == "high" else "low"
    if output_format == "webm":
        crf = "28" if quality == "high" else "38"
        encoder = _webm_encoder(ffmpeg) if ffmpeg else "libvpx-vp9"
        args = ["-c:v", encoder, "-pix_fmt", "yuv420p", "-crf", crf, "-b:v", "0"]
        if encoder == "libvpx-vp9":
            args.extend(
                [
                    "-deadline",
                    "good",
                    "-cpu-used",
                    "2" if quality == "high" else "4",
                    "-row-mt",
                    "1",
                ]
            )
        elif encoder == "libvpx":
            args.extend(["-deadline", "good", "-cpu-used", "4"])
        elif encoder == "libsvtav1":
            args.extend(["-preset", "6" if quality == "high" else "10"])
        elif encoder == "libaom-av1":
            args.extend(["-cpu-used", "4" if quality == "high" else "8"])
        return args
    if output_format == "gif":
        max_colors = 256 if quality == "high" else 128
        dither = "sierra2_4a" if quality == "high" else "bayer:bayer_scale=5"
        return [
            "-vf",
            (
                f"fps={_fps_text(fps)},split[s0][s1];"
                f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
                f"[s1][p]paletteuse=dither={dither}"
            ),
        ]

    crf = "18" if quality == "high" else "28"
    preset = "slow" if quality == "high" else "fast"
    filter_expr = ",".join(
        [
            f"fps={_fps_text(fps)}",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "format=yuv420p",
        ]
    )
    return [
        "-vf",
        filter_expr,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        crf,
        "-movflags",
        "+faststart",
    ]


@lru_cache(maxsize=4)
def _webm_encoder(ffmpeg: str | None) -> str:
    """Choose the fastest broadly compatible WebM encoder available."""

    if not ffmpeg:
        return "libvpx-vp9"
    encoders = _ffmpeg_encoders(ffmpeg)
    for encoder in ("libvpx-vp9", "libvpx", "libsvtav1", "libaom-av1"):
        if encoder in encoders:
            return encoder
    return "libvpx-vp9"


@lru_cache(maxsize=4)
def _ffmpeg_encoders(ffmpeg: str) -> str:
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    return result.stdout or ""


def _run_ffmpeg_stream(
    cmd: list[str],
    frame_iter: Iterable[Frame],
    *,
    width: int,
    height: int,
) -> bool:
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                shell=False,
            )
        except Exception as exc:
            print(f"[FFmpeg] failed to start: {exc}")
            return False

        stream_error = False
        frame_error: Exception | None = None
        try:
            assert proc.stdin is not None
            for frame, _original_size in frame_iter:
                if frame.shape != (height, width, 3):
                    raise ValueError(
                        f"Frame shape {frame.shape} does not match {(height, width, 3)}"
                    )
                if frame.dtype != np.uint8:
                    raise TypeError(f"Frame dtype must be uint8, got {frame.dtype}")
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError, ValueError, TypeError) as exc:
            if isinstance(exc, BrokenPipeError) or proc.poll() is not None:
                stream_error = True
                print(f"[FFmpeg] streaming failed: {exc}")
            else:
                frame_error = exc
        finally:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                proc.stdin = None

        if frame_error is not None:
            if proc.poll() is None:
                _terminate_process(proc)
            raise MediaProcessingError(
                "A composite frame failed while the video was being encoded.",
                str(frame_error),
            ) from frame_error

        try:
            return_code = proc.wait(timeout=FFMPEG_FINALIZE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream_error = True
            return_code = proc.returncode or -1

        if return_code == 0 and not stream_error:
            return True
        stderr_file.seek(0)
        stderr_text = stderr_file.read().decode("utf-8", errors="replace")
        print(f"[FFmpeg] failed with return code {return_code}")
        if stderr_text:
            print(stderr_text[-1000:])
        return False


def _run_ffmpeg_file(
    cmd: list[str], *, cancel_check: CancelCheck | None = None
) -> None:
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                shell=False,
            )
        except Exception as exc:
            raise MediaProcessingError(
                "FFmpeg could not start while saving the recording.", str(exc)
            ) from exc

        deadline = time.monotonic() + FFMPEG_FINALIZE_TIMEOUT_SECONDS
        while proc.poll() is None:
            if cancel_check is not None and cancel_check():
                _terminate_process(proc)
                raise MediaProcessingError("Recording canceled.")
            if time.monotonic() >= deadline:
                _terminate_process(proc)
                raise MediaProcessingError(
                    "FFmpeg timed out while saving the recording."
                )
            time.sleep(0.05)

        if proc.returncode == 0:
            return
        stderr_file.seek(0)
        stderr = _compact_ffmpeg_stderr(
            stderr_file.read().decode("utf-8", errors="replace")
        )
        raise MediaProcessingError("FFmpeg failed while saving recording.", stderr)


def _terminate_process(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _raise_if_recording_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise MediaProcessingError("Recording canceled.")


def _write_fallback_atomic(output_path: Path, writer: Callable[[Path], bool]) -> bool:
    temporary_path = _partial_output_path(output_path)
    try:
        if not writer(temporary_path):
            return False
        _raise_if_empty(temporary_path)
        _atomic_replace(temporary_path, output_path)
        return True
    except Exception as exc:
        print(f"[media] fallback writer failed: {exc}")
        return False
    finally:
        _safe_unlink(temporary_path)


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".media-{uuid.uuid4().hex}{output_path.suffix}")


def _atomic_replace(source: Path, destination: Path) -> None:
    publication_error: OSError | None = None
    try:
        _retry_windows_file_operation(lambda: os.replace(source, destination))
        _transfer_probe_cache(source, destination)
        return
    except OSError as exc:
        if not _is_windows_sharing_violation(exc):
            raise
        if destination.exists():
            raise
        publication_error = exc

    # Virus scanners and indexers can open a finished video without delete
    # sharing. The validated bytes are still readable, so publish another NTFS
    # directory entry without moving the temporarily locked source entry.
    try:
        os.link(source, destination)
        _transfer_probe_cache(source, destination)
    except OSError as link_error:
        assert publication_error is not None
        raise publication_error from link_error


def _transfer_probe_cache(source: Path, destination: Path) -> None:
    """Carry validated metadata across an atomic temporary-file publication."""

    try:
        destination_stat = destination.stat()
        source_name = str(source.resolve())
        destination_name = str(destination.resolve())
    except OSError:
        return
    with _PROBE_CACHE_LOCK:
        matches = [
            (key, value) for key, value in _PROBE_CACHE.items() if key[0] == source_name
        ]
        for key, value in matches:
            published_key = (
                destination_name,
                int(destination_stat.st_size),
                int(destination_stat.st_mtime_ns),
                key[3],
            )
            _PROBE_CACHE[published_key] = dict(value)
            _PROBE_CACHE.move_to_end(published_key)
        while len(_PROBE_CACHE) > PROBE_CACHE_MAX_ENTRIES:
            _PROBE_CACHE.popitem(last=False)


def _safe_unlink(path: Path) -> None:
    try:
        _retry_windows_file_operation(lambda: path.unlink(missing_ok=True))
    except OSError as exc:
        # Cleanup must not hide the encoder or validation result. A later run uses
        # a UUID-backed partial path, so retaining a locked stale file is harmless.
        print(f"[media] could not remove temporary file {path}: {exc}")


def _retry_windows_file_operation(operation: Callable[[], Any]) -> Any:
    """Retry short-lived Windows sharing violations from FFmpeg/FFprobe."""

    deadline = time.monotonic() + WINDOWS_FILE_OPERATION_TIMEOUT_SECONDS
    delay = WINDOWS_FILE_OPERATION_INITIAL_DELAY_SECONDS
    while True:
        try:
            return operation()
        except OSError as exc:
            if not _is_windows_sharing_violation(exc):
                raise
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 0.25)


def _is_windows_sharing_violation(exc: OSError) -> bool:
    winerror = getattr(exc, "winerror", None)
    return winerror in {32, 33} or (
        os.name == "nt" and isinstance(exc, PermissionError)
    )


def _upload_path(output_root: Path, file_prefix: str, source_format: str) -> Path:
    return output_root / (
        f".{sanitize_filename(file_prefix)}.upload-{uuid.uuid4().hex}.{source_format}"
    )


def _compact_ffmpeg_stderr(stderr: str | None) -> str:
    if not stderr:
        return "no FFmpeg stderr"
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    compact = "\n".join(lines[-12:])
    return compact[-1200:] if compact else "no FFmpeg stderr"


def _raise_if_empty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise MediaProcessingError(f"Recording output is empty: {path}")
