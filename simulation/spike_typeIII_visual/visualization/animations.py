"""GIF and MP4 generation for MHD and radio-proxy evolution."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..physics.radio import RadioResult
from ..physics.rmhd import MHDResult

NAVY = "#0B2545"
ORANGE = "#D97706"
RED = "#C9302C"

AnimationFormat = Literal["gif", "mp4"]
ANIMATION_FORMATS = frozenset({"gif", "mp4"})


def _mhd_frame_indices(result: MHDResult, render_profile: str) -> np.ndarray:
    """Select 300 distinct physical snapshots for the 4K delivery profile."""

    count = len(result.times)
    if render_profile != "presentation-4k" or count <= 300:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, 300, dtype=int)


def _imageio_module():
    """Import the optional animation writer only when an export is requested."""

    import imageio.v2 as imageio

    return imageio


def _canvas_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.asarray(rgba[:, :, :3]).copy()


def _new_figure(
    render_profile: str = "legacy",
) -> tuple[plt.Figure, plt.Axes]:
    if render_profile == "presentation-4k":
        figsize, dpi = (12.8, 7.2), 300
    else:
        figsize, dpi = (9.6, 5.4), 100
    fig, axis = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    return fig, axis


def normalize_animation_formats(selection: str) -> tuple[AnimationFormat, ...]:
    """Expand one CLI animation selection into concrete export formats."""

    mapping: dict[str, tuple[AnimationFormat, ...]] = {
        "none": (),
        "gif": ("gif",),
        "mp4": ("mp4",),
        "both": ("gif", "mp4"),
    }
    try:
        return mapping[selection]
    except KeyError as exc:
        choices = ", ".join(mapping)
        raise ValueError(
            f"Unknown animation format {selection!r}; use one of: {choices}."
        ) from exc


def validate_animation_formats(
    formats: tuple[AnimationFormat, ...],
) -> tuple[AnimationFormat, ...]:
    """Validate a concrete, ordered, duplicate-free format tuple."""

    invalid = sorted(set(formats) - ANIMATION_FORMATS)
    if invalid:
        raise ValueError(f"Unsupported animation formats: {invalid}")
    if len(set(formats)) != len(formats):
        raise ValueError("Animation formats must not contain duplicates.")
    return formats


def _load_imageio_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def require_mp4_backend() -> str:
    """Return the bundled FFmpeg executable or raise an installation hint."""

    try:
        return _load_imageio_ffmpeg_executable()
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "MP4 export requires imageio-ffmpeg. Install it with: "
            "conda install -n solar_simulation -c conda-forge "
            "imageio-ffmpeg"
        ) from exc


def _write(path: Path, frames: Iterable[np.ndarray], fps: int = 10) -> None:
    imageio = _imageio_module()
    suffix = path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(
            path,
            frames,
            duration=1000.0 / fps,
            loop=0,
        )
        return
    if suffix == ".mp4":
        require_mp4_backend()
        imageio.mimsave(
            path,
            frames,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
        )
        return
    raise ValueError(f"Unsupported animation suffix: {path.suffix!r}")


def _write_stream(path: Path, frames: Iterable[np.ndarray], fps: int) -> None:
    """Write a long presentation video without retaining RGB frames."""

    imageio = _imageio_module()
    require_mp4_backend()
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "17"],
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_lossless_master(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: int = 30,
) -> None:
    """Stream RGB frames to a lossless FFV1/MKV master."""

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot encode an empty frame sequence.") from exc
    height, width = first.shape[:2]
    command = [
        require_mp4_backend(),
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "rgb24",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        process.stdin.write(np.ascontiguousarray(first).tobytes())
        for frame in iterator:
            if frame.shape[:2] != (height, width):
                raise ValueError("All video frames must have identical dimensions.")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        returncode = process.wait()
    except BaseException:
        process.kill()
        raise
    if returncode:
        raise RuntimeError(
            f"FFV1 encoding failed: {stderr.decode(errors='replace')[-2000:]}"
        )


def _transcode_delivery(master: Path, delivery: Path) -> str:
    """Prefer NVENC CQ 17 and fall back to libx264 CRF 17."""

    executable = require_mp4_backend()
    encoder_query = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout
    if "h264_nvenc" in encoder_query:
        nvenc = [
            executable,
            "-y",
            "-i",
            str(master),
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-rc",
            "vbr",
            "-cq",
            "17",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            str(delivery),
        ]
        attempt = subprocess.run(
            nvenc,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if attempt.returncode == 0:
            return "h264_nvenc-cq17"
    fallback = [
        executable,
        "-y",
        "-i",
        str(master),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(delivery),
    ]
    subprocess.run(fallback, check=True)
    return "libx264-crf17"


def _iter_tearing_frames(
    result: MHDResult, render_profile: str = "legacy"
) -> Iterable[np.ndarray]:
    extent = (
        result.grid.x.min(),
        result.grid.x.max(),
        result.grid.y.min(),
        result.grid.y.max(),
    )
    global_limit = max(float(np.max(result.max_current)), 1.0)
    for index in _mhd_frame_indices(result, render_profile):
        time_value = result.times[index]
        _, _, _, _, current, _ = result.snapshot_fields(int(index))
        fig, axis = _new_figure(render_profile)
        axis.imshow(
            current,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-global_limit,
            vmax=global_limit,
            alpha=0.80,
        )
        axis.contour(
            result.grid.x_mesh,
            result.grid.y_mesh,
            result.psi[index],
            levels=18,
            colors="black",
            linewidths=0.52,
        )
        source_label = (
            "Athena full-MHD"
            if getattr(result, "source", "") == "athena-c"
            else "Reduced-MHD"
        )
        axis.set(
            title=f"{source_label} tearing evolution  |  t = {time_value:.2f}",
            xlabel="x (normalized)",
            ylabel="y (normalized)",
        )
        yield _canvas_rgb(fig)
        plt.close(fig)


def _render_tearing_frames(result: MHDResult) -> list[np.ndarray]:
    return list(_iter_tearing_frames(result))


def _iter_jet_frames(
    result: MHDResult, render_profile: str = "legacy"
) -> Iterable[np.ndarray]:
    extent = (
        result.grid.x.min(),
        result.grid.x.max(),
        result.grid.y.min(),
        result.grid.y.max(),
    )
    stride = max(1, result.grid.x.size // 22)
    max_speed = max(float(np.max(result.max_speed)), 1e-6)
    for index in _mhd_frame_indices(result, render_profile):
        time_value = result.times[index]
        _, _, velocity_x, velocity_y, _, _ = result.snapshot_fields(int(index))
        speed = np.hypot(velocity_x, velocity_y)
        fig, axis = _new_figure(render_profile)
        axis.imshow(
            speed,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=max_speed,
        )
        axis.quiver(
            result.grid.x_mesh[::stride, ::stride],
            result.grid.y_mesh[::stride, ::stride],
            velocity_x[::stride, ::stride],
            velocity_y[::stride, ::stride],
            color="white",
            alpha=0.82,
            scale=3.2,
            width=0.0025,
        )
        axis.set(
            title=f"Bidirectional reconnection flow  |  t = {time_value:.2f}",
            xlabel="x (normalized)",
            ylabel="y (normalized)",
        )
        yield _canvas_rgb(fig)
        plt.close(fig)


def _render_jet_frames(result: MHDResult) -> list[np.ndarray]:
    return list(_iter_jet_frames(result))


def _iter_electron_frames(
    radio: RadioResult, render_profile: str = "legacy"
) -> Iterable[np.ndarray]:
    frame_count = 300 if render_profile == "presentation-4k" else 40
    indices = np.linspace(0, len(radio.times_s) - 1, frame_count, dtype=int)
    for index in indices:
        fig, axis = _new_figure(render_profile)
        axis.plot(
            radio.times_s[: index + 1],
            radio.beam_height_mm[: index + 1],
            color=NAVY,
            linewidth=2.6,
        )
        axis.scatter(
            [radio.times_s[index]],
            [radio.beam_height_mm[index]],
            color=ORANGE,
            edgecolor="white",
            linewidth=0.8,
            s=75,
            zorder=3,
        )
        axis.fill_between(
            radio.times_s[: index + 1],
            0.0,
            radio.beam_height_mm[: index + 1],
            color="#2A7F8E",
            alpha=0.12,
        )
        axis.set(
            title="Kinematic electron-beam proxy  |  v = 0.20 c",
            xlabel="Time (s)",
            ylabel="Height (Mm)",
            xlim=(0.0, radio.times_s[-1]),
            ylim=(0.0, radio.beam_height_mm[-1] * 1.05),
        )
        axis.grid(True, alpha=0.25)
        yield _canvas_rgb(fig)
        plt.close(fig)


def _render_electron_frames(radio: RadioResult) -> list[np.ndarray]:
    return list(_iter_electron_frames(radio))


def _iter_typeiii_frames(
    radio: RadioResult, render_profile: str = "legacy"
) -> Iterable[np.ndarray]:
    frame_count = 300 if render_profile == "presentation-4k" else 60
    indices = np.linspace(1, len(radio.times_s) - 1, frame_count, dtype=int)
    for index in indices:
        fig, axis = _new_figure(render_profile)
        axis.pcolormesh(
            radio.times_s[: index + 1],
            radio.frequencies_mhz,
            radio.intensity[:, : index + 1],
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        axis.plot(
            radio.times_s[: index + 1],
            radio.ridge_frequency_mhz[: index + 1],
            color="#67E8F9",
            linewidth=1.5,
        )
        axis.set(
            title="Synthetic Type III dynamic spectrum  |  proxy model",
            xlabel="Time (s)",
            ylabel="Frequency (MHz)",
            xlim=(0.0, radio.times_s[-1]),
            ylim=(radio.frequencies_mhz.min(), radio.frequencies_mhz.max()),
        )
        yield _canvas_rgb(fig)
        plt.close(fig)


def _render_typeiii_frames(radio: RadioResult) -> list[np.ndarray]:
    return list(_iter_typeiii_frames(radio))


def save_tearing_animation(result: MHDResult, path: Path) -> None:
    """Render and save one tearing animation in the suffix-selected format."""

    _write(path, _render_tearing_frames(result))


def save_jet_animation(result: MHDResult, path: Path) -> None:
    """Render and save one reconnection-jet animation."""

    _write(path, _render_jet_frames(result))


def save_electron_animation(radio: RadioResult, path: Path) -> None:
    """Render and save one electron-beam proxy animation."""

    _write(path, _render_electron_frames(radio))


def save_typeiii_animation(radio: RadioResult, path: Path) -> None:
    """Render and save one synthetic Type III spectrum animation."""

    _write(path, _render_typeiii_frames(radio))


def save_animations(
    result: MHDResult,
    radio: RadioResult,
    animations_dir: Path,
    formats: tuple[AnimationFormat, ...] = ("gif",),
    *,
    render_profile: str = "legacy",
) -> list[Path]:
    """Render each animation once and write every requested format."""

    validate_animation_formats(formats)
    if not formats:
        return []
    if "mp4" in formats:
        require_mp4_backend()
    animations_dir.mkdir(parents=True, exist_ok=True)
    if render_profile == "presentation-4k":
        renderers = [
            ("tearing", lambda: _iter_tearing_frames(result, render_profile)),
            ("jet", lambda: _iter_jet_frames(result, render_profile)),
            (
                "electron_beam",
                lambda: _iter_electron_frames(radio, render_profile),
            ),
            ("typeIII", lambda: _iter_typeiii_frames(radio, render_profile)),
        ]
    else:
        renderers = [
            ("tearing", lambda: _render_tearing_frames(result)),
            ("jet", lambda: _render_jet_frames(result)),
            ("electron_beam", lambda: _render_electron_frames(radio)),
            ("typeIII", lambda: _render_typeiii_frames(radio)),
        ]
    paths: list[Path] = []
    encoders: dict[str, str] = {}
    for stem, render in renderers:
        if render_profile != "presentation-4k":
            frames = render()
        for animation_format in formats:
            path = animations_dir / f"{stem}.{animation_format}"
            if render_profile == "presentation-4k" and animation_format == "mp4":
                master = animations_dir / f"{stem}_master_ffv1.mkv"
                _write_lossless_master(master, render(), fps=30)
                encoders[stem] = _transcode_delivery(master, path)
                paths.append(master)
            elif render_profile == "presentation-4k":
                preview_renderers = {
                    "tearing": lambda: _iter_tearing_frames(result, "preview"),
                    "jet": lambda: _iter_jet_frames(result, "preview"),
                    "electron_beam": lambda: _iter_electron_frames(
                        radio, "preview"
                    ),
                    "typeIII": lambda: _iter_typeiii_frames(radio, "preview"),
                }
                _write(path, preview_renderers[stem]())
            else:
                _write(path, frames)
            paths.append(path)
    if encoders:
        report_path = animations_dir / "media_encoding.json"
        report_path.write_text(
            json.dumps(
                {
                    "master": "ffv1-level3",
                    "fps": 30,
                    "delivery_encoders": encoders,
                    "nvenc_fallback_policy": "libx264-crf17",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        paths.append(report_path)
    return paths
