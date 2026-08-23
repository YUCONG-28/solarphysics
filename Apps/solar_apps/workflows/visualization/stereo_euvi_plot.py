# SPDX-License-Identifier: GPL-3.0-only
"""Parametrized STEREO EUVI plotting for the App 1.0 worker."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib
import numpy as np
import sunpy
from astropy.io.fits import getheader
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from matplotlib.colors import Normalize
from sunpy.map import Map
from sunpy.util.exceptions import SunpyMetadataWarning, SunpyUserWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from solar_apps.workflows.common.image_naming import (
    build_scientific_image_filename,
)  # noqa: E402

DEFAULT_WAVELENGTHS = (171, 195, 284, 304)


@dataclass(frozen=True)
class EuvPlotConfig:
    input_dir: Path
    output_dir: Path
    wavelengths: tuple[int, ...] = DEFAULT_WAVELENGTHS
    target_time: datetime | None = None
    roi_bounds: tuple[float, float, float, float] | None = None
    fps: int = 4


def discover_euvi_files(input_dir: Path) -> list[Path]:
    """Return sorted, unique, resolved EUVI FITS paths in *input_dir*."""
    patterns = ("*.fts", "*.fits", "*.fit")
    found = {path.resolve() for pattern in patterns for path in input_dir.glob(pattern)}
    return sorted(found)


def read_euvi_record(path: Path) -> dict | None:
    """Read one EUVI FITS header and project the fields this module needs."""
    header = getheader(path)
    if "WAVELNTH" not in header or header.get("WAVELNTH") is None:
        return None
    date_obs = header.get("DATE-OBS")
    detector = header.get("DETECTOR")
    observatory = header.get("OBSRVTRY")
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "date_obs": None if date_obs is None else str(date_obs),
        "wavelength": int(float(header.get("WAVELNTH"))),
        "detector": "" if detector is None else str(detector),
        "observatory": "" if observatory is None else str(observatory),
        "exptime": header.get("EXPTIME"),
    }


def build_manifest(
    input_dir: Path, wavelengths: Sequence[int] | None = None
) -> list[dict]:
    """Build and sort the manifest for *input_dir*, optionally filtered."""
    records = [
        record
        for path in discover_euvi_files(input_dir)
        if (record := read_euvi_record(path)) is not None
    ]
    if wavelengths is not None:
        wanted = set(int(item) for item in wavelengths)
        records = [record for record in records if int(record["wavelength"]) in wanted]
    if not records:
        raise FileNotFoundError(
            f"No EUVI FITS files found in {input_dir}"
            + (f" for wavelengths {tuple(wavelengths)}" if wavelengths else "")
        )
    records.sort(
        key=lambda record: (
            int(record["wavelength"]),
            str(record.get("date_obs") or ""),
            str(record.get("filename") or ""),
        )
    )
    return records


def load_map(path: Path) -> "sunpy.map.Map":
    return Map(path)


def exposure_normalized(path: Path) -> "sunpy.map.Map":
    euvi_map = Map(path)
    exptime = getattr(euvi_map, "exposure_time", None)
    if exptime is not None:
        seconds = exptime.to_value(u.s)
        if seconds > 0:
            euvi_map = euvi_map / seconds
    euvi_map.meta["bunit"] = "DN/s"
    return euvi_map


def make_norm(euvi_map) -> ImageNormalize:
    data = np.asarray(euvi_map.data, dtype=float)
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        valid = data[np.isfinite(data)]
    return ImageNormalize(
        valid, interval=PercentileInterval(99.7), stretch=AsinhStretch()
    )


def _observatory_short(meta: Any) -> str:
    raw = str(meta.get("OBSRVTRY") or "STEREO").strip().upper()
    if "STEREO-A" in raw or raw == "A":
        return "STEREO-A"
    if "STEREO-B" in raw or raw == "B":
        return "STEREO-B"
    return "STEREO"


def _instrument_id(meta: Any) -> str:
    label = _observatory_short(meta).lower().replace("-", "_")
    if label.startswith("stereo"):
        return f"{label}_euvi"
    return "stereo_euvi"


def title_for(euvi_map) -> str:
    wave = int(round(euvi_map.wavelength.to_value(u.Angstrom)))
    return f"{_observatory_short(euvi_map.meta)} EUVI {wave} A | {euvi_map.date.strftime('%Y-%m-%d %H:%M:%S')} UT"


def out_name(euvi_map, *, sequence: int, generated_at: datetime) -> str:
    wave = int(round(euvi_map.wavelength.to_value(u.Angstrom)))
    return build_scientific_image_filename(
        sequence=sequence,
        start_time=euvi_map.date,
        instrument=_instrument_id(euvi_map.meta),
        channel=f"{wave}a",
        product="intensity",
        generated_at=generated_at,
    )


def crop_roi(euvi_map, roi_bounds) -> "sunpy.map.Map":
    from astropy.coordinates import SkyCoord

    xmin, xmax, ymin, ymax = roi_bounds
    bottom_left = SkyCoord(
        Tx=xmin * u.arcsec, Ty=ymin * u.arcsec, frame=euvi_map.coordinate_frame
    )
    top_right = SkyCoord(
        Tx=xmax * u.arcsec, Ty=ymax * u.arcsec, frame=euvi_map.coordinate_frame
    )
    return euvi_map.submap(bottom_left, top_right=top_right)


def log_map(euvi_map) -> "sunpy.map.Map":
    data = np.asarray(euvi_map.data, dtype=float)
    log_data = np.full_like(data, np.nan, dtype=float)
    positive = np.isfinite(data) & (data > 0)
    log_data[positive] = np.log10(data[positive])
    meta = euvi_map.meta.copy()
    meta["bunit"] = "log10(DN/s)"
    out = Map(log_data, meta)
    out.plot_settings["cmap"] = euvi_map.plot_settings.get("cmap", "gray")
    return out


def compute_limits(paths: Sequence[Path], roi_bounds=None) -> tuple[float, float]:
    samples = []
    for path in paths:
        euvi_map = exposure_normalized(path)
        if roi_bounds is not None:
            euvi_map = crop_roi(euvi_map, roi_bounds)
        data = np.asarray(euvi_map.data, dtype=float)
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size:
            samples.append(np.log10(valid))
    if not samples:
        raise ValueError("No valid EUVI data for compute_limits")
    combined = np.concatenate(samples)
    vmin, vmax = np.nanpercentile(combined, [1.0, 99.7])
    return float(vmin), float(vmax)


def draw_frame(
    path: Path,
    wavelength: int,
    out_png: Path,
    vmin: float,
    vmax: float,
    roi_bounds=None,
) -> None:
    euvi_map = exposure_normalized(path)
    if roi_bounds is not None:
        euvi_map = crop_roi(euvi_map, roi_bounds)
    plot_map = log_map(euvi_map)
    fig = plt.figure(figsize=(6.0, 8.0), dpi=160)
    ax = fig.add_subplot(projection=plot_map)
    im = plot_map.plot(axes=ax, norm=Normalize(vmin=vmin, vmax=vmax), title=False)
    ax.set_title(
        f"{_observatory_short(plot_map.meta)} EUVI {wavelength} A | "
        f"{plot_map.date.strftime('%Y-%m-%d %H:%M:%S')} UT",
        fontsize=10,
    )
    ax.set_xlabel("Helioprojective X (arcsec)")
    ax.set_ylabel("Helioprojective Y (arcsec)")
    ax.coords.grid(color="white", alpha=0.25, linestyle="--", linewidth=0.6)
    try:
        plot_map.draw_limb(axes=ax, color="cyan", linewidth=0.8, alpha=0.8)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.045)
    cbar.set_label("log10(DN/s)")
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.08, top=0.92)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def make_movie_like_opencv_mp4(
    frame_paths: Sequence[Path], video_path: Path, fps: int
) -> None:
    if not frame_paths:
        raise ValueError("No frames provided for movie")
    if len(frame_paths) == 0:
        raise ValueError("No frames provided for movie")
    import cv2

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not video.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Cannot read frame: {frame_path}")
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Frame size mismatch: {frame_path} has {frame.shape[:2]}, "
                f"expected {(height, width)}"
            )
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def _parse_datetime(value: Any) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _target_aware(target: datetime) -> datetime:
    if target.tzinfo is None:
        return target.replace(tzinfo=timezone.utc)
    return target


def _overview_grid(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols


def plot_euvi_overview(config: EuvPlotConfig) -> list[Path]:
    manifest = build_manifest(config.input_dir, config.wavelengths)
    by_wavelength: dict[int, list[dict]] = {}
    for record in manifest:
        by_wavelength.setdefault(int(record["wavelength"]), []).append(record)

    selected: list[dict] = []
    for wavelength in config.wavelengths:
        records = by_wavelength.get(int(wavelength), [])
        if not records:
            raise FileNotFoundError(f"No EUVI files found for {wavelength} A")
        if config.target_time is not None:
            target = _target_aware(config.target_time)
            best = min(
                records,
                key=lambda record: abs(
                    (_parse_datetime(record.get("date_obs")) - target).total_seconds()
                ),
            )
        else:
            best = min(
                records, key=lambda record: _parse_datetime(record.get("date_obs"))
            )
        selected.append(best)

    overview_dir = config.output_dir / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc)
    outputs: list[Path] = []
    items = []
    for sequence, record in enumerate(selected, start=1):
        euvi_map = exposure_normalized(Path(record["path"]))
        norm = make_norm(euvi_map)
        out_png = overview_dir / out_name(
            euvi_map, sequence=sequence, generated_at=generated_at
        )
        fig = plt.figure(figsize=(6.3, 6.0))
        ax = fig.add_subplot(projection=euvi_map)
        im = euvi_map.plot(axes=ax, norm=norm, title=False)
        ax.set_title(title_for(euvi_map), fontsize=11)
        ax.set_xlabel("Helioprojective X (arcsec)")
        ax.set_ylabel("Helioprojective Y (arcsec)")
        ax.coords.grid(color="white", alpha=0.22, linestyle="--", linewidth=0.6)
        cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.045)
        cbar.set_label(euvi_map.meta.get("bunit", "Intensity"))
        fig.tight_layout()
        fig.savefig(out_png, dpi=230, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_png)
        items.append((euvi_map, norm))

    times = [_parse_datetime(record.get("date_obs")) for record in selected]
    overview_out = overview_dir / build_scientific_image_filename(
        sequence=len(selected) + 1,
        start_time=min(times),
        end_time=max(times),
        instrument="stereo_a_euvi",
        product="multi_wavelength_overview",
        generated_at=generated_at,
    )
    rows, cols = _overview_grid(len(items))
    fig = plt.figure(figsize=(11.5, 10.5))
    for index, (euvi_map, norm) in enumerate(items, start=1):
        ax = fig.add_subplot(rows, cols, index, projection=euvi_map)
        euvi_map.plot(axes=ax, norm=norm, title=False)
        ax.set_title(title_for(euvi_map), fontsize=10)
        ax.coords.grid(color="white", alpha=0.22, linestyle="--", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("STEREO EUVI overview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(overview_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(overview_out)
    return outputs


def make_roi_movie(config: EuvPlotConfig) -> list[Path]:
    warnings.filterwarnings("ignore", category=SunpyUserWarning)
    warnings.filterwarnings("ignore", category=SunpyMetadataWarning)
    manifest = build_manifest(config.input_dir, config.wavelengths)
    by_wavelength: dict[int, list[dict]] = {}
    for record in manifest:
        by_wavelength.setdefault(int(record["wavelength"]), []).append(record)

    movie_dir = config.output_dir / "movie"
    movie_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for wavelength in config.wavelengths:
        records = by_wavelength.get(int(wavelength), [])
        if not records:
            raise FileNotFoundError(f"No EUVI files found for {wavelength} A")
        records.sort(key=lambda record: _parse_datetime(record.get("date_obs")))
        paths = [Path(record["path"]) for record in records]
        vmin, vmax = compute_limits(paths, config.roi_bounds)
        frame_dir = movie_dir / str(wavelength) / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        for index, path in enumerate(paths, start=1):
            out_png = frame_dir / f"frame_{index:03d}_{path.stem}.png"
            draw_frame(path, wavelength, out_png, vmin, vmax, config.roi_bounds)
            frame_paths.append(out_png)
        video_path = movie_dir / f"stereo_euvi_{wavelength}_roi.mp4"
        make_movie_like_opencv_mp4(frame_paths, video_path, config.fps)
        outputs.append(video_path)
    return outputs


__all__ = [
    "DEFAULT_WAVELENGTHS",
    "EuvPlotConfig",
    "discover_euvi_files",
    "read_euvi_record",
    "build_manifest",
    "load_map",
    "exposure_normalized",
    "make_norm",
    "title_for",
    "out_name",
    "crop_roi",
    "log_map",
    "compute_limits",
    "draw_frame",
    "make_movie_like_opencv_mp4",
    "plot_euvi_overview",
    "make_roi_movie",
]
