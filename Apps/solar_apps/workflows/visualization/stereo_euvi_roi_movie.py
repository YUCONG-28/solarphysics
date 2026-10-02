#!/usr/bin/env python3
"""Make fixed-canvas STEREO-A EUVI ROI log-stretched MP4 movies."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import astropy.units as u
import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from matplotlib.colors import Normalize
from sunpy.map import Map
from sunpy.util.exceptions import SunpyMetadataWarning, SunpyUserWarning

DATA_DIR = None
MANIFEST = None
OUT_ROOT = None
MP4_DIR = None
WAVELENGTHS = ("171", "195", "284", "304")

X_MIN = X_MAX = Y_MIN = Y_MAX = None

FPS = 4
DPI = 160
FIGSIZE = (6.0, 8.0)
PERCENTILES = (1.0, 99.7)

__all__ = [
    "build_parser",
    "compute_limits",
    "crop_roi",
    "draw_frame",
    "exposure_normalized",
    "files_for_wavelength",
    "load_manifest",
    "log_map",
    "main",
    "make_movie_like_opencv_mp4",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the event-recipe parser without loading FITS data."""
    parser = argparse.ArgumentParser(
        description="Generate fixed-canvas STEREO/EUVI ROI movies."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        required=True,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
    )
    return parser


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def files_for_wavelength(rows: list[dict[str, str]], wavelength: str) -> list[Path]:
    selected = [row for row in rows if row["wavelength"] == wavelength]
    selected.sort(key=lambda row: row["date_obs"])
    return [Path(row["path"]) for row in selected]


def exposure_normalized(path: Path):
    euvi_map = Map(path)
    exptime = getattr(euvi_map, "exposure_time", None)
    if exptime is not None and exptime.to_value(u.s) > 0:
        euvi_map = euvi_map / exptime.to_value(u.s)
    euvi_map.meta["bunit"] = "DN/s"
    return euvi_map


def crop_roi(euvi_map):
    bottom_left = SkyCoord(Tx=X_MIN, Ty=Y_MIN, frame=euvi_map.coordinate_frame)
    top_right = SkyCoord(Tx=X_MAX, Ty=Y_MAX, frame=euvi_map.coordinate_frame)
    return euvi_map.submap(bottom_left, top_right=top_right)


def log_map(cropped_map):
    data = np.asarray(cropped_map.data, dtype=float)
    log_data = np.full_like(data, np.nan, dtype=float)
    positive = np.isfinite(data) & (data > 0)
    log_data[positive] = np.log10(data[positive])
    meta = cropped_map.meta.copy()
    meta["bunit"] = "log10(DN/s)"
    out = Map(log_data, meta)
    out.plot_settings["cmap"] = cropped_map.plot_settings.get("cmap", "gray")
    return out


def compute_limits(paths: list[Path]) -> tuple[float, float]:
    samples = []
    for path in paths:
        cropped = crop_roi(exposure_normalized(path))
        data = np.asarray(cropped.data, dtype=float)
        valid = data[np.isfinite(data) & (data > 0)]
        if valid.size:
            samples.append(np.log10(valid))
    combined = np.concatenate(samples)
    vmin, vmax = np.nanpercentile(combined, PERCENTILES)
    return float(vmin), float(vmax)


def draw_frame(
    path: Path, wavelength: str, out_png: Path, vmin: float, vmax: float
) -> None:
    plot_map = log_map(crop_roi(exposure_normalized(path)))
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_subplot(projection=plot_map)
    im = plot_map.plot(axes=ax, norm=Normalize(vmin=vmin, vmax=vmax), title=False)
    ax.set_title(
        f"STEREO-A EUVI {wavelength} A | "
        f"{plot_map.date.strftime('%Y-%m-%d %H:%M:%S')} UT\n"
        "x=0..800, y=-1000..200 arcsec",
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
    fig.savefig(out_png)
    plt.close(fig)


def make_movie_like_opencv_mp4(frame_paths: list[Path], video_path: Path) -> None:
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    height, width, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, FPS, (width, height))
    if not video.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {video_path}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Cannot read frame: {frame_path}")
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Frame size mismatch: {frame_path} has {frame.shape[:2]}, expected {(height, width)}"
            )
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()


def main(argv: list[str] | None = None) -> int:
    global DATA_DIR, MANIFEST, OUT_ROOT, MP4_DIR, X_MIN, X_MAX, Y_MIN, Y_MAX
    args = build_parser().parse_args(argv)
    DATA_DIR = args.input_dir.expanduser()
    MANIFEST = DATA_DIR / "manifest_by_wavelength.csv"
    OUT_ROOT = args.output_dir.expanduser()
    MP4_DIR = OUT_ROOT / "mp4"
    X_MIN, X_MAX, Y_MIN, Y_MAX = (value * u.arcsec for value in args.roi)
    if X_MIN >= X_MAX or Y_MIN >= Y_MAX:
        raise ValueError("ROI bounds must increase.")
    plt.switch_backend("Agg")
    warnings.filterwarnings("ignore", category=SunpyUserWarning)
    warnings.filterwarnings("ignore", category=SunpyMetadataWarning)
    rows = load_manifest()
    MP4_DIR.mkdir(parents=True, exist_ok=True)

    summary = ["wavelength,n_frames,width,height,vmin_log10_dns,vmax_log10_dns,mp4"]
    for wavelength in WAVELENGTHS:
        paths = files_for_wavelength(rows, wavelength)
        frame_dir = OUT_ROOT / wavelength / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        vmin, vmax = compute_limits(paths)
        frame_paths = []
        for index, path in enumerate(paths, start=1):
            out_png = frame_dir / f"frame_{index:03d}_{path.stem}.png"
            draw_frame(path, wavelength, out_png, vmin, vmax)
            frame_paths.append(out_png)
            print(f"{wavelength} frame {index:03d}/{len(paths)} {out_png}", flush=True)

        video_path = (
            MP4_DIR
            / f"stereo_a_euvi_{wavelength}_roi_x0000_0800_y-1000_0200_log_fixed.mp4"
        )
        make_movie_like_opencv_mp4(frame_paths, video_path)
        first = cv2.imread(str(frame_paths[0]))
        height, width = first.shape[:2]
        summary.append(
            f"{wavelength},{len(frame_paths)},{width},{height},{vmin:.6g},{vmax:.6g},{video_path}"
        )
        print(f"movie={video_path}", flush=True)

    summary_path = OUT_ROOT / "movie_summary.csv"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
