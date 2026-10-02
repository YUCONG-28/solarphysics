#!/usr/bin/env python3
"""Plot STEREO-A EUVI images nearest an explicitly selected UTC time."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from sunpy.map import Map

from solar_apps.workflows.common.image_naming import build_scientific_image_filename

DATA_DIR = None
MANIFEST = None
OUT_DIR = None
TARGET = None
WAVELENGTHS = ("171", "195", "284", "304")

__all__ = [
    "build_parser",
    "load_manifest",
    "main",
    "make_norm",
    "nearest_file",
    "normalized_map",
    "out_name",
    "plot_overview",
    "plot_single",
    "title_for",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the event-recipe parser without reading the manifest."""
    parser = argparse.ArgumentParser(description="Plot STEREO/EUVI overview products.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-time", type=datetime.fromisoformat, required=True)
    return parser


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def nearest_file(rows: list[dict[str, str]], wavelength: str) -> Path:
    candidates = [row for row in rows if row["wavelength"] == wavelength]
    if not candidates:
        raise FileNotFoundError(f"No EUVI files found for {wavelength} A")
    best = min(
        candidates,
        key=lambda row: abs(
            (datetime.fromisoformat(row["date_obs"]) - TARGET).total_seconds()
        ),
    )
    return Path(best["path"])


def normalized_map(path: Path):
    euvi_map = Map(path)
    exptime = getattr(euvi_map, "exposure_time", None)
    if exptime is not None and exptime.to_value(u.s) > 0:
        euvi_map = euvi_map / exptime.to_value(u.s)
        euvi_map.meta["bunit"] = "DN/s"
    return euvi_map


def make_norm(euvi_map):
    data = np.asarray(euvi_map.data, dtype=float)
    valid = data[np.isfinite(data) & (data > 0)]
    if valid.size == 0:
        valid = data[np.isfinite(data)]
    return ImageNormalize(
        valid, interval=PercentileInterval(99.7), stretch=AsinhStretch()
    )


def title_for(euvi_map) -> str:
    wave = int(round(euvi_map.wavelength.to_value(u.Angstrom)))
    return f"STEREO-A EUVI {wave} A | {euvi_map.date.strftime('%Y-%m-%d %H:%M:%S')} UT"


def out_name(euvi_map, *, sequence=1, generated_at=None) -> str:
    wave = int(round(euvi_map.wavelength.to_value(u.Angstrom)))
    return build_scientific_image_filename(
        sequence=sequence,
        start_time=euvi_map.date,
        instrument="stereo_a_euvi",
        channel=f"{wave}a",
        product="intensity",
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def plot_single(euvi_map, norm, *, sequence=1, generated_at=None) -> Path:
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
    generated_at = generated_at or datetime.now(timezone.utc)
    out = OUT_DIR / out_name(
        euvi_map,
        sequence=sequence,
        generated_at=generated_at,
    )
    fig.savefig(out, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overview(items, *, sequence=1, generated_at=None) -> Path:
    fig = plt.figure(figsize=(11.5, 10.5))
    for idx, (euvi_map, norm) in enumerate(items, start=1):
        ax = fig.add_subplot(2, 2, idx, projection=euvi_map)
        euvi_map.plot(axes=ax, norm=norm, title=False)
        ax.set_title(title_for(euvi_map), fontsize=10)
        ax.coords.grid(color="white", alpha=0.22, linestyle="--", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle(f"STEREO-A EUVI nearest {TARGET.isoformat()} UT", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    generated_at = generated_at or datetime.now(timezone.utc)
    observation_times = [item[0].date for item in items]
    out = OUT_DIR / build_scientific_image_filename(
        sequence=sequence,
        start_time=min(observation_times),
        end_time=max(observation_times),
        instrument="stereo_a_euvi",
        product="multi_wavelength_overview",
        generated_at=generated_at,
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    global DATA_DIR, MANIFEST, OUT_DIR, TARGET
    args = build_parser().parse_args(argv)
    DATA_DIR = args.input_dir.expanduser()
    MANIFEST = DATA_DIR / "manifest_by_wavelength.csv"
    OUT_DIR = args.output_dir.expanduser()
    TARGET = args.target_time
    plt.switch_backend("Agg")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()
    paths = [nearest_file(rows, wavelength) for wavelength in WAVELENGTHS]
    items = []
    single_outputs = []
    selection_lines = []
    batch_generated_at = datetime.now(timezone.utc)
    for sequence, path in enumerate(paths, start=1):
        euvi_map = normalized_map(path)
        norm = make_norm(euvi_map)
        single_outputs.append(
            plot_single(
                euvi_map,
                norm,
                sequence=sequence,
                generated_at=batch_generated_at,
            )
        )
        items.append((euvi_map, norm))
        selection_lines.append(
            f"{int(round(euvi_map.wavelength.to_value(u.Angstrom)))} A,"
            f"{euvi_map.date.isot},{path}"
        )
    overview = plot_overview(
        items,
        sequence=len(paths) + 1,
        generated_at=batch_generated_at,
    )
    selected = OUT_DIR / "selected_euvi.txt"
    selected.write_text(
        "wavelength,date_obs,path\n" + "\n".join(selection_lines) + "\n",
        encoding="utf-8",
    )

    print(f"single_png={len(single_outputs)}")
    for out in single_outputs:
        print(out)
    print(f"overview={overview}")
    print(f"selected={selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
