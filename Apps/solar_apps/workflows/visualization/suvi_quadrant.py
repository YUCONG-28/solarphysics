#!/usr/bin/env python3
"""Plot SUVI lower-right quadrant images at an explicit UTC time."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from sunpy.map import Map
from sunpy.util.exceptions import SunpyMetadataWarning, SunpyUserWarning

from solar_apps.workflows.common.image_naming import build_scientific_image_filename

DATA_ROOT = None
OUT_ROOT = None
DATE_STAMP = None
TARGET_START = None
SATELLITES = ("goes16", "goes18")
CHANNELS = ("094", "131", "171", "195", "284", "304")

__all__ = [
    "SuviSelection",
    "build_parser",
    "find_suvi_files",
    "lower_right_quadrant",
    "main",
    "make_norm",
    "obs_stamp",
    "output_name",
    "plot_overview",
    "plot_single",
    "title_for",
    "write_selected_files",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the event-recipe parser without scanning the archive."""
    parser = argparse.ArgumentParser(
        description="Plot GOES/SUVI lower-right quadrant products."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-time", type=datetime.fromisoformat, required=True)
    return parser


@dataclass(frozen=True)
class SuviSelection:
    satellite: str
    sat_short: str
    channel: str
    path: Path


def find_suvi_files() -> list[SuviSelection]:
    selections: list[SuviSelection] = []
    for satellite in SATELLITES:
        sat_short = f"g{satellite.removeprefix('goes')}"
        for channel in CHANNELS:
            pattern = (
                DATA_ROOT
                / satellite
                / f"ci{channel}"
                / DATE_STAMP
                / f"dr_suvi-l2-ci{channel}_{sat_short}_s{DATE_STAMP}T{TARGET_START}Z*.fits"
            )
            matches = sorted(pattern.parent.glob(pattern.name))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one file for {satellite} ci{channel}, found {len(matches)}: {matches}"
                )
            selections.append(SuviSelection(satellite, sat_short, channel, matches[0]))
    return selections


def lower_right_quadrant(suvi_map):
    bottom_left = SkyCoord(
        Tx=0 * u.arcsec,
        Ty=suvi_map.bottom_left_coord.Ty,
        frame=suvi_map.coordinate_frame,
    )
    top_right = SkyCoord(
        Tx=suvi_map.top_right_coord.Tx,
        Ty=0 * u.arcsec,
        frame=suvi_map.coordinate_frame,
    )
    return suvi_map.submap(bottom_left, top_right=top_right)


def make_norm(cropped_map):
    data = cropped_map.data
    finite_positive = data[(data > 0) & (data == data)]
    if finite_positive.size == 0:
        return ImageNormalize(
            data, interval=PercentileInterval(99.7), stretch=AsinhStretch()
        )
    return ImageNormalize(
        finite_positive, interval=PercentileInterval(99.7), stretch=AsinhStretch()
    )


def obs_stamp(suvi_map) -> str:
    return suvi_map.date.strftime("%Y%m%d_%H%M%S")


def title_for(selection: SuviSelection, suvi_map) -> str:
    wave = int(round(suvi_map.wavelength.to_value(u.Angstrom)))
    return f"{selection.sat_short.upper()} SUVI {wave} A | {suvi_map.date.strftime('%Y-%m-%d %H:%M:%S')} UT"


def output_name(
    selection: SuviSelection, suvi_map, *, sequence=1, generated_at=None
) -> str:
    wave = int(round(suvi_map.wavelength.to_value(u.Angstrom)))
    return build_scientific_image_filename(
        sequence=sequence,
        start_time=suvi_map.date,
        instrument=f"suvi_{selection.sat_short}",
        channel=f"{wave}a",
        product="intensity",
        qualifiers="lower_right_quadrant",
        generated_at=generated_at or datetime.now(timezone.utc),
    )


def plot_single(
    selection: SuviSelection,
    cropped_map,
    norm,
    *,
    sequence=1,
    generated_at=None,
) -> Path:
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(projection=cropped_map)
    im = cropped_map.plot(axes=ax, norm=norm, title=False)
    ax.set_title(title_for(selection, cropped_map), fontsize=11)
    ax.set_xlabel("Helioprojective X (arcsec)")
    ax.set_ylabel("Helioprojective Y (arcsec)")
    ax.coords.grid(color="white", alpha=0.25, linestyle="--", linewidth=0.6)
    cbar = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05)
    cbar.set_label("Intensity")
    fig.tight_layout()

    generated_at = generated_at or datetime.now(timezone.utc)
    out_path = OUT_ROOT / output_name(
        selection,
        cropped_map,
        sequence=sequence,
        generated_at=generated_at,
    )
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overview(
    items: list[tuple[SuviSelection, object, object]],
    *,
    sequence=1,
    generated_at=None,
) -> Path:
    fig = plt.figure(figsize=(15.0, 10.0))
    for idx, (selection, cropped_map, norm) in enumerate(items, start=1):
        ax = fig.add_subplot(2, 6, idx, projection=cropped_map)
        cropped_map.plot(axes=ax, norm=norm, title=False, annotate=False)
        ax.set_title(title_for(selection, cropped_map), fontsize=9)
        ax.coords.grid(color="white", alpha=0.22, linestyle="--", linewidth=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=7)
        if idx not in (1, 7):
            ax.coords[1].set_ticklabel_visible(False)
        if idx <= 6:
            ax.coords[0].set_ticklabel_visible(False)

    fig.suptitle(
        f"SUVI lower-right quadrant | {DATE_STAMP} {TARGET_START} UTC", fontsize=15
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    generated_at = generated_at or datetime.now(timezone.utc)
    observation_times = [item[1].date for item in items]
    out_path = OUT_ROOT / build_scientific_image_filename(
        sequence=sequence,
        start_time=min(observation_times),
        end_time=max(observation_times),
        instrument="suvi",
        product="multi_satellite_channel_overview",
        qualifiers="lower_right_quadrant",
        generated_at=generated_at,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_selected_files(selections: list[SuviSelection]) -> Path:
    out_path = OUT_ROOT / "selected_files.txt"
    lines = [
        "# SUVI files used for lower-right quadrant plots",
        f"# target_start={DATE_STAMP}T{TARGET_START}Z",
        "",
    ]
    for selection in selections:
        lines.append(f"{selection.sat_short} ci{selection.channel}: {selection.path}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    global DATA_ROOT, OUT_ROOT, DATE_STAMP, TARGET_START
    args = build_parser().parse_args(argv)
    DATA_ROOT = args.input_dir.expanduser()
    OUT_ROOT = args.output_dir.expanduser()
    DATE_STAMP = args.target_time.strftime("%Y%m%d")
    TARGET_START = args.target_time.strftime("%H%M%S")
    plt.switch_backend("Agg")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=SunpyUserWarning)
    warnings.filterwarnings("ignore", category=SunpyMetadataWarning)

    selections = find_suvi_files()
    if not all(
        f"s{DATE_STAMP}T{TARGET_START}Z" in str(item.path) for item in selections
    ):
        raise RuntimeError(
            "At least one selected FITS file does not match the requested start time."
        )

    plotted_items = []
    single_outputs = []
    batch_generated_at = datetime.now(timezone.utc)
    for sequence, selection in enumerate(selections, start=1):
        full_map = Map(selection.path)
        cropped_map = lower_right_quadrant(full_map)
        norm = make_norm(cropped_map)
        single_outputs.append(
            plot_single(
                selection,
                cropped_map,
                norm,
                sequence=sequence,
                generated_at=batch_generated_at,
            )
        )
        plotted_items.append((selection, cropped_map, norm))

    overview = plot_overview(
        plotted_items,
        sequence=len(selections) + 1,
        generated_at=batch_generated_at,
    )
    selected_file_log = write_selected_files(selections)

    print(f"single_png={len(single_outputs)}")
    for out_path in single_outputs:
        print(out_path)
    print(f"overview={overview}")
    print(f"selected_files={selected_file_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
