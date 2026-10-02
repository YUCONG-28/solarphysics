"""Implementation of the historical combined SXR/HXI/AIA summary recipe."""

from __future__ import annotations

import argparse
import csv
import fnmatch
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import configured_scientific_image_path

from solar_toolkit.xray_dem.hxi import load_hxi_lightcurve
from solar_toolkit.xray_dem.processing import calculate_derivative, smooth_flux_data
from solar_toolkit.xray_dem.sxr import load_goes_sxr_dataset

DEFAULT_SXR = "data/xray/goes-sxr.nc"
DEFAULT_HXI = "data/xray/hxi.fits"
DEFAULT_AIA_DIR = "data/aia/flux"
DEFAULT_START = None
DEFAULT_END = None
DEFAULT_WINDOW = 22
DEFAULT_POLYORDER = 3


def _utc_datetimes(values) -> list[datetime]:
    return [
        datetime.utcfromtimestamp(
            value.astype("datetime64[ns]").astype("int64") / 1_000_000_000
        )
        for value in np.asarray(values)
    ]


def load_sxr_data(file_path, start_time, end_time):
    """Load and prepare SXR channels with the historical summary settings."""

    dataset = load_goes_sxr_dataset(file_path, start_time, end_time)
    times_raw = np.asarray(dataset["time"].values)
    xrsa = np.asarray(dataset["xrsa_flux"].values, dtype=float)
    xrsb = np.asarray(dataset["xrsb_flux"].values, dtype=float)
    xrsa_smooth = smooth_flux_data(
        xrsa,
        DEFAULT_WINDOW,
        DEFAULT_POLYORDER,
        method="savgol",
    )
    xrsb_smooth = smooth_flux_data(
        xrsb,
        DEFAULT_WINDOW,
        DEFAULT_POLYORDER,
        method="savgol",
    )
    return {
        "times": _utc_datetimes(times_raw),
        "xrsa": {
            "raw": xrsa,
            "smoothed": xrsa_smooth,
            "deriv": calculate_derivative(times_raw, xrsa_smooth, method="forward"),
        },
        "xrsb": {
            "raw": xrsb,
            "smoothed": xrsb_smooth,
            "deriv": calculate_derivative(times_raw, xrsb_smooth, method="forward"),
        },
        "time_subset": times_raw,
    }


def load_hxi_data(file_path, start_time, end_time):
    """Compatibility name for the canonical HXI loader."""

    return load_hxi_lightcurve(file_path, start_time, end_time)


def load_aia_data(dir_path, file_pattern, start_time, end_time):
    """Load positive AIA flux samples from matching two-column CSV files."""

    data_dir = Path(dir_path)
    all_files = sorted(
        file
        for file in data_dir.iterdir()
        if file.suffix == ".csv" and fnmatch.fnmatch(file.name, file_pattern)
    )
    if not all_files:
        print(f"No AIA CSV files in {dir_path!s} match {file_pattern!r}")
        return []

    all_aia_data = []
    for file_path in all_files:
        times: list[datetime] = []
        fluxes: list[float] = []
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) != 2:
                        continue
                    observed = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    if not start_time <= observed <= end_time:
                        continue
                    flux = float(row[1])
                    if flux > 0:
                        times.append(observed)
                        fluxes.append(flux)
        except (OSError, ValueError) as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue
        if times:
            all_aia_data.append(
                {"times": times, "flux": fluxes, "label": file_path.name}
            )
    return all_aia_data


def plot_combined_data(
    sxr_data, hxi_data, aia_data_list, start_time, end_time, vline_times
):
    """Plot SXR, HXI, SXR derivative, and optional AIA flux on shared time axes."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    figure, ax1 = plt.subplots(figsize=(18, 12))
    ax1.set_xlim(start_time, end_time)
    for index, vtime in enumerate(vline_times):
        ax1.axvline(
            x=vtime,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"Time {index + 1}: {vtime:%H:%M:%S}",
        )

    ax1.semilogy(
        sxr_data["times"],
        sxr_data["xrsa"]["smoothed"],
        label="SXR 0.5-4.0 Å (smoothed)",
        color="red",
    )
    ax1.semilogy(
        sxr_data["times"],
        sxr_data["xrsb"]["smoothed"],
        label="SXR 1.0-8.0 Å (smoothed)",
        color="green",
    )
    ax1.set_ylabel("SXR flux (W/m²)", fontsize=12, labelpad=10)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.legend(loc="upper left", fontsize=10)

    ax2 = ax1.twinx()
    colors = ["darkred", "darkgreen", "blue", "purple"]
    for index, (energy, counts) in enumerate(hxi_data["data"].items()):
        ax2.semilogy(
            hxi_data["times"],
            counts,
            label=f"HXI {energy}",
            color=colors[index],
            linestyle="--",
        )
    ax2.set_ylabel("HXI count rate (counts/s/detector)", fontsize=12, labelpad=10)
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.legend(loc="upper right", fontsize=10)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(
        sxr_data["times"][1:],
        sxr_data["xrsa"]["deriv"],
        label="SXR 0.5-4.0 Å derivative",
        color="pink",
        linestyle="-.",
    )
    ax3.plot(
        sxr_data["times"][1:],
        sxr_data["xrsb"]["deriv"],
        label="SXR 1.0-8.0 Å derivative",
        color="lightgreen",
        linestyle="-.",
    )
    ax3.set_ylabel("SXR flux derivative (W/m²/s)", fontsize=12, labelpad=10)
    ax3.tick_params(axis="y", labelcolor="black")
    ax3.set_ylim(-0.25e-15, 0.25e-15)
    ax3.legend(loc="lower left", fontsize=10)

    if aia_data_list:
        ax4 = ax1.twinx()
        ax4.spines["right"].set_position(("outward", 120))
        line_styles = ["-", "--", "-.", ":"]
        for index, aia_data in enumerate(aia_data_list):
            ax4.semilogy(
                aia_data["times"],
                aia_data["flux"],
                label=aia_data["label"],
                linestyle=line_styles[index % len(line_styles)],
                alpha=0.7,
            )
        ax4.set_ylabel("AIA flux", fontsize=12, labelpad=10)
        ax4.tick_params(axis="y", labelcolor="black")
        ax4.legend(loc="lower right", fontsize=10)

    ax1.set_xlabel("Time (UTC)", fontsize=12, labelpad=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator())
    ax1.xaxis.grid(True, which="both", linestyle="--", alpha=0.5)
    figure.autofmt_xdate(rotation=45, ha="right")
    ax1.set_title(
        f"SXR-HXR-AIA ({start_time:%Y-%m-%d %H:%M} to {end_time:%H:%M})",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def build_parser() -> argparse.ArgumentParser:
    config = load_script_config(
        "flare_aia_sxr_hxr_summary_plot",
        {"sxr": DEFAULT_SXR, "hxi": DEFAULT_HXI, "aia_dir": DEFAULT_AIA_DIR},
    )
    parser = argparse.ArgumentParser(
        description="Create a combined SXR, HXI, and AIA time-series summary"
    )
    parser.add_argument("--sxr", default=config["sxr"])
    parser.add_argument("--hxi", default=config["hxi"])
    parser.add_argument("--aia-dir", default=config["aia_dir"])
    parser.add_argument(
        "--aia-patterns", nargs="+", default=["aia_304.csv", "aia_1600.csv"]
    )
    parser.add_argument("--start-time", default=DEFAULT_START, required=True)
    parser.add_argument("--end-time", default=DEFAULT_END, required=True)
    parser.add_argument(
        "--vline-times",
        nargs="+",
        default=[],
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit image filename; omit to use the scientific naming contract.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to --output; historical default remains display-only",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start_time = datetime.fromisoformat(args.start_time.replace("T", " "))
        end_time = datetime.fromisoformat(args.end_time.replace("T", " "))
        vline_times = [
            datetime.fromisoformat(value.replace("T", " "))
            for value in args.vline_times
        ]
    except ValueError as exc:
        print(f"Invalid ISO time: {exc}")
        return 2

    sxr_data = load_sxr_data(args.sxr, args.start_time, args.end_time)
    hxi_data = load_hxi_data(args.hxi, start_time, end_time)
    aia_data_list = []
    for pattern in args.aia_patterns:
        aia_data_list.extend(load_aia_data(args.aia_dir, pattern, start_time, end_time))
    figure = plot_combined_data(
        sxr_data, hxi_data, aia_data_list, start_time, end_time, vline_times
    )
    if args.save:
        output_path = configured_scientific_image_path(
            args.output,
            sequence=1,
            start_time=start_time,
            end_time=end_time,
            instrument="goes_xrs_hxi_aia",
            product="flare_summary",
            generated_at=datetime.now(timezone.utc),
        )
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


__all__ = [
    "build_parser",
    "load_aia_data",
    "load_hxi_data",
    "load_sxr_data",
    "main",
    "plot_combined_data",
]
