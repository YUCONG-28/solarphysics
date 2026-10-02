"""Implementation of the historical GOES SXR light-curve recipe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone

from solar_apps.workflows.common.image_naming import configured_scientific_image_path
from solar_apps.platform.config import load_script_config

from solar_toolkit.xray_dem.sxr import load_goes_sxr_dataset

DEFAULT_INPUT = "data/xray/goes-sxr.nc"
DEFAULT_START = None
DEFAULT_END = None
DEFAULT_OUTPUT = "SXR.png"


def build_parser() -> argparse.ArgumentParser:
    config = load_script_config(
        "goes_sxr_lightcurve_plot", {"file_path": DEFAULT_INPUT}
    )
    parser = argparse.ArgumentParser(description="Plot a GOES soft X-ray light curve")
    parser.add_argument("--input", default=config["file_path"], help="GOES NetCDF file")
    parser.add_argument("--start-time", default=DEFAULT_START, required=True)
    parser.add_argument("--end-time", default=DEFAULT_END, required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit image filename; omit to use the scientific naming contract.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a window",
    )
    return parser


def plot_goes_lightcurve(dataset):
    """Build the historical 16-by-8 inch GOES SXR figure."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(16, 8))
    axis.semilogy(dataset["time"], dataset["xrsa_flux"], label="0.5-4.0 A")
    axis.semilogy(dataset["time"], dataset["xrsb_flux"], label="1.0-8.0 A")
    axis.legend()
    axis.set_xlabel("Time (UTC)", fontsize=12, labelpad=10)
    axis.set_ylabel("Flux (W/m²)", fontsize=12, labelpad=10)
    axis.set_title(
        "GOES Solar Soft X-ray Flux (0.5-4 Å)",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    axis.xaxis.set_minor_locator(mdates.MinuteLocator())
    axis.xaxis.grid(True, which="minor", linestyle="--", color="gray", alpha=0.5)
    figure.autofmt_xdate(rotation=45, ha="right")
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_goes_sxr_dataset(args.input, args.start_time, args.end_time)
    print(dataset)
    figure = plot_goes_lightcurve(dataset)
    times = dataset["time"].values
    output_path = configured_scientific_image_path(
        args.output,
        sequence=1,
        start_time=times[0] if len(times) else args.start_time,
        end_time=times[-1] if len(times) > 1 else None,
        instrument="goes_xrs",
        product="soft_xray_lightcurve",
        generated_at=datetime.now(timezone.utc),
    )
    figure.savefig(output_path, dpi=300)
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


__all__ = ["build_parser", "main", "plot_goes_lightcurve"]
