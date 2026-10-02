"""Implementation of the historical four-panel Neupert timing recipe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np

from solar_apps.platform.config import load_script_config

from solar_toolkit.xray_dem.processing import calculate_derivative, smooth_flux_data
from solar_toolkit.xray_dem.sxr import load_goes_sxr_dataset

DEFAULT_INPUT = "data/xray/goes-sxr.nc"
DEFAULT_START = None
DEFAULT_END = None
DEFAULT_WINDOW = 22
DEFAULT_POLYORDER = 3


def build_parser() -> argparse.ArgumentParser:
    config = load_script_config(
        "neupert_timing_error_analysis", {"file_path": DEFAULT_INPUT}
    )
    parser = argparse.ArgumentParser(
        description="Compare raw and smoothed GOES SXR channels for Neupert timing"
    )
    parser.add_argument("--input", default=config["file_path"], help="GOES NetCDF file")
    parser.add_argument("--start-time", default=DEFAULT_START, required=True)
    parser.add_argument("--end-time", default=DEFAULT_END, required=True)
    parser.add_argument("--window-length", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--polyorder", type=int, default=DEFAULT_POLYORDER)
    parser.add_argument("--output", help="Optional figure path; disabled by default")
    parser.add_argument("--no-show", action="store_true")
    return parser


def prepare_neupert_timing(
    dataset, *, window_length: int = DEFAULT_WINDOW, polyorder: int = DEFAULT_POLYORDER
) -> dict[str, np.ndarray]:
    """Prepare raw, smoothed, and forward-derivative SXR arrays."""

    times = np.asarray(dataset["time"].values)
    xrsa = np.asarray(dataset["xrsa_flux"].values, dtype=float)
    xrsb = np.asarray(dataset["xrsb_flux"].values, dtype=float)
    xrsa_smooth = smooth_flux_data(
        xrsa,
        window_length,
        polyorder,
        method="savgol",
    )
    xrsb_smooth = smooth_flux_data(
        xrsb,
        window_length,
        polyorder,
        method="savgol",
    )
    return {
        "time": times,
        "xrsa": xrsa,
        "xrsb": xrsb,
        "xrsa_smooth": xrsa_smooth,
        "xrsb_smooth": xrsb_smooth,
        "xrsa_derivative": calculate_derivative(times, xrsa_smooth, method="forward"),
        "xrsb_derivative": calculate_derivative(times, xrsb_smooth, method="forward"),
    }


def plot_neupert_timing(data: dict[str, np.ndarray]):
    """Build the historical 2-by-2 raw/smoothed GOES comparison."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    figure, ([ax1, ax3], [ax2, ax4]) = plt.subplots(2, 2, figsize=(32, 16), sharex=True)
    panels = [
        (ax1, data["xrsa"], None, "red", "GOES-16 SXR Flux (0.5-4 Å)"),
        (ax2, data["xrsb"], None, "green", "GOES-16 SXR Flux (1.0-8 Å)"),
        (
            ax3,
            data["xrsa"],
            data["xrsa_smooth"],
            "red",
            "GOES-16 SXR Flux (0.5-4 Å): raw and smoothed",
        ),
        (
            ax4,
            data["xrsb"],
            data["xrsb_smooth"],
            "green",
            "GOES-16 SXR Flux (1.0-8 Å): raw and smoothed",
        ),
    ]
    for axis, raw, smoothed, color, title in panels:
        axis.semilogy(data["time"], raw, label="Raw", color=color, alpha=0.7)
        if smoothed is not None:
            axis.semilogy(data["time"], smoothed, label="Smoothed", color=color)
        axis.legend()
        axis.set_ylabel("Flux (W/m²)", fontsize=12, labelpad=10)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.xaxis.set_minor_locator(mdates.MinuteLocator())
        axis.xaxis.grid(True, which="minor", linestyle="--", color="gray", alpha=0.5)
    figure.autofmt_xdate(rotation=45, ha="right")
    figure.tight_layout()
    return figure


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_goes_sxr_dataset(args.input, args.start_time, args.end_time)
    print(dataset)
    data = prepare_neupert_timing(
        dataset,
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    figure = plot_neupert_timing(data)
    if args.output:
        figure.savefig(args.output, dpi=300)
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


__all__ = [
    "build_parser",
    "main",
    "plot_neupert_timing",
    "prepare_neupert_timing",
]
