"""Implementation of the historical GOES SXR derivative comparison recipe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np

from solar_apps.platform.config import load_script_config
from solar_apps.workflows.common.image_naming import configured_scientific_image_path

from solar_toolkit.xray_dem.processing import (
    calculate_derivative as _calculate_derivative,
)
from solar_toolkit.xray_dem.processing import smooth_flux_data as _smooth_flux_data
from solar_toolkit.xray_dem.sxr import load_goes_sxr_dataset

DEFAULT_INPUT = "data/xray/goes-sxr.nc"
START_TIME = None
END_TIME = None
SMOOTH_WINDOW = 22
SMOOTH_POLY_ORDER = 3
FIG_SIZE = (12, 16)
DPI = 300
FIG_NAME = "SXR分析结果.png"
FONT_CANDIDATES = [
    "Arial",
    "Times New Roman",
    "PingFang SC",
    "Hiragino Sans GB",
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]


def get_available_fonts(candidates):
    """Return the requested font families that Matplotlib can resolve."""

    from matplotlib.font_manager import FontProperties, findfont

    available = []
    for font in candidates:
        try:
            findfont(FontProperties(family=font), fallback_to_default=False)
            available.append(font)
        except ValueError:
            continue
    return available if available else ["sans-serif"]


class CustomLogFormatter(mticker.LogFormatter):
    """Format log ticks with MathText-safe minus signs."""

    def __call__(self, x, pos=None):
        if x == 0:
            return ""
        exponent = np.floor(np.log10(x))
        base = x / (10**exponent)
        if abs(base - 1.0) < 1e-6:
            return rf"$10^{{{exponent}}}$"
        return rf"${base:.1f} \times 10^{{{exponent}}}$"


def init_plt_settings() -> None:
    """Apply the historical font and MathText settings."""

    import matplotlib.pyplot as plt

    available_fonts = get_available_fonts(FONT_CANDIDATES)
    print(f"Available fonts: {available_fonts}")
    plt.rcParams.update(
        {
            "font.family": available_fonts,
            "axes.unicode_minus": True,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "regular",
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def load_sxr_data(file_path, start_time, end_time):
    """Compatibility helper returning an in-memory, cropped GOES dataset."""

    dataset = load_goes_sxr_dataset(
        Path(file_path), start_time, end_time, require_data=True
    )
    print("Dataset information:")
    print(dataset)
    print(f"Selected {dataset.sizes.get('time', 0)} samples")
    return dataset


def smooth_flux_data(flux_data, window_length, polyorder):
    """Reproduce the historical odd-window, data-clamped Savitzky-Golay mode."""

    if window_length % 2 == 0:
        print(f"Adjusted smoothing window to odd length: {window_length + 1}")
    return _smooth_flux_data(
        flux_data,
        window_length,
        polyorder,
        method="savgol",
        adjust_even_window=True,
        clamp_to_data=True,
    )


def calculate_derivative(time_data, flux_data):
    """Reproduce the historical exact-time NumPy gradient."""

    return _calculate_derivative(time_data, flux_data, method="gradient")


def plot_flux_comparison(axis, time, raw_data, smooth_data, title, ylabel):
    axis.semilogy(time, raw_data, label="Raw", color="lightcoral", alpha=0.5)
    axis.semilogy(time, smooth_data, label="Smoothed", color="red")
    axis.legend()
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(True, linestyle="--", alpha=0.5)

    import matplotlib.dates as mdates

    axis.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    axis.xaxis.grid(True, which="minor", linestyle="--", color="gray", alpha=0.3)
    axis.yaxis.set_major_formatter(CustomLogFormatter())


def plot_derivative(axis, time, derivative_data, title, ylabel):
    absolute = np.abs(np.asarray(derivative_data, dtype=float)).copy()
    nonzero = absolute[absolute > 0]
    absolute[absolute == 0] = np.min(nonzero) * 0.1 if nonzero.size else 1e-20
    axis.semilogy(time, absolute, color="blue", label="Absolute derivative")
    axis.legend()
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    axis.grid(True, linestyle="--", alpha=0.5)

    import matplotlib.dates as mdates

    axis.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    axis.xaxis.grid(True, which="minor", linestyle="--", color="gray", alpha=0.3)
    axis.yaxis.set_major_formatter(CustomLogFormatter())


def visualize_results(
    time,
    xrsa_raw,
    xrsa_smooth,
    xrsb_raw,
    xrsb_smooth,
    xrsa_deriv,
    xrsb_deriv,
    *,
    save_fig: bool = False,
    fig_name: str | None = None,
    dpi: int = DPI,
    show: bool = True,
):
    """Build, optionally save, and display the four-panel comparison."""

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)
    plot_flux_comparison(
        axes[0],
        time,
        xrsa_raw,
        xrsa_smooth,
        "GOES-16 Solar Soft X-ray Flux (0.5-4 Å)",
        "Flux (W/m²)",
    )
    plot_flux_comparison(
        axes[1],
        time,
        xrsb_raw,
        xrsb_smooth,
        "GOES-16 Solar Soft X-ray Flux (1.0-8 Å)",
        "Flux (W/m²)",
    )
    plot_derivative(
        axes[2],
        time,
        xrsa_deriv,
        "GOES-16 SXR Flux Derivative (0.5-4 Å)",
        "Flux derivative (W/m²/s)",
    )
    plot_derivative(
        axes[3],
        time,
        xrsb_deriv,
        "GOES-16 SXR Flux Derivative (1.0-8 Å)",
        "Flux derivative (W/m²/s)",
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    figure.autofmt_xdate(rotation=45, ha="right")
    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    if save_fig:
        time_values = np.asarray(getattr(time, "values", time))
        output_path = configured_scientific_image_path(
            fig_name,
            sequence=1,
            start_time=time_values[0] if len(time_values) else None,
            end_time=time_values[-1] if len(time_values) > 1 else None,
            instrument="goes_xrs",
            product="soft_xray_derivative_comparison",
            generated_at=datetime.now(timezone.utc),
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")
    if show:
        plt.show()
    return figure


def build_parser() -> argparse.ArgumentParser:
    config = load_script_config(
        "neupert_sxr_derivative_hxr_comparison", {"data_file_path": DEFAULT_INPUT}
    )
    parser = argparse.ArgumentParser(
        description="Compare GOES SXR flux and its derivative for Neupert analysis"
    )
    parser.add_argument("--input", default=config["data_file_path"])
    parser.add_argument("--start-time", default=START_TIME, required=True)
    parser.add_argument("--end-time", default=END_TIME, required=True)
    parser.add_argument("--window-length", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--polyorder", type=int, default=SMOOTH_POLY_ORDER)
    parser.add_argument("--save", action="store_true", help="Save the figure")
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit image filename; omit to use the scientific naming contract.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        init_plt_settings()
        dataset = load_sxr_data(args.input, args.start_time, args.end_time)
        times = dataset["time"]
        xrsa = np.clip(dataset["xrsa_flux"].values, 1e-12, None)
        xrsb = np.clip(dataset["xrsb_flux"].values, 1e-12, None)
        xrsa_smooth = smooth_flux_data(xrsa, args.window_length, args.polyorder)
        xrsb_smooth = smooth_flux_data(xrsb, args.window_length, args.polyorder)
        visualize_results(
            times,
            xrsa,
            xrsa_smooth,
            xrsb,
            xrsb_smooth,
            calculate_derivative(times, xrsa_smooth),
            calculate_derivative(times, xrsb_smooth),
            save_fig=args.save,
            fig_name=args.output,
            show=not args.no_show,
        )
    except Exception as exc:
        print(f"Neupert comparison failed: {exc}")
        return 1
    return 0


__all__ = [
    "CustomLogFormatter",
    "build_parser",
    "calculate_derivative",
    "get_available_fonts",
    "init_plt_settings",
    "load_sxr_data",
    "main",
    "plot_derivative",
    "plot_flux_comparison",
    "smooth_flux_data",
    "visualize_results",
]
