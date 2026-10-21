"""Command-line entry point for AIA/HMI magnetic-contour overlays."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from solar_toolkit.hmi.overlay import (
    DEFAULT_AIA_DIR,
    DEFAULT_HMI_DIR,
    DEFAULT_OUTPUT_DIR,
    run_overlay_workflow,
)
from solar_apps.platform.config import load_script_config

_DEFAULT_CONFIG = {
    "input_dir_AIA": str(DEFAULT_AIA_DIR),
    "input_dir_HMI": str(DEFAULT_HMI_DIR),
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "roi_bounds": None,
    "threshold_gauss": 0.0,
    "gaussian_sigma": 3.0,
    "vmin": 16.0,
    "vmax": 6666.0,
    "contour_level_gauss": 50.0,
    "max_time_diff_seconds": 24.0,
    "dpi": 300,
    "show_plot": False,
    "show_progress": True,
}


def _build_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow hmi overlay",
        description="Render time-matched HMI magnetic contours over AIA images.",
    )
    parser.add_argument("--input-dir-aia", default=config["input_dir_AIA"])
    parser.add_argument("--input-dir-hmi", default=config["input_dir_HMI"])
    parser.add_argument("--output-dir", default=config["output_dir"])
    parser.add_argument(
        "--roi-bounds",
        nargs=4,
        type=float,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=config.get("roi_bounds"),
        required=config.get("roi_bounds") is None,
        help="Explicit helioprojective ROI in arcsec: xmin ymin xmax ymax.",
    )
    parser.add_argument(
        "--threshold-gauss",
        type=float,
        default=float(config.get("threshold_gauss", 0.0)),
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=float(config.get("gaussian_sigma", 3.0)),
    )
    parser.add_argument("--vmin", type=float, default=float(config.get("vmin", 16.0)))
    parser.add_argument(
        "--vmax",
        type=float,
        default=float(config.get("vmax", 6666.0)),
    )
    parser.add_argument(
        "--contour-level-gauss",
        type=float,
        default=float(config.get("contour_level_gauss", 50.0)),
    )
    parser.add_argument(
        "--max-time-diff-seconds",
        type=float,
        default=float(config.get("max_time_diff_seconds", 24.0)),
    )
    parser.add_argument("--dpi", type=int, default=int(config.get("dpi", 300)))
    parser.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("show_plot", False)),
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=bool(config.get("show_progress", True)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured AIA/HMI overlay workflow."""

    config = load_script_config("sdo_aia_hmi_overlay", _DEFAULT_CONFIG)
    args = _build_parser(config).parse_args(argv)
    output_paths = run_overlay_workflow(
        input_dir_aia=args.input_dir_aia,
        input_dir_hmi=args.input_dir_hmi,
        output_dir=args.output_dir,
        roi_bounds=tuple(args.roi_bounds),
        threshold_gauss=args.threshold_gauss,
        gaussian_sigma=args.gaussian_sigma,
        vmin=args.vmin,
        vmax=args.vmax,
        contour_level_gauss=args.contour_level_gauss,
        max_time_diff_seconds=args.max_time_diff_seconds,
        dpi=args.dpi,
        show_plot=args.show_plot,
        show_progress=args.show_progress,
    )
    print(f"Generated {len(output_paths)} AIA/HMI overlay image(s)")
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
