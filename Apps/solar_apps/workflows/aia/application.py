"""Command parser and application entry point for AIA EUV processing."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from pathlib import Path

from .config import AIAConfig, _normalize_wave_float_dict
from solar_toolkit.aia.processor import (
    _actual_mode,
    _configure_matplotlib_backend,
    process_aia_fits,
)

__all__ = ["build_parser", "config_from_args", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow aia",
        description="Process exposure-normalized SDO/AIA EUV FITS files.",
    )
    parser.add_argument("--root", dest="root_dir", default=None)
    parser.add_argument("--year", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. When omitted, preserve the historical "
            "behavior of writing beside the selected AIA data."
        ),
    )
    parser.add_argument("--mode", choices=("single", "mosaic", "test"), default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--test-wave", type=int, default=None)
    parser.add_argument("--test-index", type=int, default=None)
    parser.add_argument("--use-test-mode", action="store_true")
    parser.add_argument(
        "--waves",
        nargs="+",
        type=int,
        default=None,
        help="AIA wavelengths to process, e.g. 94 131 171 193 211 304 335 1600.",
    )
    parser.add_argument("--start", dest="start_idx", type=int, default=None)
    parser.add_argument("--end", dest="end_idx", type=int, default=None)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=None,
    )
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--workers", dest="max_workers", type=int, default=None)
    parser.add_argument("--mosaic-ncols", type=int, default=None)
    parser.add_argument("--mosaic-manual-layout", action="store_true", default=None)
    parser.add_argument(
        "--no-mosaic-manual-layout",
        dest="mosaic_manual_layout",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-seamless", action="store_true", default=None)
    parser.add_argument(
        "--no-mosaic-seamless",
        dest="mosaic_seamless",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--mosaic-show-outer-axes",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-outer-axes",
        dest="mosaic_show_outer_axes",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-ticklabel-fontsize", type=int, default=None)
    parser.add_argument("--mosaic-axislabel-fontsize", type=int, default=None)
    parser.add_argument("--mosaic-save-tight", action="store_true", default=None)
    parser.add_argument(
        "--no-mosaic-save-tight",
        dest="mosaic_save_tight",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-max-workers", type=int, default=None)
    parser.add_argument("--mosaic-max-slots", type=int, default=None)
    parser.add_argument("--mosaic-difference-output-subdir", default=None)
    parser.add_argument(
        "--mosaic-original-plus-difference-output-subdir",
        default=None,
    )
    parser.add_argument("--mosaic-left", type=float, default=None)
    parser.add_argument("--mosaic-right", type=float, default=None)
    parser.add_argument("--mosaic-bottom", type=float, default=None)
    parser.add_argument("--mosaic-top", type=float, default=None)
    parser.add_argument(
        "--mosaic-force-fill-axes",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-force-fill-axes",
        dest="mosaic_force_fill_axes",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-debug-layout", action="store_true", default=None)
    parser.add_argument(
        "--mosaic-reduce-tick-overlap",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-reduce-tick-overlap",
        dest="mosaic_reduce_tick_overlap",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-max-ticks-per-axis", type=int, default=None)
    parser.add_argument(
        "--mosaic-hide-boundary-ticklabels",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-hide-boundary-ticklabels",
        dest="mosaic_hide_boundary_ticklabels",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--mosaic-x-tick-strategy",
        choices=("all_bottom", "first_bottom_only", "alternating_bottom"),
        default=None,
    )
    parser.add_argument(
        "--mosaic-y-tick-strategy",
        choices=("all_left", "first_left_only", "alternating_left"),
        default=None,
    )
    parser.add_argument(
        "--mosaic-outer-axislabel-once",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-outer-axislabel-once",
        dest="mosaic_outer_axislabel_once",
        action="store_false",
        default=None,
    )
    parser.add_argument("--mosaic-global-outer-axes", action="store_true", default=None)
    parser.add_argument(
        "--no-mosaic-global-outer-axes",
        dest="mosaic_global_outer_axes",
        action="store_false",
        default=None,
    )
    parser.add_argument("--draw-original", action="store_true", default=None)
    parser.add_argument(
        "--no-draw-original",
        dest="draw_original",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--mosaic-difference-inline",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-mosaic-difference-inline",
        dest="mosaic_difference_inline",
        action="store_false",
        default=None,
    )

    parser.add_argument("--draw-difference", action="store_true", default=None)
    parser.add_argument(
        "--no-draw-difference",
        dest="draw_difference",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--difference-method",
        choices=("base", "running"),
        default=None,
    )
    parser.add_argument(
        "--difference-output-mode",
        choices=("auto", "mosaic", "single", "both"),
        default=None,
        help=(
            "Difference output target. auto: single mode saves per-band "
            "differences, mosaic mode saves mosaic differences; mosaic: save "
            "stitched mosaic; single: save per-band difference images; both: "
            "save both."
        ),
    )
    parser.add_argument(
        "--diff-waves",
        "--difference-wavelengths",
        dest="difference_wavelengths",
        nargs="+",
        type=int,
        default=None,
        help="Wavelengths for difference images only, e.g. --diff-waves 171 193 304.",
    )
    parser.add_argument(
        "--diff-base-index",
        dest="difference_base_index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--difference-norm-mode",
        choices=("auto", "fixed", "config"),
        default=None,
    )
    parser.add_argument("--difference-percentile", type=float, default=None)
    parser.add_argument("--difference-vmin", type=float, default=None)
    parser.add_argument("--difference-vmax", type=float, default=None)
    parser.add_argument(
        "--difference-vmin-by-wave",
        nargs="*",
        default=None,
        metavar="WAVE:VMIN",
        help="Per-band difference vmin, e.g. 94:-80 131:-120 171:-200.",
    )
    parser.add_argument(
        "--difference-vmax-by-wave",
        nargs="*",
        default=None,
        metavar="WAVE:VMAX",
        help="Per-band difference vmax, e.g. 94:80 131:120 171:200.",
    )
    parser.add_argument(
        "--difference-vlim-by-wave",
        nargs="*",
        default=None,
        metavar="WAVE:VLIM",
        help="Per-band symmetric difference range, e.g. 94:80 131:120 171:200.",
    )
    parser.add_argument("--difference-cmap", default=None)
    parser.add_argument(
        "--difference-cmap-mode",
        choices=("band", "diverging", "custom"),
        default=None,
    )
    parser.add_argument(
        "--warn-band-difference-cmap",
        action="store_true",
        default=None,
        help=(
            "Print a warning when difference_cmap_mode='band'. By default "
            "this warning is disabled because the user may intentionally force "
            "each AIA difference panel to use its corresponding band colormap."
        ),
    )
    parser.add_argument(
        "--difference-save-reference",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-difference-save-reference",
        dest="difference_save_reference",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--difference-colorbar",
        dest="difference_show_colorbar",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-difference-colorbar",
        dest="difference_show_colorbar",
        action="store_false",
        default=None,
    )
    parser.add_argument(
        "--difference-derotate",
        dest="difference_derotate",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-difference-derotate",
        dest="difference_derotate",
        action="store_false",
        default=None,
    )

    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument("--show-grid", dest="show_grid", action="store_true")
    grid_group.add_argument("--no-grid", dest="show_grid", action="store_false")
    parser.set_defaults(show_grid=None)

    parser.add_argument("--show-limb", action="store_true", default=None)
    parser.add_argument(
        "--colorbar", dest="show_colorbar", action="store_true", default=None
    )
    parser.add_argument(
        "--also-save-single",
        dest="multi_band_also_save_single",
        action="store_true",
        default=None,
    )
    parser.add_argument("--vmin", dest="user_vmin", type=float, default=None)
    parser.add_argument("--vmax", dest="user_vmax", type=float, default=None)
    parser.add_argument("--cmap", dest="user_cmap", default=None)
    return parser


def _set_if_provided(kwargs: dict, name: str, value) -> None:
    if value is not None:
        kwargs[name] = value


def config_from_args(args: argparse.Namespace) -> AIAConfig:
    cfg = AIAConfig()
    provided = {}
    for name in (
        "root_dir",
        "year",
        "date",
        "mode",
        "test_wave",
        "test_index",
        "start_idx",
        "end_idx",
        "dpi",
        "max_workers",
        "mosaic_ncols",
        "mosaic_manual_layout",
        "mosaic_seamless",
        "mosaic_show_outer_axes",
        "mosaic_ticklabel_fontsize",
        "mosaic_axislabel_fontsize",
        "mosaic_save_tight",
        "mosaic_max_workers",
        "mosaic_max_slots",
        "mosaic_difference_output_subdir",
        "mosaic_original_plus_difference_output_subdir",
        "mosaic_left",
        "mosaic_right",
        "mosaic_bottom",
        "mosaic_top",
        "mosaic_force_fill_axes",
        "mosaic_debug_layout",
        "mosaic_reduce_tick_overlap",
        "mosaic_max_ticks_per_axis",
        "mosaic_hide_boundary_ticklabels",
        "mosaic_x_tick_strategy",
        "mosaic_y_tick_strategy",
        "mosaic_outer_axislabel_once",
        "mosaic_global_outer_axes",
        "draw_original",
        "mosaic_difference_inline",
        "draw_difference",
        "difference_method",
        "difference_output_mode",
        "difference_base_index",
        "difference_norm_mode",
        "difference_percentile",
        "difference_vmin",
        "difference_vmax",
        "difference_vmin_by_wave",
        "difference_vmax_by_wave",
        "difference_vlim_by_wave",
        "difference_cmap",
        "difference_cmap_mode",
        "warn_band_difference_cmap",
        "difference_save_reference",
        "difference_show_colorbar",
        "difference_derotate",
        "show_grid",
        "show_limb",
        "user_vmin",
        "user_vmax",
        "user_cmap",
    ):
        _set_if_provided(provided, name, getattr(args, name))

    for name, value in provided.items():
        setattr(cfg, name, value)

    cfg.difference_vmin_by_wave = _normalize_wave_float_dict(
        cfg.difference_vmin_by_wave, "difference_vmin_by_wave"
    )
    cfg.difference_vmax_by_wave = _normalize_wave_float_dict(
        cfg.difference_vmax_by_wave, "difference_vmax_by_wave"
    )
    cfg.difference_vlim_by_wave = _normalize_wave_float_dict(
        cfg.difference_vlim_by_wave, "difference_vlim_by_wave"
    )

    if args.roi is not None:
        cfg.roi_bounds = tuple(args.roi)
    if args.waves is not None:
        cfg.multi_band_wavelengths = tuple(args.waves)
    else:
        cfg.multi_band_wavelengths = None
    if args.difference_wavelengths is not None:
        cfg.difference_wavelengths = tuple(args.difference_wavelengths)
    elif args.waves is not None:
        cfg.difference_wavelengths = tuple(args.waves)
    if args.use_test_mode:
        cfg.use_test_mode = True
    if args.test_file is not None:
        cfg.test_file = str(Path(args.test_file))

    if args.data_path is not None:
        cfg.data_path = str(Path(args.data_path))
    elif any(getattr(args, name) is not None for name in ("root_dir", "year", "date")):
        cfg.data_path = str(Path(cfg.root_dir) / cfg.year / cfg.date / "SDO" / "AIA")
    cfg.output_dir = (
        str(Path(args.output_dir)) if args.output_dir is not None else cfg.data_path
    )

    if cfg.mode == "test":
        cfg.use_test_mode = True
    if cfg.mode == "mosaic":
        cfg.multi_band_composite = True
    elif cfg.mode in ("single", "test"):
        cfg.multi_band_composite = False
    if args.show_colorbar is not None:
        cfg.show_colorbar = args.show_colorbar
    if args.multi_band_also_save_single is not None:
        cfg.multi_band_also_save_single = args.multi_band_also_save_single

    if args.difference_norm_mode is None and (
        args.difference_vmin is not None or args.difference_vmax is not None
    ):
        cfg.difference_norm_mode = "fixed"

    if cfg.difference_method not in ("base", "running"):
        raise ValueError(f"Invalid difference_method: {cfg.difference_method}")
    if cfg.difference_output_mode not in ("auto", "mosaic", "single", "both"):
        raise ValueError(
            f"Invalid difference_output_mode: {cfg.difference_output_mode}"
        )
    if cfg.difference_norm_mode not in ("auto", "fixed", "config"):
        raise ValueError(f"Invalid difference_norm_mode: {cfg.difference_norm_mode}")
    if cfg.difference_cmap_mode not in ("band", "diverging", "custom"):
        raise ValueError(f"Invalid difference_cmap_mode: {cfg.difference_cmap_mode}")
    if cfg.difference_wavelengths is not None:
        cfg.difference_wavelengths = tuple(int(w) for w in cfg.difference_wavelengths)
    if not cfg.draw_original and not cfg.draw_difference:
        raise ValueError(
            "Nothing to draw: at least one of draw_original or "
            "draw_difference must be True."
        )
    if cfg.mosaic_difference_inline and not cfg.draw_difference:
        warnings.warn(
            "mosaic_difference_inline=True but draw_difference=False; no "
            "difference panels will be added.",
            RuntimeWarning,
            stacklevel=2,
        )
    if cfg.draw_difference and cfg.difference_percentile <= 0:
        raise ValueError("difference_percentile must be positive.")

    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cli_mode = "test" if args.use_test_mode or args.mode == "test" else args.mode
    if cli_mode is not None:
        _configure_matplotlib_backend(cli_mode)

    cfg = config_from_args(args)
    actual_mode = _actual_mode(cfg)
    if cli_mode is None:
        _configure_matplotlib_backend(actual_mode)

    print("--- Starting SDO/AIA EUV FITS processing ---")
    print(f"Data path: {cfg.data_path}")
    print(f"Mode: {actual_mode}")
    print(f"Wavelengths: {cfg.multi_band_wavelengths}")
    process_aia_fits(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
