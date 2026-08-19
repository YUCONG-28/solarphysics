"""Radio source center extraction helpers.

English: CLI adapter over the reusable toolkit centers module. The science
implementation lives in solar_toolkit.radio.centers; this module only adds
the command-line parser and programmatic entry points for the Apps partition.

中文：本模块是复用 toolkit 射电源中心模块的命令行适配层，科学实现位于
solar_toolkit.radio.centers，此处仅提供 CLI 解析与程序入口。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from solar_toolkit.radio.centers import (
    FITS_SUFFIXES,
    POL_LCP,
    POL_RCP,
    POL_SUM,
    POL_UNKNOWN,
    RadioImage,
    choose_mask_component,
    compute_source_center,
    extract_radio_centers,
    filter_radio_images,
    find_files,
    first_existing_header_value,
    infer_parent_directory_polarization,
    infer_pol_from_stokes_axis,
    infer_polarization,
    iter_images_in_hdu,
    iter_radio_images,
    maybe_make_sum_images,
    normalize_pol_text,
    parse_datetime_value,
    parse_frequency_mhz,
    parse_observation_time,
    parse_time_from_filename,
    pixel_to_hpc_arcsec,
    record_from_radio_image,
    select_radio_files,
    stokes_code_to_pol,
    to_arcsec,
    write_centers_table,
)

__all__ = [
    "FITS_SUFFIXES",
    "POL_LCP",
    "POL_RCP",
    "POL_SUM",
    "POL_UNKNOWN",
    "RadioImage",
    "build_parser",
    "choose_mask_component",
    "compute_source_center",
    "extract_radio_centers",
    "filter_radio_images",
    "find_files",
    "first_existing_header_value",
    "infer_parent_directory_polarization",
    "infer_pol_from_stokes_axis",
    "infer_polarization",
    "iter_images_in_hdu",
    "iter_radio_images",
    "main",
    "maybe_make_sum_images",
    "normalize_pol_text",
    "parse_datetime_value",
    "parse_frequency_mhz",
    "parse_observation_time",
    "parse_time_from_filename",
    "pixel_to_hpc_arcsec",
    "record_from_radio_image",
    "run_center_extraction",
    "select_radio_files",
    "stokes_code_to_pol",
    "to_arcsec",
    "write_centers_table",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for threshold center extraction."""

    parser = argparse.ArgumentParser(
        prog="solar-apps workflow radio centers",
        description="Extract threshold radio-source centers from FITS files.",
    )
    parser.add_argument(
        "--radio-dir", required=True, help="Folder containing radio FITS files."
    )
    parser.add_argument(
        "--out", default="radio_centers.csv", help="Output .csv or .xlsx table."
    )
    parser.add_argument(
        "--pattern", default="*.fits", help="FITS filename glob pattern."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search subfolders recursively."
    )
    parser.add_argument(
        "--freqs",
        help="Comma-separated frequency filter in MHz, e.g. 149,164,190.",
    )
    parser.add_argument(
        "--polarizations",
        help="Comma-separated polarization filter such as LL,RR or LCP,RCP.",
    )
    parser.add_argument(
        "--time-start",
        help="Inclusive observation-time start, e.g. 2025-01-24T04:46:45.",
    )
    parser.add_argument(
        "--time-end",
        help="Inclusive observation-time end, e.g. 2025-01-24T04:50:45.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95, help="Threshold fraction, e.g. 0.95."
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["peak", "bg_peak", "percentile"],
        default="bg_peak",
        help="Threshold definition: peak, bg_peak, or percentile.",
    )
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=5.0,
        help="Background percentile used by bg_peak mode.",
    )
    parser.add_argument(
        "--centroid",
        choices=["weighted", "geometric"],
        default="weighted",
        help="Center calculation inside the threshold mask.",
    )
    parser.add_argument(
        "--component",
        choices=["peak", "largest", "brightest"],
        default="peak",
        help="Connected mask component to keep.",
    )
    parser.add_argument(
        "--use-abs", action="store_true", help="Use absolute image values."
    )
    parser.add_argument(
        "--min-pixels", type=int, default=1, help="Minimum mask pixels."
    )
    parser.add_argument(
        "--default-pol",
        choices=[POL_LCP, POL_RCP, POL_SUM, POL_UNKNOWN],
        default=POL_SUM,
        help="Fallback polarization when metadata is missing.",
    )
    parser.add_argument(
        "--make-sum",
        action="store_true",
        help="Pair LCP/RCP files and additionally extract L+R centers.",
    )
    parser.add_argument(
        "--pair-time-tolerance-sec",
        type=float,
        default=0.5,
        help="Maximum LCP/RCP pairing time difference in seconds.",
    )
    return parser


def run_center_extraction(argv: list[str] | None = None) -> pd.DataFrame:
    """Run threshold center extraction and return the generated table.

    This function preserves the former programmatic return contract of
    :func:`main`.  Command-line callers should use :func:`main`, which returns
    an integer process status code.
    """

    args = build_parser().parse_args(argv)
    freqs = _parse_float_csv(args.freqs)
    polarizations = _parse_str_csv(args.polarizations)
    df = extract_radio_centers(
        args.radio_dir,
        out=args.out,
        pattern=args.pattern,
        recursive=args.recursive,
        freqs=freqs,
        polarizations=polarizations,
        time_start=args.time_start,
        time_end=args.time_end,
        threshold_frac=args.threshold,
        threshold_mode=args.threshold_mode,
        background_percentile=args.background_percentile,
        centroid=args.centroid,
        component=args.component,
        use_abs=args.use_abs,
        min_pixels=args.min_pixels,
        default_pol=args.default_pol,
        make_sum=args.make_sum,
        pair_time_tolerance_sec=args.pair_time_tolerance_sec,
    )
    candidate_count = len(
        select_radio_files(
            args.radio_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            freqs=freqs,
            polarizations=polarizations,
            time_start=args.time_start,
            time_end=args.time_end,
        )
    )
    print(f"Candidate FITS files: {candidate_count}")
    print(f"Extracted centers: {len(df)}")
    print(f"Output table: {Path(args.out).expanduser().resolve()}")
    return df


def main(argv: list[str] | None = None) -> int:
    """Run the threshold center-extraction command."""

    run_center_extraction(argv)
    return 0


def _parse_str_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_csv(raw: str | None) -> list[float]:
    return [float(item) for item in _parse_str_csv(raw)]


if __name__ == "__main__":
    raise SystemExit(main())
