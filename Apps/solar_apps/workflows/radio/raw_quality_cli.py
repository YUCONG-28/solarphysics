"""Entrypoint for raw radio FITS quality diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from solar_toolkit.radio.config import load_radio_output_config, load_radio_user_config
from solar_toolkit.radio.raw_quality import (
    RawQualityAnalysisResult,
    analyze_radio_raw_quality,
)

from .entrypoint_utils import add_config_arguments, resolve_config_source

__all__ = ["DEFAULT_RAW_QUALITY_CONFIG", "build_parser", "main", "run_raw_quality"]


DEFAULT_RAW_QUALITY_CONFIG = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow radio raw-quality",
        description="Scan radio FITS directories and report raw-data quality issues.",
    )
    add_config_arguments(parser, default_config=DEFAULT_RAW_QUALITY_CONFIG)
    parser.add_argument(
        "--workspace-config-json", help="Explicit structured radio settings as JSON."
    )
    parser.add_argument("--root")
    parser.add_argument("--output-dir")
    parser.add_argument("--start-idx", type=int)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--freqs", help="Comma-separated frequencies in MHz.")
    parser.add_argument("--polarizations", help="Comma-separated polarizations.")
    return parser


def run_raw_quality(
    *,
    config_name: str = DEFAULT_RAW_QUALITY_CONFIG,
    root: str | Path | None = None,
    freqs: list[int | float] | None = None,
    polarizations: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path | None = None,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> RawQualityAnalysisResult:
    """Run raw radio quality diagnostics and return the analysis result."""

    user_config, _newkirk_config = load_radio_user_config(config_name)
    output_config = load_radio_output_config(config_name)
    data_cfg = user_config.get("data", {})

    resolved_root = root or data_cfg.get("multi_band_root")
    if not resolved_root:
        raise ValueError("Radio root is required; set --root or data.multi_band_root.")

    resolved_freqs = freqs or list(data_cfg.get("multi_band_freqs", []))
    if not resolved_freqs:
        raise ValueError("At least one frequency is required; set --freqs.")

    resolved_pols = polarizations or _default_polarizations(data_cfg)
    resolved_output = (
        output_dir
        or output_config.get("output_dir")
        or user_config.get("output", {}).get("output_dir")
        or "outputs"
    )

    return analyze_radio_raw_quality(
        resolved_root,
        freqs=[float(freq) for freq in resolved_freqs],
        polarizations=list(resolved_pols),
        output_dir=resolved_output,
        start_idx=int(
            start_idx if start_idx is not None else data_cfg.get("start_idx", 0)
        ),
        end_idx=(
            int(end_idx)
            if end_idx is not None
            else (
                int(data_cfg["end_idx"])
                if data_cfg.get("end_idx") is not None
                else None
            )
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run raw radio quality diagnostics from command-line arguments."""

    args = build_parser().parse_args(argv)
    result = run_raw_quality(
        config_name=resolve_config_source(args),
        root=args.root,
        freqs=_parse_freqs(args.freqs),
        polarizations=_parse_polarizations(args.polarizations),
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
    _print_summary(result)
    return 0


def _parse_freqs(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_polarizations(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _default_polarizations(data_cfg: dict) -> list[str]:
    polarization = str(data_cfg.get("polarization", "RR") or "RR")
    if polarization == "RR+LL" or data_cfg.get("combine_polarizations", False):
        return ["RR", "LL"]
    return [polarization]


def _print_summary(result) -> None:
    bad_files = sum(1 for row in result.file_rows if row.quality_flag == "bad")
    bad_slots = sum(1 for row in result.slot_rows if row.quality_flag == "bad")
    print("[Radio Raw Quality] outputs:")
    print(f"  File diagnostics: {result.file_csv_path}")
    print(f"  Slot diagnostics: {result.slot_csv_path}")
    print(f"  Bad files: {bad_files}/{len(result.file_rows)}")
    print(f"  Bad slots: {bad_slots}/{len(result.slot_rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
