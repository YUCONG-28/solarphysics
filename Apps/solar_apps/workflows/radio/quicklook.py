"""Compatibility facade for the radio quicklook workflow.

The scientific implementation lives once in solar_toolkit.radio.quicklook.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from solar_toolkit.radio import quicklook as _quicklook_impl

from .configs import DEFAULT_CONFIG_NAME

VALID_CENTERS_NAME = _quicklook_impl.VALID_CENTERS_NAME
HEIGHT_ROWS_NAME = _quicklook_impl.HEIGHT_ROWS_NAME
HEIGHT_PLOT_NAME = _quicklook_impl.HEIGHT_PLOT_NAME
TRAJECTORY_PLOT_NAME = _quicklook_impl.TRAJECTORY_PLOT_NAME
DEFAULT_ANALYSIS_SUBDIR = _quicklook_impl.DEFAULT_ANALYSIS_SUBDIR

build_quicklook_summary = _quicklook_impl.build_quicklook_summary
filter_valid_gaussian_centers = _quicklook_impl.filter_valid_gaussian_centers
plot_gaussian_center_trajectory = _quicklook_impl.plot_gaussian_center_trajectory

__all__ = [
    "build_parser",
    "build_quicklook_config",
    "build_quicklook_summary",
    "filter_valid_gaussian_centers",
    "plot_gaussian_center_trajectory",
    "resolve_gaussian_csv",
    "run_gaussian_newkirk_quicklook",
    "main",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated Gaussian/Newkirk quicklook CLI parser."""

    parser = argparse.ArgumentParser(
        prog="solar-apps workflow radio quicklook",
        description="Generate Gaussian center and Newkirk quicklook products.",
    )
    parser.add_argument("--gaussian-csv")
    parser.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--output-dir", default="quicklook_outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the quicklook workflow from command-line arguments."""

    args = build_parser().parse_args(argv)
    result = run_gaussian_newkirk_quicklook(
        gaussian_csv=args.gaussian_csv,
        config_name=args.config,
        output_dir=args.output_dir,
    )
    print(f"Quicklook input: {result['input_csv']}")
    print(f"Quicklook output: {Path(args.output_dir).resolve()}")
    return 0


def build_quicklook_config(config_name: str = DEFAULT_CONFIG_NAME) -> dict[str, Any]:
    """Merge the config sections needed for isolated quicklook products."""

    return _quicklook_impl.build_quicklook_config(config_name=config_name)


def resolve_gaussian_csv(
    *,
    gaussian_csv: str | Path | None,
    config_name: str = DEFAULT_CONFIG_NAME,
) -> Path:
    """Resolve a diagnostics CSV from an explicit path or event output config."""

    return _quicklook_impl.resolve_gaussian_csv(
        gaussian_csv=gaussian_csv,
        config_name=config_name,
    )


def run_gaussian_newkirk_quicklook(
    *,
    gaussian_csv: str | Path | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate isolated Gaussian/Newkirk quicklook CSV and PNG products."""

    return _quicklook_impl.run_gaussian_newkirk_quicklook(
        gaussian_csv=gaussian_csv,
        config_name=config_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
