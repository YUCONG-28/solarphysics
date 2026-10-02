"""Command-line contract for the full radio burst pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .configs import DEFAULT_CONFIG_NAME
from .entrypoint_utils import build_common_parser, resolve_config_source

__all__ = ["build_parser", "main"]


def build_parser():
    """Build the full-pipeline parser without importing the scientific stack."""

    return build_common_parser(
        "Run the full radio burst Gaussian, drift-rate, and Newkirk-height pipeline.",
        prog="solar-apps workflow radio pipeline",
        default_config=DEFAULT_CONFIG_NAME,
        include_pipeline_outputs=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[Sequence[str] | None], Any] | None = None,
) -> int:
    """Run the package pipeline or an explicitly supplied compatibility hook."""

    forwarded = None if argv is None else list(argv)
    args, _unknown = build_parser().parse_known_args(forwarded)
    resolve_config_source(args)
    if runner is None:
        from .pipeline_workflow import run_pipeline

        runner = run_pipeline

    result = runner(forwarded)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
