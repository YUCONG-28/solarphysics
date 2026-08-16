"""Command-line contract for radio source-map generation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from solar_toolkit.radio.config import load_radio_user_config
from .configs import DEFAULT_CONFIG_NAME
from .entrypoint_utils import apply_output_overrides, build_common_parser
from solar_toolkit.radio.provenance import (
    resolve_provenance_output_dir,
    write_radio_provenance,
)

__all__ = ["build_parser", "main"]


def build_parser():
    """Build the source-map command parser without importing plotting code."""

    return build_common_parser(
        "Run radio source maps with Gaussian overlay.",
        prog="solar-apps workflow radio source-map",
        default_config=DEFAULT_CONFIG_NAME,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[dict], Any] | None = None,
) -> int:
    """Run source-map generation or an explicit compatibility hook."""

    args, unknown = build_parser().parse_known_args(argv)

    user_config, newkirk_config = load_radio_user_config(args.config)
    resolved_config = apply_output_overrides(user_config, args)
    if runner is None:
        from .source_map_workflow import run_source_map

        result = run_source_map(resolved_config, argv=unknown)
    else:
        result = runner(resolved_config)
    output_dir = resolve_provenance_output_dir(resolved_config)
    if (not isinstance(result, int) or result == 0) and output_dir is not None:
        write_radio_provenance(
            output_dir,
            resolved_config,
            newkirk_config=newkirk_config,
            config_source=args.config,
            cli_overrides=vars(args),
        )
        if os.environ.get("APP_V1_RUN_ID"):
            _emit_app_v1_products(output_dir)
    return result if isinstance(result, int) else 0


def _emit_app_v1_products(output_dir: Path) -> None:
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    center_candidates = sorted(
        (
            path
            for path in files
            if path.suffix.casefold() == ".csv"
            and "gaussian" in path.name.casefold()
            and "diagnostic" in path.name.casefold()
        ),
        key=lambda path: (
            path.name.casefold() != "radio_gaussian_fit_diagnostics.csv",
            str(path),
        ),
    )
    if center_candidates:
        _app_v1_event(center_candidates[0], "centers")
    for path in files:
        if path.suffix.casefold() in {".png", ".jpg", ".jpeg"}:
            _app_v1_event(path, "diagnostics")


def _app_v1_event(path: Path, source_port: str) -> None:
    print(
        "APP_V1_EVENT "
        + json.dumps(
            {
                "schema_version": 1,
                "run_id": os.environ["APP_V1_RUN_ID"],
                "module_id": os.environ.get("APP_V1_MODULE_ID", "source-map"),
                "kind": "artifact",
                "payload": {"path": str(path), "source_port": source_port},
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
