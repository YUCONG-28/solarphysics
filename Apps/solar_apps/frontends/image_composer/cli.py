# SPDX-License-Identifier: MIT
"""Compatibility adapter for the App 1.0 Image Composer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps frontend image-composer",
        description="Open the Image Composer inside Solar Physics App 1.0.",
    )
    parser.add_argument(
        "--project", metavar="PATH", help="Optional .fic.json project to open."
    )
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Path-separated directories that may be opened or written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(argument in {"-h", "--help"} for argument in args):
        build_parser().print_help()
        return 0
    parsed = build_parser().parse_args(args)
    forwarded = ["--module", "image-composer"]
    if parsed.project:
        forwarded.extend(("--composer-project", parsed.project))
    if parsed.allowed_roots:
        forwarded.extend(("--allowed-roots", parsed.allowed_roots))
    from solar_apps.frontends.app_v1.cli import main as app_v1_main

    return int(app_v1_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
