# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic no-science worker used to exercise the Phase 1 task shell."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an App 1.0 shell demo task.")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--delay-ms", type=int, default=30)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument(
        "--fail-once-marker",
        help="Fail once and create this private marker, then succeed on retry.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 1 or args.delay_ms < 0:
        raise ValueError("steps must be positive and delay-ms cannot be negative")
    print("LOG Demo worker started", flush=True)
    for index in range(args.steps):
        if args.delay_ms:
            time.sleep(args.delay_ms / 1000)
        progress = round(((index + 1) / args.steps) * 100)
        print(f"PROGRESS {progress}", flush=True)
    if args.fail:
        print("LOG Demo worker failed as requested", flush=True)
        return 3
    if args.fail_once_marker:
        marker = Path(args.fail_once_marker).expanduser().resolve(strict=False)
        if not marker.is_file():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("retry may proceed\n", encoding="utf-8")
            print("LOG Demo worker recorded a recoverable first failure", flush=True)
            return 4
    print("LOG Demo worker completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
