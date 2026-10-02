"""Match observation filenames by time without external data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from solar_toolkit.time import extract_time_from_filename, nearest_by_time

REQUIRES_LOCAL_DATA = False
SAMPLE_FILES = (
    "aia.lev1_euv_12s.2000-01-01T000000Z.171.image_lev1.fits",
    "aia.lev1_euv_12s.2000-01-01T000012Z.171.image_lev1.fits",
)


def find_nearest_observation(
    target_time: str,
    *,
    max_diff_seconds: float = 12.0,
) -> str | None:
    """Return the sample filename nearest to ``target_time``."""

    observations = [(name, extract_time_from_filename(name)) for name in SAMPLE_FILES]
    match = nearest_by_time(
        target_time,
        observations,
        key=lambda item: item[1],
        max_diff_seconds=max_diff_seconds,
    )
    return None if match is None else match[0]


def build_parser() -> argparse.ArgumentParser:
    """Build the no-data example parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="2000-01-01T00:00:08Z")
    parser.add_argument("--max-diff-seconds", type=float, default=12.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the selected sample filename and return a status code."""

    args = build_parser().parse_args(argv)
    match = find_nearest_observation(
        args.target,
        max_diff_seconds=args.max_diff_seconds,
    )
    if match is None:
        print("No observation matched the requested tolerance.")
        return 1
    print(match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
