#!/usr/bin/env python3
"""Create a wavelength manifest and symlink view for STEREO-A EUVI FITS."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from astropy.io import fits

DATA_DIR = None
LINK_ROOT = None
MANIFEST = None

__all__ = ["build_parser", "main", "make_relative_symlink", "read_metadata"]


def build_parser() -> argparse.ArgumentParser:
    """Build the event-recipe parser without touching local data."""
    parser = argparse.ArgumentParser(
        prog="solar-apps workflow data stereo-manifest",
        description="Create the STEREO/EUVI wavelength manifest and symlink view.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser


def read_metadata(path: Path) -> dict[str, str]:
    header = fits.getheader(path)
    wavelength = header.get("WAVELNTH")
    if wavelength is None:
        raise ValueError(f"No WAVELNTH keyword in {path}")
    return {
        "filename": path.name,
        "path": str(path),
        "date_obs": str(header.get("DATE-OBS", "")),
        "wavelength": str(int(round(float(wavelength)))),
        "detector": str(header.get("DETECTOR", "")),
        "observatory": str(header.get("OBSRVTRY", "")),
        "exptime": str(header.get("EXPTIME", "")),
    }


def make_relative_symlink(target: Path, link: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")
    relative_target = Path("..") / ".." / target.name
    link.symlink_to(relative_target)


def main(argv: list[str] | None = None) -> int:
    global DATA_DIR, LINK_ROOT, MANIFEST
    args = build_parser().parse_args(argv)
    DATA_DIR = args.input_dir.expanduser()
    LINK_ROOT = DATA_DIR / "by_wavelength"
    MANIFEST = DATA_DIR / "manifest_by_wavelength.csv"
    fits_files = sorted(DATA_DIR.glob("*.fts"))
    if not fits_files:
        raise FileNotFoundError(f"No .fts files found in {DATA_DIR}")

    rows = [read_metadata(path) for path in fits_files]
    rows.sort(
        key=lambda row: (int(row["wavelength"]), row["date_obs"], row["filename"])
    )

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "filename",
                "date_obs",
                "wavelength",
                "detector",
                "observatory",
                "exptime",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        wave_dir = LINK_ROOT / row["wavelength"]
        wave_dir.mkdir(parents=True, exist_ok=True)
        target = DATA_DIR / row["filename"]
        link = wave_dir / row["filename"]
        make_relative_symlink(target, link)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["wavelength"]] = counts.get(row["wavelength"], 0) + 1

    print(f"manifest={MANIFEST}")
    print(f"link_root={LINK_ROOT}")
    print(f"total={len(rows)}")
    for wavelength in sorted(counts, key=int):
        print(f"{wavelength}: {counts[wavelength]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
