"""Reusable STEREO/EUVI manifest helpers with explicit filesystem inputs."""

from __future__ import annotations

import csv
import os
from pathlib import Path

from astropy.io import fits

__all__ = ["build_manifest", "make_relative_symlink", "read_metadata"]


def read_metadata(path: Path) -> dict[str, str]:
    """Read the manifest fields for one EUVI FITS product."""

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
    """Create a relative symlink without replacing ordinary files."""

    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")
    try:
        relative_target = Path(os.path.relpath(target, start=link.parent.resolve()))
    except ValueError:
        # Windows cannot express a relative path between different drives.
        relative_target = target
    link.symlink_to(relative_target)


def build_manifest(
    fits_files: list[Path],
    *,
    manifest_path: str | Path,
    link_root: str | Path | None = None,
) -> list[dict[str, str]]:
    """Write a wavelength manifest and optional symlink view.

    Discovery and event paths remain the caller's responsibility. The returned
    rows are sorted deterministically by wavelength, observation time, and name.
    """

    rows = [read_metadata(Path(path)) for path in fits_files]
    rows.sort(
        key=lambda row: (int(row["wavelength"]), row["date_obs"], row["filename"])
    )
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "date_obs",
        "wavelength",
        "detector",
        "observatory",
        "exptime",
        "path",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if link_root is not None:
        root = Path(link_root)
        for row in rows:
            wave_dir = root / row["wavelength"]
            wave_dir.mkdir(parents=True, exist_ok=True)
            make_relative_symlink(Path(row["path"]), wave_dir / row["filename"])
    return rows
