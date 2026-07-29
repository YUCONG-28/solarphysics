"""Local observation-file discovery.

English: Scan explicit directories with optional suffix, recursion, and size
filters.

中文：在明确指定的目录内按后缀、递归范围和文件大小扫描观测文件。
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from solar_toolkit.time import extract_time_from_filename

from .sorting import natural_key

PathT = TypeVar("PathT", str, Path)


def scan_files(
    folder: str | Path,
    *,
    suffixes: Sequence[str] | None = None,
    recursive: bool = False,
    min_size_kb: float | None = None,
) -> list[Path]:
    """Scan a folder for files, optionally filtering suffix and size."""

    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    normalized_suffixes = None
    if suffixes is not None:
        normalized_suffixes = {suffix.casefold() for suffix in suffixes}
    min_bytes = None if min_size_kb is None else int(float(min_size_kb) * 1024)
    iterator = root.rglob("*") if recursive else root.iterdir()
    files = []
    for path in iterator:
        if not path.is_file():
            continue
        if (
            normalized_suffixes is not None
            and path.suffix.casefold() not in normalized_suffixes
        ):
            continue
        if min_bytes is not None and path.stat().st_size < min_bytes:
            continue
        files.append(path)
    return sorted(files, key=natural_key)


def scan_fits(
    folder: str | Path,
    *,
    recursive: bool = False,
    min_size_kb: float | None = None,
) -> list[Path]:
    """Scan for FITS-like files using common suffixes."""

    return scan_files(
        folder,
        suffixes=(".fits", ".fit", ".fts"),
        recursive=recursive,
        min_size_kb=min_size_kb,
    )


def get_sorted_fits_files(
    input_dir: str | Path,
    min_size_kb: int = 1,
    *,
    recursive: bool = False,
) -> list[tuple[Path, dt.datetime]]:
    """Return valid ``.fits`` files paired with parsed times, sorted by time."""

    files: list[tuple[Path, dt.datetime]] = []
    input_path = Path(input_dir)
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    for path in iterator:
        if path.suffix.lower() != ".fits":
            continue
        try:
            if path.stat().st_size < min_size_kb * 1024:
                warnings.warn(
                    f"Skipping empty or small file: {path.name}", stacklevel=2
                )
                continue
            file_time = extract_time_from_filename(path.name)
            files.append((path, file_time))
        except ValueError as exc:
            warnings.warn(
                f"Skipping file with an invalid timestamp ({exc}): {path.name}",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not inspect file {path.name}: {exc}",
                stacklevel=2,
            )
    return sorted(files, key=lambda item: item[1])


def find_closest_file_by_time(
    target_time: dt.datetime,
    file_list: Sequence[tuple[PathT, dt.datetime]],
    max_diff_seconds: float = 3600,
) -> tuple[PathT, dt.datetime] | None:
    """Return the file nearest ``target_time`` within the maximum difference."""

    if not file_list:
        return None
    closest_file = min(
        file_list,
        key=lambda item: abs((item[1] - target_time).total_seconds()),
    )
    time_diff = abs((closest_file[1] - target_time).total_seconds())
    return None if time_diff > max_diff_seconds else closest_file


def filter_files_by_time_range(
    file_list: Sequence[tuple[PathT, dt.datetime]],
    start_time: dt.datetime,
    end_time: dt.datetime,
) -> list[tuple[PathT, dt.datetime]]:
    """Return file/time pairs inside an inclusive time range."""

    return [
        (path, file_time)
        for path, file_time in file_list
        if start_time <= file_time <= end_time
    ]


__all__ = [
    "filter_files_by_time_range",
    "find_closest_file_by_time",
    "get_sorted_fits_files",
    "scan_files",
    "scan_fits",
]
