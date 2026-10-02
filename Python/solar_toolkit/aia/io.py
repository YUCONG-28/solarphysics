"""Path, file ordering, and selection helpers for AIA workflows.

This module is intentionally lightweight: it resolves FITS paths and time
ordering before the SunPy/Astropy image stack is imported by the runtime
processor.

中文：本模块只处理 AIA FITS 路径、文件排序与选择，因此可在加载较重的
SunPy/Astropy 图像栈之前安全导入。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import AIAConfig

__all__ = [
    "build_multi_band_slots",
    "discover_wavelength_dirs",
    "parse_timestr",
    "resolve_files",
    "resolve_single_files",
    "resolve_test_file",
    "slice_band_files",
    "sorted_fits_for_band",
]


@lru_cache(maxsize=8192)
def _parse_timestr_from_name(name: str) -> str:
    """Extract a stable time string such as 2000-01-01T000000Z."""
    match = re.search(r"\d{4}-\d{2}-\d{2}T\d{6}Z", name)
    if match:
        return match.group(0)

    parts = name.split(".")
    for part in parts:
        if "T" in part and "Z" in part:
            return part
    return Path(name).stem


def parse_timestr(file_path: Path) -> str:
    """Extract a stable time string such as 2000-01-01T000000Z."""
    return _parse_timestr_from_name(Path(file_path).name)


def resolve_files(input_path: Path, start_idx: int, end_idx: int | None) -> list:
    """Return a deterministic FITS slice from either one file or a directory."""
    if input_path.is_file():
        file_list = [input_path]
    elif input_path.is_dir():
        file_list = sorted(input_path.rglob("*.fits"), key=lambda p: parse_timestr(p))
    else:
        raise ValueError(f"Invalid path: {input_path}")

    total = len(file_list)
    if total == 0:
        raise ValueError(f"No FITS files found under: {input_path}")

    end = total if end_idx is None else min(end_idx, total)
    selected = file_list[start_idx:end]
    print(
        f"Found {total} files total, selected {len(selected)} for processing "
        f"(indices: {start_idx} ~ {end - 1})"
    )
    return selected


def discover_wavelength_dirs(data_path: Path) -> tuple[int, ...]:
    """Discover numeric AIA wavelength subdirectories such as 94, 171, or 304."""
    if not data_path.is_dir():
        raise ValueError(f"AIA data directory does not exist: {data_path}")
    found = [
        int(p.name) for p in data_path.iterdir() if p.is_dir() and p.name.isdigit()
    ]
    if not found:
        raise ValueError(
            f"No numeric wavelength subdirectories found under {data_path}."
        )
    return tuple(sorted(found))


def sorted_fits_for_band(
    data_path: Path, wave: int, use_band_subdirs: bool
) -> list[Path]:
    """Resolve and time-sort all FITS files for one wavelength band."""
    band_dir = (data_path / str(wave)) if use_band_subdirs else data_path
    if not band_dir.is_dir():
        raise ValueError(f"Missing AIA band directory for {wave} Å: {band_dir}")
    files = sorted(band_dir.rglob("*.fits"), key=lambda p: parse_timestr(p))
    if not files:
        raise ValueError(f"No FITS files found in band directory: {band_dir}")
    return files


def slice_band_files(
    files: list[Path], start_idx: int, end_idx: int | None
) -> list[Path]:
    """Apply the user-selected index window while preserving time order."""
    total = len(files)
    end = total if end_idx is None else min(end_idx, total)
    return files[start_idx:end]


def resolve_single_files(cfg: AIAConfig) -> list[Path]:
    """Resolve files for single-image processing, including multi-wave batches."""
    data_path = Path(cfg.data_path)
    waves = cfg.multi_band_wavelengths

    if data_path.is_file():
        return resolve_files(data_path, cfg.start_idx, cfg.end_idx)
    if not cfg.use_band_subdirs or waves is None:
        return resolve_files(data_path, cfg.start_idx, cfg.end_idx)

    selected_files: list[Path] = []
    for wave in waves:
        files = sorted_fits_for_band(data_path, wave, cfg.use_band_subdirs)
        sliced = slice_band_files(files, cfg.start_idx, cfg.end_idx)
        if not sliced:
            raise ValueError(
                f"Band {wave} has no FITS files in index range "
                f"[{cfg.start_idx}, {cfg.end_idx})"
            )
        selected_files.extend(sliced)
        print(f"Band {wave}: selected {len(sliced)} / {len(files)} files")
    return selected_files


def resolve_test_file(cfg: AIAConfig) -> Path:
    """Resolve the single FITS file used by AIA test-mode previews."""
    if cfg.test_file:
        test_path = Path(cfg.test_file)
        if not test_path.is_file():
            raise ValueError(f"Test file does not exist: {test_path}")
        print("Test mode selected file:")
        print("Test wave: from FITS header")
        print("Test index: direct file")
        print(f"File path: {test_path}")
        return test_path

    band_dir = Path(cfg.data_path) / str(cfg.test_wave)
    if not band_dir.is_dir():
        raise ValueError(f"Test band directory does not exist: {band_dir}")

    files = sorted(band_dir.glob("*.fits"), key=lambda p: parse_timestr(p))
    if not files:
        raise ValueError(f"No FITS files found in test band directory: {band_dir}")

    if cfg.test_index < 0 or cfg.test_index >= len(files):
        raise ValueError(
            f"test_index={cfg.test_index} is out of range. Available index "
            f"range: 0 ~ {len(files) - 1}."
        )

    test_path = files[cfg.test_index]
    print("Test mode selected file:")
    print(f"Test wave: {cfg.test_wave}")
    print(f"Test index: {cfg.test_index}")
    print(f"File path: {test_path}")
    return test_path


def build_multi_band_slots(
    cfg: AIAConfig, wavelengths: tuple[int, ...]
) -> list[tuple[Path, ...]]:
    """Build per-time slots that group one FITS path from each wavelength."""
    data_path = Path(cfg.data_path)
    per_band: list[list[Path]] = []

    for wave in wavelengths:
        all_files = sorted_fits_for_band(data_path, wave, cfg.use_band_subdirs)
        sliced = slice_band_files(all_files, cfg.start_idx, cfg.end_idx)
        if not sliced:
            raise ValueError(
                f"Band {wave} has no FITS files in index range "
                f"[{cfg.start_idx}, {cfg.end_idx})"
            )
        per_band.append(sliced)

    slot_count = min(len(files) for files in per_band)
    if any(len(files) != slot_count for files in per_band):
        print(
            "Note: Available file counts differ across bands; using shortest "
            f"length {slot_count} after time sorting."
        )

    return [tuple(band[i] for band in per_band) for i in range(slot_count)]
