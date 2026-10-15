"""AIA FITS background helpers for radio-source trajectory views.

English: Scan AIA FITS folders, choose the closest image to a radio frame, and
read a lightweight helioprojective background grid for Plotly overlays.

中文：扫描 AIA FITS 文件夹，为射电帧匹配最近的 AIA 图像，并读取轻量级
日面角秒背景网格用于轨迹前端叠加。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from solar_toolkit.radio.centers import (
    FITS_SUFFIXES,
    pixel_to_hpc_arcsec,
)
from solar_toolkit.radio.time_matching import observation_stamp
from solar_toolkit.time import extract_timestamp_from_filename

__all__ = [
    "AiaBackground",
    "NearestAIA",
    "find_nearest_aia",
    "load_nearest_background",
    "read_aia_background",
    "scan_aia_folder",
]


@dataclass
class AiaBackground:
    """Downsampled AIA image plus 1D HPLN/HPLT axes for Plotly heatmaps."""

    path: str
    z: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    label: str
    obs_time: pd.Timestamp | None
    wavelength: str


@dataclass
class NearestAIA:
    """Result of matching a radio frame time to an AIA table."""

    status: str
    path: str | None = None
    obs_time: pd.Timestamp | None = None
    delta_seconds: float | None = None
    reason: str = ""
    hdu_index: int | None = None


def scan_aia_folder(
    aia_dir: str | Path,
    *,
    pattern: str = "*.fits",
    recursive: bool = False,
) -> pd.DataFrame:
    """Scan an AIA FITS folder and return ``path``, ``obs_time``, ``wavelength``."""

    folder = Path(aia_dir).expanduser()
    columns = [
        "path",
        "obs_time",
        "wavelength",
        "hdu_index",
        "utc",
        "tai_ns",
        "raw_time",
        "time_source",
        "quality",
        "exposure_seconds",
        "size_bytes",
        "mtime_ns",
    ]
    if not str(aia_dir).strip() or not folder.exists():
        return pd.DataFrame(columns=columns)

    files = folder.rglob(pattern) if recursive else folder.glob(pattern)
    rows: list[dict[str, object]] = []
    for path in sorted(files):
        if not path.is_file() or path.suffix.lower() not in FITS_SUFFIXES:
            continue
        try:
            with fits.open(path, memmap=True, ignore_missing_end=True) as hdul:
                for hdu_index, hdu in enumerate(hdul):
                    if (
                        not isinstance(
                            hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
                        )
                        or int(hdu.header.get("NAXIS", 0)) != 2
                        or int(hdu.header.get("NAXIS1", 0)) <= 0
                        or int(hdu.header.get("NAXIS2", 0)) <= 0
                    ):
                        continue
                    header = hdul[0].header.copy()
                    header.update(hdu.header)
                    stamp = _header_timestamp(header)
                    if stamp is None:
                        try:
                            stamp = extract_timestamp_from_filename(str(path))
                        except ValueError:
                            continue
                    rows.append(
                        {
                            "path": str(path),
                            "obs_time": pd.Timestamp(stamp.utc).tz_localize(None),
                            "wavelength": str(header.get("WAVELNTH", "")),
                            "hdu_index": hdu_index,
                            "utc": stamp.utc,
                            "tai_ns": stamp.tai_ns,
                            "raw_time": stamp.raw,
                            "time_source": stamp.source,
                            "quality": header.get("QUALITY"),
                            "exposure_seconds": header.get("EXPTIME"),
                            "size_bytes": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns,
                        }
                    )
        except Exception:
            continue
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["obs_time", "path", "hdu_index"], kind="stable")
        .reset_index(drop=True)
    )


def _header_time(header: fits.Header) -> pd.Timestamp | None:
    stamp = _header_timestamp(header)
    return pd.Timestamp(stamp.utc).tz_localize(None) if stamp is not None else None


def _header_timestamp(header: fits.Header):
    # AIA T_OBS is the observation epoch; DATE-OBS can be exposure start.
    try:
        return observation_stamp(header)
    except (TypeError, ValueError):
        return None


def find_nearest_aia(
    aia_table: pd.DataFrame,
    frame_time,
    *,
    max_dt_seconds: float = 3600.0,
) -> NearestAIA:
    """Find the AIA row closest to ``frame_time`` within a maximum time gap."""

    if aia_table is None or aia_table.empty:
        return NearestAIA(status="no_files", reason="AIA table is empty.")
    if "obs_time" not in aia_table.columns or "path" not in aia_table.columns:
        return NearestAIA(
            status="invalid_table", reason="AIA table lacks path/obs_time."
        )
    target = pd.Timestamp(frame_time)
    if target.tzinfo is not None:
        target = target.tz_convert("UTC").tz_localize(None)
    times = pd.to_datetime(
        aia_table["obs_time"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    valid = times.notna()
    if not valid.any():
        return NearestAIA(
            status="invalid_table", reason="AIA table has no valid times."
        )
    diffs = (times[valid] - target).abs().dt.total_seconds()
    best_index = diffs.idxmin()
    best_delta = float(diffs.loc[best_index])
    row = aia_table.loc[best_index]
    if best_delta > float(max_dt_seconds):
        return NearestAIA(
            status="too_far",
            path=str(row["path"]),
            obs_time=pd.Timestamp(row["obs_time"]),
            delta_seconds=best_delta,
            reason="Nearest AIA file exceeds max_dt_seconds.",
            hdu_index=int(row["hdu_index"]) if "hdu_index" in row else None,
        )
    return NearestAIA(
        status="matched",
        path=str(row["path"]),
        obs_time=pd.Timestamp(row["obs_time"]),
        delta_seconds=best_delta,
        hdu_index=int(row["hdu_index"]) if "hdu_index" in row else None,
    )


def read_aia_background(
    path: str | Path,
    *,
    max_pixels: int = 1024,
    percentile_limits: tuple[float, float] = (1.0, 99.7),
    log_scale: bool = True,
    wcs_mode: str = "header",
    hdu_index: int | None = None,
) -> AiaBackground:
    """Read a downsampled AIA image for Plotly background display.

    ``wcs_mode="header"`` uses a fast linear FITS-header grid. ``"sunpy"``
    validates that SunPy can read the file before using the same regular grid,
    because Plotly heatmaps need 1D axes for efficient playback.
    """

    resolved = Path(path).expanduser()
    mode = str(wcs_mode or "header").lower()
    if mode not in {"header", "sunpy"}:
        raise ValueError("wcs_mode must be 'header' or 'sunpy'")
    if mode == "sunpy":
        _validate_sunpy_map_readable(resolved)

    image, header = _read_first_image_hdu(resolved, hdu_index=hdu_index)
    image = np.where(np.isfinite(image), image, np.nan)
    ny, nx = image.shape
    step = max(1, int(math.ceil(max(ny, nx) / max(1, int(max_pixels)))))
    z = image[::step, ::step].astype(float)
    z = _scale_background(z, percentile_limits=percentile_limits, log_scale=log_scale)

    y_pix = np.arange(0, ny, step)
    x_pix = np.arange(0, nx, step)
    x_arcsec = np.array(
        [pixel_to_hpc_arcsec(header, float(x), 0.0)[0] for x in x_pix],
        dtype=float,
    )
    y_arcsec = np.array(
        [pixel_to_hpc_arcsec(header, 0.0, float(y))[1] for y in y_pix],
        dtype=float,
    )
    obs_time = _header_time(header)
    if obs_time is None:
        try:
            obs_time = pd.Timestamp(extract_timestamp_from_filename(str(resolved)).utc)
        except ValueError:
            pass
    wavelength = str(header.get("WAVELNTH", ""))
    label = resolved.name
    if obs_time is not None:
        label += f" | {pd.Timestamp(obs_time).isoformat()}"
    if wavelength:
        label += f" | {wavelength} A"
    return AiaBackground(
        path=str(resolved),
        z=z,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
        label=label,
        obs_time=pd.Timestamp(obs_time) if obs_time is not None else None,
        wavelength=wavelength,
    )


def _validate_sunpy_map_readable(path: Path) -> None:
    try:
        import sunpy.map
    except Exception as exc:  # pragma: no cover - depends on optional runtime pieces.
        raise ImportError("SunPy WCS mode requires sunpy.map.") from exc
    sunpy.map.Map(path)


def _read_first_image_hdu(
    path: Path, *, hdu_index: int | None = None
) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        candidates = [hdul[hdu_index]] if hdu_index is not None else hdul
        for hdu in candidates:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data)
            if data.dtype.fields is not None:
                continue
            data = np.squeeze(data)
            while data.ndim > 2:
                data = data[0]
            if data.ndim == 2:
                header = hdul[0].header.copy()
                header.update(hdu.header)
                return data.astype(float), header
    raise ValueError(f"No 2D image found in AIA FITS file: {path}")


def _scale_background(
    z: np.ndarray,
    *,
    percentile_limits: tuple[float, float],
    log_scale: bool,
) -> np.ndarray:
    finite = np.isfinite(z)
    if not np.any(finite):
        return np.zeros_like(z, dtype=float)
    low, high = percentile_limits
    lo = float(np.nanpercentile(z[finite], low))
    hi = float(np.nanpercentile(z[finite], high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(z[finite]))
        hi = float(np.nanmax(z[finite]))
    scaled = np.clip(z, lo, hi)
    if log_scale:
        scaled = np.log10(scaled - lo + 1.0)
    return scaled


def load_nearest_background(
    aia_table: pd.DataFrame,
    frame_time,
    *,
    max_dt_seconds: float = 3600.0,
    max_pixels: int = 1024,
    percentile_limits: tuple[float, float] = (1.0, 99.7),
    log_scale: bool = True,
    wcs_mode: str = "header",
) -> tuple[AiaBackground | None, NearestAIA]:
    """Find and read the closest AIA background, returning status either way."""

    nearest = find_nearest_aia(
        aia_table,
        frame_time,
        max_dt_seconds=max_dt_seconds,
    )
    if nearest.status != "matched" or nearest.path is None:
        return None, nearest
    return (
        read_aia_background(
            nearest.path,
            max_pixels=max_pixels,
            percentile_limits=percentile_limits,
            log_scale=log_scale,
            wcs_mode=wcs_mode,
        ),
        nearest,
    )
