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

from solar_toolkit._utils.numeric import numeric_data
from solar_toolkit.map.coordinates import hpc_world_bounds, hpc_world_to_pixel
from solar_toolkit.radio.centers import FITS_SUFFIXES
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
    """Display samples on separable, regular HPLN/HPLT arcsecond center axes.

    ``read_aia_background`` resamples through the full WCS, without conserving
    flux. Existing callers can still construct this record from a regular grid.
    """

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
    """Find the nearest AIA row within a finite nonnegative scalar gap in seconds.

    Invalid tolerances, including booleans, raise ``ValueError`` before table
    inspection. Zero accepts an exact instant only; missing tables still return
    ``no_files`` when the tolerance is valid.
    """

    try:
        if isinstance(max_dt_seconds, (bool, np.bool_)) or np.ndim(max_dt_seconds):
            raise ValueError("not a numeric scalar")
        tolerance = float(max_dt_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_dt_seconds must be a finite nonnegative scalar") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_dt_seconds must be a finite nonnegative scalar")

    if aia_table is None or aia_table.empty:
        return NearestAIA(status="no_files", reason="AIA table is empty.")
    if "obs_time" not in aia_table.columns or "path" not in aia_table.columns:
        return NearestAIA(
            status="invalid_table", reason="AIA table lacks path/obs_time."
        )
    from solar_toolkit.time import normalize_time, seconds_to_ns

    target = normalize_time(frame_time)

    def observed_time(value):
        try:
            return normalize_time(value)
        except (TypeError, ValueError):
            return None

    times = aia_table["obs_time"].map(observed_time)
    valid = times.notna()
    if not valid.any():
        return NearestAIA(
            status="invalid_table", reason="AIA table has no valid times."
        )
    diffs = times[valid].map(lambda stamp: abs(stamp.tai_ns - target.tai_ns))
    best_index = diffs.idxmin()
    best_delta_ns = int(diffs.loc[best_index])
    best_delta = best_delta_ns / 1_000_000_000
    row = aia_table.loc[best_index]
    stamp = times.loc[best_index]
    stamp.to_datetime()  # The public pandas view cannot represent a leap second.
    observed = pd.Timestamp(stamp.utc)
    if best_delta_ns > seconds_to_ns(tolerance, nonnegative=True):
        return NearestAIA(
            status="too_far",
            path=str(row["path"]),
            obs_time=observed,
            delta_seconds=best_delta,
            reason="Nearest AIA file exceeds max_dt_seconds.",
            hdu_index=int(row["hdu_index"]) if "hdu_index" in row else None,
        )
    return NearestAIA(
        status="matched",
        path=str(row["path"]),
        obs_time=observed,
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
    """Resample an AIA image to a regular HPC display grid using its full WCS.

    Both modes use complete FITS projection/matrix/distortion semantics;
    ``wcs_mode='sunpy'`` additionally checks SunPy readability. Bilinear display
    interpolation is not flux conserving and propagates missing source samples.
    Coverage outside the native pixel-edge footprint remains NaN. The grid's
    1D angular axes are pixel centers, permitting existing Plotly consumers.
    """

    resolved = Path(path).expanduser()
    mode = str(wcs_mode or "header").lower()
    if mode not in {"header", "sunpy"}:
        raise ValueError("wcs_mode must be 'header' or 'sunpy'")
    if mode == "sunpy":
        _validate_sunpy_map_readable(resolved)

    image, header = _read_first_image_hdu(resolved, hdu_index=hdu_index)
    image = numeric_data(image)
    image[~np.isfinite(image)] = np.nan
    ny, nx = image.shape
    step = max(1, int(math.ceil(max(ny, nx) / max(1, int(max_pixels)))))
    out_ny, out_nx = (ny + step - 1) // step, (nx + step - 1) // step
    left, right, bottom, top = hpc_world_bounds(header, image.shape)
    dx, dy = (right - left) / out_nx, (top - bottom) / out_ny
    if dx <= 0 or dy <= 0:
        raise ValueError("AIA WCS must span a nondegenerate angular display plane")
    x_arcsec = left + (np.arange(out_nx) + 0.5) * dx
    y_arcsec = bottom + (np.arange(out_ny) + 0.5) * dy
    grid_x, grid_y = np.meshgrid(x_arcsec, y_arcsec)
    px, py = hpc_world_to_pixel(header, grid_x, grid_y)
    covered = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= -0.5)
        & (px <= nx - 0.5)
        & (py >= -0.5)
        & (py <= ny - 0.5)
    )
    from scipy.ndimage import map_coordinates

    # Extend a boundary pixel only within its own half-pixel footprint; this
    # is not extrapolation outside the image. Missing interpolation neighbors
    # remain NaN rather than being replaced or renormalized away.
    z = map_coordinates(
        image,
        [np.where(covered, py, 0), np.where(covered, px, 0)],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    z[~covered] = np.nan
    z = _scale_background(z, percentile_limits=percentile_limits, log_scale=log_scale)
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
        return np.full_like(z, np.nan, dtype=float)
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
