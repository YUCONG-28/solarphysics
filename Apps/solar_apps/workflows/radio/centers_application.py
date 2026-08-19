"""Radio source center extraction helpers.

English: Read radio FITS images, infer observing metadata, and extract
threshold/contour source centers into CSV/XLSX-compatible tables.

中文：读取射电 FITS 图像，识别时间、频率和极化信息，并将阈值/等值线
射电源中心整理为可复用表格。
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

try:  # pragma: no cover - SciPy is expected in the project env, but optional here.
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

__all__ = [
    "FITS_SUFFIXES",
    "POL_LCP",
    "POL_RCP",
    "POL_SUM",
    "POL_UNKNOWN",
    "RadioImage",
    "build_parser",
    "choose_mask_component",
    "compute_source_center",
    "extract_radio_centers",
    "filter_radio_images",
    "find_files",
    "first_existing_header_value",
    "infer_parent_directory_polarization",
    "infer_pol_from_stokes_axis",
    "infer_polarization",
    "iter_images_in_hdu",
    "iter_radio_images",
    "main",
    "maybe_make_sum_images",
    "normalize_pol_text",
    "parse_datetime_value",
    "parse_frequency_mhz",
    "parse_observation_time",
    "parse_time_from_filename",
    "pixel_to_hpc_arcsec",
    "record_from_radio_image",
    "run_center_extraction",
    "select_radio_files",
    "stokes_code_to_pol",
    "to_arcsec",
    "write_centers_table",
]

POL_LCP = "LCP"
POL_RCP = "RCP"
POL_SUM = "L+R"
POL_UNKNOWN = "UNKNOWN"

FITS_SUFFIXES = {".fits", ".fit", ".fts"}


@dataclass
class RadioImage:
    """A single 2D radio image plane plus extracted metadata.

    中文：单个二维射电图像平面及其从 FITS/文件名推断出的元数据。
    """

    path: Path
    hdu_index: int
    image: np.ndarray
    header: fits.Header
    pol: str
    freq_mhz: float
    obs_time: datetime | None
    source_label: str = "main"


def first_existing_header_value(header: fits.Header, keys: Iterable[str]) -> str | None:
    """Return the first non-empty FITS header value among ``keys``."""

    for key in keys:
        if key in header and header.get(key) not in (None, ""):
            return str(header.get(key)).strip()
    return None


def normalize_pol_text(text: str | object) -> str:
    """Infer a normalized polarization label from free text.

    中文：从 FITS 头或文件名文本中推断标准极化标签。
    """

    if text is None:
        return POL_UNKNOWN
    raw = str(text).strip()
    if not raw:
        return POL_UNKNOWN
    low = raw.lower()
    compact = re.sub(r"[\s_\-]+", "", low)

    if any(
        token in compact
        for token in (
            "stokesi",
            "stokes1",
            "totalintensity",
            "l+r",
            "ll+rr",
            "rr+ll",
            "lcp+rcp",
            "rcp+lcp",
        )
    ):
        return POL_SUM
    if re.search(
        r"(^|[^a-z0-9])(sum|total|lr|i)([^a-z0-9]|$)",
        low,
    ):
        return POL_SUM
    if re.search(
        r"(^|[^a-z0-9])(lcp|lhcp|left[-_ ]?hand|left[-_ ]?circ|ll|l[-_ ]?pol|左旋)([^a-z0-9]|$)",
        low,
    ):
        return POL_LCP
    if re.search(
        r"(^|[^a-z0-9])(rcp|rhcp|right[-_ ]?hand|right[-_ ]?circ|rr|r[-_ ]?pol|右旋)([^a-z0-9]|$)",
        low,
    ):
        return POL_RCP

    match = re.search(r"stokes[_ -]?code[_ -]?(-?\d+)", low)
    if match:
        return stokes_code_to_pol(float(match.group(1)))
    return POL_UNKNOWN


def infer_polarization(
    path: Path, header: fits.Header, default_pol: str = POL_SUM
) -> str:
    """Infer polarization from parent folders, FITS metadata, then filename."""

    parent_pol = infer_parent_directory_polarization(path)
    if parent_pol != POL_UNKNOWN:
        return parent_pol

    header_text = " ".join(
        str(header.get(key, ""))
        for key in ("POLAR", "POL", "POLTYPE", "STOKES", "CTYPE3", "CTYPE4", "BTYPE")
    )
    pol = normalize_pol_text(header_text)
    if pol != POL_UNKNOWN:
        return pol
    pol = normalize_pol_text(path.stem)
    if pol != POL_UNKNOWN:
        return pol
    return default_pol


def infer_parent_directory_polarization(path: Path) -> str:
    """Infer LL/RR-style polarization labels from directory names."""

    for part in reversed(path.parent.parts):
        pol = normalize_pol_text(part)
        if pol != POL_UNKNOWN:
            return pol
    return POL_UNKNOWN


def parse_frequency_mhz(path: Path, header: fits.Header) -> float:
    """Parse observing frequency and return MHz."""

    for key in ("FREQ", "FREQUENCY", "RESTFRQ", "RESTFREQ", "CRVAL3", "CRVAL4"):
        if key not in header:
            continue
        try:
            value = float(header[key])
        except (TypeError, ValueError):
            continue
        unit = str(
            header.get("FREQUNIT", header.get("CUNIT3", header.get("CUNIT4", "MHz")))
        ).strip()
        return _frequency_to_mhz(value, unit)

    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ghz|mhz|khz|hz)",
        path.name,
        flags=re.IGNORECASE,
    )
    if match:
        return _frequency_to_mhz(float(match.group("value")), match.group("unit"))

    match = re.search(r"(?P<value>\d+(?:\.\d+)?)", path.stem)
    if match:
        return float(match.group("value"))
    return float("nan")


def _frequency_to_mhz(value: float, unit: str) -> float:
    unit_norm = (unit or "MHz").strip().lower()
    if unit_norm in {"hz", "hertz"}:
        return float(value) / 1e6
    if unit_norm == "khz":
        return float(value) / 1e3
    if unit_norm == "ghz":
        return float(value) * 1e3
    if float(value) > 1e5:
        return float(value) / 1e6
    return float(value)


def parse_datetime_value(value: object) -> datetime | None:
    """Parse common FITS/diagnostic datetime values as naive UTC datetimes."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip().strip("'")
    if not text or text.lower() in {"none", "nan", "nat"}:
        return None
    text = text.replace("UTC", "").replace("Z", "").strip()

    digits_only = re.fullmatch(r"\d{14,20}", text)
    if digits_only:
        base = text[:14]
        frac = text[14:20].ljust(6, "0")
        try:
            return datetime.strptime(base + frac, "%Y%m%d%H%M%S%f")
        except ValueError:
            return None

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%dT%H%M%S.%f",
        "%Y%m%dT%H%M%S",
        "%Y%m%d%H%M%S.%f",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_time_from_filename(path: Path) -> datetime | None:
    """Parse observation time from common radio/AIA-like filename patterns."""

    name = path.stem
    match = re.search(
        r"(?P<date>\d{8})[^0-9]?(?P<hms>\d{6})(?:[^0-9]?(?P<frac>\d{1,6}))?",
        name,
    )
    if match:
        frac = (match.group("frac") or "").ljust(6, "0")[:6]
        try:
            return datetime.strptime(
                match.group("date") + match.group("hms") + frac,
                "%Y%m%d%H%M%S%f",
            )
        except ValueError:
            pass

    match = re.search(
        r"(?P<year>20\d{2})(?P<month>\d{1,2})(?P<day>\d{2})"
        r"[_-](?P<hms>\d{6})(?:[_-](?P<frac>\d{1,6}))?",
        name,
    )
    if match:
        date = (
            f"{int(match.group('year')):04d}"
            f"{int(match.group('month')):02d}"
            f"{int(match.group('day')):02d}"
        )
        frac = (match.group("frac") or "").ljust(6, "0")[:6]
        try:
            return datetime.strptime(date + match.group("hms") + frac, "%Y%m%d%H%M%S%f")
        except ValueError:
            pass

    match = re.search(
        r"(?P<date>20\d{2}-\d{2}-\d{2})T(?P<hms>\d{6})(?:\.\d+)?",
        name,
    )
    if match:
        try:
            return datetime.strptime(
                match.group("date") + match.group("hms"),
                "%Y-%m-%d%H%M%S",
            )
        except ValueError:
            pass
    return None


def parse_observation_time(path: Path, header: fits.Header) -> datetime | None:
    """Infer observation time from FITS headers, falling back to filename text."""

    for key in ("DATE-OBS", "DATE_OBS", "T_OBS", "TIME-OBS", "DATE"):
        parsed = parse_datetime_value(header.get(key))
        if parsed is not None:
            return parsed
    return parse_time_from_filename(path)


def to_arcsec(value: float, unit: str) -> float:
    """Convert angular values to arcsec using common FITS units."""

    unit_norm = (unit or "arcsec").strip().lower()
    if unit_norm in {"arcsec", "arcsecs", "arcsecond", "arcseconds", "asec"}:
        return float(value)
    if unit_norm in {"deg", "degree", "degrees"}:
        return float(value) * 3600.0
    if unit_norm in {"arcmin", "arcminute", "arcminutes", "amin"}:
        return float(value) * 60.0
    if unit_norm in {"rad", "radian", "radians"}:
        return math.degrees(float(value)) * 3600.0
    return float(value)


def pixel_to_hpc_arcsec(
    header: fits.Header, x_pix0: float, y_pix0: float
) -> tuple[float, float]:
    """Approximate 0-based image pixels as HPLN/HPLT arcsec coordinates.

    中文：使用 FITS 的线性 WCS 关键字将 numpy 的 0-based 像素坐标近似转换为
    日面角秒坐标。这里保留 CRPIX/CRVAL/CDELT、PC/CD、CROTA2 的兼容逻辑。
    """

    crpix1 = float(header.get("CRPIX1", 1.0))
    crpix2 = float(header.get("CRPIX2", 1.0))
    crval1 = float(header.get("CRVAL1", 0.0))
    crval2 = float(header.get("CRVAL2", 0.0))
    cdelt1 = float(header.get("CDELT1", 1.0))
    cdelt2 = float(header.get("CDELT2", 1.0))
    cunit1 = str(header.get("CUNIT1", "arcsec"))
    cunit2 = str(header.get("CUNIT2", "arcsec"))

    dx = float(x_pix0) + 1.0 - crpix1
    dy = float(y_pix0) + 1.0 - crpix2

    if all(key in header for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        wx = float(header["CD1_1"]) * dx + float(header["CD1_2"]) * dy
        wy = float(header["CD2_1"]) * dx + float(header["CD2_2"]) * dy
    else:
        pc11 = float(header.get("PC1_1", 1.0))
        pc12 = float(header.get("PC1_2", 0.0))
        pc21 = float(header.get("PC2_1", 0.0))
        pc22 = float(header.get("PC2_2", 1.0))
        if not any(key in header for key in ("PC1_1", "PC1_2", "PC2_1", "PC2_2")):
            theta = math.radians(float(header.get("CROTA2", 0.0)))
            pc11, pc12 = math.cos(theta), -math.sin(theta)
            pc21, pc22 = math.sin(theta), math.cos(theta)
        x_int = cdelt1 * dx
        y_int = cdelt2 * dy
        wx = pc11 * x_int + pc12 * y_int
        wy = pc21 * x_int + pc22 * y_int

    return to_arcsec(crval1 + wx, cunit1), to_arcsec(crval2 + wy, cunit2)


def stokes_code_to_pol(code: float) -> str:
    """Map FITS Stokes numeric codes to project polarization labels."""

    return {1: POL_SUM, -1: POL_RCP, -2: POL_LCP}.get(
        int(round(float(code))), POL_UNKNOWN
    )


def infer_pol_from_stokes_axis(header: fits.Header, index0: int, fits_axis: int) -> str:
    """Infer polarization for one plane on a FITS Stokes axis."""

    crval = float(header.get(f"CRVAL{fits_axis}", 1.0))
    cdelt = float(header.get(f"CDELT{fits_axis}", 1.0))
    crpix = float(header.get(f"CRPIX{fits_axis}", 1.0))
    code = crval + ((float(index0) + 1.0) - crpix) * cdelt
    return stokes_code_to_pol(code)


def iter_images_in_hdu(
    path: Path,
    hdu_index: int,
    data: np.ndarray,
    header: fits.Header,
    default_pol: str = POL_SUM,
) -> Iterator[RadioImage]:
    """Yield one or more 2D image planes from a FITS HDU."""

    arr = np.asarray(data)
    if arr.size == 0 or arr.dtype.fields is not None:
        return

    naxis = int(header.get("NAXIS", arr.ndim))
    stokes_fits_axis = None
    for axis in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{axis}", "")).upper()
        if "STOKES" in ctype or "POL" in ctype:
            stokes_fits_axis = axis
            break

    if arr.ndim == 2:
        yield _radio_image(path, hdu_index, arr, header, default_pol)
        return

    squeezed = np.squeeze(arr)
    if squeezed.ndim == 2:
        yield _radio_image(path, hdu_index, squeezed, header, default_pol)
        return

    if stokes_fits_axis is not None:
        py_axis = arr.ndim - stokes_fits_axis
        if 0 <= py_axis < arr.ndim:
            yield from _iter_split_axis(
                path, hdu_index, arr, header, py_axis, True, default_pol
            )
            return

    candidate_axes = [
        axis for axis, size in enumerate(squeezed.shape[:-2]) if size <= 4
    ]
    if candidate_axes:
        yield from _iter_split_axis(
            path,
            hdu_index,
            squeezed,
            header,
            candidate_axes[-1],
            False,
            default_pol,
        )
        return

    slicer = tuple([0] * (squeezed.ndim - 2) + [slice(None), slice(None)])
    plane = np.asarray(squeezed[slicer]).squeeze()
    if plane.ndim == 2:
        yield _radio_image(path, hdu_index, plane, header, default_pol)


def _radio_image(
    path: Path,
    hdu_index: int,
    image: np.ndarray,
    header: fits.Header,
    default_pol: str,
    pol: str | None = None,
) -> RadioImage:
    return RadioImage(
        path=path,
        hdu_index=hdu_index,
        image=np.asarray(image, dtype=float),
        header=header,
        pol=pol or infer_polarization(path, header, default_pol=default_pol),
        freq_mhz=parse_frequency_mhz(path, header),
        obs_time=parse_observation_time(path, header),
    )


def _iter_split_axis(
    path: Path,
    hdu_index: int,
    arr: np.ndarray,
    header: fits.Header,
    split_axis: int,
    split_axis_is_stokes: bool,
    default_pol: str,
) -> Iterator[RadioImage]:
    for index in range(arr.shape[split_axis]):
        slicer: list[object] = []
        for axis in range(arr.ndim):
            if axis == split_axis:
                slicer.append(index)
            elif axis >= arr.ndim - 2:
                slicer.append(slice(None))
            else:
                slicer.append(0)
        plane = np.asarray(arr[tuple(slicer)]).squeeze()
        if plane.ndim != 2:
            continue
        pol = None
        if split_axis_is_stokes:
            fits_axis = arr.ndim - split_axis
            pol = infer_pol_from_stokes_axis(header, index, fits_axis)
        if pol is None or pol == POL_UNKNOWN:
            pol = infer_polarization(path, header, default_pol=default_pol)
        yield _radio_image(path, hdu_index, plane, header, default_pol, pol=pol)


def iter_radio_images(
    path: str | Path, default_pol: str = POL_SUM
) -> Iterator[RadioImage]:
    """Iterate over usable 2D radio image planes in one FITS file."""

    resolved = Path(path)
    try:
        with fits.open(resolved, memmap=False, ignore_missing_end=True) as hdul:
            for hdu_index, hdu in enumerate(hdul):
                if hdu.data is None:
                    continue
                yield from iter_images_in_hdu(
                    resolved,
                    hdu_index,
                    np.asarray(hdu.data),
                    hdu.header,
                    default_pol=default_pol,
                )
    except Exception as exc:
        warnings.warn(f"Failed to read radio FITS file {resolved}: {exc}", stacklevel=2)


def choose_mask_component(
    mask: np.ndarray, work: np.ndarray, component: str = "peak"
) -> np.ndarray:
    """Keep one connected mask component when a threshold selects multiple blobs."""

    if not np.any(mask) or ndimage is None:
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count <= 1:
        return mask

    peak_y, peak_x = np.unravel_index(np.nanargmax(work), work.shape)
    if component == "peak":
        peak_label = labels[peak_y, peak_x]
        if peak_label > 0:
            return labels == peak_label

    label_ids = np.arange(1, count + 1)
    areas = ndimage.sum(mask.astype(float), labels, index=label_ids)
    fluxes = ndimage.sum(np.nan_to_num(work, nan=0.0), labels, index=label_ids)
    if component == "brightest":
        chosen = int(np.nanargmax(fluxes)) + 1
    elif component == "largest":
        chosen = int(np.nanargmax(areas)) + 1
    else:
        raise ValueError(f"Unknown mask component mode: {component}")
    return labels == chosen


def compute_source_center(
    image: np.ndarray,
    header: fits.Header,
    threshold_frac: float = 0.95,
    threshold_mode: str = "bg_peak",
    background_percentile: float = 5.0,
    centroid: str = "weighted",
    component: str = "peak",
    use_abs: bool = False,
    min_pixels: int = 1,
) -> dict[str, float]:
    """Extract a threshold/contour source center from one radio image.

    中文：以峰值比例、背景到峰值比例或图像分位数生成掩膜，并计算几何中心或
    强度加权中心。
    """

    arr = np.asarray(image, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.all(~np.isfinite(arr)):
        raise ValueError("radio image has no finite pixels")

    work = np.abs(arr) if use_abs else arr.copy()
    finite = np.isfinite(work)
    if not np.any(finite):
        raise ValueError("radio image has no usable pixels")

    peak_value = float(np.nanmax(work))
    peak_y, peak_x = np.unravel_index(np.nanargmax(work), work.shape)
    background_value = float(np.nanpercentile(work[finite], background_percentile))

    if threshold_mode == "peak":
        threshold_value = float(threshold_frac) * peak_value
    elif threshold_mode == "bg_peak":
        threshold_value = background_value + float(threshold_frac) * (
            peak_value - background_value
        )
    elif threshold_mode == "percentile":
        percentile = float(threshold_frac) * 100.0
        threshold_value = float(np.nanpercentile(work[finite], percentile))
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

    mask = finite & (work >= threshold_value)
    if not np.any(mask):
        mask = np.zeros_like(work, dtype=bool)
        mask[peak_y, peak_x] = True
    mask = choose_mask_component(mask, work, component=component)
    if int(mask.sum()) < int(min_pixels):
        mask = np.zeros_like(work, dtype=bool)
        mask[peak_y, peak_x] = True

    ys, xs = np.where(mask)
    values = work[ys, xs]
    if centroid == "geometric":
        weights = np.ones_like(values, dtype=float)
    elif centroid == "weighted":
        weights = np.clip(values - threshold_value, 0.0, None)
        if not np.any(weights > 0):
            weights = values - np.nanmin(values)
        if not np.any(weights > 0):
            weights = np.ones_like(values, dtype=float)
    else:
        raise ValueError(f"Unknown centroid: {centroid}")

    weight_sum = float(np.sum(weights))
    center_x_pix = float(np.sum(xs * weights) / weight_sum)
    center_y_pix = float(np.sum(ys * weights) / weight_sum)
    center_x_arcsec, center_y_arcsec = pixel_to_hpc_arcsec(
        header, center_x_pix, center_y_pix
    )
    peak_x_arcsec, peak_y_arcsec = pixel_to_hpc_arcsec(
        header, float(peak_x), float(peak_y)
    )

    return {
        "center_x_pix": center_x_pix,
        "center_y_pix": center_y_pix,
        "center_x_arcsec": float(center_x_arcsec),
        "center_y_arcsec": float(center_y_arcsec),
        "peak_x_pix": float(peak_x),
        "peak_y_pix": float(peak_y),
        "peak_x_arcsec": float(peak_x_arcsec),
        "peak_y_arcsec": float(peak_y_arcsec),
        "peak_value": peak_value,
        "background_value": background_value,
        "threshold_value": float(threshold_value),
        "area_pix": int(mask.sum()),
        "flux_sum_in_mask": float(np.nansum(arr[mask])),
        "ny": int(arr.shape[0]),
        "nx": int(arr.shape[1]),
    }


def record_from_radio_image(
    item: RadioImage,
    *,
    threshold_frac: float = 0.95,
    threshold_mode: str = "bg_peak",
    background_percentile: float = 5.0,
    centroid: str = "weighted",
    component: str = "peak",
    use_abs: bool = False,
    min_pixels: int = 1,
) -> dict[str, object]:
    """Build one output table row for a ``RadioImage``."""

    center = compute_source_center(
        item.image,
        item.header,
        threshold_frac=threshold_frac,
        threshold_mode=threshold_mode,
        background_percentile=background_percentile,
        centroid=centroid,
        component=component,
        use_abs=use_abs,
        min_pixels=min_pixels,
    )
    obs_time = item.obs_time
    row: dict[str, object] = {
        "obs_time": obs_time.isoformat(timespec="milliseconds") if obs_time else "",
        "time_unix": (
            obs_time.replace(tzinfo=timezone.utc).timestamp() if obs_time else np.nan
        ),
        "freq_mhz": item.freq_mhz,
        "polarization": item.pol,
        "source_label": item.source_label,
        "center_method": f"threshold_{centroid}_{threshold_mode}_{threshold_frac:g}",
        "threshold_frac": float(threshold_frac),
        "quality_flag": "ok",
        "quality_flag_detail": "",
        "hdu_index": item.hdu_index,
        "filename": item.path.name,
        "filepath": str(item.path),
        "ctype1": str(item.header.get("CTYPE1", "")),
        "ctype2": str(item.header.get("CTYPE2", "")),
        "cunit1": str(item.header.get("CUNIT1", "arcsec")),
        "cunit2": str(item.header.get("CUNIT2", "arcsec")),
    }
    row.update(center)
    return row


def find_files(
    radio_dir: str | Path, pattern: str = "*.fits", recursive: bool = False
) -> list[Path]:
    """Return sorted FITS-like files from a radio folder."""

    folder = Path(radio_dir).expanduser()
    files = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return sorted(
        path
        for path in files
        if path.is_file() and path.suffix.lower() in FITS_SUFFIXES
    )


def select_radio_files(
    radio_dir: str | Path,
    *,
    pattern: str = "*.fits",
    recursive: bool = False,
    freqs: list[float] | tuple[float, ...] | None = None,
    polarizations: list[str] | tuple[str, ...] | None = None,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
) -> list[Path]:
    """Return FITS files matching path-inferable filters."""

    files = find_files(radio_dir, pattern=pattern, recursive=recursive)
    freq_set = {float(freq) for freq in freqs or []}
    pol_set = {normalize_pol_text(pol) for pol in polarizations or []}
    pol_set.discard(POL_UNKNOWN)
    start = _parse_optional_datetime(time_start, "time_start")
    end = _parse_optional_datetime(time_end, "time_end")
    selected: list[Path] = []
    blank_header = fits.Header()
    for path in files:
        if freq_set:
            freq = parse_frequency_mhz(path, blank_header)
            if np.isfinite(freq) and not _frequency_in_set(freq, freq_set):
                continue
        if pol_set:
            pol = infer_polarization(path, blank_header)
            if pol != POL_UNKNOWN and pol not in pol_set:
                continue
        if start is not None or end is not None:
            obs_time = parse_time_from_filename(path)
            if start is not None and obs_time is not None and obs_time < start:
                continue
            if end is not None and obs_time is not None and obs_time > end:
                continue
        selected.append(path)
    return selected


def maybe_make_sum_images(
    items: list[RadioImage], tolerance_sec: float = 0.5
) -> list[RadioImage]:
    """Pair nearby LCP/RCP images and synthesize L+R image planes."""

    l_items = [
        item
        for item in items
        if item.pol == POL_LCP
        and item.obs_time is not None
        and np.isfinite(item.freq_mhz)
    ]
    r_items = [
        item
        for item in items
        if item.pol == POL_RCP
        and item.obs_time is not None
        and np.isfinite(item.freq_mhz)
    ]
    sums: list[RadioImage] = []
    used_r: set[int] = set()
    for left in l_items:
        best_index = None
        best_dt = None
        for index, right in enumerate(r_items):
            if index in used_r:
                continue
            if not _same_frequency(left.freq_mhz, right.freq_mhz):
                continue
            dt = abs((left.obs_time - right.obs_time).total_seconds())
            if dt <= tolerance_sec and (best_dt is None or dt < best_dt):
                best_index = index
                best_dt = dt
        if best_index is None:
            continue
        right = r_items[best_index]
        used_r.add(best_index)
        if left.image.shape != right.image.shape:
            warnings.warn(
                f"Skipping L+R pair with mismatched shapes: {left.path.name}, {right.path.name}",
                stacklevel=2,
            )
            continue
        header = left.header.copy()
        header["POLAR"] = POL_SUM
        midpoint = left.obs_time + (right.obs_time - left.obs_time) / 2
        sums.append(
            RadioImage(
                path=left.path,
                hdu_index=left.hdu_index,
                image=np.asarray(left.image, dtype=float)
                + np.asarray(right.image, dtype=float),
                header=header,
                pol=POL_SUM,
                freq_mhz=left.freq_mhz,
                obs_time=midpoint,
                source_label="paired_L_plus_R",
            )
        )
    return sums


def _same_frequency(left_mhz: float, right_mhz: float) -> bool:
    if not (np.isfinite(left_mhz) and np.isfinite(right_mhz)):
        return False
    tolerance = max(1e-6, 1e-5 * abs(float(left_mhz)))
    return abs(float(left_mhz) - float(right_mhz)) <= tolerance


def extract_radio_centers(
    radio_dir: str | Path,
    *,
    out: str | Path | None = None,
    pattern: str = "*.fits",
    recursive: bool = False,
    freqs: list[float] | tuple[float, ...] | None = None,
    polarizations: list[str] | tuple[str, ...] | None = None,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
    threshold_frac: float = 0.95,
    threshold_mode: str = "bg_peak",
    background_percentile: float = 5.0,
    centroid: str = "weighted",
    component: str = "peak",
    use_abs: bool = False,
    min_pixels: int = 1,
    default_pol: str = POL_SUM,
    make_sum: bool = False,
    pair_time_tolerance_sec: float = 0.5,
) -> pd.DataFrame:
    """Extract source centers from all matching FITS files in ``radio_dir``."""

    folder = Path(radio_dir).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Radio data folder does not exist: {folder}")
    files = select_radio_files(
        folder,
        pattern=pattern,
        recursive=recursive,
        freqs=freqs,
        polarizations=polarizations,
        time_start=time_start,
        time_end=time_end,
    )
    if not files:
        raise FileNotFoundError(f"No FITS files found under {folder} matching filters.")

    images: list[RadioImage] = []
    for path in files:
        images.extend(iter_radio_images(path, default_pol=default_pol))
    images = filter_radio_images(
        images,
        freqs=freqs,
        polarizations=polarizations,
        time_start=time_start,
        time_end=time_end,
    )
    if make_sum:
        images.extend(
            maybe_make_sum_images(images, tolerance_sec=pair_time_tolerance_sec)
        )

    rows: list[dict[str, object]] = []
    for item in images:
        try:
            rows.append(
                record_from_radio_image(
                    item,
                    threshold_frac=threshold_frac,
                    threshold_mode=threshold_mode,
                    background_percentile=background_percentile,
                    centroid=centroid,
                    component=component,
                    use_abs=use_abs,
                    min_pixels=min_pixels,
                )
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to extract source center from {item.path.name} "
                f"(pol={item.pol}): {exc}",
                stacklevel=2,
            )
    if not rows:
        raise RuntimeError("No radio source centers were extracted.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["freq_mhz", "polarization", "obs_time", "filename"],
        kind="mergesort",
    ).reset_index(drop=True)
    if out is not None:
        write_centers_table(df, out)
    return df


def filter_radio_images(
    images: list[RadioImage],
    *,
    freqs: list[float] | tuple[float, ...] | None = None,
    polarizations: list[str] | tuple[str, ...] | None = None,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
) -> list[RadioImage]:
    """Filter radio image planes before center extraction."""

    freq_set = {float(freq) for freq in freqs or []}
    pol_set = {normalize_pol_text(pol) for pol in polarizations or []}
    pol_set.discard(POL_UNKNOWN)
    start = _parse_optional_datetime(time_start, "time_start")
    end = _parse_optional_datetime(time_end, "time_end")

    filtered: list[RadioImage] = []
    for item in images:
        if freq_set and not _frequency_in_set(item.freq_mhz, freq_set):
            continue
        if pol_set and item.pol not in pol_set:
            continue
        if start is not None and (item.obs_time is None or item.obs_time < start):
            continue
        if end is not None and (item.obs_time is None or item.obs_time > end):
            continue
        filtered.append(item)
    return filtered


def _parse_optional_datetime(
    value: str | datetime | None, label: str
) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = parse_datetime_value(value)
    if parsed is None:
        raise ValueError(f"Invalid {label}: {value}")
    return parsed


def _frequency_in_set(value: float, freq_set: set[float]) -> bool:
    if not np.isfinite(value):
        return False
    return any(_same_frequency(float(value), freq) for freq in freq_set)


def write_centers_table(df: pd.DataFrame, out: str | Path) -> Path:
    """Write a source-center table to CSV or Excel based on suffix."""

    output_path = Path(out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for threshold center extraction."""

    parser = argparse.ArgumentParser(
        prog="solar-apps workflow radio centers",
        description="Extract threshold radio-source centers from FITS files.",
    )
    parser.add_argument(
        "--radio-dir", required=True, help="Folder containing radio FITS files."
    )
    parser.add_argument(
        "--out", default="radio_centers.csv", help="Output .csv or .xlsx table."
    )
    parser.add_argument(
        "--pattern", default="*.fits", help="FITS filename glob pattern."
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search subfolders recursively."
    )
    parser.add_argument(
        "--freqs",
        help="Comma-separated frequency filter in MHz, e.g. 149,164,190.",
    )
    parser.add_argument(
        "--polarizations",
        help="Comma-separated polarization filter such as LL,RR or LCP,RCP.",
    )
    parser.add_argument(
        "--time-start",
        help="Inclusive observation-time start, e.g. 2025-01-24T04:46:45.",
    )
    parser.add_argument(
        "--time-end",
        help="Inclusive observation-time end, e.g. 2025-01-24T04:50:45.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.95, help="Threshold fraction, e.g. 0.95."
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["peak", "bg_peak", "percentile"],
        default="bg_peak",
        help="Threshold definition: peak, bg_peak, or percentile.",
    )
    parser.add_argument(
        "--background-percentile",
        type=float,
        default=5.0,
        help="Background percentile used by bg_peak mode.",
    )
    parser.add_argument(
        "--centroid",
        choices=["weighted", "geometric"],
        default="weighted",
        help="Center calculation inside the threshold mask.",
    )
    parser.add_argument(
        "--component",
        choices=["peak", "largest", "brightest"],
        default="peak",
        help="Connected mask component to keep.",
    )
    parser.add_argument(
        "--use-abs", action="store_true", help="Use absolute image values."
    )
    parser.add_argument(
        "--min-pixels", type=int, default=1, help="Minimum mask pixels."
    )
    parser.add_argument(
        "--default-pol",
        choices=[POL_LCP, POL_RCP, POL_SUM, POL_UNKNOWN],
        default=POL_SUM,
        help="Fallback polarization when metadata is missing.",
    )
    parser.add_argument(
        "--make-sum",
        action="store_true",
        help="Pair LCP/RCP files and additionally extract L+R centers.",
    )
    parser.add_argument(
        "--pair-time-tolerance-sec",
        type=float,
        default=0.5,
        help="Maximum LCP/RCP pairing time difference in seconds.",
    )
    return parser


def run_center_extraction(argv: list[str] | None = None) -> pd.DataFrame:
    """Run threshold center extraction and return the generated table.

    This function preserves the former programmatic return contract of
    :func:`main`.  Command-line callers should use :func:`main`, which returns
    an integer process status code.
    """

    args = build_parser().parse_args(argv)
    freqs = _parse_float_csv(args.freqs)
    polarizations = _parse_str_csv(args.polarizations)
    df = extract_radio_centers(
        args.radio_dir,
        out=args.out,
        pattern=args.pattern,
        recursive=args.recursive,
        freqs=freqs,
        polarizations=polarizations,
        time_start=args.time_start,
        time_end=args.time_end,
        threshold_frac=args.threshold,
        threshold_mode=args.threshold_mode,
        background_percentile=args.background_percentile,
        centroid=args.centroid,
        component=args.component,
        use_abs=args.use_abs,
        min_pixels=args.min_pixels,
        default_pol=args.default_pol,
        make_sum=args.make_sum,
        pair_time_tolerance_sec=args.pair_time_tolerance_sec,
    )
    candidate_count = len(
        select_radio_files(
            args.radio_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            freqs=freqs,
            polarizations=polarizations,
            time_start=args.time_start,
            time_end=args.time_end,
        )
    )
    print(f"Candidate FITS files: {candidate_count}")
    print(f"Extracted centers: {len(df)}")
    print(f"Output table: {Path(args.out).expanduser().resolve()}")
    return df


def main(argv: list[str] | None = None) -> int:
    """Run the threshold center-extraction command."""

    run_center_extraction(argv)
    return 0


def _parse_str_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_csv(raw: str | None) -> list[float]:
    return [float(item) for item in _parse_str_csv(raw)]


if __name__ == "__main__":
    raise SystemExit(main())
