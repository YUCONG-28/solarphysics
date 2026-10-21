"""Raw FITS quality diagnostics for radio source-map inputs.

English: Classify obvious raw radio image artifacts before source fitting or
overlay rendering.

中文：在射电源拟合或叠加绘图前，识别原始射电 FITS 图像中明显的坏帧和
伪影。
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from solar_toolkit._utils.numeric import numeric_data

__all__ = [
    "RawFileQualityRow",
    "RawQualityAnalysisResult",
    "RawQualityClassification",
    "RawQualityFilterResult",
    "RawQualityMetrics",
    "RawQualityThresholds",
    "RawSlotQualityRow",
    "analyze_radio_raw_quality",
    "classify_raw_metrics",
    "compute_raw_quality_metrics",
    "filter_bad_radio_fits_files",
    "read_radio_fits_image",
]


@dataclass(frozen=True)
class RawQualityThresholds:
    """Conservative defaults for obvious raw radio image artifacts."""

    high_percentile: float = 99.7
    p95_delta_bad: float = 0.75
    p997_delta_bad: float = 0.75
    min_bright_pixels_bad: int = 12
    min_component_count_bad: int = 4
    min_distributed_bright_fraction_bad: float = 0.55
    p997_delta_extended_bad: float = 0.75
    min_extended_component_pixels_bad: int = 45
    min_extended_component_span_bad: int = 16
    max_extended_component_fill_fraction_bad: float = 0.50


@dataclass(frozen=True)
class RawQualityMetrics:
    total_pixel_count: int
    finite_pixel_count: int
    positive_pixel_count: int
    finite_fraction: float
    positive_fraction: float
    p50: float
    p95: float
    p99: float
    p997: float
    p999: float
    max_log10: float
    bright_threshold_log10: float
    bright_pixel_count: int
    bright_component_count: int
    largest_component_pixels: int
    largest_component_bbox_width: int
    largest_component_bbox_height: int
    largest_component_fill_fraction: float
    distributed_bright_fraction: float
    valid: bool
    reason: str


@dataclass(frozen=True)
class RawQualityClassification:
    quality_flag: str
    reason: str
    p95_delta: float
    p997_delta: float
    high_tail: bool
    distributed_bright_pixels: bool
    extended_bright_component: bool


@dataclass(frozen=True)
class RawFileQualityRow:
    source_file: str
    frequency_mhz: float
    polarization: str
    file_index: int
    slot_index: int
    time: str
    quality_flag: str
    reason: str
    total_pixel_count: int
    finite_pixel_count: int
    positive_pixel_count: int
    finite_fraction: float
    positive_fraction: float
    p50: float
    p95: float
    p99: float
    p997: float
    p999: float
    max_log10: float
    baseline_p95: float
    baseline_p997: float
    p95_delta: float
    p997_delta: float
    bright_threshold_log10: float
    bright_pixel_count: int
    bright_component_count: int
    largest_component_pixels: int
    largest_component_bbox_width: int
    largest_component_bbox_height: int
    largest_component_fill_fraction: float
    distributed_bright_fraction: float


@dataclass(frozen=True)
class RawSlotQualityRow:
    slot_index: int
    quality_flag: str
    bad_file_count: int
    total_file_count: int
    time: str
    bad_items: str
    source_files: str


@dataclass(frozen=True)
class RawQualityAnalysisResult:
    file_rows: list[RawFileQualityRow]
    slot_rows: list[RawSlotQualityRow]
    file_csv_path: Path
    slot_csv_path: Path


@dataclass(frozen=True)
class RawQualityFilterResult:
    accepted_files: list[str]
    rejected_rows: list[RawFileQualityRow]
    file_rows: list[RawFileQualityRow]


@dataclass(frozen=True)
class _PendingFile:
    path: Path
    frequency_mhz: float
    polarization: str
    file_index: int
    slot_index: int
    time: str
    metrics: RawQualityMetrics


def read_radio_fits_image(file_path: str | Path) -> tuple[np.ndarray, fits.Header]:
    """Read a radio FITS image, preferring the first ImageHDU after primary."""

    path = Path(file_path)
    with fits.open(path, memmap=True) as hdul:
        hdu = None
        for candidate in hdul[1:]:
            if isinstance(candidate, fits.ImageHDU) and candidate.data is not None:
                hdu = candidate
                break
        if hdu is None:
            hdu = hdul[0]
        if hdu.data is None:
            raise ValueError("no FITS image data")
        data = np.squeeze(np.array(hdu.data, dtype=np.float64, copy=True))
        header = hdu.header.copy()
    return data, header


def compute_raw_quality_metrics(
    data: np.ndarray,
    thresholds: RawQualityThresholds | None = None,
) -> RawQualityMetrics:
    """Compute image metrics, excluding masked and non-finite samples alike."""

    cfg = thresholds or RawQualityThresholds()
    arr = np.squeeze(numeric_data(data))
    if arr.ndim != 2:
        raise ValueError(f"expected 2D FITS image, got {arr.ndim}D")

    total = int(arr.size)
    finite_mask = np.isfinite(arr)
    finite_count = int(finite_mask.sum())
    positive_mask = finite_mask & (arr > 0)
    positive_count = int(positive_mask.sum())
    finite_fraction = _safe_fraction(finite_count, total)
    positive_fraction = _safe_fraction(positive_count, total)

    if positive_count == 0:
        return _invalid_metrics(
            total,
            finite_count,
            positive_count,
            finite_fraction,
            positive_fraction,
            "no_finite_positive_pixels",
        )

    log_values = np.log10(arr[positive_mask])
    p50, p95, p99, p997, p999 = np.percentile(
        log_values, [50.0, 95.0, 99.0, cfg.high_percentile, 99.9]
    )
    max_log10 = float(np.max(log_values))

    log_image = np.full(arr.shape, np.nan, dtype=np.float64)
    log_image[positive_mask] = log_values
    bright_mask = np.isfinite(log_image) & (log_image >= p997)
    component_stats = _connected_component_stats(bright_mask)
    component_sizes = [item["size"] for item in component_stats]
    bright_pixel_count = int(bright_mask.sum())
    largest_component_stat = max(
        component_stats, key=lambda item: item["size"], default=None
    )
    largest_component = (
        int(largest_component_stat["size"]) if largest_component_stat else 0
    )
    largest_bbox_width = (
        int(largest_component_stat["width"]) if largest_component_stat else 0
    )
    largest_bbox_height = (
        int(largest_component_stat["height"]) if largest_component_stat else 0
    )
    largest_fill_fraction = (
        float(largest_component_stat["fill_fraction"])
        if largest_component_stat
        else 0.0
    )
    bright_component_count = len(component_sizes)
    distributed_fraction = (
        _safe_fraction(bright_pixel_count - largest_component, bright_pixel_count)
        if bright_pixel_count
        else 0.0
    )

    return RawQualityMetrics(
        total_pixel_count=total,
        finite_pixel_count=finite_count,
        positive_pixel_count=positive_count,
        finite_fraction=finite_fraction,
        positive_fraction=positive_fraction,
        p50=float(p50),
        p95=float(p95),
        p99=float(p99),
        p997=float(p997),
        p999=float(p999),
        max_log10=max_log10,
        bright_threshold_log10=float(p997),
        bright_pixel_count=bright_pixel_count,
        bright_component_count=int(bright_component_count),
        largest_component_pixels=int(largest_component),
        largest_component_bbox_width=int(largest_bbox_width),
        largest_component_bbox_height=int(largest_bbox_height),
        largest_component_fill_fraction=float(largest_fill_fraction),
        distributed_bright_fraction=float(distributed_fraction),
        valid=True,
        reason="ok",
    )


def classify_raw_metrics(
    metrics: RawQualityMetrics,
    baseline: dict[str, float] | None,
    thresholds: RawQualityThresholds | None = None,
) -> RawQualityClassification:
    """Classify one metrics row against a robust frequency/polarization baseline."""

    cfg = thresholds or RawQualityThresholds()
    if not metrics.valid:
        return RawQualityClassification(
            quality_flag="bad",
            reason=metrics.reason,
            p95_delta=math.nan,
            p997_delta=math.nan,
            high_tail=False,
            distributed_bright_pixels=False,
            extended_bright_component=False,
        )

    baseline = baseline or {}
    baseline_p95 = _finite_or_nan(baseline.get("p95"))
    baseline_p997 = _finite_or_nan(baseline.get("p997"))
    p95_delta = _delta_or_zero(metrics.p95, baseline_p95)
    p997_delta = _delta_or_zero(metrics.p997, baseline_p997)
    high_tail = p95_delta >= cfg.p95_delta_bad or p997_delta >= cfg.p997_delta_bad
    distributed = (
        metrics.bright_pixel_count >= cfg.min_bright_pixels_bad
        and metrics.bright_component_count >= cfg.min_component_count_bad
        and metrics.distributed_bright_fraction
        >= cfg.min_distributed_bright_fraction_bad
    )
    distributed_high_tail = distributed and p997_delta >= cfg.p997_delta_bad
    extended_component = (
        p997_delta >= cfg.p997_delta_extended_bad
        and metrics.largest_component_pixels >= cfg.min_extended_component_pixels_bad
        and max(
            metrics.largest_component_bbox_width,
            metrics.largest_component_bbox_height,
        )
        >= cfg.min_extended_component_span_bad
        and metrics.largest_component_fill_fraction
        <= cfg.max_extended_component_fill_fraction_bad
    )

    if high_tail and (distributed_high_tail or extended_component):
        morphology_reasons = []
        if distributed_high_tail:
            morphology_reasons.append(
                "distributed_bright_pixels:"
                f"components={metrics.bright_component_count},"
                f"off_fraction={metrics.distributed_bright_fraction:.3f}"
            )
        if extended_component:
            morphology_reasons.append(
                "extended_bright_component:"
                f"pixels={metrics.largest_component_pixels},"
                "bbox="
                f"{metrics.largest_component_bbox_width}x"
                f"{metrics.largest_component_bbox_height},"
                f"fill={metrics.largest_component_fill_fraction:.3f}"
            )
        reason = (
            f"high_tail:p95_delta={p95_delta:.3f},p997_delta={p997_delta:.3f};"
            + ";".join(morphology_reasons)
        )
        return RawQualityClassification(
            quality_flag="bad",
            reason=reason,
            p95_delta=float(p95_delta),
            p997_delta=float(p997_delta),
            high_tail=True,
            distributed_bright_pixels=bool(distributed),
            extended_bright_component=bool(extended_component),
        )

    return RawQualityClassification(
        quality_flag="ok",
        reason="ok",
        p95_delta=float(p95_delta),
        p997_delta=float(p997_delta),
        high_tail=bool(high_tail),
        distributed_bright_pixels=bool(distributed),
        extended_bright_component=bool(extended_component),
    )


def analyze_radio_raw_quality(
    root: str | Path,
    *,
    freqs: list[int | float],
    polarizations: list[str] | tuple[str, ...] = ("RR", "LL"),
    output_dir: str | Path,
    start_idx: int = 0,
    end_idx: int | None = None,
    thresholds: RawQualityThresholds | None = None,
) -> RawQualityAnalysisResult:
    """Scan radio FITS directories and write file-level and slot-level CSV reports."""

    cfg = thresholds or RawQualityThresholds()
    root_path = Path(root)
    out_dir = Path(output_dir) / "raw_quality_diagnostics"
    file_csv = out_dir / "radio_raw_file_quality.csv"
    slot_csv = out_dir / "radio_raw_slot_quality.csv"

    pending = _scan_pending_files(
        root_path,
        freqs=freqs,
        polarizations=polarizations,
        start_idx=start_idx,
        end_idx=end_idx,
        thresholds=cfg,
    )
    baselines = _build_group_baselines(pending)
    file_rows = _build_file_rows(pending, baselines, cfg)
    slot_rows = _build_slot_rows(file_rows)

    _write_rows(file_csv, file_rows, RawFileQualityRow)
    _write_rows(slot_csv, slot_rows, RawSlotQualityRow)
    return RawQualityAnalysisResult(
        file_rows=file_rows,
        slot_rows=slot_rows,
        file_csv_path=file_csv,
        slot_csv_path=slot_csv,
    )


def filter_bad_radio_fits_files(
    files: list[str | Path],
    *,
    frequency_mhz: int | float,
    polarization: str,
    thresholds: RawQualityThresholds | None = None,
) -> RawQualityFilterResult:
    """Return FITS files whose raw image quality is not flagged as bad."""

    cfg = thresholds or RawQualityThresholds()
    pending = [
        _analyze_one_file(
            Path(path),
            frequency_mhz=float(frequency_mhz),
            polarization=str(polarization),
            file_index=index,
            slot_index=index,
            thresholds=cfg,
        )
        for index, path in enumerate(files)
    ]
    baselines = _build_group_baselines(pending)
    rows = _build_file_rows(pending, baselines, cfg)
    rejected = [row for row in rows if row.quality_flag == "bad"]
    rejected_paths = {row.source_file for row in rejected}
    accepted = [str(path) for path in files if str(Path(path)) not in rejected_paths]
    return RawQualityFilterResult(
        accepted_files=accepted,
        rejected_rows=rejected,
        file_rows=rows,
    )


def _scan_pending_files(
    root: Path,
    *,
    freqs: list[int | float],
    polarizations: list[str] | tuple[str, ...],
    start_idx: int,
    end_idx: int | None,
    thresholds: RawQualityThresholds,
) -> list[_PendingFile]:
    pending: list[_PendingFile] = []
    for freq in freqs:
        freq_value = float(freq)
        for polarization in polarizations:
            band_dir = root / f"{_format_freq(freq_value)}MHz" / str(polarization)
            files = sorted(band_dir.glob("*.fits"))
            selected = files[start_idx:end_idx]
            for slot_index, path in enumerate(selected):
                file_index = start_idx + slot_index
                pending.append(
                    _analyze_one_file(
                        path,
                        frequency_mhz=freq_value,
                        polarization=str(polarization),
                        file_index=file_index,
                        slot_index=slot_index,
                        thresholds=thresholds,
                    )
                )
    return pending


def _analyze_one_file(
    path: Path,
    *,
    frequency_mhz: float,
    polarization: str,
    file_index: int,
    slot_index: int,
    thresholds: RawQualityThresholds,
) -> _PendingFile:
    time_value = ""
    try:
        data, header = read_radio_fits_image(path)
        time_value = _time_from_header(header)
        metrics = compute_raw_quality_metrics(data, thresholds)
    except (
        Exception
    ) as exc:  # noqa: BLE001 - report bad FITS rows without aborting scan
        metrics = _invalid_metrics(0, 0, 0, math.nan, math.nan, f"invalid_data: {exc}")
    return _PendingFile(
        path=path,
        frequency_mhz=frequency_mhz,
        polarization=polarization,
        file_index=file_index,
        slot_index=slot_index,
        time=time_value,
        metrics=metrics,
    )


def _build_group_baselines(
    pending: list[_PendingFile],
) -> dict[tuple[float, str], dict[str, float]]:
    grouped: dict[tuple[float, str], list[RawQualityMetrics]] = defaultdict(list)
    for item in pending:
        if item.metrics.valid:
            grouped[(item.frequency_mhz, item.polarization)].append(item.metrics)

    baselines: dict[tuple[float, str], dict[str, float]] = {}
    for key, metrics_rows in grouped.items():
        baselines[key] = {
            "p95": _lower_half_median(row.p95 for row in metrics_rows),
            "p997": _lower_half_median(row.p997 for row in metrics_rows),
        }
    return baselines


def _build_file_rows(
    pending: list[_PendingFile],
    baselines: dict[tuple[float, str], dict[str, float]],
    thresholds: RawQualityThresholds,
) -> list[RawFileQualityRow]:
    rows: list[RawFileQualityRow] = []
    for item in pending:
        baseline = baselines.get((item.frequency_mhz, item.polarization), {})
        classification = classify_raw_metrics(item.metrics, baseline, thresholds)
        rows.append(
            RawFileQualityRow(
                source_file=str(item.path),
                frequency_mhz=item.frequency_mhz,
                polarization=item.polarization,
                file_index=item.file_index,
                slot_index=item.slot_index,
                time=item.time,
                quality_flag=classification.quality_flag,
                reason=classification.reason,
                total_pixel_count=item.metrics.total_pixel_count,
                finite_pixel_count=item.metrics.finite_pixel_count,
                positive_pixel_count=item.metrics.positive_pixel_count,
                finite_fraction=item.metrics.finite_fraction,
                positive_fraction=item.metrics.positive_fraction,
                p50=item.metrics.p50,
                p95=item.metrics.p95,
                p99=item.metrics.p99,
                p997=item.metrics.p997,
                p999=item.metrics.p999,
                max_log10=item.metrics.max_log10,
                baseline_p95=_finite_or_nan(baseline.get("p95")),
                baseline_p997=_finite_or_nan(baseline.get("p997")),
                p95_delta=classification.p95_delta,
                p997_delta=classification.p997_delta,
                bright_threshold_log10=item.metrics.bright_threshold_log10,
                bright_pixel_count=item.metrics.bright_pixel_count,
                bright_component_count=item.metrics.bright_component_count,
                largest_component_pixels=item.metrics.largest_component_pixels,
                largest_component_bbox_width=(
                    item.metrics.largest_component_bbox_width
                ),
                largest_component_bbox_height=(
                    item.metrics.largest_component_bbox_height
                ),
                largest_component_fill_fraction=(
                    item.metrics.largest_component_fill_fraction
                ),
                distributed_bright_fraction=item.metrics.distributed_bright_fraction,
            )
        )
    return rows


def _build_slot_rows(file_rows: list[RawFileQualityRow]) -> list[RawSlotQualityRow]:
    grouped: dict[int, list[RawFileQualityRow]] = defaultdict(list)
    for row in file_rows:
        grouped[row.slot_index].append(row)

    slot_rows: list[RawSlotQualityRow] = []
    for slot_index in sorted(grouped):
        rows = sorted(
            grouped[slot_index],
            key=lambda item: (item.frequency_mhz, item.polarization, item.source_file),
        )
        bad_rows = [row for row in rows if row.quality_flag == "bad"]
        bad_items = "; ".join(
            f"{_format_freq(row.frequency_mhz)}MHz/{row.polarization}:"
            f"{Path(row.source_file).name}:{row.reason}"
            for row in bad_rows
        )
        source_files = "|".join(row.source_file for row in rows)
        time_value = next((row.time for row in rows if row.time), "")
        slot_rows.append(
            RawSlotQualityRow(
                slot_index=slot_index,
                quality_flag="bad" if bad_rows else "ok",
                bad_file_count=len(bad_rows),
                total_file_count=len(rows),
                time=time_value,
                bad_items=bad_items,
                source_files=source_files,
            )
        )
    return slot_rows


def _write_rows(path: Path, rows: list[Any], row_type: type) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _connected_component_stats(mask: np.ndarray) -> list[dict[str, float]]:
    try:
        from scipy.ndimage import find_objects, label

        labels, count = label(mask, structure=np.ones((3, 3), dtype=np.int8))
        if count == 0:
            return []
        stats = []
        for label_value, component_slice in enumerate(find_objects(labels), start=1):
            if component_slice is None:
                continue
            y_slice, x_slice = component_slice
            component = labels[component_slice] == label_value
            size = int(component.sum())
            width = int(x_slice.stop - x_slice.start)
            height = int(y_slice.stop - y_slice.start)
            area = max(width * height, 1)
            stats.append(
                {
                    "size": size,
                    "width": width,
                    "height": height,
                    "fill_fraction": float(size) / float(area),
                }
            )
        return stats
    except Exception:  # pragma: no cover - fallback for constrained installs
        return _connected_component_stats_fallback(mask)


def _connected_component_stats_fallback(mask: np.ndarray) -> list[dict[str, float]]:
    visited = np.zeros(mask.shape, dtype=bool)
    stats: list[dict[str, float]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            min_y = max_y = y
            min_x = max_x = x
            while stack:
                cy, cx = stack.pop()
                size += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            bbox_width = max_x - min_x + 1
            bbox_height = max_y - min_y + 1
            area = max(bbox_width * bbox_height, 1)
            stats.append(
                {
                    "size": int(size),
                    "width": int(bbox_width),
                    "height": int(bbox_height),
                    "fill_fraction": float(size) / float(area),
                }
            )
    return stats


def _invalid_metrics(
    total: int,
    finite_count: int,
    positive_count: int,
    finite_fraction: float,
    positive_fraction: float,
    reason: str,
) -> RawQualityMetrics:
    return RawQualityMetrics(
        total_pixel_count=int(total),
        finite_pixel_count=int(finite_count),
        positive_pixel_count=int(positive_count),
        finite_fraction=float(finite_fraction),
        positive_fraction=float(positive_fraction),
        p50=math.nan,
        p95=math.nan,
        p99=math.nan,
        p997=math.nan,
        p999=math.nan,
        max_log10=math.nan,
        bright_threshold_log10=math.nan,
        bright_pixel_count=0,
        bright_component_count=0,
        largest_component_pixels=0,
        largest_component_bbox_width=0,
        largest_component_bbox_height=0,
        largest_component_fill_fraction=0.0,
        distributed_bright_fraction=0.0,
        valid=False,
        reason=reason,
    )


def _time_from_header(header: fits.Header) -> str:
    for key in ("DATE-OBS", "DATEOBS", "TIME-OBS", "TIMEOBS"):
        value = header.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _lower_half_median(values) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return math.nan
    ordered = np.sort(arr)
    lower_count = max(1, int(math.ceil(ordered.size * 0.10)))
    return float(np.median(ordered[:lower_count]))


def _delta_or_zero(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline):
        return 0.0
    return float(value - baseline)


def _finite_or_nan(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if np.isfinite(number) else math.nan


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def _format_freq(freq: float) -> str:
    value = float(freq)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
