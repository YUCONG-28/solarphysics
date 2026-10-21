"""Pure drift-rate calculations, persistence, and plot-overlay helpers.

The browser selector and HTTP server live in ``solar_apps``. Interactive
selection can be injected with ``launch_func`` without coupling the public
library to an application runtime.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import warnings
from dataclasses import dataclass

import matplotlib.dates as mdates
import numpy as np

from .io import DRIFT_RATE_DIAGNOSTIC_FIELDS, _sample_float
from .io import drift_output_path as _drift_output_path
from .io import parse_datetime_value as _parse_datetime_value
from .spectrogram import _date_num_to_datetime

__all__ = [
    "DriftRateResult",
    "assert_spectrogram_mapping_not_flipped",
    "calculate_drift_rate_from_line",
    "get_or_load_drift_rate_results",
    "load_drift_selection_json",
    "overlay_drift_rate_results",
    "save_drift_rate_diagnostics_once",
    "save_drift_selection_json",
]

_DRIFT_RATE_RESULTS_CACHE: dict[tuple, list[DriftRateResult]] = {}
_DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS: set[tuple] = set()


@dataclass
class DriftRateResult:
    label: str
    mode: str
    t_start: datetime.datetime
    t_end: datetime.datetime
    f_start_mhz: float
    f_end_mhz: float
    drift_rate_mhz_s: float
    abs_drift_rate_mhz_s: float
    duration_s: float
    bandwidth_mhz: float
    color: str = "white"
    quality_flag: str = "ok"
    warning: str = ""


@dataclass(frozen=True)
class _DriftRateCalculationProfile:
    sort_endpoints_by_time: bool
    zero_duration_policy: str
    zero_duration_tolerance_s: float
    default_label: str
    default_mode: str
    default_on_falsy: bool
    warn_positive_drift: bool
    strict_frequency_fields: bool


_CANONICAL_DRIFT_RATE_PROFILE = _DriftRateCalculationProfile(
    sort_endpoints_by_time=True,
    zero_duration_policy="quality_flag",
    zero_duration_tolerance_s=0.0,
    default_label="drift_001",
    default_mode="manual",
    default_on_falsy=True,
    warn_positive_drift=True,
    strict_frequency_fields=False,
)
_CSO_DRIFT_RATE_PROFILE = _DriftRateCalculationProfile(
    sort_endpoints_by_time=False,
    zero_duration_policy="raise",
    zero_duration_tolerance_s=1e-9,
    default_label="drift",
    default_mode="manual_endpoint",
    default_on_falsy=False,
    warn_positive_drift=False,
    strict_frequency_fields=True,
)
_UNSET_DRIFT_TIME = object()


def _datetime_iso_ms(value: datetime.datetime) -> str:
    return value.replace(tzinfo=None).isoformat(timespec="milliseconds")


def _drift_line_time(line: dict, key: str) -> datetime.datetime:
    parsed = _parse_datetime_value(line.get(key))
    if parsed is None:
        raise ValueError(f"Invalid drift-rate time field {key}: {line.get(key)!r}")
    return parsed


def _calculate_drift_rate_from_line(
    line: dict,
    *,
    profile: _DriftRateCalculationProfile = _CANONICAL_DRIFT_RATE_PROFILE,
    t_start=_UNSET_DRIFT_TIME,
    t_end=_UNSET_DRIFT_TIME,
) -> DriftRateResult:
    """Build a result under an explicit endpoint compatibility policy."""

    if t_start is _UNSET_DRIFT_TIME:
        t_start = _drift_line_time(line, "t_start")
    if t_end is _UNSET_DRIFT_TIME:
        t_end = _drift_line_time(line, "t_end")
    if profile.strict_frequency_fields:
        f_start = _sample_float(line["f_start_mhz"])
        f_end = _sample_float(line["f_end_mhz"])
    else:
        f_start = _sample_float(line.get("f_start_mhz"))
        f_end = _sample_float(line.get("f_end_mhz"))
    warning = ""
    if profile.sort_endpoints_by_time and t_end < t_start:
        t_start, t_end = t_end, t_start
        f_start, f_end = f_end, f_start
        warning = "endpoints_sorted_by_time"
    duration_s = float((t_end - t_start).total_seconds())
    bandwidth_mhz = float(f_end - f_start)
    if not np.isfinite([f_start, f_end]).all() or min(f_start, f_end) <= 0:
        drift_rate = np.nan
        quality_flag = "invalid_frequency"
        warning = ";".join(
            filter(None, [warning, "frequency_must_be_finite_and_positive"])
        )
    elif abs(duration_s) <= profile.zero_duration_tolerance_s:
        if profile.zero_duration_policy == "raise":
            raise ValueError(
                f"Cannot calculate drift rate for zero-duration line: {line}"
            )
        drift_rate = np.nan
        quality_flag = "invalid_zero_duration"
        warning = ";".join(filter(None, [warning, "zero_duration"]))
    else:
        drift_rate = float(bandwidth_mhz / duration_s)
        quality_flag = "ok"
        if profile.warn_positive_drift and drift_rate > 0:
            warning = ";".join(filter(None, [warning, "positive_drift_rate"]))
    if profile.default_on_falsy:
        label = str(line.get("label") or profile.default_label)
        mode = str(line.get("mode") or profile.default_mode)
    else:
        label = str(line.get("label", profile.default_label))
        mode = str(line.get("mode", profile.default_mode))
    return DriftRateResult(
        label=label,
        mode=mode,
        t_start=t_start,
        t_end=t_end,
        f_start_mhz=f_start,
        f_end_mhz=f_end,
        drift_rate_mhz_s=drift_rate,
        abs_drift_rate_mhz_s=abs(drift_rate) if np.isfinite(drift_rate) else np.nan,
        duration_s=duration_s,
        bandwidth_mhz=bandwidth_mhz,
        color=str(line.get("color") or "white"),
        quality_flag=quality_flag,
        warning=warning,
    )


def calculate_drift_rate_from_line(line: dict) -> DriftRateResult:
    """Calculate MHz/s from UTC endpoints without modifying ``line``.

    Endpoints are sorted by time. Nonpositive/nonfinite frequencies and a zero
    duration produce NaN drift with an explicit quality flag; malformed times
    raise ValueError. A positive drift is retained with a diagnostic warning.
    """

    return _calculate_drift_rate_from_line(line)


def _mark_drift_range_warnings(results, cache):
    x_start, x_end = cache.display_time_nums
    f_min = float(np.nanmin(cache.freq))
    f_max = float(np.nanmax(cache.freq))
    for result in results:
        t1 = mdates.date2num(result.t_start)
        t2 = mdates.date2num(result.t_end)
        out = (
            max(t1, t2) < x_start
            or min(t1, t2) > x_end
            or max(result.f_start_mhz, result.f_end_mhz) < f_min
            or min(result.f_start_mhz, result.f_end_mhz) > f_max
        )
        if out and "out_of_range" not in result.warning:
            result.warning = ";".join(filter(None, [result.warning, "out_of_range"]))
    return results


def _spectrogram_coord_from_pixel(
    metadata: dict, x_pixel: float, y_pixel: float
) -> dict:
    bbox = metadata["axes_bbox_px"]
    left, right = float(bbox["left"]), float(bbox["right"])
    top, bottom = float(bbox["top"]), float(bbox["bottom"])
    if x_pixel < left or x_pixel > right or y_pixel < top or y_pixel > bottom:
        raise ValueError("click outside spectrogram axes")
    x_fraction = (float(x_pixel) - left) / max(right - left, 1e-12)
    y_fraction = (float(y_pixel) - top) / max(bottom - top, 1e-12)
    x_number = float(metadata["x_start_num"]) + x_fraction * (
        float(metadata["x_end_num"]) - float(metadata["x_start_num"])
    )
    f_max, f_min = float(metadata["f_max_mhz"]), float(metadata["f_min_mhz"])
    frequency = f_max - y_fraction * (f_max - f_min)
    return {
        "time_num": x_number,
        "time_iso": _datetime_iso_ms(_date_num_to_datetime(x_number)),
        "frequency_mhz": float(frequency),
    }


def assert_spectrogram_mapping_not_flipped(metadata: dict) -> None:
    bbox = metadata["axes_bbox_px"]
    top = _spectrogram_coord_from_pixel(metadata, bbox["left"], bbox["top"])
    bottom = _spectrogram_coord_from_pixel(metadata, bbox["left"], bbox["bottom"])
    if top["frequency_mhz"] <= bottom["frequency_mhz"]:
        raise AssertionError("Spectrogram selector mapping is flipped")


def save_drift_selection_json(path, lines, cache, cfg) -> None:
    """Persist explicitly supplied selection lines and source metadata."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_file": cache.source_file,
        "source_files": cache.source_files or [cache.source_file],
        "created_at": _datetime_iso_ms(datetime.datetime.now()),
        "spectrogram_time_start": _datetime_iso_ms(
            _date_num_to_datetime(cache.display_time_nums[0])
        ),
        "spectrogram_time_end": _datetime_iso_ms(
            _date_num_to_datetime(cache.display_time_nums[1])
        ),
        "spectrogram_f_start": float(cfg.get("spectrogram_f_start", np.nan)),
        "spectrogram_f_end": float(cfg.get("spectrogram_f_end", np.nan)),
        "lines": list(lines or []),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_drift_selection_json(path) -> list[dict]:
    """Load selection lines from either the current payload or a legacy list."""

    payload = _load_drift_selection_payload(path)
    return list(payload.get("lines", []) or [])


def _load_drift_selection_payload(path) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Drift-rate selection JSON does not exist: {path}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"lines": payload} if isinstance(payload, list) else dict(payload)


def get_or_load_drift_rate_results(
    cache, cfg, launch_func=None
) -> list[DriftRateResult]:
    """Load drift lines or use a caller-injected interactive selector."""

    if not cfg.get("enable_drift_rate_overlay", False):
        return []
    mode = str(cfg.get("drift_rate_mode", "off") or "off").lower()
    if mode == "off":
        return []
    selection_path = cfg.get("_drift_selection_cli_path") or _drift_output_path(
        cfg, "drift_rate_selection_json"
    )
    interactive = dict(cfg.get("drift_rate_interactive", {}) or {})
    launch_policy = str(interactive.get("launch_policy", "cli_only") or "cli_only")
    cache_key = (mode, os.path.abspath(selection_path), launch_policy)
    if cache_key in _DRIFT_RATE_RESULTS_CACHE:
        return _DRIFT_RATE_RESULTS_CACHE[cache_key]
    selection_exists = os.path.exists(selection_path)

    def load_lines() -> list[dict]:
        payload = _load_drift_selection_payload(selection_path)
        source_file = payload.get("source_file")
        if source_file and os.path.abspath(str(source_file)) != os.path.abspath(
            cache.source_file
        ):
            warnings.warn(
                "selection source differs from current spectrogram", stacklevel=2
            )
        return list(payload.get("lines", []) or [])

    if mode == "interactive_manual":
        should_launch = cfg.get("_select_drift_now", False) or launch_policy == "always"
        should_launch = should_launch or (
            launch_policy == "auto_if_missing" and not selection_exists
        )
        if should_launch:
            if launch_func is None:
                warnings.warn(
                    "interactive drift selection requires a solar_apps launch_func",
                    stacklevel=2,
                )
                return []
            lines = launch_func(cache, cfg)
        elif selection_exists:
            lines = load_lines()
        else:
            return []
    elif mode == "manual_json":
        if not selection_exists:
            warnings.warn(f"selection JSON not found: {selection_path}", stacklevel=2)
            return []
        lines = load_lines()
    else:
        return []
    results = _mark_drift_range_warnings(
        [calculate_drift_rate_from_line(line) for line in lines], cache
    )
    _DRIFT_RATE_RESULTS_CACHE[cache_key] = results
    return results


def overlay_drift_rate_results(ax, results, cfg) -> None:
    """Draw calculated line segments on an existing Matplotlib axis."""

    if not results:
        return
    colors = cfg.get("drift_rate_interactive", {}).get(
        "line_color_cycle", ["white", "cyan", "lime", "yellow", "magenta", "orange"]
    )
    for index, result in enumerate(results):
        color = result.color or colors[index % len(colors)]
        x_start, x_end = mdates.date2num(result.t_start), mdates.date2num(result.t_end)
        if cfg.get("draw_drift_rate_lines", True):
            ax.plot(
                [x_start, x_end],
                [result.f_start_mhz, result.f_end_mhz],
                color=color,
                linewidth=float(cfg.get("drift_rate_line_width", 2.2)),
                alpha=0.95,
                clip_on=True,
                zorder=4,
            )
        if cfg.get("draw_drift_rate_endpoints", True):
            ax.scatter(
                [x_start, x_end],
                [result.f_start_mhz, result.f_end_mhz],
                marker=cfg.get("drift_rate_endpoint_marker", "o"),
                s=float(cfg.get("drift_rate_endpoint_size", 30)),
                c=color,
                edgecolors="black",
                linewidths=0.5,
                clip_on=True,
                zorder=5,
            )


def save_drift_rate_diagnostics_once(results, cfg, source_file) -> None:
    """Append unique calculated results to the configured diagnostics CSV."""

    if not results:
        return
    csv_path = _drift_output_path(cfg, "drift_rate_diagnostics_csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
        except OSError:
            header = []
        if header and header != DRIFT_RATE_DIAGNOSTIC_FIELDS:
            shutil.copy2(csv_path, f"{csv_path}.bak")
            os.remove(csv_path)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DRIFT_RATE_DIAGNOSTIC_FIELDS, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        for result in results:
            key = (source_file, result.label, result.t_start, result.t_end)
            if key in _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS:
                continue
            _DRIFT_RATE_DIAGNOSTIC_WRITTEN_KEYS.add(key)
            writer.writerow(
                {
                    "source_file": source_file,
                    "label": result.label,
                    "mode": result.mode,
                    "t_start": _datetime_iso_ms(result.t_start),
                    "t_end": _datetime_iso_ms(result.t_end),
                    "f_start_mhz": result.f_start_mhz,
                    "f_end_mhz": result.f_end_mhz,
                    "duration_s": result.duration_s,
                    "bandwidth_mhz": result.bandwidth_mhz,
                    "drift_rate_mhz_s": result.drift_rate_mhz_s,
                    "abs_drift_rate_mhz_s": result.abs_drift_rate_mhz_s,
                    "color": result.color,
                    "quality_flag": result.quality_flag,
                    "warning": result.warning,
                }
            )
