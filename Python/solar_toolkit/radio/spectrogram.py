"""Dynamic spectrogram helpers for radio overlays.

English: This module owns CSO/dynamic-spectrogram caching, rebinning, and
overlay-panel drawing for radio source-map workflows.

中文：本模块负责射电源图工作流中的动态频谱缓存、重采样和叠加面板绘制。
"""

from __future__ import annotations

import datetime
import math
import os
import threading
import warnings
from dataclasses import dataclass

import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits

from .io import (
    index_range_from_time_values,
    index_range_from_values,
    parse_datetime_value,
)

__all__ = [
    "SpectrogramCache",
    "build_spectrogram_cache",
    "clear_spectrogram_cache",
    "get_spectrogram_cache",
    "overlay_spectrogram_panel",
    "resolve_spectrogram_time_window_multi",
]

_parse_datetime_value = parse_datetime_value
_index_range_from_values = index_range_from_values
_index_range_from_time_values = index_range_from_time_values

# Keyed cache: multiple radio events/configs can share one process without
# silently overwriting each other's rebinned dynamic spectrum.
_SPECTROGRAM_CACHE: dict[tuple, SpectrogramCache] = {}
_SPECTROGRAM_CACHE_LOCK = threading.Lock()
_SPECTROGRAM_CACHE_MAX_ITEMS = 8


@dataclass
class SpectrogramCache:
    """Cached, rebinned dynamic spectrum used by all radio-image frames."""

    data: np.ndarray
    time_nums: np.ndarray
    display_time_nums: tuple[float, float]
    time_datetimes: list[datetime.datetime]
    freq: np.ndarray
    title: str
    cmap: str
    vmin: float | None
    vmax: float | None
    cbar_label: str
    source_file: str
    source_files: list[str] | None = None


def _spectrogram_panel_enabled(cfg: dict) -> bool:
    return bool(cfg.get("enable_spectrogram_panel", False))


def _normalize_spectrogram_paths(cfg: dict) -> list[str]:
    """
    Return spectrogram FITS paths.

    Priority:
    1. cfg["spectrogram_file_paths"] if non-empty
    2. cfg["spectrogram_file_path"] as single-file fallback
    """
    raw_paths = cfg.get("spectrogram_file_paths")
    if isinstance(raw_paths, str):
        candidates = [raw_paths]
    elif isinstance(raw_paths, (list, tuple)):
        candidates = list(raw_paths)
    else:
        candidates = []

    candidates = [str(path).strip() for path in candidates if str(path).strip()]
    if not candidates:
        fallback = str(cfg.get("spectrogram_file_path") or "").strip()
        candidates = [fallback] if fallback else []

    paths = []
    for path in candidates:
        if os.path.isfile(path):
            paths.append(path)
        else:
            warnings.warn(
                f"Spectrogram FITS file does not exist and will be skipped: {path}",
                stacklevel=2,
            )
    return paths


def _read_spectrogram_file_metadata(path: str) -> dict:
    """
    Read only header/time/frequency metadata from one CSO spectrogram FITS.
    """
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        time_arr = np.ravel(hdul[1].data["time"]).astype(np.float64)
        freq_arr = np.ravel(hdul[1].data["frequency"]).astype(np.float32)
        dateobs = header.get("DATE-OBS") or header.get("DATE_OBS")
        if not dateobs:
            raise ValueError(f"Spectrogram FITS header lacks DATE-OBS/DATE_OBS: {path}")
        dt_base = datetime.datetime.fromisoformat(str(dateobs)[:10])
        if time_arr.size and time_arr[0] < 0:
            dt_base += datetime.timedelta(days=1)

    finite_time = time_arr[np.isfinite(time_arr)]
    if finite_time.size == 0:
        raise ValueError(f"Spectrogram FITS time axis has no finite values: {path}")
    file_start = dt_base + datetime.timedelta(seconds=float(np.nanmin(finite_time)))
    file_end = dt_base + datetime.timedelta(seconds=float(np.nanmax(finite_time)))
    print(f"  - {path}")
    print(f"    file_start / file_end: {file_start} / {file_end}")
    return {
        "path": path,
        "dt_base": dt_base,
        "time_arr": time_arr,
        "freq_arr": freq_arr,
        "file_start": file_start,
        "file_end": file_end,
    }


def resolve_spectrogram_time_window_multi(cfg, radio_time_range, file_metas):
    """
    Resolve requested spectrogram time window for one or multiple FITS files.
    """
    global_file_start = min(meta["file_start"] for meta in file_metas)
    global_file_end = max(meta["file_end"] for meta in file_metas)
    mode = str(cfg.get("spectrogram_time_display_mode", "user") or "user").lower()
    if mode not in {"user", "auto_radio", "full"}:
        mode = "user"

    if mode == "full":
        t_start, t_end = global_file_start, global_file_end
    elif mode == "auto_radio":
        if radio_time_range is not None:
            margin = datetime.timedelta(
                seconds=float(cfg.get("spectrogram_time_margin_seconds", 30.0))
            )
            t_start = radio_time_range[0] - margin
            t_end = radio_time_range[1] + margin
        else:
            t_start, t_end = global_file_start, global_file_end
    else:
        t_start = (
            _parse_datetime_value(cfg.get("spectrogram_time_start"))
            or global_file_start
        )
        t_end = (
            _parse_datetime_value(cfg.get("spectrogram_time_end")) or global_file_end
        )

    if t_start > t_end:
        t_start, t_end = t_end, t_start
    print(f"[Spectrogram time window] mode={mode}, start={t_start}, end={t_end}")
    return t_start, t_end, mode


def _spectrogram_overlap_segments(cfg, t_start, t_end, metas):
    segments = []
    for meta in metas:
        overlap_start = max(t_start, meta["file_start"])
        overlap_end = min(t_end, meta["file_end"])
        if overlap_start <= overlap_end:
            t1s = (overlap_start - meta["dt_base"]).total_seconds()
            t2s = (overlap_end - meta["dt_base"]).total_seconds()
            time_range = _index_range_from_time_values(meta["time_arr"], t1s, t2s)
            if time_range is None:
                continue
            ti0, ti1 = time_range
            actual_start = meta["dt_base"] + datetime.timedelta(
                seconds=float(meta["time_arr"][ti0])
            )
            actual_end = meta["dt_base"] + datetime.timedelta(
                seconds=float(meta["time_arr"][ti1])
            )
            print("  selected spectrogram segment:")
            print(f"    path: {meta['path']}")
            print(f"    overlap_start / overlap_end: {overlap_start} / {overlap_end}")
            print(f"    ti0 / ti1: {ti0} / {ti1}")
            print(
                "    actual selected datetime range: " f"{actual_start} / {actual_end}"
            )
            segments.append(
                {
                    "meta": meta,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "ti0": ti0,
                    "ti1": ti1,
                }
            )

    segments.sort(key=lambda item: item["overlap_start"])
    gaps = []
    tolerance = datetime.timedelta(seconds=1.0)
    cursor = t_start
    for segment in segments:
        if segment["overlap_start"] > cursor + tolerance:
            gaps.append((cursor, segment["overlap_start"]))
        if segment["overlap_end"] > cursor:
            cursor = segment["overlap_end"]
    if cursor < t_end - tolerance:
        gaps.append((cursor, t_end))

    if gaps:
        msg = "Spectrogram files do not fully cover requested time window; gaps: " + (
            "; ".join(f"{start} -- {end}" for start, end in gaps)
        )
        warnings.warn(msg, stacklevel=2)
        if cfg.get("spectrogram_disable_on_time_mismatch", True):
            return []
    return segments


def _rebinned_axis_values(
    arr: np.ndarray, i0: int, i1: int, bin_size: int
) -> np.ndarray:
    n_raw = i1 - i0 + 1
    bin_eff = max(1, min(int(bin_size), n_raw))
    n_trim = (n_raw // bin_eff) * bin_eff
    if n_trim <= 0:
        raise ValueError("Cannot rebin an empty spectrogram axis slice")
    return arr[i0 : i0 + n_trim].reshape(-1, bin_eff).mean(axis=1)


def _read_rebinned_plane(
    plane,
    fi0: int,
    fi1: int,
    ti0: int,
    ti1: int,
    f_bin: int,
    t_bin: int,
    chunk_mem_mb: int,
) -> np.ndarray:
    """Read a frequency-time slice from FITS memmap and downsample immediately."""
    n_freq_raw = fi1 - fi0 + 1
    n_time_raw = ti1 - ti0 + 1
    if n_freq_raw <= 0 or n_time_raw <= 0:
        raise ValueError(
            f"Invalid spectrogram slice shape: n_freq={n_freq_raw}, n_time={n_time_raw}"
        )
    f_bin_eff = max(1, min(int(f_bin), n_freq_raw))
    t_bin_eff = max(1, min(int(t_bin), n_time_raw))
    n_freq_trim = (n_freq_raw // f_bin_eff) * f_bin_eff
    n_time_trim = (n_time_raw // t_bin_eff) * t_bin_eff
    if n_freq_trim <= 0 or n_time_trim <= 0:
        raise ValueError(
            f"Cannot rebin empty spectrogram slice: n_freq_trim={n_freq_trim}, "
            f"n_time_trim={n_time_trim}"
        )
    n_freq_out = n_freq_trim // f_bin_eff
    n_time_out = n_time_trim // t_bin_eff
    bytes_per_col = max(n_freq_trim * 4, 1)
    cols_per_chunk = max(
        t_bin_eff,
        (int(chunk_mem_mb * 1e6 / bytes_per_col) // t_bin_eff) * t_bin_eff,
    )
    out = np.empty((n_freq_out, n_time_out), dtype=np.float32)
    out_col = 0
    for col0 in range(0, n_time_trim, cols_per_chunk):
        col1 = min(col0 + cols_per_chunk, n_time_trim)
        n_cols = ((col1 - col0) // t_bin_eff) * t_bin_eff
        if n_cols <= 0:
            continue
        chunk = np.array(
            plane[fi0 : fi0 + n_freq_trim, ti0 + col0 : ti0 + col0 + n_cols],
            dtype=np.float32,
        )
        n_t_chunk = n_cols // t_bin_eff
        rb = chunk.reshape(n_freq_out, f_bin_eff, n_t_chunk, t_bin_eff).mean(
            axis=(1, 3), dtype=np.float32
        )
        out[:, out_col : out_col + n_t_chunk] = rb
        out_col += n_t_chunk
    return out


def build_spectrogram_cache(
    cfg: dict, radio_time_range=None
) -> SpectrogramCache | None:
    """Load and downsample the large CSO dynamic spectrum once per script run."""
    paths = _normalize_spectrogram_paths(cfg)
    if not paths:
        warnings.warn(
            "Spectrogram panel enabled but no valid spectrogram FITS files were found.",
            stacklevel=2,
        )
        return None

    print("=" * 60)
    print("Loading dynamic spectrum once for all frames")
    print("Spectrogram files:")
    metas = [_read_spectrogram_file_metadata(path) for path in paths]
    metas.sort(key=lambda item: item["file_start"])
    paths = [meta["path"] for meta in metas]

    t_start, t_end, mode_used = resolve_spectrogram_time_window_multi(
        cfg, radio_time_range, metas
    )
    if t_start is None or t_end is None:
        return None
    if t_start >= t_end:
        raise ValueError(f"Invalid spectrogram time range: {t_start} >= {t_end}")
    print(f"  radio_time_range: {radio_time_range}")
    print(f"  requested t_start / t_end: {t_start} / {t_end}")

    segments = _spectrogram_overlap_segments(cfg, t_start, t_end, metas)
    if not segments:
        warnings.warn(
            "No spectrogram FITS file overlaps the requested time window; "
            "spectrogram panel disabled.",
            stacklevel=2,
        )
        return None

    f_lo = float(cfg.get("spectrogram_f_start", 80.0))
    f_hi = float(cfg.get("spectrogram_f_end", 340.0))
    first_freq_range = _index_range_from_values(
        segments[0]["meta"]["freq_arr"], f_lo, f_hi
    )
    if first_freq_range is None:
        warnings.warn(
            f"No spectrogram frequency data in requested range {f_lo} -- {f_hi} MHz.",
            stacklevel=2,
        )
        return None
    fi0_ref, fi1_ref = first_freq_range
    n_f = fi1_ref - fi0_ref + 1
    n_t_total = sum(segment["ti1"] - segment["ti0"] + 1 for segment in segments)
    t_target = int(cfg.get("spectrogram_rebin_t_target", 1000) or n_t_total)
    f_target = int(cfg.get("spectrogram_rebin_f_target", 700) or n_f)
    t_bin = max(1, n_t_total // max(t_target, 1))
    f_bin = max(1, n_f // max(f_target, 1))
    chunk_mem_mb = int(cfg.get("spectrogram_chunk_mem_mb", 64))
    requested_pol = str(cfg.get("spectrogram_polarization", "LL")).lower()

    print(f"  Frequency request: {f_lo:.1f} -- {f_hi:.1f} MHz")
    print(f"  Estimated raw slice: {n_f} x {n_t_total}; rebin f={f_bin}, t={t_bin}")

    data_parts = []
    time_num_parts = []
    time_datetime_parts = []
    freq_ref = None
    plot_kind = None

    for segment in segments:
        meta = segment["meta"]
        path = meta["path"]
        time_arr = meta["time_arr"]
        freq_arr = meta["freq_arr"]
        fi_range = _index_range_from_values(freq_arr, f_lo, f_hi)
        if fi_range is None:
            raise ValueError(
                f"Spectrogram file has no frequency data in requested range: {path}"
            )
        fi0, fi1 = fi_range
        ti0, ti1 = segment["ti0"], segment["ti1"]

        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            raw = hdul[0].data
            if raw.ndim == 2:
                planes = {"single": raw}
            elif raw.ndim == 3:
                polars = str(header.get("POLARIZA", "RL"))
                if polars == "RCP and LCP":
                    polars = "RL"
                planes = {}
                for i in range(raw.shape[0]):
                    key = polars[i].upper() * 2 if i < len(polars) else f"P{i}"
                    planes[key] = raw[i]
            else:
                raise ValueError(f"Unsupported spectrogram raw ndim={raw.ndim}")

            if requested_pol in {"ll", "lcp"}:
                plane = planes["LL"] if "LL" in planes else next(iter(planes.values()))
                data_i = _read_rebinned_plane(
                    plane, fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                plot_kind = "LL"
            elif requested_pol in {"rr", "rcp"}:
                plane = planes["RR"] if "RR" in planes else next(iter(planes.values()))
                data_i = _read_rebinned_plane(
                    plane, fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                plot_kind = "RR"
            elif (
                requested_pol in {"sum", "rr+ll", "ll+rr"}
                and "RR" in planes
                and "LL" in planes
            ):
                rr = _read_rebinned_plane(
                    planes["RR"], fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                ll = _read_rebinned_plane(
                    planes["LL"], fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                data_i = rr + ll
                plot_kind = "RR+LL"
            elif (
                requested_pol in {"ratio", "polarization_ratio"}
                and "RR" in planes
                and "LL" in planes
            ):
                rr = _read_rebinned_plane(
                    planes["RR"], fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                ll = _read_rebinned_plane(
                    planes["LL"], fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                denom = np.where(rr + ll == 0, np.nan, rr + ll)
                data_i = np.clip((rr - ll) / denom, -1.0, 1.0).astype(np.float32)
                plot_kind = "(RR-LL)/(RR+LL)"
            else:
                plane = next(iter(planes.values()))
                data_i = _read_rebinned_plane(
                    plane, fi0, fi1, ti0, ti1, f_bin, t_bin, chunk_mem_mb
                )
                plot_kind = next(iter(planes.keys()))

        freq_out_i = _rebinned_axis_values(freq_arr, fi0, fi1, f_bin).astype(np.float32)
        time_out_i = _rebinned_axis_values(time_arr, ti0, ti1, t_bin)
        time_datetimes_i = [
            meta["dt_base"] + datetime.timedelta(seconds=float(sec))
            for sec in time_out_i
        ]
        time_nums_i = mdates.date2num(time_datetimes_i)

        if freq_ref is None:
            freq_ref = freq_out_i
        elif not np.allclose(freq_out_i, freq_ref, rtol=0, atol=1e-3):
            raise ValueError(
                "Spectrogram frequency axes mismatch between files "
                f"{segments[0]['meta']['path']} and {path}"
            )

        data_parts.append(data_i)
        time_num_parts.append(np.asarray(time_nums_i, dtype=np.float64))
        time_datetime_parts.extend(time_datetimes_i)

    data = np.concatenate(data_parts, axis=1)
    time_nums = np.concatenate(time_num_parts)
    order = np.argsort(time_nums)
    data = data[:, order]
    time_nums = time_nums[order]
    time_datetimes = [time_datetime_parts[int(idx)] for idx in order]

    keep = np.concatenate([[True], np.diff(time_nums) > 1e-10])
    keep_idx = np.flatnonzero(keep)
    data = data[:, keep]
    time_nums = time_nums[keep]
    time_datetimes = [time_datetimes[int(idx)] for idx in keep_idx]
    freq_out = np.asarray(freq_ref, dtype=np.float32)

    if freq_out[0] > freq_out[-1]:
        freq_out = freq_out[::-1]
        data = data[::-1, :]

    print(f"  final concatenated shape: {data.shape}")
    print(
        "  final concatenated time range: "
        f"{time_datetimes[0]} / {time_datetimes[-1]}"
    )
    print(f"  final frequency range: {freq_out[0]:.3f} / {freq_out[-1]:.3f} MHz")

    use_log = bool(cfg.get("spectrogram_use_log10", True)) and requested_pol not in {
        "ratio",
        "polarization_ratio",
    }
    if use_log:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.log10(np.where(data > 0, data, np.nan))

    finite = data[np.isfinite(data)]
    vmin = cfg.get("spectrogram_vmin")
    vmax = cfg.get("spectrogram_vmax")
    if finite.size:
        if vmin is None:
            vmin = float(np.nanpercentile(finite, 1.0))
        if vmax is None:
            vmax = float(np.nanpercentile(finite, 99.5))
    title = cfg.get("spectrogram_title") or f"CSO dynamic spectrum {plot_kind}"
    cbar_label = cfg.get("spectrogram_colorbar_label") or (
        "log10 intensity" if use_log else "intensity"
    )
    print("  Spectrogram cache ready; subsequent frames reuse this cached array.")
    print("=" * 60)
    return SpectrogramCache(
        data=np.asarray(data, dtype=np.float32),
        time_nums=np.asarray(time_nums, dtype=np.float64),
        display_time_nums=(mdates.date2num(t_start), mdates.date2num(t_end)),
        time_datetimes=time_datetimes,
        freq=np.asarray(freq_out, dtype=np.float32),
        title=f"{title} | {plot_kind}",
        cmap=cfg.get("spectrogram_cmap", "jet"),
        vmin=vmin,
        vmax=vmax,
        cbar_label=cbar_label,
        source_file=" | ".join(paths),
        source_files=paths,
    )


def _spectrogram_cache_key(cfg: dict, radio_time_range=None) -> tuple | None:
    """Return a stable cache key for one spectrogram configuration window."""
    if not _spectrogram_panel_enabled(cfg):
        return None
    paths = _normalize_spectrogram_paths(cfg)
    if not paths:
        return None
    return (
        tuple(str(path) for path in paths),
        float(cfg.get("spectrogram_f_start", 80.0)),
        float(cfg.get("spectrogram_f_end", 340.0)),
        int(cfg.get("spectrogram_rebin_t_target", 1000) or 1000),
        int(cfg.get("spectrogram_rebin_f_target", 700) or 700),
        str(cfg.get("spectrogram_polarization", "LL")).lower(),
        None if radio_time_range is None else repr(radio_time_range),
    )


def get_spectrogram_cache(cfg: dict, radio_time_range=None) -> SpectrogramCache | None:
    """Return a cached rebinned dynamic spectrum for one config, building if needed."""
    key = _spectrogram_cache_key(cfg, radio_time_range)
    if key is None:
        return None
    with _SPECTROGRAM_CACHE_LOCK:
        cached = _SPECTROGRAM_CACHE.get(key)
        if cached is not None:
            return cached
        cache = build_spectrogram_cache(cfg, radio_time_range)
        if cache is not None:
            if len(_SPECTROGRAM_CACHE) >= _SPECTROGRAM_CACHE_MAX_ITEMS:
                _SPECTROGRAM_CACHE.pop(next(iter(_SPECTROGRAM_CACHE)))
            _SPECTROGRAM_CACHE[key] = cache
        return cache


def clear_spectrogram_cache() -> None:
    """Drop all cached dynamic spectra, e.g. between independent events."""
    with _SPECTROGRAM_CACHE_LOCK:
        _SPECTROGRAM_CACHE.clear()


def _spectrogram_time_locator(cfg: dict, span_seconds: float):
    max_ticks = max(2, int(cfg.get("spectrogram_max_time_ticks", 8) or 8))
    if not cfg.get("spectrogram_auto_time_locator", True):
        interval = max(1, int(cfg.get("spectrogram_major_tick_seconds", 10) or 10))
        interval = max(interval, int(math.ceil(span_seconds / max_ticks)) or interval)
        return mdates.SecondLocator(interval=interval)
    if span_seconds <= 300:
        requested = max(1, int(cfg.get("spectrogram_major_tick_seconds", 10) or 10))
        interval = max(requested, int(math.ceil(span_seconds / max_ticks)) or 1)
        return mdates.SecondLocator(interval=interval)
    if span_seconds <= 3600:
        interval = max(1, int(math.ceil(span_seconds / 60.0 / max_ticks)))
        return mdates.MinuteLocator(interval=interval)
    if span_seconds <= 86400:
        interval = max(1, int(math.ceil(span_seconds / 3600.0 / max_ticks)))
        return mdates.HourLocator(interval=interval)
    return mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)


def _date_num_to_datetime(value: float) -> datetime.datetime:
    return mdates.num2date(float(value)).replace(tzinfo=None)


def _spectrogram_display_data_extent(cache):
    f_min = float(np.nanmin(cache.freq))
    f_max = float(np.nanmax(cache.freq))
    data = cache.data
    if cache.freq.size >= 2 and float(cache.freq[0]) > float(cache.freq[-1]):
        data = np.flipud(data)
    x_start, x_end = cache.display_time_nums
    return data, [x_start, x_end, f_min, f_max], f_min, f_max


def overlay_spectrogram_panel(
    ax,
    cfg: dict,
    current_time: datetime.datetime | None,
    cache: SpectrogramCache | None = None,
):
    """Draw cached dynamic spectrum and the current-time vertical dashed line."""
    if cache is None:
        cache = get_spectrogram_cache(cfg)
    if cache is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Spectrogram unavailable", ha="center", va="center")
        return None
    x_start, x_end = cache.display_time_nums
    display_data, extent, f_min, f_max = _spectrogram_display_data_extent(cache)
    im = ax.imshow(
        display_data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cache.cmap,
        vmin=cache.vmin,
        vmax=cache.vmax,
    )
    if cfg.get("enable_drift_rate_overlay", False):
        from .drift_rate import (
            get_or_load_drift_rate_results,
            overlay_drift_rate_results,
            save_drift_rate_diagnostics_once,
        )

        drift_results = get_or_load_drift_rate_results(cache, cfg)
        overlay_drift_rate_results(ax, drift_results, cfg)
        if cfg.get("save_drift_rate_diagnostics", False):
            save_drift_rate_diagnostics_once(drift_results, cfg, cache.source_file)
    if current_time is not None:
        current_num = mdates.date2num(current_time)
        in_range = x_start <= current_num <= x_end
        if in_range or not cfg.get("spectrogram_clip_current_time_line", True):
            ax.axvline(
                current_num,
                color=cfg.get("spectrogram_line_color", "white"),
                linestyle=cfg.get("spectrogram_line_style", "--"),
                linewidth=cfg.get("spectrogram_line_width", 1.6),
                alpha=cfg.get("spectrogram_line_alpha", 0.95),
                zorder=7,
            )
        elif cfg.get("spectrogram_show_out_of_range_time_note", True):
            ax.text(
                0.02,
                0.94,
                "Current frame time outside displayed spectrum range",
                transform=ax.transAxes,
                fontsize=max(cfg.get("annotation_fontsize", 20) - 10, 8),
                color=cfg.get("tick_color", "black"),
                va="top",
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none"),
            )
    ax.set_title(cache.title, fontsize=max(cfg.get("title_fontsize", 24) - 8, 9), pad=4)
    ax.set_ylabel(
        "Frequency (MHz)", fontsize=max(cfg.get("label_fontsize", 28) - 12, 9)
    )
    ax.set_xlabel("Time (UT)", fontsize=max(cfg.get("label_fontsize", 28) - 12, 9))
    ax.xaxis_date()
    span_seconds = float((x_end - x_start) * 86400.0)
    if np.isfinite(span_seconds) and span_seconds > 0:
        ax.xaxis.set_major_locator(_spectrogram_time_locator(cfg, span_seconds))
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(cfg.get("spectrogram_xtick_format", "%H:%M:%S"))
    )
    ax.tick_params(
        labelsize=max(cfg.get("tick_fontsize", 22) - 10, 8),
        colors=cfg.get("tick_color", "black"),
    )
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(f_min, f_max)
    if cfg.get("spectrogram_draw_colorbar", True):
        cbar = ax.figure.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label(
            cache.cbar_label, fontsize=max(cfg.get("tick_fontsize", 22) - 11, 8)
        )
        cbar.ax.tick_params(labelsize=max(cfg.get("tick_fontsize", 22) - 12, 8))
    return im
