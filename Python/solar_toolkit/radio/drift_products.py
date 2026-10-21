"""Persistent drift-rate selection science products.

English: Save drift-selection previews, CSV tables, metadata JSON, and cutout
figures without depending on the historical plotting monolith.

中文：保存漂移率选点预览图、CSV 表格、元数据 JSON 和局部裁剪图，避免新
代码继续依赖历史大型绘图脚本。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._image_naming import build_radio_image_filename, chronological_rows
from .io import (
    DRIFT_RATE_DIAGNOSTIC_FIELDS,
    ensure_output_dir,
    parse_datetime_value,
    parse_timestamp_value,
    write_csv_rows,
    write_json_file,
)

__all__ = [
    "ANNOTATED_PREVIEW_NAME",
    "DEFAULT_DRIFT_PRODUCT_CONFIG",
    "METADATA_JSON_NAME",
    "RAW_PREVIEW_NAME",
    "SELECTION_CSV_NAME",
    "save_drift_selection_artifacts",
]

RAW_PREVIEW_NAME = "spectrogram_drift_rate_selection_preview_raw.png"
ANNOTATED_PREVIEW_NAME = "spectrogram_drift_rate_selection_preview_annotated.png"
SELECTION_CSV_NAME = "spectrogram_drift_rate_selection_points.csv"
METADATA_JSON_NAME = "spectrogram_drift_rate_selection_metadata.json"


DEFAULT_DRIFT_PRODUCT_CONFIG = {
    "enable": True,
    "save_raw_preview": True,
    "save_annotated_preview": True,
    "save_selection_csv": True,
    "save_metadata_json": True,
    "save_per_drift_cutouts": True,
    "cutout_time_padding_s": 2.0,
    "cutout_frequency_padding_mhz": 20.0,
    "annotate_drift_rate": True,
    "annotate_endpoints": True,
    "preserve_existing": True,
    "dpi": 200,
    "output_subdir": "drift_selection",
}


def save_drift_selection_artifacts(
    spectrogram_data,
    time_axis,
    frequency_axis_mhz,
    selections,
    output_dir,
    source_file=None,
    config=None,
):
    """Save reproducible drift-selection previews, tables, metadata, and cutouts."""
    cfg = dict(DEFAULT_DRIFT_PRODUCT_CONFIG)
    cfg.update(dict(config or {}))
    if not cfg.get("enable", True):
        return {"status": "skipped", "reason": "disabled", "saved": []}

    out_dir = ensure_output_dir(output_dir)
    display = _prepare_display(spectrogram_data, time_axis, frequency_axis_mhz)
    rows = _normalize_selections(selections)
    batch_generated_at = datetime.now(timezone.utc)
    display_frame = pd.DataFrame(
        {
            "obs_time": [
                _num_to_iso(display["extent"][0]),
                _num_to_iso(display["extent"][1]),
            ]
        }
    )

    saved = []
    next_sequence = 1
    raw_name = build_radio_image_filename(
        display_frame,
        sequence=next_sequence,
        product="dynamic_spectrum",
        qualifiers="raw",
        generated_at=batch_generated_at,
    )
    if cfg.get("save_raw_preview", True):
        next_sequence += 1
    annotated_name = build_radio_image_filename(
        display_frame,
        sequence=next_sequence,
        product="dynamic_spectrum",
        qualifiers="annotated",
        generated_at=batch_generated_at,
    )
    if cfg.get("save_annotated_preview", True):
        next_sequence += 1
    raw_path = out_dir / raw_name
    annotated_path = out_dir / annotated_name
    csv_path = out_dir / SELECTION_CSV_NAME
    metadata_path = out_dir / METADATA_JSON_NAME

    raw_meta = {}
    if cfg.get("save_raw_preview", True):
        raw_meta = _plot_spectrogram(
            display,
            raw_path,
            cfg,
            annotated_rows=None,
            title="Drift-rate selection preview",
        )
        if raw_meta.get("saved"):
            saved.append(str(raw_path))

    annotated_meta = raw_meta
    if cfg.get("save_annotated_preview", True):
        annotated_meta = _plot_spectrogram(
            display,
            annotated_path,
            cfg,
            annotated_rows=rows,
            title="Drift-rate selection preview with selected drift lines",
        )
        if annotated_meta.get("saved"):
            saved.append(str(annotated_path))

    if cfg.get("save_selection_csv", True) and _should_write(csv_path, cfg):
        write_csv_rows(csv_path, rows, DRIFT_RATE_DIAGNOSTIC_FIELDS[1:])
        saved.append(str(csv_path))

    cutout_paths = []
    if cfg.get("save_per_drift_cutouts", True):
        cutout_paths = _save_cutouts(
            display,
            chronological_rows(rows),
            out_dir / "cutouts",
            cfg,
            sequence_start=next_sequence,
            generated_at=batch_generated_at,
        )
        saved.extend(str(path) for path in cutout_paths)

    metadata = _metadata_payload(
        display,
        rows,
        source_file,
        raw_path,
        annotated_path,
        annotated_meta,
        cfg,
        cutout_paths,
    )
    if cfg.get("save_metadata_json", True) and _should_write(metadata_path, cfg):
        write_json_file(metadata_path, metadata)
        saved.append(str(metadata_path))

    return {
        "status": "saved" if saved else "skipped",
        "reason": "" if saved else "preserve_existing",
        "output_dir": str(out_dir),
        "raw_preview_png": str(raw_path),
        "annotated_preview_png": str(annotated_path),
        "selection_csv": str(csv_path),
        "metadata_json": str(metadata_path),
        "cutouts": [str(path) for path in cutout_paths],
        "saved": saved,
    }


def _prepare_display(spectrogram_data, time_axis, frequency_axis_mhz):
    data = np.asarray(spectrogram_data, dtype=float)
    time_nums = _time_axis_to_mpl_nums(time_axis)
    freq = np.asarray(frequency_axis_mhz, dtype=float).ravel()

    if data.ndim != 2:
        raise ValueError("spectrogram_data must be a 2D array")
    if data.shape[1] != len(time_nums) and data.shape[0] == len(time_nums):
        data = data.T
    if data.shape[0] != len(freq) or data.shape[1] != len(time_nums):
        raise ValueError(
            "spectrogram_data shape must match frequency_axis_mhz x time_axis"
        )

    order = "ascending" if len(freq) < 2 or freq[0] <= freq[-1] else "descending"
    display_data = data
    display_freq = freq
    if order == "descending":
        display_data = data[::-1, :]
        display_freq = freq[::-1]

    finite_t = time_nums[np.isfinite(time_nums)]
    finite_f = display_freq[np.isfinite(display_freq)]
    if finite_t.size == 0 or finite_f.size == 0:
        raise ValueError("time_axis and frequency_axis_mhz must contain finite values")

    return {
        "data": display_data,
        "time_nums": time_nums,
        "frequency_mhz": display_freq,
        "freq_axis_order": order,
        "extent": [
            float(np.nanmin(finite_t)),
            float(np.nanmax(finite_t)),
            float(np.nanmin(finite_f)),
            float(np.nanmax(finite_f)),
        ],
    }


def _plot_spectrogram(display, path, cfg, annotated_rows=None, title=""):
    saved = False
    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (10, 6)), dpi=int(cfg.get("dpi", 200))
    )
    im = ax.imshow(
        display["data"],
        extent=display["extent"],
        origin="lower",
        aspect="auto",
        cmap=cfg.get("cmap", "viridis"),
        vmin=cfg.get("vmin"),
        vmax=cfg.get("vmax"),
    )
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(title)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(cfg.get("time_format", "%H:%M:%S"))
    )
    fig.colorbar(im, ax=ax, label=cfg.get("colorbar_label", "Intensity"))
    if annotated_rows:
        _draw_selection_rows(ax, annotated_rows, cfg)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    bbox = ax.get_window_extent()
    metadata = {
        "fig_width_px": int(width_px),
        "fig_height_px": int(height_px),
        "axes_bbox_px": {
            "left": float(bbox.x0),
            "right": float(bbox.x1),
            "top": float(height_px - bbox.y1),
            "bottom": float(height_px - bbox.y0),
        },
        "saved": False,
    }
    if _should_write(path, cfg):
        fig.savefig(path, dpi=fig.dpi, bbox_inches=None)
        saved = True
    plt.close(fig)
    metadata["saved"] = saved
    return metadata


def _draw_selection_rows(ax, rows, cfg):
    for idx, row in enumerate(rows):
        color = row.get("color") or f"C{idx % 10}"
        t1 = _datetime_to_num(parse_datetime_value(row.get("t_start")))
        t2 = _datetime_to_num(parse_datetime_value(row.get("t_end")))
        f1 = _float_or_nan(row.get("f_start_mhz"))
        f2 = _float_or_nan(row.get("f_end_mhz"))
        if not (
            np.isfinite(t1) and np.isfinite(t2) and np.isfinite(f1) and np.isfinite(f2)
        ):
            continue
        ax.plot([t1, t2], [f1, f2], color=color, linewidth=1.6, zorder=4)
        if cfg.get("annotate_endpoints", True):
            ax.scatter(
                [t1, t2], [f1, f2], s=22, color=color, edgecolors="black", zorder=5
            )
        xm = 0.5 * (t1 + t2)
        ym = 0.5 * (f1 + f2)
        label = str(row.get("label") or f"drift_{idx + 1:03d}")
        if cfg.get("annotate_drift_rate", True):
            rate = _float_or_nan(row.get("drift_rate_mhz_s"))
            if np.isfinite(rate):
                label = f"{label}\ndf/dt = {rate:.1f} MHz/s"
        ax.annotate(
            label,
            xy=(xm, ym),
            xytext=(7, 6 + 4 * (idx % 3)),
            textcoords="offset points",
            color=color,
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"),
            zorder=6,
        )


def _save_cutouts(
    display,
    rows,
    output_dir,
    cfg,
    *,
    sequence_start,
    generated_at,
):
    out_dir = ensure_output_dir(output_dir)
    saved = []
    time_pad_days = float(cfg.get("cutout_time_padding_s", 2.0) or 0.0) / 86400.0
    freq_pad = float(cfg.get("cutout_frequency_padding_mhz", 20.0) or 0.0)
    x0, x1, y0, y1 = display["extent"]
    for idx, row in enumerate(rows, start=1):
        label = str(row.get("label") or f"drift_{idx:03d}")
        row_frame = pd.DataFrame(
            {
                "obs_time": [row.get("t_start"), row.get("t_end")],
            }
        )
        path = out_dir / build_radio_image_filename(
            row_frame,
            sequence=sequence_start + idx - 1,
            product="drift_rate_cutout",
            qualifiers=label or f"drift_{idx:03d}",
            generated_at=generated_at,
        )
        if not _should_write(path, cfg):
            continue
        t1 = _datetime_to_num(parse_datetime_value(row.get("t_start")))
        t2 = _datetime_to_num(parse_datetime_value(row.get("t_end")))
        f1 = _float_or_nan(row.get("f_start_mhz"))
        f2 = _float_or_nan(row.get("f_end_mhz"))
        if not (
            np.isfinite(t1) and np.isfinite(t2) and np.isfinite(f1) and np.isfinite(f2)
        ):
            continue
        fig, ax = plt.subplots(
            figsize=cfg.get("cutout_figsize", (5, 3.5)), dpi=int(cfg.get("dpi", 200))
        )
        ax.imshow(
            display["data"],
            extent=display["extent"],
            origin="lower",
            aspect="auto",
            cmap=cfg.get("cmap", "viridis"),
            vmin=cfg.get("vmin"),
            vmax=cfg.get("vmax"),
        )
        _draw_selection_rows(ax, [row], cfg)
        ax.set_xlim(
            max(min(t1, t2) - time_pad_days, x0), min(max(t1, t2) + time_pad_days, x1)
        )
        ax.set_ylim(max(min(f1, f2) - freq_pad, y0), min(max(f1, f2) + freq_pad, y1))
        ax.set_xlabel("Time (UT)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(label)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(cfg.get("time_format", "%H:%M:%S"))
        )
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        saved.append(path)
    return saved


def _metadata_payload(
    display,
    rows,
    source_file,
    raw_path,
    png_path,
    plot_meta,
    cfg,
    cutout_paths,
):
    x_start, x_end, f_min, f_max = display["extent"]
    return {
        "source_file": str(source_file or ""),
        "png_path": str(png_path),
        "x_start_iso": _num_to_iso(x_start),
        "x_end_iso": _num_to_iso(x_end),
        "f_min_mhz": f_min,
        "f_max_mhz": f_max,
        "freq_axis_order": display["freq_axis_order"],
        "fig_width_px": int(plot_meta.get("fig_width_px", 0) or 0),
        "fig_height_px": int(plot_meta.get("fig_height_px", 0) or 0),
        "axes_bbox_px": plot_meta.get("axes_bbox_px", {}),
        "display": {
            "cmap": cfg.get("cmap", "viridis"),
            "vmin": cfg.get("vmin"),
            "vmax": cfg.get("vmax"),
            "colorbar_label": cfg.get("colorbar_label", "Intensity"),
        },
        "raw_preview_png": Path(raw_path).name,
        "annotated_preview_png": Path(png_path).name,
        "selection_csv": SELECTION_CSV_NAME,
        "cutouts": [str(Path(path).name) for path in cutout_paths],
        "selections": rows,
        "config": {
            "cutout_time_padding_s": cfg.get("cutout_time_padding_s"),
            "cutout_frequency_padding_mhz": cfg.get("cutout_frequency_padding_mhz"),
            "annotate_drift_rate": cfg.get("annotate_drift_rate"),
            "annotate_endpoints": cfg.get("annotate_endpoints"),
        },
    }


def _normalize_selections(selections):
    rows = []
    if selections is None:
        return rows
    if isinstance(selections, pd.DataFrame):
        iterable = selections.to_dict("records")
    else:
        iterable = list(selections or [])
    for idx, item in enumerate(iterable, start=1):
        row = _selection_to_dict(item, idx)
        rows.append(row)
    return rows


def _selection_to_dict(item, idx):
    if hasattr(item, "__dict__") and not isinstance(item, dict):
        raw = dict(item.__dict__)
    else:
        raw = dict(item or {})
    label = _text_or_empty(raw.get("label")) or f"drift_{idx:03d}"
    t_start = parse_datetime_value(raw.get("t_start"))
    t_end = parse_datetime_value(raw.get("t_end"))
    f_start = _float_or_nan(raw.get("f_start_mhz"))
    f_end = _float_or_nan(raw.get("f_end_mhz"))
    duration = _float_or_nan(raw.get("duration_s"))
    if not np.isfinite(duration) and t_start is not None and t_end is not None:
        start_stamp = parse_timestamp_value(raw.get("t_start"))
        end_stamp = parse_timestamp_value(raw.get("t_end"))
        duration = (end_stamp.tai_ns - start_stamp.tai_ns) / 1_000_000_000
    bandwidth = _float_or_nan(raw.get("bandwidth_mhz"))
    if not np.isfinite(bandwidth) and np.isfinite(f_start) and np.isfinite(f_end):
        bandwidth = f_end - f_start
    drift_rate = _float_or_nan(raw.get("drift_rate_mhz_s"))
    if (
        not np.isfinite(drift_rate)
        and np.isfinite(duration)
        and abs(duration) > 1e-12
        and np.isfinite(bandwidth)
    ):
        drift_rate = bandwidth / duration
    warning = _text_or_empty(raw.get("warning"))
    quality_flag = _text_or_empty(raw.get("quality_flag"))
    if not quality_flag:
        quality_flag = "ok" if np.isfinite(drift_rate) else "invalid"
    return {
        "label": label,
        "mode": _text_or_empty(raw.get("mode"))
        or _text_or_empty(raw.get("selection_mode"))
        or "manual",
        "t_start": _datetime_iso(t_start),
        "t_end": _datetime_iso(t_end),
        "f_start_mhz": f_start,
        "f_end_mhz": f_end,
        "duration_s": duration,
        "bandwidth_mhz": bandwidth,
        "drift_rate_mhz_s": drift_rate,
        "abs_drift_rate_mhz_s": abs(drift_rate) if np.isfinite(drift_rate) else np.nan,
        "color": _text_or_empty(raw.get("color")),
        "quality_flag": quality_flag,
        "warning": warning,
    }


def _time_axis_to_mpl_nums(time_axis):
    values = []
    if time_axis is None:
        iterable = []
    else:
        iterable = list(time_axis)
    for value in iterable:
        if isinstance(value, (int, float, np.integer, np.floating)):
            values.append(float(value))
            continue
        parsed = parse_datetime_value(value)
        if parsed is None and hasattr(value, "to_pydatetime"):
            parsed = value.to_pydatetime().replace(tzinfo=None)
        if parsed is None:
            values.append(np.nan)
        else:
            values.append(mdates.date2num(parsed))
    return np.asarray(values, dtype=float)


def _datetime_to_num(value):
    if value is None:
        return np.nan
    return float(mdates.date2num(value))


def _datetime_iso(value):
    if value is None:
        return ""
    return value.isoformat(timespec="milliseconds")


def _num_to_iso(value):
    if not np.isfinite(value):
        return ""
    return (
        mdates.num2date(float(value))
        .replace(tzinfo=None)
        .isoformat(timespec="milliseconds")
    )


def _float_or_nan(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _text_or_empty(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _should_write(path, cfg):
    candidate = Path(path)
    if bool(cfg.get("preserve_existing", True)) and candidate.exists():
        return False
    ensure_output_dir(candidate.parent or ".")
    return True
