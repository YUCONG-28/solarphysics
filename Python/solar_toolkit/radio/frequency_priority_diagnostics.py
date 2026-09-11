"""Frequency-priority diagnostic products for Gaussian/Newkirk comparisons."""

# ruff: noqa: I001

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import ensure_output_dir, parse_datetime_value, truthy
from ._image_naming import build_radio_image_filename
from .newkirk import (
    SPEED_OF_LIGHT_KM_S,
    effective_density_factor,
    newkirk_assumption_label,
    newkirk_height_from_frequency_mhz,
    newkirk_radius_from_frequency_mhz,
    newkirk_speed_from_drift_rate,
    plasma_density_from_frequency_mhz,
)

__all__ = [
    "DEFAULT_COMPARISON_FREQUENCIES_MHZ",
    "apply_frequency_priority_drift_matching",
    "build_frequency_priority_summary",
    "build_newkirk_physical_consistency_report",
    "build_selected_band_newkirk_height_speed_table",
    "format_newkirk_case_label",
    "model_label",
    "plot_drift_frequency_band_matching",
    "plot_event_gaussian_newkirk_height_comparison",
    "plot_event_newkirk_speed_frequency",
    "plot_frequency_priority_summary",
    "plot_gaussian_center_by_frequency_facets",
    "plot_gaussian_center_trajectory_by_frequency",
    "plot_height_time_by_frequency_facets",
    "resolve_comparison_frequencies",
    "resolve_drift_frequency_tolerance",
    "save_frequency_priority_summary_csv",
    "save_newkirk_physical_consistency_report",
    "save_selected_band_newkirk_height_speed_table",
    "write_frequency_priority_dashboard",
]

DEFAULT_COMPARISON_FREQUENCIES_MHZ = [149, 164, 190, 205, 223, 238]


def model_label(multiplier, harmonic) -> str:
    return f"{float(multiplier):g}× Newkirk, s={float(harmonic):g}"


def format_newkirk_case_label(multiplier, harmonic, compact=False) -> str:
    multiplier_value = float(multiplier)
    harmonic_value = float(harmonic)
    factor = effective_density_factor(multiplier_value, harmonic_value)
    mode = "fundamental" if harmonic_value == 1.0 else "harmonic"
    harmonic_text = f"H={harmonic_value:g}"
    if compact:
        return (
            f"{multiplier_value:g}x Newkirk\n"
            f"{mode} {harmonic_text}\n"
            f"N*s^2={factor:g}"
        )
    return f"{multiplier_value:g}x Newkirk / {mode} ({harmonic_text}), N*s^2={factor:g}"


def _short_newkirk_case_label(multiplier, harmonic, reference=False) -> str:
    mode = "F" if float(harmonic) == 1.0 else "H"
    label = f"{float(multiplier):g}x {mode}"
    return f"{label} ref" if reference else label


def _format_drift_rate(value) -> str:
    drift_rate = _float_or_nan(value)
    if not np.isfinite(drift_rate):
        return ""
    return f"df/dt={drift_rate:.2f} MHz/s"


def _format_newkirk_label_from_row(row, fallback="Newkirk model") -> str:
    multiplier = _float_or_nan(row.get("newkirk_multiplier"))
    harmonic = _float_or_nan(row.get("harmonic", row.get("newkirk_harmonic")))
    if np.isfinite(multiplier) and np.isfinite(harmonic):
        return format_newkirk_case_label(multiplier, harmonic)
    text = str(row.get("newkirk_assumption_label") or fallback)
    match = re.search(
        r"(?P<multiplier>\d+(?:\.\d+)?)x\s+Newkirk,\s+H=(?P<harmonic>\d+(?:\.\d+)?)",
        text,
    )
    if match:
        return format_newkirk_case_label(
            match.group("multiplier"), match.group("harmonic")
        )
    return text


def build_frequency_priority_summary(
    height_df, gaussian_df, drift_df=None, config=None
):
    """Build frequency-band summaries using only configured Gaussian bands."""
    cfg = dict(config or {})
    frequencies = resolve_comparison_frequencies(cfg, height_df, gaussian_df)
    height = _clean_height_table(height_df)
    gaussian = _clean_gaussian_table(gaussian_df)
    drift = pd.DataFrame(drift_df).copy() if drift_df is not None else pd.DataFrame()
    height = _filter_to_comparison_frequencies(height, frequencies, "frequency_mhz")
    gaussian = _filter_to_comparison_frequencies(gaussian, frequencies, "frequency_mhz")
    if "height_valid" in height:
        height = height[height["height_valid"].fillna(False).map(truthy)].copy()

    unique_gaussian_height = height.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    )
    gaussian_height_summary = _frequency_summary(
        unique_gaussian_height, frequencies, "gaussian_height_rsun"
    )
    residual_summary = _residual_summary(height, frequencies)
    model_ranking = _model_ranking(height)
    center_summary = _center_summary(gaussian, frequencies)
    drift_match_summary = _drift_match_summary(
        apply_frequency_priority_drift_matching(gaussian, drift, cfg), frequencies
    )
    missing = [
        float(freq)
        for freq in frequencies
        if _frequency_group_empty(gaussian, float(freq))
        and _frequency_group_empty(height, float(freq))
    ]
    return {
        "comparison_frequency_mhz": [float(freq) for freq in frequencies],
        "missing_frequency_mhz": missing,
        "gaussian_height_summary": gaussian_height_summary,
        "residual_summary": residual_summary,
        "model_ranking": model_ranking,
        "center_summary": center_summary,
        "drift_match_summary": drift_match_summary,
    }


def apply_frequency_priority_drift_matching(gaussian_df, drift_df, config=None):
    """Assign drift labels/source types to Gaussian rows only on confident matches."""
    cfg = dict(config or {})
    gaussian = _clean_gaussian_table(gaussian_df)
    drift = pd.DataFrame(drift_df).copy()
    if gaussian.empty:
        return gaussian
    frequencies = resolve_comparison_frequencies(cfg, gaussian, gaussian)
    freq_tol = resolve_drift_frequency_tolerance(cfg, frequencies)
    time_tol = float(cfg.get("drift_time_tolerance_s", 0.75) or 0.0)
    label_map = {
        str(key): str(value)
        for key, value in (cfg.get("drift_source_type_map") or {}).items()
    }

    rows = []
    for _, row in gaussian.iterrows():
        candidates = []
        row_time = parse_datetime_value(row.get("time"))
        row_freq = _float_or_nan(row.get("frequency_mhz"))
        if row_time is not None and np.isfinite(row_freq):
            for _, drift_row in drift.iterrows():
                expected = _frequency_on_drift_line(row_time, drift_row, time_tol)
                if expected is None or not np.isfinite(expected):
                    continue
                delta_f = abs(float(row_freq) - expected)
                if delta_f <= freq_tol:
                    label = str(drift_row.get("label") or "").strip()
                    candidates.append(
                        {
                            "label": label,
                            "source_type": label_map.get(label, "")
                            or _drift_row_source_type(drift_row),
                            "distance": delta_f / max(freq_tol, 1e-9),
                        }
                    )
        out = row.to_dict()
        if not candidates:
            out["drift_label"] = ""
            out["source_type"] = _existing_source_type(row)
            out["drift_match_warning"] = ""
        else:
            candidates = sorted(candidates, key=lambda item: item["distance"])
            if (
                len(candidates) > 1
                and abs(candidates[1]["distance"] - candidates[0]["distance"]) < 0.05
            ):
                out["drift_label"] = ""
                out["source_type"] = _existing_source_type(row)
                out["drift_match_warning"] = "ambiguous_drift_match"
            else:
                best = candidates[0]
                out["drift_label"] = best["label"]
                out["source_type"] = best["source_type"] or _existing_source_type(row)
                out["drift_match_warning"] = ""
        rows.append(out)
    return pd.DataFrame(rows)


def resolve_comparison_frequencies(config=None, height_df=None, gaussian_df=None):
    cfg = dict(config or {})
    raw = cfg.get("comparison_frequency_mhz")
    if raw:
        return [float(freq) for freq in raw]
    for df, column in (
        (gaussian_df, "freq"),
        (gaussian_df, "frequency_mhz"),
        (height_df, "frequency_mhz"),
    ):
        data = pd.DataFrame(df)
        if not data.empty and column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce").dropna().unique()
            if len(values):
                return [float(freq) for freq in sorted(values)]
    return [float(freq) for freq in DEFAULT_COMPARISON_FREQUENCIES_MHZ]


def resolve_drift_frequency_tolerance(config, frequencies):
    value = (config or {}).get(
        "drift_frequency_tolerance_mhz", "adaptive_half_band_spacing"
    )
    if isinstance(value, str) and value == "adaptive_half_band_spacing":
        freqs = np.asarray(sorted(float(freq) for freq in frequencies), dtype=float)
        if freqs.size < 2:
            return float(
                (config or {}).get("min_adaptive_frequency_tolerance_mhz", 5.0)
            )
        half_spacing = 0.5 * float(np.nanmin(np.diff(freqs)))
        min_tol = float((config or {}).get("min_adaptive_frequency_tolerance_mhz", 5.0))
        max_tol = float(
            (config or {}).get("max_adaptive_frequency_tolerance_mhz", 15.0)
        )
        return min(max(half_spacing, min_tol), max_tol)
    return float(value)


def plot_frequency_priority_summary(
    height_df, gaussian_df, drift_df, output_path, config=None
):
    cfg = dict(config or {})
    summary = build_frequency_priority_summary(height_df, gaussian_df, drift_df, cfg)
    height = _filter_to_comparison_frequencies(
        _clean_height_table(height_df),
        summary["comparison_frequency_mhz"],
        "frequency_mhz",
    )
    if height.empty:
        return {"status": "skipped", "reason": "no_frequency_priority_height_rows"}

    fig, axes = plt.subplots(
        2, 2, figsize=cfg.get("summary_figsize", (13, 9)), dpi=int(cfg.get("dpi", 170))
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    freqs = summary["comparison_frequency_mhz"]
    unique = height.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    )
    _plot_height_box_by_frequency(ax_a, unique, freqs)
    _plot_newkirk_curves_with_band_samples(
        ax_b, height, freqs, summary["model_ranking"]
    )
    _plot_top_residual_summaries(
        ax_c, summary["residual_summary"], summary["model_ranking"], cfg
    )
    _plot_model_ranking_heatmap(ax_d, summary["residual_summary"], freqs)
    fig.suptitle("Frequency-priority Gaussian/Newkirk diagnostics", y=0.995)
    _save(fig, output_path)
    return {"status": "saved", "path": str(output_path), "summary": summary}


def plot_gaussian_center_by_frequency_facets(gaussian_df, output_path, config=None):
    cfg = dict(config or {})
    gaussian = _clean_gaussian_table(gaussian_df)
    freqs = resolve_comparison_frequencies(cfg, gaussian, gaussian)
    gaussian = _filter_to_comparison_frequencies(gaussian, freqs, "frequency_mhz")
    if gaussian.empty:
        return {"status": "skipped", "reason": "no_gaussian_centers"}
    solar_radius = _solar_radius_arcsec(cfg)
    fig, axes = plt.subplots(
        2, 3, figsize=cfg.get("center_figsize", (12, 7.5)), dpi=int(cfg.get("dpi", 170))
    )
    scatter = None
    for ax, freq in zip(axes.ravel(), freqs, strict=False):
        group = gaussian[gaussian["frequency_mhz"].eq(float(freq))]
        ax.set_title(f"{freq:g} MHz")
        if group.empty:
            ax.text(
                0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center"
            )
            continue
        time_num = _date_numbers_for_group(group)
        scatter = ax.scatter(
            group["center_x_arcsec"],
            group["center_y_arcsec"],
            c=time_num,
            cmap="plasma",
            s=18,
            alpha=0.55,
            edgecolors="none",
        )
        mx = float(group["center_x_arcsec"].median())
        my = float(group["center_y_arcsec"].median())
        ax.scatter([mx], [my], marker="+", s=130, color="black", linewidths=2.0)
        _draw_iqr_ellipse(ax, group)
        _draw_solar_radius_reference(ax, solar_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("Solar X (arcsec)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Solar Y (arcsec)")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
        cbar.set_label("Time (UT)")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.suptitle("Gaussian centers by fitted frequency band", y=0.99)
    _save(fig, output_path)
    return {
        "status": "saved",
        "path": str(output_path),
        "solar_radius_arcsec": solar_radius,
        "color_meaning": "Time (UT)",
    }


def plot_gaussian_center_trajectory_by_frequency(gaussian_df, output_dir, config=None):
    cfg = dict(config or {})
    gaussian = _clean_gaussian_table(gaussian_df)
    freqs = resolve_comparison_frequencies(cfg, gaussian, gaussian)
    gaussian = _filter_to_comparison_frequencies(gaussian, freqs, "frequency_mhz")
    if gaussian.empty:
        return {"status": "skipped", "reason": "no_gaussian_centers"}

    output_dir = Path(output_dir)
    ensure_output_dir(output_dir)
    name_template = cfg.get(
        "trajectory_by_frequency_name_template",
        "source_trajectory",
    )
    paths = []
    groups = []
    for freq in freqs:
        group = gaussian[gaussian["frequency_mhz"].eq(float(freq))].dropna(
            subset=["time_dt", "center_x_arcsec", "center_y_arcsec"]
        )
        if group.empty:
            continue
        group = group.sort_values("time_dt")
        groups.append((float(freq), group))
    batch_generated_at = datetime.now(timezone.utc)
    for sequence, (freq, group) in enumerate(groups, start=1):
        configured_name = str(name_template).format(frequency=freq)
        if Path(configured_name).suffix.casefold() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".webp",
        }:
            filename = configured_name
        else:
            naming_frame = group.rename(columns={"time_dt": "obs_time"})
            filename = build_radio_image_filename(
                naming_frame,
                sequence=sequence,
                product=configured_name,
                frequency_mhz=freq,
                generated_at=batch_generated_at,
            )
        path = output_dir / filename
        _plot_single_frequency_time_trajectory(group, float(freq), path, cfg)
        paths.append(str(path))

    if not paths:
        return {"status": "skipped", "reason": "no_frequency_trajectory_rows"}
    return {
        "status": "saved",
        "paths": paths,
        "cross_frequency_line_count": 0,
        "color_meaning": "Time (UT)",
    }


def plot_height_time_by_frequency_facets(height_df, output_path, config=None):
    cfg = dict(config or {})
    height = _clean_height_table(height_df)
    freqs = resolve_comparison_frequencies(cfg, height, height)
    height = _filter_to_comparison_frequencies(height, freqs, "frequency_mhz")
    unique = height.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    )
    if unique.empty:
        return {"status": "skipped", "reason": "no_height_time_rows"}
    fig, axes = plt.subplots(
        2,
        3,
        figsize=cfg.get("height_time_figsize", (13, 7.5)),
        dpi=int(cfg.get("dpi", 170)),
        sharex=True,
        sharey=True,
    )
    cross_frequency_line_count = 0
    newkirk_reference_line_count = 0
    selected_multiplier, selected_harmonic = _selected_newkirk_model(cfg)
    selected_label = model_label(selected_multiplier, selected_harmonic)
    for ax, freq in zip(axes.ravel(), freqs, strict=False):
        group = unique[unique["frequency_mhz"].eq(float(freq))].dropna(
            subset=["time_dt", "gaussian_height_rsun"]
        )
        ax.set_title(f"{freq:g} MHz")
        if group.empty:
            ax.text(
                0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center"
            )
            continue
        for source_type, part in group.groupby("source_type", dropna=False):
            ax.scatter(
                part["time_dt"],
                part["gaussian_height_rsun"],
                s=18,
                alpha=0.65,
                label=str(source_type),
            )
        if "drift_label" in group.columns:
            for drift_label, part in group.groupby("drift_label", dropna=False):
                label = str(drift_label).strip()
                part = part.sort_values("time_dt")
                if label and len(part) > 1:
                    ax.plot(
                        part["time_dt"],
                        part["gaussian_height_rsun"],
                        linewidth=1.0,
                        alpha=0.75,
                    )
        reference_height = _newkirk_reference_height(
            freq, selected_multiplier, selected_harmonic
        )
        if np.isfinite(reference_height):
            ax.axhline(
                reference_height,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=f"{selected_label} height",
            )
            newkirk_reference_line_count += 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.grid(True, linestyle=":", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles, strict=False))
            ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (UT)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Gaussian projected height (Rsun)")
    fig.autofmt_xdate()
    fig.suptitle("Gaussian projected height by fitted frequency band", y=0.99)
    _save(fig, output_path)
    return {
        "status": "saved",
        "path": str(output_path),
        "cross_frequency_line_count": cross_frequency_line_count,
        "newkirk_reference_line_count": newkirk_reference_line_count,
        "newkirk_reference_model_label": selected_label,
    }


def build_selected_band_newkirk_height_speed_table(drift_df=None, config=None):
    cfg = dict(config or {})
    freqs = resolve_comparison_frequencies(cfg, None, None)
    multiplier, harmonic = _selected_newkirk_model(cfg)
    drift = pd.DataFrame(drift_df).copy() if drift_df is not None else pd.DataFrame()
    freq_tol = resolve_drift_frequency_tolerance(cfg, freqs)
    rows = []
    for freq in freqs:
        density = _safe_newkirk_value(
            plasma_density_from_frequency_mhz, freq, harmonic=harmonic
        )
        radius = _safe_newkirk_value(
            newkirk_radius_from_frequency_mhz,
            freq,
            multiplier=multiplier,
            harmonic=harmonic,
        )
        height = radius - 1.0 if np.isfinite(radius) else np.nan
        match = _match_drift_for_frequency(float(freq), drift, freq_tol)
        row = {
            "frequency_mhz": float(freq),
            "newkirk_multiplier": float(multiplier),
            "harmonic": (
                int(harmonic) if float(harmonic).is_integer() else float(harmonic)
            ),
            "density_multiplier": float(multiplier),
            "emission_harmonic": float(harmonic),
            "effective_density_factor": effective_density_factor(multiplier, harmonic),
            "newkirk_assumption_label": newkirk_assumption_label(multiplier, harmonic),
            "electron_density_cm3": density,
            "newkirk_radius_rsun": radius,
            "newkirk_height_rsun": height,
            "drift_label": "",
            "drift_rate_mhz_s": np.nan,
            "dr_dt_rsun_s": np.nan,
            "speed_km_s": np.nan,
            "v_over_c": np.nan,
            "newkirk_speed_km_s": np.nan,
            "newkirk_speed_c": np.nan,
            "speed_status": "no_matching_drift_rate",
        }
        if match is not None and np.isfinite(radius):
            drift_rate = _drift_rate_from_row(match)
            if np.isfinite(drift_rate):
                speed = newkirk_speed_from_drift_rate(float(freq), drift_rate, radius)
                row.update(
                    {
                        "drift_label": str(match.get("label") or ""),
                        "drift_rate_mhz_s": drift_rate,
                        "dr_dt_rsun_s": speed["dr_dt_rsun_s"],
                        "speed_km_s": speed["speed_km_s"],
                        "v_over_c": (
                            speed["speed_km_s"] / SPEED_OF_LIGHT_KM_S
                            if np.isfinite(speed["speed_km_s"])
                            else np.nan
                        ),
                        "newkirk_speed_km_s": speed["speed_km_s"],
                        "newkirk_speed_c": (
                            speed["speed_km_s"] / SPEED_OF_LIGHT_KM_S
                            if np.isfinite(speed["speed_km_s"])
                            else np.nan
                        ),
                        "speed_status": "ok",
                    }
                )
            else:
                row["drift_label"] = str(match.get("label") or "")
                row["speed_status"] = "invalid_drift_rate"
        elif not np.isfinite(radius):
            row["speed_status"] = "invalid_newkirk_inversion"
        rows.append(row)
    return pd.DataFrame(rows)


def save_selected_band_newkirk_height_speed_table(drift_df, output_path, config=None):
    table = build_selected_band_newkirk_height_speed_table(drift_df, config)
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    table.to_csv(path, index=False)
    return {"status": "saved", "path": str(path), "rows": len(table)}


def plot_event_gaussian_newkirk_height_comparison(height_df, output_path, config=None):
    """Plot projected Gaussian source height against Newkirk radial heights."""
    cfg = dict(config or {})
    height = _clean_height_table(height_df)
    freqs = resolve_comparison_frequencies(cfg, height, height)
    height = _filter_to_comparison_frequencies(height, freqs, "frequency_mhz")
    if height.empty:
        return {"status": "skipped", "reason": "no_height_rows"}

    reference = str(cfg.get("reference_newkirk_assumption") or "2xH2")
    gaussian = height.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    ).dropna(subset=["frequency_mhz", "gaussian_height_rsun"])
    if "gaussian_projected_height_valid" not in gaussian.columns:
        gaussian["gaussian_projected_height_valid"] = (
            pd.to_numeric(gaussian["gaussian_height_rsun"], errors="coerce") >= 0
        )
    newkirk = height.drop_duplicates(
        subset=["frequency_mhz", "newkirk_multiplier", "harmonic"]
    ).dropna(subset=["frequency_mhz", "newkirk_height_rsun"])
    if gaussian.empty and newkirk.empty:
        return {"status": "skipped", "reason": "no_plottable_height_rows"}

    fig, ax = plt.subplots(
        figsize=cfg.get("event_height_figsize", (8.2, 5.4)),
        dpi=int(cfg.get("dpi", 180)),
    )
    positions = np.arange(len(freqs), dtype=float)
    pos_by_freq = {float(freq): idx for idx, freq in enumerate(freqs)}

    plotted_gaussian_freqs = set()
    invalid_plotted = False
    for freq in freqs:
        group = gaussian[gaussian["frequency_mhz"].eq(float(freq))]
        valid = group[group["gaussian_projected_height_valid"].map(_truthy)]
        invalid = group[~group["gaussian_projected_height_valid"].map(_truthy)]
        values = pd.to_numeric(valid["gaussian_height_rsun"], errors="coerce").dropna()
        if not values.empty:
            x = np.full(len(values), pos_by_freq[float(freq)], dtype=float)
            jitter = (
                np.linspace(-0.08, 0.08, len(values))
                if len(values) > 1
                else np.array([0.0])
            )
            ax.scatter(
                x + jitter,
                values.to_numpy(dtype=float),
                s=18,
                alpha=0.45,
                color="#2f80ed",
                edgecolors="none",
                label="Gaussian centers" if not plotted_gaussian_freqs else None,
                zorder=2,
            )
            median = float(values.median())
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            ax.errorbar(
                pos_by_freq[float(freq)],
                median,
                yerr=[[median - q1], [q3 - median]],
                fmt="o",
                color="#0b3d91",
                capsize=3,
                markersize=5,
                label="Gaussian median +/- IQR" if not plotted_gaussian_freqs else None,
                zorder=4,
            )
            plotted_gaussian_freqs.add(float(freq))
        invalid_values = pd.to_numeric(
            invalid["gaussian_height_rsun"], errors="coerce"
        ).dropna()
        if not invalid_values.empty:
            x = np.full(len(invalid_values), pos_by_freq[float(freq)], dtype=float)
            ax.scatter(
                x,
                invalid_values.to_numpy(dtype=float),
                s=28,
                marker="x",
                alpha=0.65,
                color="0.45",
                label="Invalid projected height" if not invalid_plotted else None,
                zorder=1,
            )
            invalid_plotted = True

    model_count = 0
    newkirk_legend_added = False
    label_y_positions = []
    for multiplier, harmonic in _model_pairs(height):
        key = f"{float(multiplier):g}xH{float(harmonic):g}"
        group = newkirk[
            pd.to_numeric(newkirk["newkirk_multiplier"], errors="coerce").eq(
                float(multiplier)
            )
            & pd.to_numeric(newkirk["harmonic"], errors="coerce").eq(float(harmonic))
        ].sort_values("frequency_mhz")
        if group.empty:
            continue
        x = [
            pos_by_freq[float(freq)]
            for freq in group["frequency_mhz"]
            if float(freq) in pos_by_freq
        ]
        y = [
            float(row["newkirk_height_rsun"])
            for _, row in group.iterrows()
            if float(row["frequency_mhz"]) in pos_by_freq
        ]
        if not x:
            continue
        is_reference = key == reference
        ax.plot(
            x,
            y,
            marker="s",
            color="black" if is_reference else "0.55",
            linestyle="-" if is_reference else "--",
            linewidth=2.0 if is_reference else 0.9,
            markersize=5 if is_reference else 3.5,
            alpha=0.95 if is_reference else 0.58,
            label=(
                "Reference Newkirk model"
                if is_reference and not newkirk_legend_added
                else None
            ),
            zorder=5 if is_reference else 3,
        )
        if is_reference:
            newkirk_legend_added = True
        label_y = y[-1]
        for existing_y in label_y_positions:
            if abs(label_y - existing_y) < 0.035:
                label_y = existing_y + 0.035
        label_y_positions.append(label_y)
        label_x = x[-1] + (
            0.08 if not cfg.get("reverse_frequency_axis", False) else -0.08
        )
        ax.text(
            label_x,
            label_y,
            _short_newkirk_case_label(multiplier, harmonic, reference=is_reference),
            color="black" if is_reference else "0.35",
            fontsize=7,
            va="center",
            ha="left" if not cfg.get("reverse_frequency_axis", False) else "right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4},
            zorder=6,
        )
        model_count += 1

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{freq:g}" for freq in freqs])
    ax.set_xlabel("Selected frequency (MHz)")
    ax.set_ylabel("Height above photosphere (Rsun)")
    ax.set_title("Projected Gaussian source height vs Newkirk radial height")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    if cfg.get("reverse_frequency_axis", False):
        ax.invert_xaxis()
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.text(
        0.5,
        0.01,
        "F = fundamental (H=1), H = harmonic (H=2); prefix = Newkirk density multiplier.",
        ha="center",
        fontsize=8,
    )
    _save(fig, output_path)
    return {
        "status": "saved",
        "path": str(output_path),
        "selected_model_label": reference,
        "newkirk_model_count": model_count,
        "gaussian_frequency_count": len(plotted_gaussian_freqs),
    }


def plot_event_newkirk_speed_frequency(speed_df, output_path, config=None):
    """Plot Newkirk-inferred exciter speed without connecting unrelated drifts."""
    cfg = dict(config or {})
    df = pd.DataFrame(speed_df).copy()
    if "newkirk_speed_km_s" not in df.columns and "speed_km_s" in df.columns:
        df["newkirk_speed_km_s"] = df["speed_km_s"]
    if "newkirk_speed_c" not in df.columns and "newkirk_speed_km_s" in df.columns:
        df["newkirk_speed_c"] = (
            pd.to_numeric(df["newkirk_speed_km_s"], errors="coerce")
            / SPEED_OF_LIGHT_KM_S
        )
    required = {"frequency_mhz", "newkirk_speed_km_s", "speed_status"}
    if df.empty or not required.issubset(df.columns):
        return {"status": "skipped", "reason": "no_speed_rows"}
    df["frequency_mhz"] = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    df["newkirk_speed_km_s"] = pd.to_numeric(df["newkirk_speed_km_s"], errors="coerce")
    df["newkirk_speed_c"] = pd.to_numeric(df.get("newkirk_speed_c"), errors="coerce")
    if "drift_label" not in df.columns:
        df["drift_label"] = ""
    if "newkirk_assumption_label" not in df.columns:
        df["newkirk_assumption_label"] = "Newkirk assumption"
    ok = (
        df["speed_status"].astype(str).eq("ok")
        & df["frequency_mhz"].notna()
        & df["newkirk_speed_km_s"].notna()
    )
    plotted = df[ok].copy().sort_values("frequency_mhz")
    skipped_count = int((~ok).sum())
    unmatched = df[
        ~ok
        & df["frequency_mhz"].notna()
        & df["speed_status"].astype(str).eq("no_matching_drift_rate")
    ].copy()
    if plotted.empty:
        return {
            "status": "skipped",
            "reason": "no_matched_speed_rows",
            "skipped_frequency_count": skipped_count,
        }

    fig, ax = plt.subplots(
        figsize=cfg.get("event_speed_figsize", (7.8, 5.2)), dpi=int(cfg.get("dpi", 180))
    )
    cross_drift_line_count = 0
    markers = ["o", "s", "^", "D", "P", "X"]
    model_label_text = _format_newkirk_label_from_row(plotted.iloc[0])
    for idx, (assumption, group) in enumerate(
        plotted.groupby("newkirk_assumption_label", dropna=False)
    ):
        marker = markers[idx % len(markers)]
        for drift_label, part in group.groupby("drift_label", dropna=False):
            part = part.sort_values("frequency_mhz")
            drift_rate_text = (
                _format_drift_rate(part["drift_rate_mhz_s"].iloc[0])
                if "drift_rate_mhz_s" in part
                else ""
            )
            label_parts = [str(drift_label).strip() or str(assumption).strip()]
            if drift_rate_text:
                label_parts.append(drift_rate_text)
            label = ", ".join(part for part in label_parts if part)
            ax.scatter(
                part["frequency_mhz"],
                part["newkirk_speed_km_s"],
                marker=marker,
                s=42,
                alpha=0.85,
                label=label,
            )
    for _, row in plotted.iterrows():
        label = str(row.get("drift_label") or "").strip()
        if label:
            drift_rate_text = _format_drift_rate(row.get("drift_rate_mhz_s"))
            text = f"{label}\n{drift_rate_text}" if drift_rate_text else label
            ax.annotate(
                text,
                (row["frequency_mhz"], row["newkirk_speed_km_s"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
    y_values = plotted["newkirk_speed_km_s"].dropna().to_numpy(dtype=float)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_span = max(y_max - y_min, abs(y_max) * 0.08, 1.0)
    y_pad = y_span * 0.18
    missing_y = y_min - y_pad * 0.55
    if not unmatched.empty:
        unmatched = unmatched.drop_duplicates(subset=["frequency_mhz"]).sort_values(
            "frequency_mhz"
        )
        ax.scatter(
            unmatched["frequency_mhz"],
            np.full(len(unmatched), missing_y, dtype=float),
            marker="o",
            s=42,
            facecolors="none",
            edgecolors="0.45",
            linewidths=1.0,
            alpha=0.8,
            label="No matching drift rate",
        )
        for _, row in unmatched.iterrows():
            freq = float(row["frequency_mhz"])
            ax.annotate(
                f"{freq:g} MHz: no drift",
                (freq, missing_y),
                xytext=(4, -10),
                textcoords="offset points",
                fontsize=7,
                color="0.35",
            )
    all_freqs = pd.concat(
        [plotted["frequency_mhz"], unmatched["frequency_mhz"]]
    ).dropna()
    x_min = float(all_freqs.min())
    x_max = float(all_freqs.max())
    x_span = max(x_max - x_min, abs(x_max) * 0.04, 1.0)
    x_pad = x_span * 0.08
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(missing_y - y_pad * 0.45, y_max + y_pad)
    ax.set_xlabel("Selected frequency (MHz)")
    ax.set_ylabel("Newkirk-inferred exciter speed (km/s)")
    ax.set_title("Newkirk-inferred exciter speed by selected frequency")
    ax.grid(True, linestyle=":", alpha=0.35)
    if cfg.get("reverse_frequency_axis", False):
        ax.invert_xaxis()
    ax.legend(fontsize=7, title=f"Model: {model_label_text}", title_fontsize=7)
    fig.text(
        0.5,
        0.01,
        "Points are grouped by drift label; connecting lines are omitted to avoid implying branch continuity.",
        ha="center",
        fontsize=8,
    )
    _save(fig, output_path, bbox_inches="tight")
    return {
        "status": "saved",
        "path": str(output_path),
        "plotted_frequency_count": int(len(plotted)),
        "skipped_frequency_count": skipped_count,
        "unmatched_frequency_count": int(len(unmatched)),
        "cross_drift_line_count": cross_drift_line_count,
    }


def build_newkirk_physical_consistency_report(speed_df, height_summary_df, config=None):
    """Build a Markdown report describing Newkirk physical consistency checks."""
    cfg = dict(config or {})
    speed = pd.DataFrame(speed_df).copy()
    height = pd.DataFrame(height_summary_df).copy()
    if "newkirk_speed_km_s" not in speed.columns and "speed_km_s" in speed.columns:
        speed["newkirk_speed_km_s"] = speed["speed_km_s"]
    if "newkirk_speed_c" not in speed.columns and "newkirk_speed_km_s" in speed.columns:
        speed["newkirk_speed_c"] = (
            pd.to_numeric(speed["newkirk_speed_km_s"], errors="coerce")
            / SPEED_OF_LIGHT_KM_S
        )

    speed_values = pd.to_numeric(
        speed.get("newkirk_speed_km_s"), errors="coerce"
    ).dropna()
    c_values = pd.to_numeric(speed.get("newkirk_speed_c"), errors="coerce").dropna()
    reference = str(cfg.get("reference_newkirk_assumption") or "2xH2")
    lines = [
        "# Newkirk Physical Consistency Report",
        "",
        "## Speed range summary",
        f"- min speed km/s: {_fmt_float(speed_values.min() if not speed_values.empty else np.nan)}",
        f"- max speed km/s: {_fmt_float(speed_values.max() if not speed_values.empty else np.nan)}",
        f"- min v/c: {_fmt_float(c_values.min() if not c_values.empty else np.nan, 3)}",
        f"- max v/c: {_fmt_float(c_values.max() if not c_values.empty else np.nan, 3)}",
        "",
        "## Speed classification",
    ]
    if c_values.empty:
        lines.append("- no finite Newkirk-inferred exciter speeds available")
    else:
        for _, row in speed.iterrows():
            v_c = _float_or_nan(row.get("newkirk_speed_c"))
            if not np.isfinite(v_c):
                continue
            lines.append(
                f"- {row.get('frequency_mhz', '')} MHz {row.get('drift_label', '')}: "
                f"{_classify_speed_c(v_c)}"
            )
    lines.extend(["", "## Height consistency summary"])
    ref_col = f"newkirk_height_rsun_{reference}"
    if height.empty:
        lines.append("- no height summary rows available")
    else:
        for _, row in height.iterrows():
            gaussian = _float_or_nan(row.get("gaussian_projected_height_median_rsun"))
            ref_height = _float_or_nan(row.get(ref_col))
            delta = _float_or_nan(row.get("abs_delta_reference_rsun"))
            q25 = _float_or_nan(row.get("gaussian_projected_height_q25_rsun"))
            q75 = _float_or_nan(row.get("gaussian_projected_height_q75_rsun"))
            in_iqr = (
                np.isfinite(ref_height)
                and np.isfinite(q25)
                and np.isfinite(q75)
                and q25 <= ref_height <= q75
            )
            lines.append(
                f"- {row.get('frequency_mhz', '')} MHz: Gaussian projected median="
                f"{_fmt_float(gaussian)}, reference Newkirk radial height={_fmt_float(ref_height)}, "
                f"delta={_fmt_float(delta)}, within Gaussian IQR={bool(in_iqr)}"
            )
    lines.extend(["", "## Invalid Gaussian points summary"])
    if height.empty:
        lines.append("- no invalid point summary available")
    else:
        for _, row in height.iterrows():
            invalid_count = _float_or_nan(row.get("gaussian_invalid_count"))
            invalid_count = int(invalid_count) if np.isfinite(invalid_count) else 0
            lines.append(
                f"- {row.get('frequency_mhz', '')} MHz: "
                f"{invalid_count} invalid projected-height points"
            )
    lines.extend(
        [
            "",
            "## Conservative physical interpretation",
            (
                "The drift-rate-derived Newkirk speeds are treated as model-inferred "
                "exciter speeds rather than direct radio source bulk motions. The Gaussian "
                "center heights are projected plane-of-sky heights, while Newkirk heights "
                "are radial heights inferred from a one-dimensional density model. Therefore, "
                "agreement between the two should be interpreted as order-of-magnitude "
                "consistency rather than a direct three-dimensional spatial validation."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def save_newkirk_physical_consistency_report(
    speed_df, height_summary_df, output_path, config=None
):
    """Save the Newkirk physical consistency report as Markdown."""
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    content = build_newkirk_physical_consistency_report(
        speed_df, height_summary_df, config
    )
    path.write_text(content, encoding="utf-8")
    return {"status": "saved", "path": str(path)}


def plot_drift_frequency_band_matching(
    spectrogram_data,
    time_axis,
    frequency_axis_mhz,
    drift_df,
    output_path,
    config=None,
):
    cfg = dict(config or {})
    drift = pd.DataFrame(drift_df).copy()
    if drift.empty:
        return {"status": "skipped", "reason": "no_drift_rows"}
    data, time_nums, freqs, extent = _prepare_spectrogram(
        spectrogram_data, time_axis, frequency_axis_mhz
    )
    bands = resolve_comparison_frequencies(cfg, None, None)
    fig, (ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=cfg.get("drift_matching_figsize", (13, 6)),
        dpi=int(cfg.get("dpi", 170)),
        gridspec_kw={"width_ratios": [4.4, 1.6]},
    )
    ax.imshow(
        data,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap=cfg.get("cmap", "viridis"),
    )
    for freq in bands:
        ax.axhspan(float(freq) - 1.5, float(freq) + 1.5, color="white", alpha=0.13)
        ax.axhline(float(freq), color="white", linewidth=0.7, alpha=0.75)
    rows = []
    for _, row in drift.iterrows():
        t1 = parse_datetime_value(row.get("t_start"))
        t2 = parse_datetime_value(row.get("t_end"))
        f1 = _float_or_nan(row.get("f_start_mhz"))
        f2 = _float_or_nan(row.get("f_end_mhz"))
        label = str(row.get("label") or "")
        color = str(row.get("color") or "cyan")
        if t1 is None or t2 is None or not np.isfinite(f1) or not np.isfinite(f2):
            continue
        ax.plot(
            [mdates.date2num(t1), mdates.date2num(t2)],
            [f1, f2],
            color=color,
            linewidth=1.6,
        )
        ax.scatter(
            [mdates.date2num(t1), mdates.date2num(t2)],
            [f1, f2],
            color=color,
            s=18,
            edgecolors="black",
        )
        lo, hi = sorted((f1, f2))
        matched = [f"{freq:g}" for freq in bands if lo <= float(freq) <= hi]
        rows.append([label, f"{f1:.0f}-{f2:.0f}", ", ".join(matched) or "-"])
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("Drift selections against Gaussian fitted frequency bands")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    table_ax.axis("off")
    table = table_ax.table(
        cellText=rows,
        colLabels=["Drift", "MHz", "Bands"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    fig.autofmt_xdate()
    _save(fig, output_path)
    return {"status": "saved", "path": str(output_path)}


def write_frequency_priority_dashboard(
    height_df, gaussian_df, drift_df, output_path, config=None
):
    cfg = dict(config or {})
    summary = build_frequency_priority_summary(height_df, gaussian_df, drift_df, cfg)
    height = _filter_to_comparison_frequencies(
        _clean_height_table(height_df),
        summary["comparison_frequency_mhz"],
        "frequency_mhz",
    )
    gaussian = apply_frequency_priority_drift_matching(gaussian_df, drift_df, cfg)
    payload = {
        "comparison_frequency_mhz": summary["comparison_frequency_mhz"],
        "model_ranking": _json_records(summary["model_ranking"]),
        "gaussian_height_summary": _json_records(summary["gaussian_height_summary"]),
        "residual_summary": _json_records(summary["residual_summary"]),
        "center_summary": _json_records(summary["center_summary"]),
        "drift_match_summary": _json_records(summary["drift_match_summary"]),
        "height_points": _json_records(
            height[
                [
                    "time",
                    "frequency_mhz",
                    "source_type",
                    "drift_label",
                    "gaussian_height_rsun",
                    "newkirk_multiplier",
                    "harmonic",
                    "newkirk_height_rsun",
                    "height_residual_rsun",
                ]
            ].head(int(cfg.get("dashboard_max_height_rows", 5000)))
        ),
        "gaussian_points": _json_records(
            gaussian[
                [
                    "time",
                    "frequency_mhz",
                    "center_x_arcsec",
                    "center_y_arcsec",
                    "source_type",
                    "drift_label",
                    "drift_match_warning",
                ]
            ].head(int(cfg.get("dashboard_max_gaussian_rows", 5000)))
        ),
    }
    content = _dashboard_html(payload)
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    path.write_text(content, encoding="utf-8")
    return {"status": "saved", "path": str(path)}


def save_frequency_priority_summary_csv(summary, output_path):
    rows = []
    for _, row in summary["model_ranking"].iterrows():
        rows.append(
            {
                "section": "model_ranking",
                "model_label": row.get("model_label"),
                "newkirk_multiplier": row.get("newkirk_multiplier"),
                "harmonic": row.get("harmonic"),
                "median_abs_residual_rsun": row.get("median_abs_residual_rsun"),
                "n": row.get("n"),
            }
        )
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _plot_height_box_by_frequency(ax, unique, freqs):
    data = [
        unique.loc[unique["frequency_mhz"].eq(float(freq)), "gaussian_height_rsun"]
        .dropna()
        .to_numpy()
        for freq in freqs
    ]
    positions = np.arange(len(freqs))
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{freq:g}" for freq in freqs])
    ax.set_xlabel("Gaussian fitted frequency (MHz)")
    ax.set_ylabel("Gaussian projected height (Rsun)")
    ax.set_title("A. Gaussian heights at observed bands")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)


def _plot_newkirk_curves_with_band_samples(ax, height, freqs, model_ranking):
    grid = np.linspace(min(freqs) * 0.95, max(freqs) * 1.05, 180)
    best = str(model_ranking.iloc[0]["model_label"]) if not model_ranking.empty else ""
    for multiplier, harmonic in _model_pairs(height):
        label = model_label(multiplier, harmonic)
        curve = newkirk_height_from_frequency_mhz(
            grid, multiplier=multiplier, harmonic=harmonic
        )
        sample = newkirk_height_from_frequency_mhz(
            np.asarray(freqs), multiplier=multiplier, harmonic=harmonic
        )
        ax.plot(
            grid,
            curve,
            linewidth=2.2 if label == best else 1.0,
            alpha=0.9 if label == best else 0.55,
            label=label,
        )
        ax.scatter(freqs, sample, s=20, alpha=0.85)
    for freq in freqs:
        ax.axvline(freq, color="0.8", linewidth=0.7, zorder=0)
    ax.invert_xaxis()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Newkirk height (Rsun)")
    ax.set_title("B. Newkirk curves sampled at observed bands")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


def _plot_top_residual_summaries(ax, residual_summary, model_ranking, cfg):
    if residual_summary.empty or model_ranking.empty:
        ax.text(
            0.5,
            0.5,
            "No residual summary",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return
    top_n = int(cfg.get("top_residual_models", 3) or 3)
    labels = set(model_ranking.head(top_n)["model_label"])
    data = residual_summary[residual_summary["model_label"].isin(labels)]
    for label, group in data.groupby("model_label", sort=False):
        ax.errorbar(
            group["frequency_mhz"],
            group["median_residual_rsun"],
            yerr=0.5 * group["iqr_residual_rsun"],
            marker="o",
            linewidth=1.1,
            capsize=2,
            label=label,
        )
    ax.axhline(0, color="black", linestyle=":", linewidth=0.9)
    ax.invert_xaxis()
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gaussian - Newkirk height (Rsun)")
    ax.set_title("C. Residual median ± half-IQR")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(fontsize=7)


def _plot_model_ranking_heatmap(ax, residual_summary, freqs):
    if residual_summary.empty:
        ax.text(
            0.5,
            0.5,
            "No model ranking",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return
    matrix = residual_summary.pivot_table(
        index="model_label",
        columns="frequency_mhz",
        values="median_abs_residual_rsun",
        aggfunc="median",
    ).reindex(columns=freqs)
    order = matrix.median(axis=1).sort_values().index
    matrix = matrix.loc[order]
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(
        [f"{freq:g}" for freq in matrix.columns], rotation=35, ha="right"
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("D. Median |residual| by model and band")
    ax.set_xlabel("Frequency (MHz)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iat[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.figure.colorbar(im, ax=ax, label="Median |residual| (Rsun)")


def _frequency_summary(df, freqs, value_col):
    rows = []
    for freq in freqs:
        series = pd.to_numeric(
            df.loc[df["frequency_mhz"].eq(float(freq)), value_col], errors="coerce"
        ).dropna()
        rows.append(_summary_row(freq, series, value_col))
    return pd.DataFrame(rows)


def _summary_row(freq, series, value_col):
    if series.empty:
        return {
            "frequency_mhz": float(freq),
            "n": 0,
            f"{value_col}_median": np.nan,
            f"{value_col}_q25": np.nan,
            f"{value_col}_q75": np.nan,
            f"{value_col}_iqr": np.nan,
        }
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    return {
        "frequency_mhz": float(freq),
        "n": int(len(series)),
        f"{value_col}_median": float(series.median()),
        f"{value_col}_q25": q25,
        f"{value_col}_q75": q75,
        f"{value_col}_iqr": q75 - q25,
    }


def _residual_summary(height, freqs):
    rows = []
    for (multiplier, harmonic), model_group in height.groupby(
        ["newkirk_multiplier", "harmonic"], dropna=False
    ):
        label = model_label(multiplier, harmonic)
        for freq in freqs:
            residual = pd.to_numeric(
                model_group.loc[
                    model_group["frequency_mhz"].eq(float(freq)), "height_residual_rsun"
                ],
                errors="coerce",
            ).dropna()
            if residual.empty:
                rows.append(
                    {
                        "newkirk_multiplier": float(multiplier),
                        "harmonic": harmonic,
                        "model_label": label,
                        "frequency_mhz": float(freq),
                        "n": 0,
                        "median_residual_rsun": np.nan,
                        "iqr_residual_rsun": np.nan,
                        "median_abs_residual_rsun": np.nan,
                    }
                )
                continue
            q25 = float(residual.quantile(0.25))
            q75 = float(residual.quantile(0.75))
            rows.append(
                {
                    "newkirk_multiplier": float(multiplier),
                    "harmonic": harmonic,
                    "model_label": label,
                    "frequency_mhz": float(freq),
                    "n": int(len(residual)),
                    "median_residual_rsun": float(residual.median()),
                    "iqr_residual_rsun": q75 - q25,
                    "median_abs_residual_rsun": float(residual.abs().median()),
                }
            )
    return pd.DataFrame(rows)


def _model_ranking(height):
    rows = []
    for (multiplier, harmonic), group in height.groupby(
        ["newkirk_multiplier", "harmonic"], dropna=False
    ):
        residual = pd.to_numeric(
            group["height_residual_rsun"], errors="coerce"
        ).dropna()
        if residual.empty:
            continue
        rows.append(
            {
                "newkirk_multiplier": float(multiplier),
                "harmonic": harmonic,
                "model_label": model_label(multiplier, harmonic),
                "n": int(len(residual)),
                "median_abs_residual_rsun": float(residual.abs().median()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("median_abs_residual_rsun")
        .reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )


def _center_summary(gaussian, freqs):
    rows = []
    for freq in freqs:
        if gaussian.empty or "frequency_mhz" not in gaussian.columns:
            group = pd.DataFrame()
        else:
            group = gaussian[gaussian["frequency_mhz"].eq(float(freq))]
        rows.append(
            {
                "frequency_mhz": float(freq),
                "n": int(len(group)),
                "median_x_arcsec": _median_or_nan(group.get("center_x_arcsec")),
                "median_y_arcsec": _median_or_nan(group.get("center_y_arcsec")),
                "iqr_x_arcsec": _iqr_or_nan(group.get("center_x_arcsec")),
                "iqr_y_arcsec": _iqr_or_nan(group.get("center_y_arcsec")),
            }
        )
    return pd.DataFrame(rows)


def _drift_match_summary(matched_gaussian, freqs):
    rows = []
    for freq in freqs:
        if matched_gaussian.empty or "frequency_mhz" not in matched_gaussian.columns:
            group = pd.DataFrame()
        else:
            group = matched_gaussian[matched_gaussian["frequency_mhz"].eq(float(freq))]
        if group.empty:
            rows.append(
                {
                    "frequency_mhz": float(freq),
                    "drift_label": "",
                    "source_type": "missing",
                    "n": 0,
                }
            )
            continue
        counts = group.groupby(["drift_label", "source_type"], dropna=False).size()
        for (label, source_type), n in counts.items():
            rows.append(
                {
                    "frequency_mhz": float(freq),
                    "drift_label": str(label or ""),
                    "source_type": str(source_type or "unknown"),
                    "n": int(n),
                }
            )
    return pd.DataFrame(rows)


def _filter_to_comparison_frequencies(df, freqs, column):
    data = pd.DataFrame(df).copy()
    if data.empty or column not in data.columns:
        return data
    values = pd.to_numeric(data[column], errors="coerce")
    mask = np.zeros(len(data), dtype=bool)
    for freq in freqs:
        mask |= np.isclose(values, float(freq), rtol=0, atol=1e-6)
    return data.loc[mask].copy()


def _frequency_group_empty(df, freq):
    data = pd.DataFrame(df)
    if data.empty or "frequency_mhz" not in data.columns:
        return True
    return data[data["frequency_mhz"].eq(float(freq))].empty


def _clean_height_table(height_df):
    df = pd.DataFrame(height_df).copy()
    if df.empty:
        return df
    for col in (
        "frequency_mhz",
        "gaussian_height_rsun",
        "newkirk_height_rsun",
        "height_residual_rsun",
        "newkirk_multiplier",
        "harmonic",
        "gaussian_x_arcsec",
        "gaussian_y_arcsec",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["source_type"] = (
        df["source_type"].fillna("unknown").astype(str)
        if "source_type" in df.columns
        else "unknown"
    )
    df["drift_label"] = (
        df["drift_label"].fillna("").astype(str) if "drift_label" in df.columns else ""
    )
    df["time_dt"] = _parse_time_series(df.get("time"))
    return df


def _clean_gaussian_table(gaussian_df):
    df = pd.DataFrame(gaussian_df).copy()
    if df.empty:
        return df
    if "frequency_mhz" not in df.columns:
        if "freq" in df.columns:
            df["frequency_mhz"] = pd.to_numeric(df["freq"], errors="coerce")
        elif "freq_mhz" in df.columns:
            df["frequency_mhz"] = pd.to_numeric(df["freq_mhz"], errors="coerce")
    for src, dest in (
        ("center_x_arcsec", "center_x_arcsec"),
        ("center_y_arcsec", "center_y_arcsec"),
    ):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce")
    df["source_type"] = (
        df["source_type"].fillna("unknown").astype(str)
        if "source_type" in df.columns
        else "unknown"
    )
    df["drift_label"] = (
        df["drift_label"].fillna("").astype(str) if "drift_label" in df.columns else ""
    )
    df["drift_match_warning"] = (
        df["drift_match_warning"].fillna("").astype(str)
        if "drift_match_warning" in df.columns
        else ""
    )
    df["time_dt"] = _parse_time_series(df.get("time"))
    return df


def _frequency_on_drift_line(row_time, drift_row, time_tolerance_s):
    t_start = parse_datetime_value(drift_row.get("t_start"))
    t_end = parse_datetime_value(drift_row.get("t_end"))
    f_start = _float_or_nan(drift_row.get("f_start_mhz"))
    f_end = _float_or_nan(drift_row.get("f_end_mhz"))
    if (
        t_start is None
        or t_end is None
        or not np.isfinite(f_start)
        or not np.isfinite(f_end)
    ):
        return None
    if t_end < t_start:
        t_start, t_end = t_end, t_start
        f_start, f_end = f_end, f_start
    duration = (t_end - t_start).total_seconds()
    offset = (row_time - t_start).total_seconds()
    if offset < -time_tolerance_s or offset > duration + time_tolerance_s:
        return None
    if abs(duration) <= 1e-12:
        return 0.5 * (f_start + f_end)
    fraction = min(1.0, max(0.0, offset / duration))
    return f_start + fraction * (f_end - f_start)


def _prepare_spectrogram(spectrogram_data, time_axis, frequency_axis_mhz):
    data = np.asarray(spectrogram_data, dtype=float)
    time_nums = np.asarray(
        [mdates.date2num(parse_datetime_value(t) or t) for t in time_axis], dtype=float
    )
    freqs = np.asarray(frequency_axis_mhz, dtype=float)
    if data.shape[0] != freqs.size and data.shape[1] == freqs.size:
        data = data.T
    if freqs[0] > freqs[-1]:
        freqs = freqs[::-1]
        data = data[::-1, :]
    extent = [
        float(np.nanmin(time_nums)),
        float(np.nanmax(time_nums)),
        float(np.nanmin(freqs)),
        float(np.nanmax(freqs)),
    ]
    return data, time_nums, freqs, extent


def _parse_time_series(series):
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.Series(series).map(_parse_radio_datetime)


def _parse_radio_datetime(value):
    parsed = parse_datetime_value(value)
    if parsed is not None:
        return pd.Timestamp(parsed)
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 14:
        return pd.NaT
    base = digits[:14]
    suffix = digits[14:]
    microsecond = 0
    if suffix:
        if len(suffix) <= 3:
            microsecond = int(suffix.zfill(3) + "000")
        else:
            microsecond = int(suffix[:6].ljust(6, "0"))
    try:
        ts = pd.to_datetime(base, format="%Y%m%d%H%M%S", errors="raise")
    except Exception:
        return pd.NaT
    return ts + pd.Timedelta(microseconds=microsecond)


def _date_numbers_for_group(group):
    times = pd.to_datetime(group["time_dt"], errors="coerce")
    if times.notna().any():
        values = [
            value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
            for value in times
        ]
        return mdates.date2num(values)
    return np.arange(len(group), dtype=float)


def _model_pairs(height):
    pairs = height[["newkirk_multiplier", "harmonic"]].dropna().drop_duplicates()
    return [
        (float(row["newkirk_multiplier"]), row["harmonic"])
        for _, row in pairs.sort_values(["newkirk_multiplier", "harmonic"]).iterrows()
    ]


def _draw_iqr_ellipse(ax, group):
    x = pd.to_numeric(group["center_x_arcsec"], errors="coerce").dropna()
    y = pd.to_numeric(group["center_y_arcsec"], errors="coerce").dropna()
    if x.empty or y.empty:
        return
    mx = float(x.median())
    my = float(y.median())
    width = max(float(x.quantile(0.75) - x.quantile(0.25)), 1e-6)
    height = max(float(y.quantile(0.75) - y.quantile(0.25)), 1e-6)
    from matplotlib.patches import Ellipse

    ax.add_patch(
        Ellipse(
            (mx, my),
            width=width,
            height=height,
            angle=0,
            facecolor="none",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
        )
    )


def _draw_solar_radius_reference(ax, solar_radius_arcsec):
    radius = _float_or_nan(solar_radius_arcsec)
    if not np.isfinite(radius) or radius <= 0:
        return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_values = np.linspace(max(min(xlim), -radius), min(max(xlim), radius), 200)
    if x_values.size:
        y_abs = np.sqrt(np.maximum(radius**2 - x_values**2, 0.0))
        for y_values in (y_abs, -y_abs):
            mask = (y_values >= min(ylim)) & (y_values <= max(ylim))
            if mask.any():
                ax.plot(
                    x_values[mask],
                    y_values[mask],
                    color="0.25",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.45,
                    zorder=0,
                )
    y_values = np.linspace(max(min(ylim), -radius), min(max(ylim), radius), 200)
    if y_values.size:
        x_abs = np.sqrt(np.maximum(radius**2 - y_values**2, 0.0))
        for x_values in (x_abs, -x_abs):
            mask = (x_values >= min(xlim)) & (x_values <= max(xlim))
            if mask.any():
                ax.plot(
                    x_values[mask],
                    y_values[mask],
                    color="0.25",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.45,
                    zorder=0,
                )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(
        0.02,
        0.98,
        f"1 Rsun = {radius:g} arcsec",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
    )


def _plot_single_frequency_time_trajectory(group, frequency_mhz, path, config):
    fig, ax = plt.subplots(
        figsize=config.get("trajectory_figsize", (7, 6)),
        dpi=int(config.get("dpi", 180)),
    )
    time_nums = _date_numbers_for_group(group)
    ax.plot(
        group["center_x_arcsec"],
        group["center_y_arcsec"],
        color="0.55",
        linewidth=1.0,
        alpha=0.75,
        zorder=1,
    )
    sc = ax.scatter(
        group["center_x_arcsec"],
        group["center_y_arcsec"],
        c=time_nums,
        cmap="plasma",
        s=34,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    if len(group) >= 2:
        start = group.iloc[-2]
        end = group.iloc[-1]
        ax.annotate(
            "",
            xy=(end["center_x_arcsec"], end["center_y_arcsec"]),
            xytext=(start["center_x_arcsec"], start["center_y_arcsec"]),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
        )
    cbar = fig.colorbar(sc, ax=ax, label="Time (UT)")
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title(f"Gaussian center trajectory colored by time - {frequency_mhz:g} MHz")
    ax.grid(True, linestyle=":", alpha=0.35)
    _save(fig, path)


def _selected_newkirk_model(config):
    multiplier = config.get(
        "selected_newkirk_multiplier", config.get("newkirk_multiplier", 2.0)
    )
    harmonic = config.get("selected_newkirk_harmonic", config.get("harmonic", 2))
    return float(multiplier), float(harmonic)


def _newkirk_reference_height(freq, multiplier, harmonic):
    return _safe_newkirk_value(
        newkirk_height_from_frequency_mhz,
        freq,
        multiplier=multiplier,
        harmonic=harmonic,
    )


def _solar_radius_arcsec(config):
    value = _float_or_nan(config.get("solar_radius_arcsec", 959.63))
    return value if np.isfinite(value) and value > 0 else 959.63


def _safe_newkirk_value(func, *args, **kwargs):
    try:
        return _float_or_nan(func(*args, **kwargs))
    except Exception:
        return np.nan


def _match_drift_for_frequency(freq, drift, freq_tol):
    if drift.empty:
        return None
    best = None
    best_distance = np.inf
    for _, row in drift.iterrows():
        f_start = _float_or_nan(row.get("f_start_mhz"))
        f_end = _float_or_nan(row.get("f_end_mhz"))
        if not np.isfinite(f_start) or not np.isfinite(f_end):
            continue
        lo = min(f_start, f_end) - float(freq_tol)
        hi = max(f_start, f_end) + float(freq_tol)
        if lo <= freq <= hi:
            distance = (
                0.0
                if min(f_start, f_end) <= freq <= max(f_start, f_end)
                else min(abs(freq - f_start), abs(freq - f_end))
            )
            if distance < best_distance:
                best = row
                best_distance = distance
    return best


def _drift_rate_from_row(row):
    drift_rate = _float_or_nan(row.get("drift_rate_mhz_s"))
    if np.isfinite(drift_rate):
        return drift_rate
    f_start = _float_or_nan(row.get("f_start_mhz"))
    f_end = _float_or_nan(row.get("f_end_mhz"))
    duration = _float_or_nan(row.get("duration_s"))
    if not np.isfinite(duration) or abs(duration) <= 1e-12:
        t_start = parse_datetime_value(row.get("t_start"))
        t_end = parse_datetime_value(row.get("t_end"))
        if t_start is not None and t_end is not None:
            duration = (t_end - t_start).total_seconds()
    if (
        np.isfinite(f_start)
        and np.isfinite(f_end)
        and np.isfinite(duration)
        and abs(duration) > 1e-12
    ):
        return (f_end - f_start) / duration
    return np.nan


def _classify_speed_c(v_c):
    if v_c >= 1.0:
        return "physically invalid"
    if v_c > 0.5:
        return "suspiciously high, check frequency drift or density assumption"
    if v_c < 0.01:
        return "too slow for typical type III exciter, check drift selection"
    if v_c <= 0.3:
        return "plausible type III / spike-associated exciter range"
    return "high but sub-relativistic, inspect event context"


def _truthy(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _fmt_float(value, digits=3):
    value = _float_or_nan(value)
    if not np.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def _dashboard_html(payload):
    payload_json = json.dumps(_json_safe(payload), ensure_ascii=False)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Radio Newkirk Frequency Priority Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 18px; color: #202124; }}
.bar {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }}
button {{ padding: 6px 10px; border: 1px solid #888; background: #f5f5f5; cursor: pointer; }}
button.active {{ background: #174ea6; color: white; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 7px; text-align: right; }}
th {{ background: #f1f3f4; }}
td:first-child, th:first-child {{ text-align: left; }}
.panel {{ border: 1px solid #ddd; padding: 10px; }}
svg {{ width: 100%; height: 320px; border: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>Radio/Newkirk Frequency-Priority Dashboard</h1>
<p>Comparisons are grouped by the Gaussian-fitted radio bands. Newkirk is treated as a 1D frequency-to-height model.</p>
<div id="freqButtons" class="bar"></div>
<div class="grid">
  <div class="panel"><h2>Height residual points</h2><svg id="residualPlot"></svg></div>
  <div class="panel"><h2>Model ranking</h2><div id="ranking"></div></div>
  <div class="panel"><h2>Gaussian center summary</h2><div id="centers"></div></div>
  <div class="panel"><h2>Drift matches</h2><div id="drifts"></div></div>
</div>
<script>
const payload = {payload_json};
let selectedFreq = payload.comparison_frequency_mhz[0];
function fmt(v) {{ return Number.isFinite(+v) ? (+v).toFixed(3) : ""; }}
function table(rows, cols) {{
  if (!rows.length) return "<p>No rows</p>";
  return "<table><thead><tr>" + cols.map(c => `<th>${{c}}</th>`).join("") +
    "</tr></thead><tbody>" + rows.map(r => "<tr>" + cols.map(c => `<td>${{r[c] ?? ""}}</td>`).join("") + "</tr>").join("") + "</tbody></table>";
}}
function renderButtons() {{
  const wrap = document.getElementById("freqButtons");
  wrap.innerHTML = "";
  payload.comparison_frequency_mhz.forEach(f => {{
    const b = document.createElement("button");
    b.textContent = `${{f}} MHz`;
    b.className = f === selectedFreq ? "active" : "";
    b.onclick = () => {{ selectedFreq = f; render(); }};
    wrap.appendChild(b);
  }});
}}
function renderPlot() {{
  const svg = document.getElementById("residualPlot");
  const rows = payload.height_points.filter(r => +r.frequency_mhz === +selectedFreq);
  svg.innerHTML = "";
  if (!rows.length) return;
  const w = svg.clientWidth || 500, h = 320, pad = 38;
  const vals = rows.map(r => +r.height_residual_rsun).filter(Number.isFinite);
  const lo = Math.min(-0.05, ...vals), hi = Math.max(0.05, ...vals);
  const models = [...new Set(rows.map(r => `${{r.newkirk_multiplier}}×s=${{r.harmonic}}`))];
  function x(i) {{ return pad + i * (w - 2 * pad) / Math.max(models.length - 1, 1); }}
  function y(v) {{ return h - pad - (v - lo) * (h - 2 * pad) / (hi - lo); }}
  svg.insertAdjacentHTML("beforeend", `<line x1="${{pad}}" x2="${{w-pad}}" y1="${{y(0)}}" y2="${{y(0)}}" stroke="black" stroke-dasharray="3,3"/>`);
  rows.forEach(r => {{
    const idx = models.indexOf(`${{r.newkirk_multiplier}}×s=${{r.harmonic}}`);
    const cx = x(idx) + (Math.random() - 0.5) * 8;
    const cy = y(+r.height_residual_rsun);
    svg.insertAdjacentHTML("beforeend", `<circle cx="${{cx}}" cy="${{cy}}" r="3" fill="#1f77b4" opacity="0.42"><title>${{r.time}} residual=${{fmt(r.height_residual_rsun)}}</title></circle>`);
  }});
  models.forEach((m, i) => svg.insertAdjacentHTML("beforeend", `<text x="${{x(i)}}" y="${{h-10}}" text-anchor="middle" font-size="11">${{m}}</text>`));
}}
function render() {{
  renderButtons();
  renderPlot();
  document.getElementById("ranking").innerHTML = table(payload.model_ranking, ["model_label","median_abs_residual_rsun","n"]);
  document.getElementById("centers").innerHTML = table(payload.center_summary.filter(r => +r.frequency_mhz === +selectedFreq), ["frequency_mhz","n","median_x_arcsec","median_y_arcsec","iqr_x_arcsec","iqr_y_arcsec"]);
  document.getElementById("drifts").innerHTML = table(payload.drift_match_summary.filter(r => +r.frequency_mhz === +selectedFreq), ["frequency_mhz","drift_label","source_type","n"]);
}}
render();
</script>
</body>
</html>
"""


def _json_records(df):
    return _json_safe(pd.DataFrame(df).replace({np.nan: None}).to_dict("records"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def _existing_source_type(row):
    for key in ("source_type", "burst_type", "type"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() != "nan":
            return value
    return "unknown"


def _drift_row_source_type(row):
    for key in ("source_type", "burst_type", "type"):
        value = str(row.get(key) or "").strip()
        if value and value.lower() != "nan":
            return value
    return ""


def _median_or_nan(series):
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if not values.empty else np.nan


def _iqr_or_nan(series):
    if series is None:
        return np.nan
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.quantile(0.75) - values.quantile(0.25))


def _float_or_nan(value):
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _save(fig, output_path, **savefig_kwargs):
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    fig.tight_layout()
    fig.savefig(path, **savefig_kwargs)
    plt.close(fig)
