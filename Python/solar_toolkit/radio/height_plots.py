"""Plot helpers for Gaussian-Newkirk height comparison diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .height_comparison import model_label
from .io import ensure_output_dir
from .newkirk import newkirk_height_from_frequency_mhz

__all__ = [
    "plot_gaussian_vs_newkirk_height_frequency",
    "plot_gaussian_vs_newkirk_height_time",
    "plot_height_residual_vs_frequency",
]


def plot_gaussian_vs_newkirk_height_frequency(height_df, output_path, config=None):
    cfg = dict(config or {})
    df = _clean_height_df(height_df)
    if df.empty or not _has_columns(df, ["frequency_mhz", "gaussian_height_rsun"]):
        return {"status": "skipped", "reason": "no_valid_height_rows"}
    valid = df.dropna(subset=["frequency_mhz", "gaussian_height_rsun"])
    if valid.empty:
        return {"status": "skipped", "reason": "no_valid_height_rows"}
    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (8.5, 6)), dpi=int(cfg.get("dpi", 180))
    )
    _scatter_gaussian_height_by_frequency(ax, df, cfg)
    _plot_gaussian_frequency_median_iqr(ax, df)
    best_pair = _best_residual_model_pair(df)
    freqs = _frequency_grid(df)
    for multiplier, harmonic in _model_pairs(df):
        heights = newkirk_height_from_frequency_mhz(
            freqs, multiplier=multiplier, harmonic=harmonic
        )
        is_best = best_pair == (float(multiplier), float(harmonic))
        ax.plot(
            freqs,
            heights,
            linewidth=2.4 if is_best else 1.2,
            alpha=0.95 if is_best else 0.70,
            label=(
                f"{model_label(multiplier, harmonic)} Newkirk radial height (best median |residual|)"
                if is_best
                else f"{model_label(multiplier, harmonic)} Newkirk radial height"
            ),
            zorder=4 if is_best else 2,
        )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Height above photosphere (Rsun)")
    ax.set_title("Projected Gaussian source height vs Newkirk radial height")
    if cfg.get("reverse_frequency_axis", True):
        ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.35)
    _legend_if_needed(ax, fontsize=8, ncol=2)
    _save(fig, output_path)
    return {"status": "saved", "path": str(output_path), "best_model": best_pair}


def plot_gaussian_vs_newkirk_height_time(height_df, output_path, config=None):
    cfg = dict(config or {})
    df = _clean_height_df(height_df)
    if df.empty or not _has_columns(df, ["time_dt", "gaussian_height_rsun"]):
        return {"status": "skipped", "reason": "no_valid_time_height_rows"}
    valid = df.dropna(subset=["time_dt", "gaussian_height_rsun"])
    if valid.empty:
        return {"status": "skipped", "reason": "no_valid_time_height_rows"}
    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (9, 5.5)), dpi=int(cfg.get("dpi", 180))
    )
    connected_line_count = 0
    one_per_source = df.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    ).copy()
    for _source_type, group in one_per_source.groupby("source_type", dropna=False):
        group = group.dropna(subset=["time_dt", "gaussian_height_rsun"]).sort_values(
            "time_dt"
        )
        if group.empty:
            continue
        ax.scatter(
            group["time_dt"],
            group["gaussian_height_rsun"],
            s=24,
            marker="o",
            label="Gaussian center projected height",
            alpha=0.65,
            edgecolors="none",
        )
    newkirk = df.dropna(subset=["time_dt", "newkirk_height_rsun"]).copy()
    for (multiplier, harmonic, source_type), group in newkirk.groupby(
        ["newkirk_multiplier", "harmonic", "source_type"], dropna=False
    ):
        group = group.sort_values("time_dt")
        ax.scatter(
            group["time_dt"],
            group["newkirk_height_rsun"],
            s=14,
            marker="x",
            alpha=0.35,
            label=f"{model_label(multiplier, harmonic)} {source_type} raw",
        )
    if "drift_label" in newkirk.columns:
        ridge = newkirk[newkirk["drift_label"].astype(str).str.strip().ne("")]
        for (multiplier, harmonic, source_type, drift_label), group in ridge.groupby(
            ["newkirk_multiplier", "harmonic", "source_type", "drift_label"],
            dropna=False,
        ):
            group = group.dropna(subset=["time_dt", "newkirk_height_rsun"]).sort_values(
                "time_dt"
            )
            if len(group) < 2:
                continue
            ax.plot(
                group["time_dt"],
                group["newkirk_height_rsun"],
                linewidth=1.35,
                linestyle="--",
                label=(
                    f"{model_label(multiplier, harmonic)} {source_type} "
                    f"{drift_label}"
                ),
                alpha=0.90,
            )
            connected_line_count += 1
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Height (Rsun above photosphere)")
    ax.set_title("Gaussian and Newkirk height evolution")
    ax.grid(True, linestyle=":", alpha=0.35)
    _legend_if_needed(ax, fontsize=7, ncol=2)
    _save(fig, output_path)
    return {
        "status": "saved",
        "path": str(output_path),
        "connected_line_count": connected_line_count,
    }


def plot_height_residual_vs_frequency(height_df, output_path, config=None):
    cfg = dict(config or {})
    df = _clean_height_df(height_df)
    if df.empty or not _has_columns(df, ["frequency_mhz", "height_residual_rsun"]):
        return {"status": "skipped", "reason": "no_valid_residual_rows"}
    valid = df.dropna(subset=["frequency_mhz", "height_residual_rsun"])
    if valid.empty:
        return {"status": "skipped", "reason": "no_valid_residual_rows"}
    summary_path = Path(output_path).parent / cfg.get(
        "height_residual_summary_name",
        "gaussian_newkirk_height_residual_summary.csv",
    )
    summary = _write_residual_summary(valid, summary_path)
    fig, ax = plt.subplots(
        figsize=cfg.get("figsize", (8.5, 5.5)), dpi=int(cfg.get("dpi", 180))
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":", label="zero residual")
    for (multiplier, harmonic, source_type), group in valid.groupby(
        ["newkirk_multiplier", "harmonic", "source_type"], dropna=False
    ):
        outlier = group["height_residual_rsun"].abs() > float(
            cfg.get("outlier_residual_rsun", 0.5)
        )
        ax.scatter(
            group.loc[~outlier, "frequency_mhz"],
            group.loc[~outlier, "height_residual_rsun"],
            s=18,
            alpha=0.35,
            label=f"{model_label(multiplier, harmonic)} {source_type}",
        )
        if outlier.any() and cfg.get("mark_outliers", True):
            ax.scatter(
                group.loc[outlier, "frequency_mhz"],
                group.loc[outlier, "height_residual_rsun"],
                s=32,
                marker="x",
                alpha=0.85,
                label=f"{model_label(multiplier, harmonic)} outlier",
            )
    _plot_residual_median_iqr(ax, summary)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Gaussian height - Newkirk height (Rsun)")
    ax.set_title("Gaussian-Newkirk height residual vs frequency")
    if cfg.get("reverse_frequency_axis", True):
        ax.invert_xaxis()
    ax.grid(True, linestyle=":", alpha=0.35)
    _legend_if_needed(ax, fontsize=7, ncol=2)
    _save(fig, output_path)
    return {
        "status": "saved",
        "path": str(output_path),
        "summary_csv": str(summary_path),
    }


def _scatter_gaussian_height_by_frequency(ax, df: pd.DataFrame, cfg: dict):
    one_per_source = df.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    ).copy()
    if "gaussian_projected_height_valid" in one_per_source.columns:
        valid_mask = one_per_source["gaussian_projected_height_valid"].map(_truthy)
    else:
        valid_mask = one_per_source["gaussian_height_rsun"] >= 0
    invalid = one_per_source[~valid_mask].dropna(subset=["gaussian_height_rsun"])
    one_per_source = one_per_source[valid_mask]
    if not invalid.empty and cfg.get("show_invalid_projected_heights", True):
        ax.scatter(
            invalid["frequency_mhz"],
            invalid["gaussian_height_rsun"],
            s=28,
            marker="x",
            alpha=0.65,
            color="0.5",
            label="Invalid projected height",
            zorder=2,
        )
    color_by = str(cfg.get("color_by", "source_type")).lower()
    if color_by == "time":
        colors = mdates.date2num(one_per_source["time_dt"])
        sc = ax.scatter(
            one_per_source["frequency_mhz"],
            one_per_source["gaussian_height_rsun"],
            c=colors,
            cmap=cfg.get("gaussian_cmap", "plasma"),
            s=36,
            edgecolors="black",
            linewidths=0.35,
            label="Gaussian center projected height",
            zorder=3,
        )
        cbar = ax.figure.colorbar(sc, ax=ax)
        cbar.set_label("Time (UT)")
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        return
    for _source_type, group in one_per_source.groupby("source_type", dropna=False):
        ax.scatter(
            group["frequency_mhz"],
            group["gaussian_height_rsun"],
            s=20,
            alpha=0.45,
            edgecolors="black",
            linewidths=0.35,
            label="Gaussian center projected height",
            zorder=3,
        )


def _clean_height_df(height_df) -> pd.DataFrame:
    df = pd.DataFrame(height_df).copy()
    if df.empty:
        return df
    numeric_cols = [
        "frequency_mhz",
        "gaussian_height_rsun",
        "newkirk_height_rsun",
        "height_residual_rsun",
        "newkirk_multiplier",
        "harmonic",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    if "source_type" in df.columns:
        df["source_type"] = df["source_type"].fillna("unknown").astype(str)
    else:
        df["source_type"] = "unknown"
    if "drift_label" in df.columns:
        df["drift_label"] = df["drift_label"].fillna("").astype(str)
    else:
        df["drift_label"] = ""
    df["time_dt"] = pd.to_datetime(df.get("time"), errors="coerce")
    return df.dropna(subset=["frequency_mhz"])


def _plot_gaussian_frequency_median_iqr(ax, df: pd.DataFrame):
    one_per_source = df.drop_duplicates(
        subset=["time", "frequency_mhz", "gaussian_x_arcsec", "gaussian_y_arcsec"]
    ).dropna(subset=["frequency_mhz", "gaussian_height_rsun"])
    if "gaussian_projected_height_valid" in one_per_source.columns:
        one_per_source = one_per_source[
            one_per_source["gaussian_projected_height_valid"].map(_truthy)
        ]
    else:
        one_per_source = one_per_source[one_per_source["gaussian_height_rsun"] >= 0]
    summary = _median_iqr(
        one_per_source,
        ["frequency_mhz"],
        "gaussian_height_rsun",
    )
    if summary.empty:
        return
    color = "#d62728"
    ax.errorbar(
        summary["frequency_mhz"],
        summary["median"],
        yerr=[summary["median"] - summary["q25"], summary["q75"] - summary["median"]],
        fmt="o",
        markersize=9,
        color=color,
        ecolor=color,
        elinewidth=3.0,
        capsize=6,
        capthick=3.0,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="Gaussian median ± IQR (Q1–Q3)",
        zorder=6,
    )


def _plot_residual_median_iqr(ax, summary: pd.DataFrame):
    if summary.empty:
        return
    for (multiplier, harmonic), group in summary.groupby(
        ["newkirk_multiplier", "harmonic"], dropna=False
    ):
        half_iqr = 0.5 * group["iqr_residual_rsun"]
        ax.errorbar(
            group["frequency_mhz"],
            group["median_residual_rsun"],
            yerr=half_iqr,
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1.0,
            label=f"{model_label(multiplier, harmonic)} median ± IQR",
        )


def _write_residual_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(
        ["newkirk_multiplier", "harmonic", "frequency_mhz"], dropna=False
    )
    for (multiplier, harmonic, frequency), group in grouped:
        residual = pd.to_numeric(
            group["height_residual_rsun"], errors="coerce"
        ).dropna()
        if residual.empty:
            continue
        q25 = float(residual.quantile(0.25))
        q75 = float(residual.quantile(0.75))
        rows.append(
            {
                "newkirk_multiplier": float(multiplier),
                "harmonic": harmonic,
                "frequency_mhz": float(frequency),
                "n": int(len(residual)),
                "median_residual_rsun": float(residual.median()),
                "iqr_residual_rsun": q75 - q25,
                "mean_abs_residual_rsun": float(residual.abs().mean()),
                "outlier_count": int((residual.abs() > 0.5).sum()),
            }
        )
    summary = pd.DataFrame(
        rows,
        columns=[
            "newkirk_multiplier",
            "harmonic",
            "frequency_mhz",
            "n",
            "median_residual_rsun",
            "iqr_residual_rsun",
            "mean_abs_residual_rsun",
            "outlier_count",
        ],
    )
    ensure_output_dir(Path(path).parent or ".")
    summary.to_csv(path, index=False)
    return summary


def _median_iqr(
    df: pd.DataFrame, group_cols: list[str], value_col: str
) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        series = pd.to_numeric(group[value_col], errors="coerce").dropna()
        if series.empty:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys, strict=False)}
        row.update(
            {
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "n": int(len(series)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _best_residual_model_pair(df: pd.DataFrame):
    valid = df.dropna(subset=["height_residual_rsun", "newkirk_multiplier", "harmonic"])
    if valid.empty:
        return None
    scores = []
    for (multiplier, harmonic), group in valid.groupby(
        ["newkirk_multiplier", "harmonic"], dropna=False
    ):
        residual = (
            pd.to_numeric(group["height_residual_rsun"], errors="coerce").abs().dropna()
        )
        if residual.empty:
            continue
        scores.append((float(residual.median()), float(multiplier), float(harmonic)))
    if not scores:
        return None
    _, multiplier, harmonic = min(scores)
    return (multiplier, harmonic)


def _legend_if_needed(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(**kwargs)


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok"}
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _model_pairs(df: pd.DataFrame):
    pairs = df[["newkirk_multiplier", "harmonic"]].dropna().drop_duplicates()
    return [
        (float(row["newkirk_multiplier"]), row["harmonic"])
        for _, row in pairs.sort_values(["newkirk_multiplier", "harmonic"]).iterrows()
    ]


def _frequency_grid(df: pd.DataFrame):
    freqs = pd.to_numeric(df["frequency_mhz"], errors="coerce")
    freqs = freqs[np.isfinite(freqs)]
    if freqs.empty:
        return np.asarray([], dtype=float)
    lo, hi = float(freqs.min()), float(freqs.max())
    if lo == hi:
        lo = max(lo * 0.85, 1e-6)
        hi = hi * 1.15
    return np.linspace(lo, hi, 160)


def _save(fig, output_path):
    path = Path(output_path)
    ensure_output_dir(path.parent or ".")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
