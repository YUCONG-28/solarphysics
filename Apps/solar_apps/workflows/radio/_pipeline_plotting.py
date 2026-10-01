"""Radio pipeline diagnostic presentation with lazy scientific imports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ._pipeline_products import _pd

if TYPE_CHECKING:
    import pandas as pd


def _np():
    """Load NumPy lazily to keep entrypoint imports independent of BLAS."""
    import numpy as np

    return np


def _plt():
    """Load Matplotlib with the non-interactive backend used for saved figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mdates():
    """Load Matplotlib date helpers only for plotting/time-axis conversion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates

    return mdates


def parse_radio_time_value(value):
    pd = _pd()
    digits = "".join(re.findall(r"\d", str(value)))
    if len(digits) < 14:
        return pd.NaT

    base = digits[:14]
    suffix = digits[14:]
    if suffix:
        if len(suffix) <= 3:
            microsecond = suffix.zfill(3) + "000"
        else:
            microsecond = suffix[:6].ljust(6, "0")
    else:
        microsecond = "000000"

    return pd.to_datetime(base + microsecond, format="%Y%m%d%H%M%S%f", errors="coerce")


def _plot_gaussian_center_trajectory(df: pd.DataFrame, path: Path) -> None:
    from solar_toolkit.radio.quicklook import plot_gaussian_center_trajectory

    plot_gaussian_center_trajectory(df, path)


def _plot_gaussian_center_trajectory_time_colored(df: pd.DataFrame, path: Path) -> None:
    pd = _pd()
    plt = _plt()
    mdates = _mdates()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    if not df.empty:
        data = df.copy()
        data["time_dt"] = data["time"].map(parse_radio_time_value)
        data["freq_num"] = pd.to_numeric(data.get("freq"), errors="coerce")
        data["center_x_num"] = pd.to_numeric(data["center_x_arcsec"], errors="coerce")
        data["center_y_num"] = pd.to_numeric(data["center_y_arcsec"], errors="coerce")
        data = data.dropna(
            subset=["time_dt", "freq_num", "center_x_num", "center_y_num"]
        ).sort_values("time_dt")
        if not data.empty:
            time_nums = mdates.date2num(data["time_dt"])
            ax.plot(
                data["center_x_num"],
                data["center_y_num"],
                color="0.55",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
            sc = ax.scatter(
                data["center_x_num"],
                data["center_y_num"],
                c=time_nums,
                cmap="plasma",
                s=34,
                edgecolors="black",
                linewidths=0.3,
                zorder=2,
            )
            for _, row in data.iterrows():
                ax.annotate(
                    f"{row['freq_num']:.0f}",
                    (row["center_x_num"], row["center_y_num"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6,
                    color="black",
                )
            if len(data) >= 2:
                start = data.iloc[-2]
                end = data.iloc[-1]
                ax.annotate(
                    "",
                    xy=(end["center_x_num"], end["center_y_num"]),
                    xytext=(start["center_x_num"], start["center_y_num"]),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
                )
            cbar = fig.colorbar(sc, ax=ax, label="Time (UT)")
            cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Gaussian center trajectory colored by time")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_gaussian_newkirk_height_time(df: pd.DataFrame, path: Path) -> None:
    pd = _pd()
    np = _np()
    plt = _plt()
    mdates = _mdates()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    if not df.empty:
        data = df.copy()
        data["time_dt"] = data["time"].map(parse_radio_time_value)
        data["newkirk_height_rsun_num"] = pd.to_numeric(
            data["newkirk_height_rsun"], errors="coerce"
        )
        data = data[
            data["time_dt"].notna() & np.isfinite(data["newkirk_height_rsun_num"])
        ]
        for (multiplier, harmonic), group in data.groupby(
            ["newkirk_multiplier", "newkirk_harmonic"]
        ):
            group = group.sort_values("time_dt")
            ax.plot(
                group["time_dt"],
                group["newkirk_height_rsun_num"],
                marker="o",
                linewidth=1.2,
                markersize=3,
                label=f"{multiplier:g}x H{harmonic:g}",
            )
        if not data.empty:
            ax.legend(fontsize=8, ncol=2)
            ax.set_xlim(data["time_dt"].min(), data["time_dt"].max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel("Time (UT)")
    ax.set_ylabel("Newkirk height (Rsun above photosphere)")
    ax.set_title("Gaussian Newkirk height evolution")
    ax.grid(True, linestyle=":", alpha=0.35)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_drift_speed_comparison(df: pd.DataFrame, path: Path) -> None:
    from solar_toolkit.radio.frequency_priority_diagnostics import (
        format_newkirk_case_label,
    )

    pd = _pd()
    np = _np()
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=180)
    if not df.empty:
        data = df.copy()
        data["newkirk_case"] = (
            data["newkirk_multiplier"].map(lambda v: f"{float(v):g}x")
            + "H"
            + data["newkirk_harmonic"].map(lambda v: f"{float(v):g}")
        )
        case_labels = {
            row["newkirk_case"]: format_newkirk_case_label(
                row["newkirk_multiplier"], row["newkirk_harmonic"], compact=True
            )
            for _, row in data.drop_duplicates(subset=["newkirk_case"]).iterrows()
        }
        drift_rates = (
            data.assign(
                drift_rate_num=pd.to_numeric(
                    data.get("drift_rate_mhz_s"), errors="coerce"
                )
            )
            .drop_duplicates(subset=["label"])
            .set_index("label")["drift_rate_num"]
            .to_dict()
        )
        speed_col = (
            "newkirk_speed_km_s"
            if "newkirk_speed_km_s" in data.columns
            else "speed_km_s"
        )
        data["speed_km_s_num"] = pd.to_numeric(data[speed_col], errors="coerce")
        if "newkirk_speed_c" in data.columns:
            data["speed_c_num"] = pd.to_numeric(
                data["newkirk_speed_c"], errors="coerce"
            )
        else:
            data["speed_c_num"] = data["speed_km_s_num"] / 299792.458
        heatmap = data.pivot_table(
            index="label",
            columns="newkirk_case",
            values="speed_km_s_num",
            aggfunc="mean",
        )
        heatmap_c = data.pivot_table(
            index="label",
            columns="newkirk_case",
            values="speed_c_num",
            aggfunc="mean",
        )
        desired_cols = ["1xH1", "1xH2", "2xH1", "2xH2", "4xH1", "4xH2"]
        existing_cols = [col for col in desired_cols if col in heatmap.columns]
        extra_cols = [col for col in heatmap.columns if col not in existing_cols]
        heatmap = heatmap.reindex(columns=existing_cols + extra_cols)
        heatmap_c = heatmap_c.reindex(index=heatmap.index, columns=heatmap.columns)
        values = heatmap.to_numpy(dtype=float)
        c_values = heatmap_c.to_numpy(dtype=float)
        if values.size:
            finite_mean = np.nanmean(values) if np.isfinite(values).any() else np.nan
            im = ax.imshow(values, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(heatmap.columns)))
            ax.set_xticklabels(
                [case_labels.get(col, col) for col in heatmap.columns],
                rotation=40,
                ha="right",
                fontsize=8,
            )
            ax.set_yticks(np.arange(len(heatmap.index)))
            ax.set_yticklabels(
                [
                    (
                        f"{label}\n{drift_rates[label]:.2f} MHz/s"
                        if label in drift_rates and np.isfinite(drift_rates[label])
                        else label
                    )
                    for label in heatmap.index
                ]
            )
            for y_idx in range(values.shape[0]):
                for x_idx in range(values.shape[1]):
                    value = values[y_idx, x_idx]
                    c_value = c_values[y_idx, x_idx] if c_values.size else np.nan
                    if np.isfinite(value):
                        ax.text(
                            x_idx,
                            y_idx,
                            (
                                f"{value:.0f}\n{c_value:.2f}c"
                                if np.isfinite(c_value)
                                else f"{value:.0f}"
                            ),
                            ha="center",
                            va="center",
                            color="white" if value > finite_mean else "black",
                            fontsize=7,
                        )
            fig.colorbar(im, ax=ax, label="Newkirk-inferred exciter speed (km/s)")
    ax.set_xlabel("Density / emission assumption")
    ax.set_ylabel("Drift label")
    ax.set_title("Drift-rate-derived Newkirk exciter speed comparison")
    fig.text(
        0.5,
        0.02,
        "Note: 1xH2 and 4xH1 are degenerate because the inferred height depends on N*s^2.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
