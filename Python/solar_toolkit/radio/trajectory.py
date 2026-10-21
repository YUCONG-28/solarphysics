"""Normalize and select radio-source trajectory tables.

English: Provide one table contract for threshold/contour centers and existing
Gaussian diagnostics, then select the centers visible at a playback frame.

中文：统一阈值中心表和既有 Gaussian 诊断表字段，并为前端播放帧选择当前、
前 N 帧或截至当前的全部轨迹点。
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..time import normalize_time
from ._time import table_timestamp
from .centers import POL_LCP, POL_RCP, POL_SUM, POL_UNKNOWN
from .io import truthy

__all__ = [
    "FRAME_MODE_ALL",
    "FRAME_MODE_CURRENT",
    "FRAME_MODE_LABELS",
    "FRAME_MODE_TAIL",
    "STANDARD_COLUMNS",
    "filter_centers",
    "filter_time_range",
    "frame_times",
    "load_centers_table",
    "make_lr_compare_table",
    "normalize_centers_dataframe",
    "normalize_frame_mode",
    "normalize_polarization_label",
    "select_visible_centers",
    "summarize_motion",
]

FRAME_MODE_CURRENT = "current"
FRAME_MODE_TAIL = "tail"
FRAME_MODE_ALL = "all"

FRAME_MODE_LABELS = {
    FRAME_MODE_CURRENT: "Current centers",
    FRAME_MODE_TAIL: "Previous N-frame trail",
    FRAME_MODE_ALL: "All points up to current time",
}

STANDARD_COLUMNS = [
    "obs_time",
    "freq_mhz",
    "polarization",
    "center_x_arcsec",
    "center_y_arcsec",
    "center_method",
    "quality_flag",
    "source_label",
]


def load_centers_table(path_or_buffer, *, valid_only: bool = True) -> pd.DataFrame:
    """Read CSV/XLSX radio center tables and normalize their columns.

    CSV ``time``/``obs_time`` columns are read as text before numeric inference,
    preserving compact calendar digits even when another row has no timestamp.
    Spreadsheet compact timestamps should likewise be stored as text to avoid
    precision loss in the spreadsheet itself.
    """

    if isinstance(path_or_buffer, str | Path):
        path = Path(path_or_buffer).expanduser()
        if path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, dtype={"time": "string", "obs_time": "string"})
    else:
        name = str(getattr(path_or_buffer, "name", ""))
        if name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path_or_buffer)
        else:
            df = pd.read_csv(
                path_or_buffer, dtype={"time": "string", "obs_time": "string"}
            )
    return normalize_centers_dataframe(df, valid_only=valid_only)


def normalize_centers_dataframe(
    df: pd.DataFrame, *, valid_only: bool = True
) -> pd.DataFrame:
    """Return a standard trajectory DataFrame.

    The input may be a threshold-center table with ``obs_time``/``freq_mhz`` or
    a Gaussian diagnostics table with ``time``/``freq``. Returned times are
    timezone-naive UTC. Absent validity columns retain the legacy table
    contract; present but unknown flags are rejected when ``valid_only`` is true.
    Compact calendar integers are parsed like strings. Floats, unlabelled
    numeric epochs and missing/invalid times are excluded for both validity
    modes, since their observation instant is unknown. The input DataFrame is
    not modified.
    """

    if df is None:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    source = df.copy()
    original_columns = set(source.columns)
    if "obs_time" not in source.columns and "time" in source.columns:
        source["obs_time"] = source["time"]
    if "freq_mhz" not in source.columns and "freq" in source.columns:
        source["freq_mhz"] = source["freq"]
    if "polarization" not in source.columns:
        source["polarization"] = POL_UNKNOWN
    if "center_method" not in source.columns:
        source["center_method"] = (
            "gaussian" if {"time", "freq"} & original_columns else "threshold"
        )
    if "quality_flag" not in source.columns:
        source["quality_flag"] = "ok"
    if "source_label" not in source.columns:
        source["source_label"] = "main"

    missing = [
        column
        for column in (
            "obs_time",
            "freq_mhz",
            "polarization",
            "center_x_arcsec",
            "center_y_arcsec",
        )
        if column not in source.columns
    ]
    if missing:
        raise ValueError(f"Center table missing required columns: {missing}")

    source["obs_time"] = source["obs_time"].map(_to_timestamp)
    for column in ("freq_mhz", "center_x_arcsec", "center_y_arcsec"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["polarization"] = source["polarization"].map(normalize_polarization_label)
    source["center_method"] = source["center_method"].fillna("unknown").astype(str)
    source["quality_flag"] = source["quality_flag"].fillna("unknown").astype(str)
    source["source_label"] = source["source_label"].fillna("main").astype(str)

    source = source.dropna(
        subset=["obs_time", "freq_mhz", "center_x_arcsec", "center_y_arcsec"]
    )
    if valid_only:
        mask = source["quality_flag"].str.strip().str.lower().eq("ok")
        mask &= source["freq_mhz"].gt(0)
        for column in ("freq_mhz", "center_x_arcsec", "center_y_arcsec"):
            mask &= source[column].map(math.isfinite)
        if "overlay_valid" in source.columns:
            mask &= source["overlay_valid"].map(truthy)
        if "trajectory_valid" in source.columns:
            mask &= source["trajectory_valid"].map(truthy)
        source = source.loc[mask].copy()

    extras = [column for column in source.columns if column not in STANDARD_COLUMNS]
    return (
        source[STANDARD_COLUMNS + extras]
        .sort_values(["obs_time", "freq_mhz", "polarization", "center_method"])
        .reset_index(drop=True)
    )


def normalize_polarization_label(value: object) -> str:
    """Normalize common L/R/Stokes-I labels without losing unknown labels."""

    if value is None:
        return POL_UNKNOWN
    text = str(value).strip()
    if not text:
        return POL_UNKNOWN
    low = text.lower().replace("_", "").replace("-", "").replace(" ", "")
    if low in {"l+r", "rr+ll", "ll+rr", "r+l", "lr", "stokesi", "i", "total"}:
        return POL_SUM
    if low in {"lcp", "lhcp", "ll", "left", "lefthand", "leftcirc"}:
        return POL_LCP
    if low in {"rcp", "rhcp", "rr", "right", "righthand", "rightcirc"}:
        return POL_RCP
    return text


def _to_timestamp(value) -> pd.Timestamp:
    return table_timestamp(value)


def filter_centers(
    df: pd.DataFrame,
    *,
    freqs: list[float] | tuple[float, ...] | None = None,
    polarizations: list[str] | tuple[str, ...] | None = None,
    center_methods: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Filter normalized centers by frequency, polarization, and method."""

    result = df.copy()
    if freqs:
        freq_set = {float(freq) for freq in freqs}
        result = result[result["freq_mhz"].astype(float).isin(freq_set)]
    if polarizations:
        pol_set = {normalize_polarization_label(pol) for pol in polarizations}
        result = result[result["polarization"].isin(pol_set)]
    if center_methods:
        method_set = {str(method) for method in center_methods}
        result = result[result["center_method"].isin(method_set)]
    return result.reset_index(drop=True)


def filter_time_range(
    df: pd.DataFrame,
    *,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Filter centers to an inclusive observation-time range."""

    if df.empty:
        return df.copy()
    result = df.copy()
    times = result["obs_time"].map(table_timestamp)
    mask = times.notna()
    start_ts = _optional_timestamp(start)
    end_ts = _optional_timestamp(end)
    if start_ts is not None:
        mask &= times >= start_ts
    if end_ts is not None:
        mask &= times <= end_ts
    return result.loc[mask].reset_index(drop=True)


def summarize_motion(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize start-to-end center displacement for each trajectory group."""

    columns = [
        "freq_mhz",
        "polarization",
        "center_method",
        "source_label",
        "point_count",
        "start_time",
        "end_time",
        "duration_sec",
        "start_x_arcsec",
        "start_y_arcsec",
        "end_x_arcsec",
        "end_y_arcsec",
        "dx_arcsec",
        "dy_arcsec",
        "distance_arcsec",
        "mean_speed_arcsec_s",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    group_cols = [
        column
        for column in ("freq_mhz", "polarization", "center_method", "source_label")
        if column in df.columns
    ]
    for group_key, group in df.groupby(group_cols, dropna=False, sort=True):
        ordered = group.sort_values("obs_time")
        first = ordered.iloc[0]
        last = ordered.iloc[-1]
        start_time = pd.Timestamp(first["obs_time"])
        end_time = pd.Timestamp(last["obs_time"])
        duration_sec = (
            normalize_time(end_time).tai_ns - normalize_time(start_time).tai_ns
        ) / 1_000_000_000
        dx = float(last["center_x_arcsec"]) - float(first["center_x_arcsec"])
        dy = float(last["center_y_arcsec"]) - float(first["center_y_arcsec"])
        distance = float(math.hypot(dx, dy))
        row = {
            "point_count": int(len(ordered)),
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration_sec,
            "start_x_arcsec": float(first["center_x_arcsec"]),
            "start_y_arcsec": float(first["center_y_arcsec"]),
            "end_x_arcsec": float(last["center_x_arcsec"]),
            "end_y_arcsec": float(last["center_y_arcsec"]),
            "dx_arcsec": dx,
            "dy_arcsec": dy,
            "distance_arcsec": distance,
            "mean_speed_arcsec_s": distance / duration_sec if duration_sec > 0 else 0.0,
        }
        _attach_group_values(row, group_cols, group_key)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _optional_timestamp(value) -> pd.Timestamp | None:
    if value is None or isinstance(value, str) and value == "":
        return None
    timestamp = table_timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Time bounds must be valid explicit calendar timestamps")
    return timestamp


def frame_times(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return sorted unique playback frame times."""

    if df.empty:
        return []
    return sorted(df["obs_time"].map(table_timestamp).dropna().unique().tolist())


def select_visible_centers(
    df: pd.DataFrame,
    frame_time,
    *,
    mode: str = FRAME_MODE_TAIL,
    tail_n: int = 5,
) -> pd.DataFrame:
    """Select centers visible at one playback frame."""

    if df.empty:
        return df.copy()
    resolved_mode = normalize_frame_mode(mode)
    timestamp = _optional_timestamp(frame_time)
    if timestamp is None:
        raise ValueError("frame_time must be a valid explicit calendar timestamp")
    past = df[df["obs_time"].map(table_timestamp) <= timestamp].copy()
    if past.empty:
        return past
    group_cols = [
        column
        for column in ("freq_mhz", "polarization", "center_method", "source_label")
        if column in past.columns
    ]
    if resolved_mode == FRAME_MODE_CURRENT:
        return past.groupby(group_cols, group_keys=False).tail(1).reset_index(drop=True)
    if resolved_mode == FRAME_MODE_TAIL:
        return (
            past.groupby(group_cols, group_keys=False)
            .tail(max(1, int(tail_n)))
            .reset_index(drop=True)
        )
    return past.reset_index(drop=True)


def normalize_frame_mode(mode: str) -> str:
    """Normalize English or Chinese UI frame-mode labels."""

    text = str(mode or "").strip().lower()
    if text in {FRAME_MODE_CURRENT, "current center", "当前中心"}:
        return FRAME_MODE_CURRENT
    if text in {FRAME_MODE_TAIL, "tail", "前 n 帧尾迹", "前n帧尾迹"}:
        return FRAME_MODE_TAIL
    if text in {FRAME_MODE_ALL, "all", "全部轨迹", "截至当前全部轨迹"}:
        return FRAME_MODE_ALL
    return FRAME_MODE_TAIL


def make_lr_compare_table(
    visible: pd.DataFrame, *, tolerance_sec: float = 1.0
) -> pd.DataFrame:
    """Pair visible LCP/RCP centers and calculate separation vectors."""

    rows: list[dict[str, object]] = []
    if visible.empty:
        return pd.DataFrame(rows)
    group_cols = [
        column
        for column in ("freq_mhz", "center_method", "source_label")
        if column in visible.columns
    ]
    for group_key, group in visible.groupby(group_cols):
        ldf = group[group["polarization"] == POL_LCP].sort_values("obs_time")
        rdf = group[group["polarization"] == POL_RCP].sort_values("obs_time")
        if ldf.empty or rdf.empty:
            continue
        merged = pd.merge_asof(
            ldf,
            rdf,
            on="obs_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=float(tolerance_sec)),
            suffixes=("_L", "_R"),
        ).dropna(subset=["center_x_arcsec_R", "center_y_arcsec_R"])
        for _, row in merged.iterrows():
            dx = float(row["center_x_arcsec_L"] - row["center_x_arcsec_R"])
            dy = float(row["center_y_arcsec_L"] - row["center_y_arcsec_R"])
            result = {
                "obs_time": row["obs_time"],
                "freq_mhz": row["freq_mhz_L"],
                "center_method": row.get("center_method_L", ""),
                "source_label": row.get("source_label_L", ""),
                "L_x_arcsec": row["center_x_arcsec_L"],
                "L_y_arcsec": row["center_y_arcsec_L"],
                "R_x_arcsec": row["center_x_arcsec_R"],
                "R_y_arcsec": row["center_y_arcsec_R"],
                "dx_L_minus_R_arcsec": dx,
                "dy_L_minus_R_arcsec": dy,
                "distance_arcsec": float(math.hypot(dx, dy)),
            }
            _attach_group_values(result, group_cols, group_key)
            rows.append(result)
    return pd.DataFrame(rows)


def _attach_group_values(
    row: dict[str, object], group_cols: list[str], group_key
) -> None:
    values = group_key if isinstance(group_key, tuple) else (group_key,)
    for column, value in zip(group_cols, values, strict=False):
        row[column] = value
