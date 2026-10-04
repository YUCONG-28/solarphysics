"""Tabular time-series normalization and selection.

English: Standardize pandas time columns and select an inclusive time range.

中文：规范 pandas 时间列，并按闭区间筛选时间序列。
"""

from __future__ import annotations

import pandas as pd


def normalize_time_column(
    frame: pd.DataFrame,
    *,
    source_column: str = "time",
    target_column: str = "obs_time",
) -> pd.DataFrame:
    """Return a copy with ``target_column`` as timezone-naive UTC timestamps."""

    result = frame.copy()
    times = pd.to_datetime(result[source_column], utc=True, errors="raise")
    result[target_column] = times.dt.tz_convert(None)
    return result


def crop_time_range(
    frame: pd.DataFrame,
    start_time=None,
    end_time=None,
    *,
    time_column: str = "obs_time",
) -> pd.DataFrame:
    """Copy rows inside optional inclusive UTC bounds, preserving row order.

    Naive bounds are interpreted as UTC; aware bounds are converted to UTC.
    Missing bounds leave that side open. Invalid or reversed bounds raise
    ``ValueError``.
    """
    start, end = _normalize_time_bounds(start_time, end_time)
    if start is None and end is None:
        return frame.copy()
    times = pd.to_datetime(frame[time_column], utc=True).dt.tz_convert(None)
    mask = times.notna()
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times <= end
    return frame.loc[mask].copy()


def _normalize_time_bounds(start_time, end_time):
    """Normalize optional scalar bounds to timezone-naive UTC timestamps."""
    bounds = []
    for name, value in (("start_time", start_time), ("end_time", end_time)):
        if value is None:
            bounds.append(None)
            continue
        try:
            bound = pd.Timestamp(value)
            if pd.isna(bound):
                raise ValueError("missing timestamp")
            if bound.tzinfo is None:
                bound = bound.tz_localize("UTC")
            bounds.append(bound.tz_convert(None))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a valid scalar UTC timestamp") from exc
    start, end = bounds
    if start is not None and end is not None and start > end:
        raise ValueError("start_time must not be after end_time")
    return start, end


__all__ = ["crop_time_range", "normalize_time_column"]
