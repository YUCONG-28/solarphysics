"""Time-based selection helpers.

English: Select the nearest observation or filter observations to an inclusive
time range.

中文：选择最接近目标时间的观测，或按闭区间筛选观测。
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from .parsing import parse_time
from .precision import normalize_time, seconds_to_ns

T = TypeVar("T")


def nearest_by_time(
    target_time: str | dt.datetime | dt.date,
    items: Sequence[T],
    *,
    key: Callable[[T], str | dt.datetime | dt.date] | None = None,
    max_diff_seconds: float | None = None,
) -> T | None:
    """Return the item nearest to ``target_time`` or ``None`` past a threshold."""

    limit = None
    if max_diff_seconds is not None:
        limit = seconds_to_ns(max_diff_seconds, nonnegative=True)
    if not items:
        return None
    target = _timestamp_ns(target_time)
    resolver = key or _default_time_key
    closest = min(
        items,
        key=lambda item: abs(_timestamp_ns(resolver(item)) - target),
    )
    difference = abs(_timestamp_ns(resolver(closest)) - target)
    if limit is not None and difference > limit:
        return None
    return closest


def filter_by_time_range(
    items: Iterable[T],
    start_time: str | dt.datetime | dt.date,
    end_time: str | dt.datetime | dt.date,
    *,
    key: Callable[[T], str | dt.datetime | dt.date] | None = None,
) -> list[T]:
    """Return items whose resolved time falls within the inclusive range."""

    start = _timestamp_ns(start_time)
    end = _timestamp_ns(end_time)
    if start > end:
        raise ValueError("start_time must be before or equal to end_time")
    resolver = key or _default_time_key
    return [item for item in items if start <= _timestamp_ns(resolver(item)) <= end]


def _timestamp_ns(value):
    # Preserve explicitly date-only legacy calls as midnight. Complete strings
    # go directly to the high-precision parser, never through datetime first.
    if (isinstance(value, dt.date) and not isinstance(value, dt.datetime)) or (
        isinstance(value, str)
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{8}", value.strip())
    ):
        value = parse_time(value)
    return normalize_time(value).tai_ns


def _default_time_key(item):
    if isinstance(item, tuple) and len(item) >= 2:
        return item[1]
    if isinstance(item, dict):
        for name in ("obs_time", "time", "date_obs", "DATE-OBS"):
            if name in item:
                return item[name]
    return item


__all__ = ["filter_by_time_range", "nearest_by_time"]
