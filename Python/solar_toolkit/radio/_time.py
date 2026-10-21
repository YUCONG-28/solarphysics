"""Legacy table views of the shared exact radio clock.

Scientific durations and matching use TimeStamp.tai_ns. Pandas views preserve
nanoseconds for display/tables, but cannot represent a UTC leap second.
"""

from datetime import timezone

import pandas as pd

from solar_toolkit.time import TimeStamp, normalize_time

from .io import parse_timestamp_value


def table_timestamp(value, *, utc=False):
    """Return a UTC pandas view, NaT for invalid input, or reject a leap second."""
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC")
        return (
            (value.tz_localize("UTC") if value.tzinfo is None else value)
            if utc
            else (value.tz_localize(None) if value.tzinfo is not None else value)
        )
    stamp = parse_timestamp_value(value)
    if stamp is None:
        return pd.NaT
    stamp.to_datetime()  # Explicit datetime representability check, not a conversion.
    return pd.Timestamp(stamp.utc, tz="UTC" if utc else None)


def shift_datetime(value, seconds):
    """Shift by SI seconds, returning a legacy UTC datetime view.

    Aware inputs stay UTC aware. Finer than microsecond precision is truncated
    only at this display boundary; a result inside a leap second is rejected.
    """
    result = normalize_time(value).shift(seconds).to_datetime()
    return (
        result.replace(tzinfo=timezone.utc)
        if getattr(value, "tzinfo", None)
        else result
    )


def midpoint_datetime(start, end):
    """Return the SI midpoint as a legacy UTC datetime, rejecting leap seconds."""
    summed_ns = normalize_time(start).tai_ns + normalize_time(end).tai_ns
    midpoint_ns, remainder = divmod(summed_ns, 2)
    midpoint_ns += remainder * (midpoint_ns % 2)  # Half-even at one nanosecond.
    result = TimeStamp.from_tai_ns(midpoint_ns).to_datetime()
    return (
        result.replace(tzinfo=timezone.utc)
        if getattr(start, "tzinfo", None)
        else result
    )
