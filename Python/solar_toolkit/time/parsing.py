"""Timestamp parsing helpers for solar-observation workflows.

English: Parse timestamps supplied directly or embedded in common local file
names without depending on instrument-specific packages.

中文：解析直接提供的时间戳，以及常见本地观测文件名中包含的时间戳。
"""

from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache

from .precision import TimeStamp, normalize_time


def parse_time(value: str | dt.datetime | dt.date) -> dt.datetime:
    """Parse common values into naive UTC datetimes (microsecond resolution).

    TAI-labelled inputs are converted to UTC, rather than merely losing their
    label. Use ``normalize_time`` when nanoseconds or leap seconds must survive.
    """

    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time())
    else:
        text = str(value).strip()
        if re.search(r"(?:_TAI|\s+TAI)$", text, re.IGNORECASE):
            from .precision import normalize_time

            text = normalize_time(text).utc
        elif re.search(r"(?:_UTC|\s+UTC)$", text, re.IGNORECASE):
            text = re.sub(r"(?:_UTC|\s+UTC)$", "", text, flags=re.IGNORECASE)
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = dt.datetime.fromisoformat(text)
        except ValueError:
            from .precision import normalize_time

            parsed = normalize_time(text).to_datetime()

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_isot_time(time_str: str) -> dt.datetime:
    """Parse an ISO-like timestamp into the same naive UTC legacy type."""

    return parse_time(time_str)


@lru_cache(maxsize=8192)
def extract_timestamp_from_filename(filename: str) -> TimeStamp:
    """Extract a complete observation timestamp, preserving nanoseconds.

    Underscore fractions require a radio/polarization or time-scale label.
    Known AIA wavelength suffixes remain wavelengths; other ambiguous numeric
    suffixes raise rather than silently losing potential fractional seconds.
    """

    text = str(filename)
    separated = re.search(
        r"(?<!\d)(\d{8}[T_-]\d{6}|\d{4}-\d{2}-\d{2}T\d{6})_(\d+)"
        r"(?:_(TAI|UTC))?(?!\d)",
        text,
        re.IGNORECASE,
    )
    if separated:
        basename = text.replace("\\", "/").rsplit("/", 1)[-1]
        aia = re.search(r"(?:^|[_.-])aia(?:[_.-]|$)", basename, re.IGNORECASE)
        radio = re.search(
            r"(?:^|[/\\_.-])(?:radio|cso|dart|muser|rr|ll|rcp|lcp)(?:[/\\_.-]|$)",
            text,
            re.IGNORECASE,
        )
        is_aia_band = aia and int(separated[2]) in {
            94,
            131,
            171,
            193,
            211,
            304,
            335,
            1600,
            1700,
            4500,
        }
        if separated[3] or (radio and not is_aia_band):
            if len(separated[2]) > 9:
                raise ValueError(
                    f"Could not extract time from filename: {filename}: fraction exceeds nanoseconds"
                )
            try:
                return normalize_time(separated.group(), source="filename")
            except ValueError as exc:
                raise ValueError(
                    f"Could not extract time from filename: {filename}"
                ) from exc
        if not is_aia_band:
            raise ValueError(
                f"Ambiguous numeric timestamp suffix in {filename}: use a dotted fraction "
                "or an explicit radio/time-scale label"
            )
    patterns = [
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:?\d{2}|_TAI|_UTC)?(?!\d|\.\d)",
        r"\d{4}-\d{2}-\d{2}T\d{6}(?:\.\d{1,9})?(?:Z|_TAI|_UTC)?(?!\d|\.\d)",
        r"(?<!\d)\d{8}[T_-]\d{6}(?:\.\d{1,9})?(?:_TAI|_UTC|Z)?(?!\d|\.\d)",
        r"(?<!\d)\d{14}[ \t]+\d{1,3}(?:_TAI|_UTC)?(?!\d)",
        r"(?<!\d)\d{14}(?:\.\d{1,9}|\d{1,9})?(?:_TAI|_UTC|Z)?(?!\d|\.\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            return normalize_time(match.group(), source="filename")
        except ValueError as exc:
            # A malformed full timestamp must not quietly become midnight.
            raise ValueError(
                f"Could not extract time from filename: {filename}"
            ) from exc
    raise ValueError(
        f"Could not extract time from filename: {filename}: complete observation time required"
    )


@lru_cache(maxsize=8192)
def extract_time_from_filename(filename: str) -> dt.datetime:
    """Legacy UTC datetime wrapper; explicit date-only context names stay valid.

    Datetime cannot carry sub-microsecond digits. Scientific FITS fallbacks
    should call ``extract_timestamp_from_filename`` directly instead.
    """
    text = str(filename)
    if re.search(r"\d{4}-\d{2}-\d{2}T|\d{8}[T_-]\d|\d{14}", text):
        return extract_timestamp_from_filename(filename).to_datetime()
    for pattern, fmt in (
        (r"(\d{4}-\d{2}-\d{2})", "%Y-%m-%d"),
        (r"(?<!\d)(\d{8})(?!\d)", "%Y%m%d"),
    ):
        match = re.search(pattern, text)
        if match:
            try:
                return dt.datetime.strptime(match.group(1), fmt)
            except ValueError as exc:
                raise ValueError(
                    f"Could not extract time from filename: {filename}"
                ) from exc
    raise ValueError(f"Could not extract time from filename: {filename}")


__all__ = [
    "extract_time_from_filename",
    "extract_timestamp_from_filename",
    "parse_isot_time",
    "parse_time",
]
