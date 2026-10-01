"""Deterministic filenames for generated scientific images.

The helpers in this module are side-effect free.  In particular, they never
read the clock: callers that lack observation timestamps must capture one UTC
batch timestamp and pass it with ``time_source="generated"``.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ImageFilenameSpec",
    "build_image_filename",
    "format_utc_filename_time",
]

_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".svg", ".tif", ".tiff", ".webp"}
)
_TIME_SOURCES = frozenset({"observation", "generated"})
_SEPARATORS_RE = re.compile(r"[^a-z0-9]+")
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_CHANNEL_UNIT_RE = re.compile(r"^(\d+(?:p\d+)?)_(a|angstrom|ghz|khz|mhz)$")

_TOKEN_ALIASES = {
    "left_circular": "lcp",
    "left_circular_polarization": "lcp",
    "ll": "lcp",
    "lcp": "lcp",
    "right_circular": "rcp",
    "right_circular_polarization": "rcp",
    "rr": "rcp",
    "rcp": "rcp",
    "ll_plus_rr": "lcp_plus_rcp",
    "rr_plus_ll": "lcp_plus_rcp",
    "lcp_plus_rcp": "lcp_plus_rcp",
    "rcp_plus_lcp": "lcp_plus_rcp",
    "stokes_i": "stokes_i",
    "i": "stokes_i",
    "stokes_v_over_i": "stokes_v_over_i",
    "v_over_i": "stokes_v_over_i",
}


@dataclass(frozen=True)
class ImageFilenameSpec:
    """Declarative components of one generated scientific-image filename."""

    sequence: int
    start_time: Any
    instrument: str
    product: str | Sequence[str]
    end_time: Any | None = None
    channel: str | float | int | None = None
    polarization: str | Sequence[str] | None = None
    qualifiers: str | Sequence[str] = ()
    extension: str = ".png"
    time_source: Literal["observation", "generated"] = "observation"


def format_utc_filename_time(value: Any, end: Any | None = None) -> str:
    """Format one UTC timestamp or inclusive UTC range for a filename.

    A timezone-naive value is interpreted as UTC for compatibility with FITS
    and table readers that already expose UTC-like naive datetimes.  Fractional
    seconds are deliberately truncated.
    """

    start_utc = _coerce_utc_datetime(value).replace(microsecond=0)
    start_text = start_utc.strftime("%Y%m%dT%H%M%SZ")
    if end is None:
        return start_text
    end_utc = _coerce_utc_datetime(end).replace(microsecond=0)
    if end_utc < start_utc:
        raise ValueError("Image filename end time precedes start time.")
    if end_utc == start_utc:
        return start_text
    return f"{start_text}-{end_utc.strftime('%Y%m%dT%H%M%SZ')}"


def build_image_filename(spec: ImageFilenameSpec) -> str:
    """Validate and build a deterministic generated-image filename."""

    if isinstance(spec.sequence, bool) or not isinstance(spec.sequence, int):
        raise TypeError("Image filename sequence must be an integer.")
    if not 1 <= spec.sequence <= 9999:
        raise ValueError("Image filename sequence must be between 1 and 9999.")
    if spec.time_source not in _TIME_SOURCES:
        raise ValueError(f"Unsupported image time source: {spec.time_source!r}")

    time_text = format_utc_filename_time(spec.start_time, spec.end_time)
    tokens = [f"{spec.sequence:04d}", time_text]
    if spec.time_source == "generated":
        tokens.append("generated")
    tokens.append(_normalize_token(spec.instrument, role="instrument"))
    if spec.channel is not None and str(spec.channel).strip():
        tokens.append(_normalize_token(spec.channel, role="channel"))
    tokens.extend(_normalize_many(spec.polarization, role="polarization"))
    product_tokens = _normalize_many(spec.product, role="product")
    if not product_tokens:
        raise ValueError("Image filename product is required.")
    tokens.extend(product_tokens)
    tokens.extend(_normalize_many(spec.qualifiers, role="qualifier"))
    return "_".join(tokens) + _normalize_extension(spec.extension)


def _coerce_utc_datetime(value: Any) -> dt.datetime:
    if value is None:
        raise ValueError("Image filename time is required.")
    if isinstance(value, dt.datetime):
        result = value
    elif isinstance(value, dt.date):
        result = dt.datetime.combine(value, dt.time())
    elif isinstance(value, str):
        result = _parse_datetime_text(value)
    elif value.__class__.__name__ == "datetime64":
        result = _parse_datetime_text(str(value))
    elif hasattr(value, "to_pydatetime"):
        result = value.to_pydatetime()
    elif hasattr(value, "to_datetime"):
        try:
            result = value.to_datetime(timezone=dt.timezone.utc)
        except TypeError:
            result = value.to_datetime()
    else:
        raise TypeError(f"Unsupported image filename time value: {type(value)!r}")
    if not isinstance(result, dt.datetime):
        raise TypeError(f"Image filename time did not resolve to datetime: {value!r}")
    if result.tzinfo is None:
        return result.replace(tzinfo=dt.timezone.utc)
    return result.astimezone(dt.timezone.utc)


def _parse_datetime_text(value: str) -> dt.datetime:
    text = value.strip()
    if not text:
        raise ValueError("Image filename time is required.")
    iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        return dt.datetime.fromisoformat(iso_text)
    except ValueError:
        pass
    for pattern in (
        "%Y%m%dT%H%M%S",
        "%Y%m%d_%H%M%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"Unsupported image filename time: {value!r}")


def _normalize_many(value: Any, *, role: str) -> list[str]:
    if value is None:
        return []
    values = [value] if isinstance(value, (str, int, float)) else list(value)
    return [_normalize_token(item, role=role) for item in values if str(item).strip()]


def _normalize_token(value: Any, *, role: str) -> str:
    text = str(value).strip().casefold()
    text = text.replace("å", "a").replace("Å", "a")
    text = text.replace("+", " plus ").replace("/", " over ")
    text = _DECIMAL_RE.sub("p", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    token = _SEPARATORS_RE.sub("_", text).strip("_")
    token = _TOKEN_ALIASES.get(token, token)
    if role == "channel":
        match = _CHANNEL_UNIT_RE.fullmatch(token)
        if match:
            unit = "a" if match.group(2) == "angstrom" else match.group(2)
            token = f"{match.group(1)}{unit}"
    if not token or not token.isascii() or token.startswith("_") or token.endswith("_"):
        raise ValueError(f"Invalid {role} token: {value!r}")
    return token


def _normalize_extension(value: str) -> str:
    extension = str(value).strip().casefold()
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension not in _IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image filename extension: {value!r}")
    return extension
