"""Lossless observation timestamps with a continuous integer TAI clock.

Astropy is imported only when a timestamp needs astronomical scale conversion.
UTC text is retained for display; comparisons use integer TAI nanoseconds.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from numbers import Integral

_NS = Decimal(1_000_000_000)


def seconds_to_ns(seconds, *, nonnegative=False) -> int:
    """Convert scalar seconds, time quantities or TimeDelta to integer ns.

    Bare numbers and decimal strings denote seconds. Explicit duration objects
    are converted to seconds before rounding once, half-even. Masks and arrays
    cannot supply a scalar scientific duration. ``nonnegative`` validates the
    sign before rounding, so a small negative tolerance cannot become zero.
    """
    from astropy.time import TimeDelta
    from astropy.units import Quantity

    if isinstance(seconds, (TimeDelta, Quantity)):
        if not seconds.isscalar or bool(getattr(seconds, "mask", False)):
            raise ValueError("Seconds must be an unmasked scalar duration")
        seconds = (
            seconds.to_value("sec", subfmt="decimal")
            if isinstance(seconds, TimeDelta)
            else seconds.to_value("s")
        )
    if isinstance(seconds, bool):
        raise ValueError("Seconds must be finite numbers, not booleans")
    try:
        value = Decimal(str(seconds))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Seconds must be finite numbers") from exc
    if not value.is_finite():
        raise ValueError("Seconds must be finite numbers")
    if nonnegative and value < 0:
        raise ValueError("Seconds must be nonnegative")
    with localcontext() as context:
        context.prec = max(
            40, len(value.as_tuple().digits) + abs(value.adjusted()) + 12
        )
        return int((value * _NS).to_integral_value(rounding=ROUND_HALF_EVEN))


@dataclass(frozen=True)
class TimeStamp:
    raw: str
    utc: str
    tai_ns: int
    source: str | None = None
    scale: str = "utc"

    def __post_init__(self):
        if isinstance(self.tai_ns, bool) or not isinstance(self.tai_ns, Integral):
            raise ValueError("tai_ns must be an integer")
        object.__setattr__(self, "tai_ns", int(self.tai_ns))

    @classmethod
    def from_tai_ns(cls, tai_ns: int, *, source=None, raw=None) -> TimeStamp:
        if isinstance(tai_ns, bool) or not isinstance(tai_ns, Integral):
            raise ValueError("tai_ns must be an integer")
        with localcontext() as context:
            context.prec = max(40, len(str(abs(int(tai_ns)))) + 12)
            value = Decimal(int(tai_ns)) / _NS
        result = normalize_time(value, source=source, scale="tai", format="unix_tai")
        return replace(result, raw=result.utc if raw is None else str(raw))

    def shift(self, seconds, *, source=None) -> TimeStamp:
        """Apply an explicit offset; no clock correction is inferred."""
        return self.from_tai_ns(
            self.tai_ns + seconds_to_ns(seconds),
            source=self.source if source is None else source,
            raw=f"{self.raw} + ({seconds}) s",
        )

    def to_datetime(self) -> dt.datetime:
        """Return naive UTC at microsecond resolution, truncating finer digits.

        The exact timestamp remains unchanged. Leap seconds cannot be expressed
        by ``datetime`` and raise ``ValueError``; use this object for those times.
        """
        # Normalized UTC has nine fractional digits; Python 3.10's ISO parser
        # only accepts three or six. Truncate explicitly at this legacy boundary.
        return dt.datetime.strptime(self.utc[:26], "%Y-%m-%dT%H:%M:%S.%f")

    def to_dict(self):
        return {
            "raw": self.raw,
            "utc": self.utc,
            "tai_ns": int(self.tai_ns),
            "source": self.source,
            "scale": self.scale,
        }


def _text_time(text, scale):
    """Canonicalize complete ISO/compact times without discarding fractions."""
    text = text.strip().strip("'")
    suffix = re.search(r"(?:_TAI|_UTC|\s+TAI|\s+UTC)$", text, re.IGNORECASE)
    if suffix:
        suffix_scale = suffix.group().strip("_ ").lower()
        if scale is not None and scale.lower() != suffix_scale:
            raise ValueError("Timestamp suffix and explicit time scale disagree")
        scale = suffix_scale
        text = text[: suffix.start()]
    scale = (scale or "utc").lower()
    padded_milliseconds = re.fullmatch(r"(\d{14})[ \t]+(\d{1,3})", text)
    if padded_milliseconds:
        # DART headers encode a separate integer millisecond field, sometimes
        # left padded with spaces. 11 is 11 ms, never 0.11 fractional seconds.
        text = padded_milliseconds[1] + f".{int(padded_milliseconds[2]):03d}"
    compact = re.fullmatch(
        r"(\d{4})[.-]?(\d{2})[.-]?(\d{2})[T_ ]?(\d{2}):?(\d{2}):?(\d{2})"
        r"(?:[._](\d{1,9})|(\d{1,9}))?(Z|[+-]\d{2}:?\d{2})?",
        text,
    )
    if compact:
        year, month, day, hour, minute, second, dotted, undotted, zone = (
            compact.groups()
        )
        text = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        if dotted or undotted:
            text += "." + (dotted or undotted)
        text += zone or ""
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?"
        r"(Z|[+-]\d{2}:?\d{2})?",
        text,
    )
    if not match:
        raise ValueError("A complete observation timestamp including time is required")
    date, clock, fraction, zone = match.groups()
    result = f"{date}T{clock}" + ("." + fraction if fraction else "")
    if zone:
        if scale != "utc":
            raise ValueError("Timezone offsets are only valid with UTC timestamps")
        if zone != "Z":
            # Preserve the fractional nanoseconds independently of datetime's
            # microsecond limit. Offsets themselves are whole minutes.
            if len(zone) == 5:
                zone = f"{zone[:3]}:{zone[3:]}"
            if int(zone[1:3]) >= 24 or int(zone[4:6]) >= 60:
                raise ValueError(
                    "Timezone offsets require hours below 24 and minutes below 60"
                )
            # datetime cannot represent a leap second. Shift its preceding
            # second's calendar fields; the scale-aware parser validates :60.
            leap = int(clock[-2:]) >= 60
            ordinary_clock = clock[:-2] + "59" if leap else clock
            offset = dt.datetime.fromisoformat(f"{date}T{ordinary_clock}{zone}")
            base = offset.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            if leap:
                base = base[:-2] + clock[-2:]
            result = base + ("." + fraction if fraction else "")
    return result, scale


def _calendar_time(text, scale, *, format="isot"):
    """Use Astropy's SI-second arithmetic to validate calendar second 60."""
    from astropy.time import Time, TimeDelta

    match = re.fullmatch(r"(.*:)(\d{2})(?:\.(\d+))?", text)
    if match is None or int(match[2]) < 60:
        return Time(text, format=format, scale=scale)
    prefix, seconds, fraction = match.groups()
    if fraction and len(fraction) > 9:
        raise ValueError("Leap-second timestamps support at most nanosecond precision")
    second = int(seconds)
    fraction = fraction or ""
    preceding = Time(
        prefix + "59." + fraction.ljust(9, "0"), format=format, scale=scale
    )
    parsed = preceding + TimeDelta(second - 59, format="sec")
    parsed.precision = 9
    expected = prefix + seconds + "." + fraction.ljust(9, "0")
    if parsed.to_value(format) != expected:
        raise ValueError("Invalid leap second for the declared time scale and calendar")
    return parsed


def normalize_time(value, *, source=None, scale=None, format=None) -> TimeStamp:
    """Normalize a complete timestamp without a float Unix-seconds round trip.

    Naive datetimes and unlabelled observation strings use UTC. A numeric value
    requires an explicit Astropy format, such as ``unix`` or ``unix_tai``.
    Date-only values are deliberately not scientific observation timestamps.
    A complete 14-digit time followed by whitespace and 1-3 digits uses the
    explicit DART integer-milliseconds field convention.
    """
    if isinstance(value, TimeStamp):
        if scale is not None and scale.lower() != value.scale:
            raise ValueError("Cannot reinterpret an already normalized timestamp")
        return value if source is None else replace(value, source=source)
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        raise ValueError("A complete observation timestamp including time is required")
    import numpy as np
    from astropy.time import Time

    if isinstance(value, Time):
        if not value.isscalar:
            raise ValueError("An observation timestamp must be scalar")
        if bool(value.mask):
            raise ValueError("A masked observation timestamp is not valid")
        if scale is not None and scale.lower() != value.scale:
            raise ValueError("Timestamp and explicit time scale disagree")
        parsed = value.copy()
        parsed.precision = 9
        raw, source_scale = str(parsed.value), parsed.scale
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("An observation timestamp must not be NaT")
        if format not in (None, "datetime64"):
            raise ValueError("datetime64 input requires its own timestamp format")
        if np.datetime_data(value.dtype)[0] in {"Y", "M", "W", "D"}:
            raise ValueError(
                "A complete observation timestamp including time is required"
            )
        raw, source_scale = str(value), (scale or "utc").lower()
        text = np.datetime_as_string(value, unit="auto")
        if "T" not in text:
            text += "T00:00:00"
        elif len(text.split("T")[1]) == 2:
            text += ":00:00"
        elif len(text.split("T")[1]) == 5:
            text += ":00"
        parsed = _calendar_time(text, source_scale)
    elif isinstance(value, dt.datetime):
        raw = value.isoformat()
        if value.tzinfo is not None and scale is not None and scale.lower() != "utc":
            raise ValueError("Timezone offsets are only valid with UTC timestamps")
        # datetime supports second-level historical offsets as well as modern
        # whole-minute zones. Convert its zone before the FITS-string parser.
        canonical = (
            value.astimezone(dt.timezone.utc).isoformat()
            if value.tzinfo is not None
            else raw
        )
        text, source_scale = _text_time(canonical, scale)
        parsed = _calendar_time(text, source_scale)
    elif isinstance(value, str) and format in (None, "iso", "isot", "fits"):
        raw = value
        text, source_scale = _text_time(value, scale)
        parsed = _calendar_time(text, source_scale)
    elif format is not None:
        raw = str(value)
        source_scale = (scale or ("tai" if format == "unix_tai" else "utc")).lower()
        try:
            numeric = Decimal(str(value))
        except InvalidOperation:
            parsed = (
                _calendar_time(value, source_scale, format=format)
                if isinstance(value, str)
                else Time(value, format=format, scale=source_scale)
            )
        else:
            if not numeric.is_finite():
                raise ValueError("Observation timestamps must be finite")
            parsed = Time(
                numeric, format=format, scale=source_scale, in_subfmt="decimal"
            )
    else:
        raise ValueError("Numeric timestamps require an explicit format")
    utc = parsed.utc
    utc.precision = 9
    exact_seconds = parsed.to_value("unix_tai", subfmt="decimal")
    tai_ns = seconds_to_ns(exact_seconds)
    return TimeStamp(
        raw=raw, utc=utc.isot, tai_ns=tai_ns, source=source, scale=source_scale
    )


def elapsed_seconds(start, end) -> float:
    """Return SI seconds from start to end after subtracting integer TAI ns."""
    return (normalize_time(end).tai_ns - normalize_time(start).tai_ns) / 1_000_000_000


__all__ = ["TimeStamp", "elapsed_seconds", "normalize_time", "seconds_to_ns"]
