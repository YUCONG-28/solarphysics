"""Header-based observation times without filename or photospheric assumptions."""

import math
import re


def header_time(value, *, default_scale="utc"):
    from astropy.time import Time

    text = str(value).strip()
    match = re.fullmatch(
        r"(\d{4})[.-](\d{2})[.-](\d{2})[T_ ](\d{2}:\d{2}:\d{2}(?:\.\d+)?)(?:_(TAI|UTC))?Z?",
        text,
        re.I,
    )
    if not match:
        raise ValueError(f"Unrecognized explicit FITS time: {text}")
    year, month, day, clock, scale = match.groups()
    return Time(
        f"{year}-{month}-{day}T{clock}", scale=(scale or default_scale).lower()
    ).utc


def observation_midpoint(smap):
    import astropy.units as u

    meta = smap.meta
    if str(meta.get("telescop", "")).upper().endswith("HMI") and meta.get("t_obs"):
        return header_time(meta["t_obs"], default_scale="tai")
    value = meta.get("date-obs") or meta.get("date_obs")
    if not value:
        raise ValueError("Observation start time missing")
    exposure = float(meta.get("exptime", -1))
    if not math.isfinite(exposure) or exposure <= 0:
        raise ValueError("Positive exposure required for an image midpoint")
    return (
        header_time(value, default_scale=str(meta.get("timesys", "UTC")))
        + exposure / 2 * u.s
    )


__all__ = ["header_time", "observation_midpoint"]
