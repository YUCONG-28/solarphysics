"""Private leaf helpers extracted from roi_lightcurve.py.

No behavior change: these functions only use imports and builtins.
"""

from __future__ import annotations


def _parse_float_csv(raw: str | None) -> list[float]:
    if not raw:
        return []
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_text_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


__all__ = ["_parse_float_csv", "_parse_text_csv"]
