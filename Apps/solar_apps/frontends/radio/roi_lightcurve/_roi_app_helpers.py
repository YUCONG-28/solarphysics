"""Private leaf helpers extracted from roi_lightcurve_app.py.

No behavior change: these functions only use imports and builtins.
"""

from __future__ import annotations


def _expanded_lightcurve_limits(lower: float, upper: float) -> tuple[float, float]:
    lower = float(lower)
    upper = float(upper)
    if lower < upper:
        return lower, upper
    padding = max(abs(lower) * 0.05, 1.0)
    return lower - padding, upper + padding


def _frequency_state_key(freq_mhz: float) -> str:
    return format(float(freq_mhz), ".12g")


def _option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


__all__ = ["_expanded_lightcurve_limits", "_frequency_state_key", "_option_index"]
