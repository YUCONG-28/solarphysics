"""Private leaf helpers extracted from review.py.

No behavior change: these functions only use imports and builtins.
"""

from __future__ import annotations


def _format_frequency(value: float) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


__all__ = ["_format_frequency"]
