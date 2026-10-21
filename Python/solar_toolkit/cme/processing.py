"""Numerical CME image processing.

English: Provide array-level CME products without plotting or workflow side
effects.

中文：提供不包含绘图或工作流副作用的 CME 数组级处理结果。
"""

from __future__ import annotations

import numpy as np


def running_difference(current, previous) -> np.ndarray:
    """Return a floating-point difference of identically shaped arrays.

    Conversion occurs before subtraction, preserving negative changes in
    unsigned detector images. Broadcasting raises ``ValueError``; nonfinite
    measurements propagate according to NumPy subtraction. Inputs are unchanged.
    """
    current_values = np.asarray(current, dtype=float)
    previous_values = np.asarray(previous, dtype=float)
    if current_values.shape != previous_values.shape:
        raise ValueError("current and previous must have the same shape")
    return current_values - previous_values


__all__ = ["running_difference"]
