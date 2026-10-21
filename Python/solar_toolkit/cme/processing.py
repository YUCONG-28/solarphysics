"""Numerical CME image processing.

English: Provide array-level CME products without plotting or workflow side
effects.

中文：提供不包含绘图或工作流副作用的 CME 数组级处理结果。
"""

from __future__ import annotations

import numpy as np

from solar_toolkit._utils.numeric import as_value, numeric_data


def running_difference(current, previous) -> np.ndarray:
    """Return a floating-point difference of identically shaped arrays.

    Conversion occurs before subtraction, preserving negative changes in
    unsigned detector images. Broadcasting raises ``ValueError``. Masks and
    nonfinite samples remain NaN. Unit-bearing inputs are converted to the
    current image's unit, or the previous image's unit if only it declares one;
    bare arrays share that unit. The legacy ndarray return contains magnitudes
    in the selected unit. Incompatible dimensions raise ``UnitConversionError``.
    Inputs are unchanged.
    """
    unit = getattr(current, "unit", None)
    if unit is None:
        unit = getattr(previous, "unit", None)
    current_values = (
        numeric_data(current) if unit is None else as_value(current, unit, "current")
    )
    previous_values = (
        numeric_data(previous) if unit is None else as_value(previous, unit, "previous")
    )
    if current_values.shape != previous_values.shape:
        raise ValueError("current and previous must have the same shape")
    return current_values - previous_values


__all__ = ["running_difference"]
