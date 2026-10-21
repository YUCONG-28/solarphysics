"""Array operations for map images.

English: Crop pixel-space regions of interest and normalize image arrays for
display.

中文：裁剪像素坐标中的感兴趣区域，并归一化图像数组以供显示。
"""

from __future__ import annotations

import numpy as np

from .._utils.numeric import as_value, numeric_data


def crop_roi(
    data: np.ndarray,
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
) -> np.ndarray:
    """Copy half-open pixel bounds, excluding masked samples as NaN."""

    array = numeric_data(data)
    x0, x1 = map(int, x_range)
    y0, y1 = map(int, y_range)
    return array[y0:y1, x0:x1]


def normalize_image(
    data: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = True,
) -> np.ndarray:
    """Normalize finite unmasked pixels to 0..1 without filling missing data.

    Finite constant images become zero; masked/nonfinite samples remain NaN.
    Quantity bounds are converted to the input unit. Inputs are unchanged.
    """

    import astropy.units as u

    array = numeric_data(data)
    valid = np.isfinite(array)
    array[~valid] = np.nan
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return array
    unit = getattr(data, "unit", None) or u.dimensionless_unscaled

    def bound(value, default, name):
        if value is None:
            return default
        result = as_value(value, unit, name)
        if result.ndim or not np.isfinite(result):
            raise ValueError(f"{name} must be a finite scalar")
        return float(result)

    low = bound(vmin, float(finite.min()), "vmin")
    high = bound(vmax, float(finite.max()), "vmax")
    if high < low:
        raise ValueError("vmax must not be below vmin")
    if high == low:
        array[valid] = 0.0
        return array
    normalized = (array - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0) if clip else normalized


__all__ = ["crop_roi", "normalize_image"]
