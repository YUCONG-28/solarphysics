"""Internal numerical boundaries for explicit units and missing samples.

These helpers return independent float64 arrays, including zero-dimensional
arrays for scalar input. They do not propagate metadata or uncertainties.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.nddata import NDDataBase

__all__ = ["as_value", "numeric_data"]


def numeric_data(value, mask=None) -> np.ndarray:
    """Copy numerical magnitudes, replacing masked samples with NaN.

    NumPy and Astropy masks, ``NDData.mask``, and an optional broadcastable
    boolean ``mask`` are combined. Nonfinite values are represented as NaN;
    no samples are interpolated or filled with zero. Quantity magnitudes retain
    their current unit here: use :func:`as_value` at physical-unit boundaries.
    Input data and masks are never modified.
    """
    masks = [mask]
    if isinstance(value, NDDataBase):
        masks.append(value.mask)
        value = value.data
    masks.append(getattr(value, "mask", None))
    if np.ma.isMaskedArray(value):
        value = value.data
    elif hasattr(value, "unmasked"):
        # Astropy MaskedNDArray and MaskedQuantity expose their unmasked data.
        value = value.unmasked
    if isinstance(value, u.Quantity):
        value = value.value
    array = np.array(value, dtype=np.float64, copy=True)
    for item in masks:
        # pandas Series.mask is an operation, not a missing-data mask.
        if item is not None and not callable(item):
            try:
                invalid = np.broadcast_to(np.asarray(item, dtype=bool), array.shape)
            except ValueError as exc:
                raise ValueError(
                    "mask must be broadcastable to the data shape"
                ) from exc
            array[invalid] = np.nan
    array[~np.isfinite(array)] = np.nan
    return array


def as_value(value, unit, name: str) -> np.ndarray:
    """Return numerical data in ``unit``, preserving masks as NaN.

    Bare numbers already use the caller's documented canonical unit. Quantity
    and unit-bearing NDData inputs are explicitly converted; incompatible
    dimensions raise ``UnitConversionError`` naming the argument. This helper
    does not enable implicit spectral or other physical equivalencies.
    """
    target = u.Unit(unit)
    mask = None
    if isinstance(value, NDDataBase):
        mask = value.mask
        value = (
            u.Quantity(value.data, value.unit) if value.unit is not None else value.data
        )
    if isinstance(value, u.Quantity):
        try:
            value = value.to_value(target)
        except u.UnitConversionError as exc:
            raise u.UnitConversionError(
                f"{name} must have units equivalent to {target}"
            ) from exc
    return numeric_data(value, mask=mask)
