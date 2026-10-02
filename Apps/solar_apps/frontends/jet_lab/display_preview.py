"""Bounded display sampling in native pixel coordinates, never measurement data."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class NativePreview:
    data: np.ndarray
    extent: tuple
    slices: tuple | None


def _reduce_mask(mask, shape, operation):
    """Reduce every source pixel, including partial edge blocks."""
    rows = np.floor(np.arange(shape[0]) * mask.shape[0] / shape[0]).astype(int)
    cols = np.floor(np.arange(shape[1]) * mask.shape[1] / shape[1]).astype(int)
    return operation.reduceat(operation.reduceat(mask, rows, axis=0), cols, axis=1)


def sample_native_view(data, limits, physical_pixels, *, max_edge=1024):
    """Return a display-only image with exact native-coordinate outer edges.

    Pixel-cell edges are at half integers. Coarse samples are interpolated at
    the centres of those displayed cells; zooming below the cap uses the native
    array slice without interpolation. A block containing any missing input is
    transparent, so a narrow missing stripe cannot disappear through sampling.
    ``physical_pixels`` is the Matplotlib axes bbox, already device-pixel scaled.
    """
    h, w = data.shape
    x0, x1 = sorted(limits[0])
    y0, y1 = sorted(limits[1])
    left, right = max(0, int(np.floor(x0 + 0.5))), min(w, int(np.ceil(x1 + 0.5)))
    bottom, top = max(0, int(np.floor(y0 + 0.5))), min(h, int(np.ceil(y1 + 0.5)))
    if right <= left or top <= bottom:
        return NativePreview(
            np.full((1, 1), np.nan), (-0.5, w - 0.5, -0.5, h - 0.5), None
        )
    slices = (slice(bottom, top), slice(left, right))
    view = data[slices]
    # One sample per physical screen pixel (below the two-sample ceiling),
    # with a fixed 1M-pixel cap. Native detail returns when the viewport narrows.
    nx = min(right - left, max_edge, max(1, int(np.ceil(physical_pixels[0]))))
    ny = min(top - bottom, max_edge, max(1, int(np.ceil(physical_pixels[1]))))
    extent = (left - 0.5, right - 0.5, bottom - 0.5, top - 0.5)
    if (ny, nx) == view.shape:
        return NativePreview(view, extent, slices)
    x = (np.arange(nx) + 0.5) * view.shape[1] / nx - 0.5
    y = (np.arange(ny) + 0.5) * view.shape[0] / ny - 0.5
    xx, yy = np.broadcast_arrays(x[None, :], y[:, None])
    sampled = map_coordinates(
        view, (yy, xx), order=1, prefilter=False, mode="nearest", output=np.float32
    )
    valid = _reduce_mask(np.isfinite(view), sampled.shape, np.logical_and)
    sampled[~valid] = np.nan
    return NativePreview(sampled, extent, slices)


def preview_region_mask(mask, preview):
    """Conservative display of a selected region; no new scientific mask."""
    if preview.slices is None:
        return np.zeros(preview.data.shape, dtype=bool)
    source = np.asarray(mask[preview.slices], dtype=bool)
    shown = (
        source
        if source.shape == preview.data.shape
        else _reduce_mask(source, preview.data.shape, np.logical_or)
    )
    return shown & np.isfinite(preview.data)
