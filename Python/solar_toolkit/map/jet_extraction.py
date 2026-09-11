"""Independent, two-dimensional jet annotation algorithms (no height inversion).

All pixel coordinates are zero-based (x=column, y=row) pixel centres. Display
stretch is deliberately absent. SJET inspired the interaction workflow, not
this implementation: https://github.com/songsolarphysics/SJET .
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.morphology import disk, skeletonize

__all__ = [
    "SegmentationParameters",
    "roi_mask",
    "segment",
    "component_at",
    "retain_component",
    "skeleton_paths",
    "dense_polyline",
    "validate_axis",
    "safe_smooth",
    "axis_metrics",
    "gaussian_width",
    "cross_sections",
]


@dataclass
class SegmentationParameters:
    method: str = "percentile"
    value: float = 85.0
    opening: int = 0
    closing: int = 0
    min_area: int = 8
    polarity: str = "positive"

    def validate(self):
        if self.method not in {"percentile", "manual", "otsu"}:
            raise ValueError("Unknown threshold method")
        if self.polarity not in {"positive", "negative"}:
            raise ValueError("Select positive enhancement or negative dimming")
        if not math.isfinite(self.value):
            raise ValueError("Threshold must be finite")
        if self.method == "percentile" and not 0 <= self.value <= 100:
            raise ValueError("Percentile must be in [0, 100]")
        if any(
            int(v) != v or v < 0 for v in (self.opening, self.closing, self.min_area)
        ):
            raise ValueError("Morphology sizes must be non-negative integers")


def roi_mask(shape, vertices=None):
    if vertices is None:
        return np.ones(shape, dtype=bool)
    xy = np.asarray(vertices, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("ROI requires at least three finite (x, y) vertices")
    result = np.zeros(shape, dtype=bool)
    rr, cc = polygon(xy[:, 1], xy[:, 0], shape)
    result[rr, cc] = True
    return result


def segment(data, parameters=None, roi=None):
    """Threshold finite samples only; retain unknown pixels through morphology."""
    p = parameters or SegmentationParameters()
    p.validate()
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected a two-dimensional image")
    valid = np.isfinite(data) & roi_mask(data.shape, roi)
    signed = data if p.polarity == "positive" else -data
    values = signed[valid]
    if not len(values):
        return np.zeros(data.shape, dtype=np.int32), float("nan")
    threshold = (
        np.percentile(values, p.value)
        if p.method == "percentile"
        else threshold_otsu(values) if p.method == "otsu" else p.value
    )
    mask = valid & (signed > threshold)
    # Difference polarity is literal: never include negative enhancement or zero.
    mask &= signed > 0
    if p.opening:
        mask = ndi.binary_opening(mask, structure=disk(p.opening)) & valid
    if p.closing:
        mask = ndi.binary_closing(mask, structure=disk(p.closing)) & valid
    labels, _ = ndi.label(mask, np.ones((3, 3)))
    counts = np.bincount(labels.ravel())
    keep = counts >= p.min_area
    keep[0] = False
    return ndi.label(keep[labels] & valid, np.ones((3, 3)))[0], float(threshold)


def component_at(labels, xy):
    x, y = np.rint(xy).astype(int)
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        raise ValueError("Click is outside the image")
    label = labels[y, x]
    if not label:
        raise ValueError("Click inside a thresholded component")
    return labels == label


def retain_component(old_labels, old_mask, new_labels, seed):
    """Fail closed on vanished/split/merged selection, including a lost seed."""
    if old_mask is None or seed is None:
        return None, "select_component"
    overlapping = np.unique(new_labels[old_mask])
    overlapping = overlapping[overlapping != 0]
    if len(overlapping) != 1:
        return None, (
            "component_split" if len(overlapping) > 1 else "component_disappeared"
        )
    candidate = new_labels == overlapping[0]
    ancestors = np.unique(old_labels[candidate])
    if len(ancestors[ancestors != 0]) > 1:
        return None, "component_merged"
    x, y = np.rint(seed).astype(int)
    if not candidate[y, x]:
        return None, "selection_point_disappeared"
    return candidate, "retained_reconfirm_axis"


def skeleton_paths(mask, inner_xy):
    """Enumerate shortest connected skeleton paths to every endpoint.

    The longest candidate is displayed first, never automatically confirmed.
    A cyclic skeleton with no endpoints is explicitly unresolved.
    """
    skel = skeletonize(np.asarray(mask, dtype=bool))
    yx = np.argwhere(skel)
    if len(yx) < 2:
        return skel, [], {"reason": "insufficient_skeleton", "branches": []}
    xy = yx[:, ::-1]
    lookup = {tuple(p): i for i, p in enumerate(yx)}
    neighbours = [[] for _ in yx]
    for i, (y, x) in enumerate(yx):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = lookup.get((y + dy, x + dx))
                if j is not None and j != i:
                    neighbours[i].append((j, math.hypot(dx, dy)))
    start = int(np.argmin(np.sum((xy - inner_xy) ** 2, axis=1)))
    dist = np.full(len(yx), np.inf)
    previous = np.full(len(yx), -1, dtype=int)
    dist[start] = 0
    queue = [(0.0, start)]
    while queue:
        distance, i = heapq.heappop(queue)
        if distance != dist[i]:
            continue
        for j, step in neighbours[i]:
            trial = distance + step
            if trial < dist[j]:
                dist[j], previous[j] = trial, i
                heapq.heappush(queue, (trial, j))
    ends = [
        i
        for i, n in enumerate(neighbours)
        if len(n) == 1 and i != start and np.isfinite(dist[i])
    ]
    paths = []
    for end in sorted(ends, key=lambda i: dist[i], reverse=True):
        chain = [end]
        while chain[-1] != start:
            chain.append(previous[chain[-1]])
        paths.append(xy[chain[::-1]].astype(float))
    return (
        skel,
        paths,
        {
            "reason": "pending_confirmation" if paths else "cyclic_or_no_endpoint",
            "branches": xy[
                [i for i, n in enumerate(neighbours) if len(n) > 2]
            ].tolist(),
            "anchor_pixel": xy[start].tolist(),
        },
    )


def dense_polyline(points, step=0.25):
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError("An axis requires two or more finite points")
    result = []
    for a, b in zip(xy[:-1], xy[1:], strict=True):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        result.extend(np.linspace(a, b, count, endpoint=False))
    return np.vstack([*result, xy[-1]])


def validate_axis(points, mask):
    dense = dense_polyline(points)
    ij = np.rint(dense).astype(int)
    if (
        (ij < 0).any()
        or (ij[:, 0] >= mask.shape[1]).any()
        or (ij[:, 1] >= mask.shape[0]).any()
    ):
        raise ValueError("Axis leaves the image")
    if not mask[ij[:, 1], ij[:, 0]].all():
        raise ValueError("Axis leaves the selected region or crosses a gap")
    return dense


def safe_smooth(points, mask, sigma=1):
    """Accept an optional display smoother only if its entire path stays inside."""
    original = validate_axis(points, mask)
    smoothed = ndi.gaussian_filter1d(original, sigma, axis=0)
    smoothed[[0, -1]] = original[[0, -1]]
    try:
        return validate_axis(smoothed, mask), "smoothed_inside_mask"
    except ValueError:
        return original, "smoothing_rejected"


def axis_metrics(hpc_xy):
    """Angular projected lengths/directions; +Tx is 0°, +Ty is +90°."""
    xy = np.asarray(hpc_xy, dtype=float)
    if len(xy) < 2:
        return {"length_arcsec": 0.0, "terminal_directions": []}
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    output = []
    for requested in (20.0, 30.0, 40.0):
        length = min(requested, distance[-1])
        start = np.array(
            [np.interp(distance[-1] - length, distance, xy[:, i]) for i in range(2)]
        )
        delta = xy[-1] - start
        output.append(
            {
                "requested_arcsec": requested,
                "available_arcsec": float(length),
                "direction_deg": float(np.degrees(np.arctan2(delta[1], delta[0]))),
                "definition": "terminal_chord_hpc",
                "short_segment": bool(length < requested),
            }
        )
    return {"length_arcsec": float(distance[-1]), "terminal_directions": output}


def gaussian_width(offset, intensity):
    """Single positive Gaussian + constant; invalid fits have no valid FWHM."""
    x, y = np.asarray(offset, float), np.asarray(intensity, float)

    def failure(reason):
        return {"status": reason, "fwhm_pixel": None}

    if len(x) < 9 or not np.isfinite(y).all():
        return failure("missing_or_insufficient_profile")
    span = float(np.ptp(y))
    if span <= 0:
        return failure("flat_profile")
    noise = np.median(np.abs(np.diff(y, n=2))) / 1.65
    filtered = ndi.gaussian_filter1d(y, 1)
    peaks, _ = find_peaks(filtered, prominence=max(0.15 * span, 3 * noise), distance=3)
    if len(peaks) > 1:
        return failure("multiple_peaks")
    if not len(peaks):
        return failure("boundary_truncated_or_no_peak")
    if span < max(4 * noise, np.finfo(float).eps):
        return failure("low_signal")

    def model(xx, amplitude, centre, sigma, background):
        return background + amplitude * np.exp(-0.5 * ((xx - centre) / sigma) ** 2)

    try:
        dx = float(np.min(np.diff(x)))
        fit, covariance = curve_fit(
            model,
            x,
            y,
            p0=[span, x[np.argmax(y)], max(dx, np.ptp(x) / 8), np.min(y)],
            bounds=([0, x[0], dx / 2, -np.inf], [np.inf, x[-1], np.ptp(x), np.inf]),
            maxfev=5000,
        )
        amp, centre, sigma, background = fit
        if centre - 3 * sigma <= x[0] or centre + 3 * sigma >= x[-1]:
            return failure("boundary_truncated")
        residual = float(np.sqrt(np.mean((model(x, *fit) - y) ** 2)) / amp)
        if not np.isfinite(covariance).all() or residual > 0.2:
            return failure("poor_single_gaussian_fit")
        return {
            "status": "valid",
            "fwhm_pixel": float(2 * np.sqrt(2 * np.log(2)) * sigma),
            "centre_pixel": float(centre),
            "sigma_pixel": float(sigma),
            "amplitude": float(amp),
            "background": float(background),
            "residual_relative": residual,
        }
    except (ValueError, RuntimeError, FloatingPointError):
        return failure("fit_failed")


def cross_sections(data, mask, points, count=10, half_span=20.0):
    """Sample on raw intensity; report connected mask width and fit separately."""
    xy = validate_axis(points, mask)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    samples = np.linspace(0.1, 0.9, count) * distances[-1]
    output = []
    offsets = np.arange(-half_span, half_span + 0.125, 0.25)
    for target in samples:
        i = int(np.argmin(abs(distances - target)))
        tangent = xy[min(len(xy) - 1, i + 4)] - xy[max(0, i - 4)]
        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        normal = np.array([-tangent[1], tangent[0]]) / norm
        locations = xy[i] + offsets[:, None] * normal
        profile = ndi.map_coordinates(
            np.asarray(data, float),
            locations.T[::-1],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        inside = (
            ndi.map_coordinates(
                mask.astype(float),
                locations.T[::-1],
                order=0,
                mode="constant",
                cval=0,
                prefilter=False,
            )
            > 0
        )
        centre = int(np.argmin(abs(offsets)))
        lo = hi = centre
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        while hi < len(inside) - 1 and inside[hi + 1]:
            hi += 1
        bounded = lo > 0 and hi < len(inside) - 1
        output.append(
            {
                "axis_pixel": xy[i].tolist(),
                "normal_pixel": normal.tolist(),
                "offset_pixel": offsets.tolist(),
                "intensity": profile.tolist(),
                "mask_width_pixel": (
                    float(offsets[hi] - offsets[lo] + 0.25) if bounded else None
                ),
                "mask_status": "valid" if bounded else "boundary_truncated",
                "mask_edge_pixels": [
                    (xy[i] + (offsets[lo] - 0.125) * normal).tolist(),
                    (xy[i] + (offsets[hi] + 0.125) * normal).tolist(),
                ],
                "gaussian": gaussian_width(offsets, profile),
            }
        )
    return output
