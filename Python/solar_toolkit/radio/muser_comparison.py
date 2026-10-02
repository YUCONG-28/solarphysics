"""Header-based indexing and strict temporal matching for MUSER/DART previews."""

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .centers import iter_radio_images, maybe_make_sum_images, parse_observation_time

__all__ = [
    "FrameRef",
    "index_images",
    "nearest_index",
    "pair_dart",
    "load_dart_sum",
    "select_peak_frame",
]


@dataclass(frozen=True)
class FrameRef:
    seconds: float
    paths: tuple[Path, ...]


def index_images(paths):
    entries = []
    for path in sorted(paths):
        h = fits.getheader(path)
        time = parse_observation_time(Path(path), h)
        if time is None:
            raise ValueError(f"Missing observation time: {path}")
        entries.append(FrameRef(float(Time(time, scale="utc").unix), (Path(path),)))
    return sorted(entries, key=lambda ref: ref.seconds)


def nearest_index(times, target, tolerance):
    if not len(times):
        return None
    i = bisect_left(times, target)
    choices = range(max(0, i - 1), min(len(times), i + 1))
    best = min(choices, key=lambda j: abs(times[j] - target))
    return best if abs(times[best] - target) <= tolerance + 1e-6 else None


def pair_dart(left, right, tolerance=0.1):
    times = [r.seconds for r in right]
    used, pairs = set(), []
    for left_frame in left:
        i = nearest_index(times, left_frame.seconds, tolerance)
        if i is not None and i not in used:
            used.add(i)
            pairs.append(
                FrameRef(
                    (left_frame.seconds + right[i].seconds) / 2,
                    left_frame.paths + right[i].paths,
                )
            )
    return pairs


def load_dart_sum(ref):
    images = [item for p in ref.paths for item in iter_radio_images(p)]
    if len(images) != 2 or images[0].image.shape != images[1].image.shape:
        raise ValueError("A matching LL/RR pair is required")
    for k in [
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CDELT1",
        "CDELT2",
        "CROTA2",
        "PC1_1",
        "PC1_2",
        "PC2_1",
        "PC2_2",
        "BUNIT",
    ]:
        if images[0].header.get(k) != images[1].header.get(k):
            raise ValueError(f"LL/RR metadata mismatch: {k}")
    paired = maybe_make_sum_images(images, tolerance_sec=0.1)
    if len(paired) != 1:
        raise ValueError("Could not construct DART RR+LL")
    return paired[0]


def select_peak_frame(spectrum_times, score, muser, dart, aia, bounds):
    """Rank original DART spectral samples; require all image matches before selection."""
    groups = [muser, *dart, *aia]
    axes = [[r.seconds for r in group] for group in groups]
    tolerances = [0.1] * (1 + len(dart)) + [12.0] * len(aia)
    for i in np.argsort(-np.asarray(score), kind="stable"):
        target = float(spectrum_times[i])
        if not np.isfinite(score[i]) or not bounds[0] <= target <= bounds[1]:
            continue
        matched = [
            nearest_index(t, target, tol)
            for t, tol in zip(axes, tolerances, strict=True)
        ]
        if any(j is None for j in matched):
            continue
        refs = [g[j] for g, j in zip(groups, matched, strict=True)]
        # Require both actual polarization frames, not just their midpoint.
        for ref in refs[1 : 1 + len(dart)]:
            for path in ref.paths:
                t = parse_observation_time(path, fits.getheader(path))
                if abs(float(Time(t).unix) - target) > 0.100001:
                    break
            else:
                continue
            break
        else:
            return target, refs, int(i)
    raise ValueError("No complete image/spectrum/AIA match within the time tolerances")
