"""Header-based indexing and strict temporal matching for MUSER/DART previews."""

from bisect import bisect_left
from collections.abc import Sequence
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
    """Legacy seconds view plus an exact original FITS timestamp and HDU identity."""

    seconds: float
    paths: tuple[Path, ...]
    stamp: object = None
    payload: dict | None = None

    @property
    def time(self):
        from solar_toolkit.time import normalize_time

        if self.stamp is not None:
            return self.stamp
        return normalize_time(Time(self.seconds, format="unix", scale="utc"))

    @property
    def time_ns(self):
        return self.time.tai_ns

    @property
    def frame_id(self):
        return (self.payload or {}).get("frame_id") or f"{self.paths}#legacy"


def index_images(paths):
    from .time_matching import index_radio_frames

    indexed = index_radio_frames(paths)
    grouped = {}
    for frame in indexed:
        payload = frame.payload
        key = (payload["path"], payload["hdu_index"])
        if key not in grouped:
            grouped[key] = [frame, []]
        grouped[key][1].append(
            {
                "frame_id": frame.frame_id,
                "frequency_mhz": payload["frequency_mhz"],
                "polarization": payload["polarization"],
                "plane_indices": payload["plane_indices"],
            }
        )
    entries = []
    for frame, planes in grouped.values():
        payload = dict(frame.payload)
        payload["frame_id"] = (
            f"{payload['path']}#hdu={payload['hdu_index']};size={payload['size_bytes']};mtime_ns={payload['mtime_ns']}"
        )
        payload["logical_planes"] = planes
        entries.append(
            FrameRef(
                float(Time(frame.time.utc, scale="utc").unix),
                (Path(payload["path"]),),
                stamp=frame.time,
                payload=payload,
            )
        )
    return sorted(entries, key=lambda ref: (ref.time_ns, str(ref.paths)))


def nearest_index(
    times: Sequence[float], target: float, tolerance: float
) -> int | None:
    """Match sorted finite UTC seconds, keeping input indices and earlier ties.

    Times must be a one-dimensional, nondecreasing sequence. The target must be
    finite and the tolerance finite and non-negative, including for empty inputs.
    The existing one-microsecond tolerance allowance is retained.
    """
    try:
        if any(np.iscomplexobj(value) for value in (times, target, tolerance)):
            raise ValueError("times, target and tolerance must be real")
        timestamps = np.asarray(times, dtype=np.float64)
        target = float(target)
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("times, target and tolerance must be real numbers") from exc
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all():
        raise ValueError(
            "times must be a one-dimensional sequence of finite timestamps"
        )
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("times must be sorted in nondecreasing order")
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not len(timestamps):
        return None
    i = bisect_left(timestamps, target)
    choices = range(max(0, i - 1), min(len(timestamps), i + 1))
    best = min(choices, key=lambda j: abs(float(timestamps[j]) - target))
    return best if abs(float(timestamps[best]) - target) <= tolerance + 1e-6 else None


def pair_dart(left, right, tolerance=0.1):
    from solar_toolkit.time import TimeFrame, TimeStamp, monotone_one_to_one

    from .time_matching import compatible_radio_payloads

    left, right = list(left), list(right)
    identities = [ref.frame_id for ref in (*left, *right)]
    if len(identities) != len(set(identities)):
        raise ValueError("An original polarization frame may appear only once")

    def frames(values):
        return [
            TimeFrame(f"{ref.paths}#{index}", ref.time, ref)
            for index, ref in enumerate(values)
        ]

    def compatible(a, b):
        if a.payload.payload is None or b.payload.payload is None:
            return True  # Compatibility for legacy in-memory timestamp-only references.
        return compatible_radio_payloads(a.payload.payload, b.payload.payload)

    pairs = monotone_one_to_one(
        frames(left),
        frames(right),
        tolerance_seconds=tolerance,
        compatibility=compatible,
    )
    output = []
    for pair in pairs:
        a, b = pair.left.payload, pair.right.payload
        stamp = TimeStamp.from_tai_ns(
            (a.time_ns + b.time_ns) // 2, source="RR+LL midpoint"
        )
        payload = dict(a.payload or {})
        payload.update(
            {
                "paths": tuple(map(str, a.paths + b.paths)),
                "frame_id": f"{a.frame_id}+{b.frame_id}",
                "actual_times_utc": (a.time.utc, b.time.utc),
                "actual_times_tai_ns": (a.time_ns, b.time_ns),
                "pair_delta_ns": pair.delta_ns,
                "polarization": "L+R",
                "pair_members": (a.payload, b.payload),
            }
        )
        output.append(
            FrameRef(
                float(Time(stamp.utc, scale="utc").unix),
                a.paths + b.paths,
                stamp=stamp,
                payload=payload,
            )
        )
    return output


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
