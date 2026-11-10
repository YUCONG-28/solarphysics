"""Internal native-time CSO measurements, exported through cso_extract."""

from __future__ import annotations

import operator
from contextlib import ExitStack
from numbers import Real

import numpy as np
from astropy.time import Time

from solar_toolkit._utils.numeric import as_value
from solar_toolkit.time import TimeStamp, normalize_time

__all__ = ["extract_native_polarization"]


def _stamp(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Time bounds must be observation timestamps, not booleans")
    if isinstance(value, Real):
        return normalize_time(value, format="unix", scale="utc")
    return normalize_time(value)


def _unix_ns(values):
    values = np.asarray(values, dtype=np.int64)
    return Time(
        values // 1_000_000_000,
        (values % 1_000_000_000) / 1e9,
        format="unix_tai",
        scale="tai",
    ).utc.to_value("unix")


def _support(times, cadences, lower, upper, gap_factor):
    """Construct separate mesh edges for supported segments on the TAI clock."""
    if not times.size:
        return [], [], [[lower, upper]]
    breaks = [0]
    for index in range(1, len(times)):
        if int(times[index]) - int(times[index - 1]) > gap_factor * max(
            cadences[index - 1], cadences[index]
        ):
            breaks.append(index)
    breaks.append(len(times))
    segments = []
    for first, stop in zip(breaks[:-1], breaks[1:], strict=True):
        centers = times[first:stop]
        edges = [max(lower, int(centers[0]) - int(cadences[first]) // 2)]
        edges.extend(
            (int(a) + int(b)) // 2
            for a, b in zip(centers[:-1], centers[1:], strict=True)
        )
        edges.append(
            min(upper, int(centers[-1]) + max(1, int(cadences[stop - 1]) // 2))
        )
        segments.append(
            {
                "start_index": first,
                "stop_index": stop,
                "time_edges_unix": _unix_ns(edges).tolist(),
                "time_edges_tai_ns": edges,
                "start_utc": TimeStamp.from_tai_ns(int(centers[0])).utc,
                "end_utc": TimeStamp.from_tai_ns(int(centers[-1])).utc,
                "support_start_tai_ns": edges[0],
                "support_end_tai_ns": edges[-1],
                "cadence_seconds": float(np.median(cadences[first:stop])) / 1e9,
                "native_time_samples": stop - first,
            }
        )
    gaps = [
        [left["support_end_tai_ns"], right["support_start_tai_ns"]]
        for left, right in zip(segments[:-1], segments[1:], strict=True)
    ]
    boundary = []
    if segments[0]["support_start_tai_ns"] > lower:
        boundary.append([lower, segments[0]["support_start_tai_ns"]])
    if segments[-1]["support_end_tai_ns"] < upper:
        boundary.append([segments[-1]["support_end_tai_ns"], upper])
    return segments, gaps, boundary


def extract_native_polarization(
    paths,
    time_bounds,
    frequency_bounds,
    *,
    frequency_bin_mhz,
    cache_root=None,
    chunk_columns=128,
    gap_factor=1.5,
):
    """Extract native UTC samples, aggregating only frequency channels.

    Input floating FITS cubes must declare paired RCP/LCP, UTC, and SFU. FITS
    scaling is applied exactly once. Bare frequency bounds/bin widths mean MHz;
    Quantities convert explicitly. Time bounds accept canonical timestamps or
    explicitly UTC Unix-second numbers, with both endpoints included. All
    ordering, duplicate checks and temporal support use integer TAI nanoseconds.

    The arrays have shape ``(frequency_bin, native_time)``. ``sum_rcp`` and
    ``sum_lcp`` use the same finite native pairs and ``sample_count``; empty sums
    are zero while missing means/intensity/polarization are NaN. Intensity is
    mean R+mean L, not half-sum Stokes I. Polarization is the ratio of paired
    sums, not a mean of native ratios. Negative averaged channels/nonpositive
    total are masked in ``polarization_percent``; the unmasked diagnostic is
    ``raw_polarization_percent``. No clipping, threshold or calibration is added.

    Only exactly identical physical paired records with matching selected
    frequency coordinates may share a timestamp. Conflicts raise ValueError.
    Segments supply separate mesh edges, including singleton segments; render
    each separately rather than connecting a time gap. Full-source cadence must
    be inferable from at least two distinct timestamps. Temporary gzip expansion
    is scoped to this call; multiple compressed inputs may need staging space
    together. Returned arrays are detached and source files are never modified.
    """
    # Delayed import permits either public module to be imported first.
    from astropy import units as u

    from .cso_extract import _edges, _native_edges, _read_index, _source_read_paths

    try:
        chunk = operator.index(chunk_columns)
    except TypeError as exc:
        raise ValueError("chunk_columns must be a positive integer") from exc
    if isinstance(chunk_columns, (bool, np.bool_)) or chunk <= 0:
        raise ValueError("chunk_columns must be a positive integer")
    if isinstance(gap_factor, (bool, np.bool_)) or np.ndim(gap_factor):
        raise ValueError("gap_factor must be finite and greater than one")
    gap_factor = float(gap_factor)
    if not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must be finite and greater than one")
    if len(time_bounds) != 2:
        raise ValueError("time_bounds must contain two increasing timestamps")
    start, end = map(_stamp, time_bounds)
    lower, upper = start.tai_ns, end.tai_ns
    if not np.iinfo(np.int64).min <= lower < upper <= np.iinfo(np.int64).max:
        raise ValueError("time_bounds must increase within int64 TAI nanoseconds")
    frequency = as_value(frequency_bounds, u.MHz, "frequency_bounds")
    if (
        frequency.shape != (2,)
        or not np.isfinite(frequency).all()
        or not 0 < frequency[0] < frequency[1]
    ):
        raise ValueError(
            "frequency_bounds must contain two increasing positive MHz values"
        )
    if frequency_bin_mhz is not None:
        width = as_value(frequency_bin_mhz, u.MHz, "frequency_bin_mhz")
        if width.ndim or not np.isfinite(width) or width <= 0:
            raise ValueError("frequency_bin_mhz must be finite and positive or None")
        width = float(width)
    else:
        width = None
    if not paths:
        raise ValueError("At least one CSO source is required")

    with (
        _source_read_paths(paths, cache_root) as (resolved, reads),
        ExitStack() as handles,
    ):
        sources = [
            _read_index(path, False, read_path=reads[str(path)], continuous_time=True)
            for path in resolved
        ]
        sources.sort(key=lambda item: (int(item["time_tai_ns"].min()), item["path"]))
        try:
            records, duplicate_count = _merge_records(
                sources, handles, reads, frequency, lower, upper
            )
            selected_frequency = [
                s["frequency"][s["selected_frequency"]]
                for s in sources
                if s["selected_frequency"].size
            ]
            if not selected_frequency:
                raise ValueError("No native frequencies fall inside frequency_bounds")
            if width is None:
                centers = np.unique(np.concatenate(selected_frequency))
                edges = _native_edges(centers, frequency)
            else:
                edges = _edges(*frequency, width)
                centers = (edges[:-1] + edges[1:]) / 2
            times = np.asarray(sorted(records), dtype=np.int64)
            cadences = np.asarray(
                [records[int(t)][0]["cadence_ns"] for t in times], dtype=np.int64
            )
            right_sum, left_sum, counts = _aggregate(
                sources, times, centers, edges, width, chunk
            )
            arrays = _measurements(right_sum, left_sum, counts)
            arrays.update(
                time_tai_ns=times,
                time_unix=_unix_ns(times),
                frequency_edges_mhz=edges,
                frequency_centers_mhz=centers,
            )
            segments, gaps, boundary = _support(
                times, cadences, lower, upper, gap_factor
            )
            reports = [
                {
                    k: value
                    for k, value in s.items()
                    if k
                    not in {
                        "time_unix",
                        "time_tai_ns",
                        "frequency",
                        "bscale",
                        "bzero",
                        "cube",
                        "handle",
                        "kept",
                        "selected_frequency",
                        "cadence_ns",
                    }
                }
                for s in sources
            ]
            metadata = {
                "sources": reports,
                "unit": "SFU",
                "time_scale": "TAI nanoseconds; UTC display",
                "requested_time_bounds_utc": [start.utc, end.utc],
                "requested_time_bounds_tai_ns": [lower, upper],
                "requested_time_bounds_unix": _unix_ns([lower, upper]).tolist(),
                "requested_frequency_bounds_mhz": frequency.tolist(),
                "frequency_bin_mhz": width,
                "segments": segments,
                "actual_coverage": segments,
                "gaps_unix": [_unix_ns(pair).tolist() for pair in gaps],
                "boundary_missing_support_unix": [
                    _unix_ns(pair).tolist() for pair in boundary
                ],
                "missing_support_unix": [
                    _unix_ns(pair).tolist() for pair in sorted(gaps + boundary)
                ],
                "missing_support_tai_ns": sorted(gaps + boundary),
                "gap_factor": gap_factor,
                "duplicate_time_samples_removed": duplicate_count,
                "duplicate_policy": "Require exactly identical physical paired values and selected frequency coordinates; otherwise reject",
                "native_samples_aggregated": int(counts.sum()),
                "time_aggregation": "none",
                "intensity_definition": "mean RCP + mean LCP; not half-sum Stokes I",
                "polarization_definition": "100*(sum_rcp-sum_lcp)/(sum_rcp+sum_lcp), same finite native pairs",
                "calibration_caveat": "SFU and RCP/LCP order are file declarations; absolute intensity and polarization calibration are not independently verified",
                "source_mutation": "none",
            }
            return arrays, metadata
        finally:
            for source in sources:
                source.pop("cube", None)


def _merge_records(sources, handles, reads, frequency, lower, upper):
    """Index stable native columns and reject conflicting simultaneous records."""
    from .cso_extract import _raw_hdus

    records = {}
    duplicate_count = 0
    for source in sources:
        if not np.isfinite([source["bscale"], source["bzero"]]).all():
            raise ValueError("FITS scale and offset must be finite")
        source["selected_frequency"] = np.flatnonzero(
            (source["frequency"] >= frequency[0])
            & (source["frequency"] <= frequency[1])
        )
        steps = np.diff(np.unique(source["time_tai_ns"]))
        if not steps.size:
            raise ValueError(
                "At least two distinct source timestamps are required to infer cadence"
            )
        source["cadence_ns"] = int(np.median(steps))
        source["handle"] = handles.enter_context(_raw_hdus(reads[source["path"]]))
        source["cube"] = source["handle"][0].data
        source["kept"] = []
        indices = np.flatnonzero(
            (source["time_tai_ns"] >= lower) & (source["time_tai_ns"] <= upper)
        )
        for column in indices[
            np.argsort(source["time_tai_ns"][indices], kind="stable")
        ]:
            if not source["selected_frequency"].size:
                continue
            stamp = int(source["time_tai_ns"][column])
            if stamp in records:
                previous, old_column = records[stamp]
                same_grid = np.array_equal(
                    previous["frequency"][previous["selected_frequency"]],
                    source["frequency"][source["selected_frequency"]],
                )
                if not same_grid or not np.array_equal(
                    _column_values(previous, old_column),
                    _column_values(source, column),
                    equal_nan=True,
                ):
                    raise ValueError(
                        "Conflicting paired CSO records at the same timestamp: "
                        f"UTC={TimeStamp.from_tai_ns(stamp).utc}, tai_ns={stamp}; "
                        f"{previous['path']} column {old_column} versus "
                        f"{source['path']} column {int(column)}"
                    )
                duplicate_count += 1
            else:
                records[stamp] = source, int(column)
                source["kept"].append(int(column))
    return records, duplicate_count


def _aggregate(sources, times, centers, edges, width, chunk):
    """Reduce paired native frequencies, retaining independent time columns."""
    shape = len(centers), len(times)
    right_sum, left_sum = np.zeros(shape), np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    for source in sources:
        fi = source["selected_frequency"]
        if not source["kept"]:
            continue
        cols = np.asarray(source["kept"], dtype=np.int64)
        labels = (
            np.searchsorted(centers, source["frequency"][fi])
            if width is None
            else np.minimum(
                np.searchsorted(edges, source["frequency"][fi], side="right") - 1,
                len(centers) - 1,
            )
        )
        groups = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
        for first in range(0, len(cols), chunk):
            selected = cols[first : first + chunk]
            block = np.array(
                source["cube"][:, fi[0] : fi[-1] + 1, :][:, :, selected],
                dtype=np.float64,
                copy=True,
            )
            block *= source["bscale"]
            block += source["bzero"]
            with np.errstate(over="ignore", invalid="ignore"):
                valid = np.isfinite(block).all(axis=0) & np.isfinite(
                    block[0] + block[1]
                )
            block[:, ~valid] = 0
            with np.errstate(over="ignore", invalid="ignore"):
                grouped = np.add.reduceat(block, groups, axis=1)
            if not np.isfinite(grouped).all():
                raise ValueError("CSO paired frequency aggregation overflow")
            target = np.ix_(
                labels[groups],
                np.searchsorted(times, source["time_tai_ns"][selected]),
            )
            right_sum[target], left_sum[target] = grouped[0], grouped[1]
            counts[target] = np.add.reduceat(valid.astype(np.int64), groups, axis=0)
    return right_sum, left_sum, counts


def _column_values(source, column):
    values = np.array(
        source["cube"][:, source["selected_frequency"], column],
        dtype=np.float64,
        copy=True,
    )
    return values * source["bscale"] + source["bzero"]


def _measurements(right_sum, left_sum, counts):
    right = np.divide(
        right_sum, counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )
    left = np.divide(
        left_sum, counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )
    with np.errstate(over="ignore", invalid="ignore"):
        total = right + left
    total[~np.isfinite(total)] = np.nan
    scale = np.maximum(np.abs(right), np.abs(left))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_right, normalized_left = right / scale, left / scale
        raw = (
            100
            * (normalized_right - normalized_left)
            / (normalized_right + normalized_left)
        )
    valid = np.isfinite(total) & (total > 0)
    raw[~valid] = np.nan
    polarization = np.where(valid & (right >= 0) & (left >= 0), raw, np.nan)
    return {
        "sum_rcp": right_sum,
        "sum_lcp": left_sum,
        "sample_count": counts,
        "rcp": right,
        "lcp": left,
        "intensity": total,
        "polarization_percent": polarization,
        "raw_polarization_percent": raw,
    }
