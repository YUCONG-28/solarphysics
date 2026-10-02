"""Read CSO spectra without changing the source FITS files.

The returned image has frequency rows and time columns.  Empty bins remain NaN;
``sample_count`` counts native frequency/time samples, not merely time samples.
Float products are paired RCP + LCP before averaging.  Eight-bit products retain
the original channel-zero RCP encoding by default, or sum the native RCP/LCP
codes after conversion to float64.  They are never described as calibrated SFU.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

__all__ = ["extract_spectrum"]


def _unix(value: Any) -> float:
    text = str(value).strip()
    compact = re.fullmatch(r"(\d{14})(\d*)", text)
    if compact:
        date = datetime.strptime(compact[1], "%Y%m%d%H%M%S")
        fraction = float("0." + compact[2]) if compact[2] else 0.0
        return date.replace(tzinfo=timezone.utc).timestamp() + fraction
    date = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    elif date.utcoffset().total_seconds() != 0:
        raise ValueError(f"FITS date has a non-UTC offset: {text}")
    return date.timestamp()


def _iso(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _date_card(header: fits.Header, *keys: str) -> tuple[str, Any]:
    for key in keys:
        if key in header:
            return key, header[key]
    raise ValueError(f"Missing required FITS date: {' or '.join(keys)}")


def _column(table: Any, name: str) -> tuple[np.ndarray, str | None]:
    names = {n.lower(): n for n in table.columns.names}
    if name not in names:
        raise ValueError(f"CSO table has no {name!r} coordinate")
    actual = names[name]
    return (
        np.asarray(table.data[actual], dtype=np.float64).ravel(),
        table.columns[actual].unit,
    )


def _read_index(path: str, quantized: bool) -> dict[str, Any]:
    p = Path(path)
    stat = p.stat()
    with fits.open(p, memmap=True, do_not_scale_image_data=True) as hdul:
        header = hdul[0].header
        if str(header.get("TIMESYS", "")).strip().upper() != "UTC":
            raise ValueError(f"{p}: TIMESYS must explicitly be UTC")
        polarization = str(header.get("POLARIZA", "")).strip()
        if polarization.casefold() != "rcp and lcp":
            raise ValueError(
                f"{p}: expected POLARIZA='RCP and LCP', got {polarization!r}"
            )
        bitpix = int(header["BITPIX"])
        if quantized and bitpix != 8:
            raise ValueError(f"{p}: encoded product must have BITPIX=8")
        if not quantized and bitpix not in (-32, -64):
            raise ValueError(f"{p}: calibrated product must be floating point")
        unit = str(header.get("BUNIT", "")).strip()
        if not quantized and unit.lower().replace(" ", "") not in {
            "solarfluxunit(sfu)",
            "sfu",
        }:
            raise ValueError(f"{p}: floating product does not explicitly declare SFU")
        if int(header.get("NAXIS", 0)) != 3 or int(header.get("NAXIS3", 0)) != 2:
            raise ValueError(f"{p}: expected a two-polarization frequency/time cube")
        time_seconds, time_unit = _column(hdul[1], "time")
        frequency, frequency_unit = _column(hdul[1], "frequency")
        if time_unit and time_unit.strip().lower() not in {
            "s",
            "sec",
            "second",
            "seconds",
        }:
            raise ValueError(f"{p}: unexpected time-coordinate unit {time_unit!r}")
        if frequency_unit and frequency_unit.strip().lower() != "mhz":
            raise ValueError(
                f"{p}: unexpected frequency-coordinate unit {frequency_unit!r}"
            )
        if time_seconds.size != int(header["NAXIS1"]) or frequency.size != int(
            header["NAXIS2"]
        ):
            raise ValueError(f"{p}: table coordinates do not match image dimensions")
        if not time_seconds.size or not frequency.size:
            raise ValueError(f"{p}: empty spectrum")
        if not np.isfinite(time_seconds).all():
            raise ValueError(f"{p}: time coordinates must be finite")
        time_steps = np.diff(time_seconds)
        backward_steps = time_steps[time_steps < 0]
        stable_order = np.argsort(time_seconds, kind="stable")
        reordered_native_columns = int(
            np.count_nonzero(stable_order != np.arange(time_seconds.size))
        )
        if not np.isfinite(frequency).all() or np.any(np.diff(frequency) <= 0):
            raise ValueError(
                f"{p}: frequency coordinates must be finite and strictly increasing"
            )
        start_key, start_value = _date_card(header, "DATE_BEG", "DATE-BEG", "DATE-OBS")
        end_key, end_value = _date_card(header, "DATE_END", "DATE-END")
        declared_start = _unix(start_value)
        declared_end = _unix(end_value)
        origin = declared_start - float(time_seconds[0])
        midnight = round(origin / 86400.0) * 86400.0
        if abs(origin - midnight) > 0.001:
            raise ValueError(
                f"{p}: time origin is {origin - midnight:.6f} s from UTC midnight"
            )
        # The declared first timestamp can have rounded fractions.  An explicitly
        # validated midnight is the stable origin for negative time coordinates.
        time_unix = midnight + time_seconds
        end_error = float(time_unix[-1] - declared_end)
        if abs(end_error) > 0.1:
            raise ValueError(
                f"{p}: final table time disagrees with {end_key} by {end_error:.6f} s"
            )
        index_hash = hashlib.sha256()
        index_hash.update(header.tostring().encode("ascii"))
        index_hash.update(hdul[1].header.tostring().encode("ascii"))
        index_hash.update(time_seconds.astype("<f8", copy=False).tobytes())
        index_hash.update(frequency.astype("<f8", copy=False).tobytes())
        cards = {
            key: header[key]
            for key in (
                "TIMESYS",
                "POLARIZA",
                "BUNIT",
                "BITPIX",
                "NAXIS1",
                "NAXIS2",
                "NAXIS3",
                "BSCALE",
                "BZERO",
                "TELESCOP",
                "DEV_NAME",
                "QUAL_FLG",
                start_key,
                end_key,
            )
            if key in header
        }
        return {
            "path": str(p.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "header": cards,
            "bitpix": bitpix,
            "declared_unit": unit,
            "index_sha256": index_hash.hexdigest(),
            "index_hash_scope": "primary and table headers plus complete time/frequency coordinates; excludes image payload",
            "first_actual_utc": _iso(float(np.min(time_unix))),
            "last_actual_utc": _iso(float(np.max(time_unix))),
            "native_time_axis_inversion_count": int(backward_steps.size),
            "largest_backward_step_seconds": (
                float(-backward_steps.min()) if backward_steps.size else 0.0
            ),
            "native_columns_reordered_count": reordered_native_columns,
            "time_order_policy": "stable ascending native timestamps, carrying each original paired image column; no timestamp shifts or interpolation",
            "origin_midnight_utc": _iso(midnight),
            "origin_validation_error_seconds": float(origin - midnight),
            "end_validation_error_seconds": end_error,
            "time_unix": time_unix,
            "frequency": frequency,
            "bscale": float(header.get("BSCALE", 1.0)),
            "bzero": float(header.get("BZERO", 0.0)),
        }


def _edges(lower: float, upper: float, width: float) -> np.ndarray:
    units = (upper - lower) / width
    # Subtracting Unix timestamps loses sub-microsecond precision.  A nominal
    # seven-bin interval must not gain an eighth bin from that rounding alone.
    roundoff = max(1e-10, 4.0 * abs(np.spacing(max(abs(lower), abs(upper)))) / width)
    nearest = round(units)
    count = int(nearest if abs(units - nearest) <= roundoff else math.ceil(units))
    count = max(1, count)
    edges = lower + np.arange(count + 1, dtype=np.float64) * width
    edges[-1] = upper
    return edges


def _native_edges(centers: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    if centers.size == 1:
        return np.asarray(bounds, dtype=np.float64)
    middle = (centers[:-1] + centers[1:]) / 2.0
    return np.r_[
        max(bounds[0], centers[0] - (centers[1] - centers[0]) / 2.0),
        middle,
        min(bounds[1], centers[-1] + (centers[-1] - centers[-2]) / 2.0),
    ]


def _segments(times: np.ndarray, cadence: float) -> list[dict[str, Any]]:
    if not times.size:
        return []
    breaks = np.flatnonzero(np.diff(times) > max(3.0 * cadence, 0.000001)) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, times.size]
    return [
        {
            "start_utc": _iso(float(times[a])),
            "end_utc": _iso(float(times[b - 1])),
            "native_time_samples": int(b - a),
        }
        for a, b in zip(starts, stops, strict=True)
    ]


def extract_spectrum(
    paths: list[str],
    bounds_unix: tuple[float, float],
    *,
    quantized: bool,
    encoded_polarization: str = "RCP",
    time_bin_seconds: float = 0.1,
    frequency_bounds: tuple[float, float] = (90.0, 300.0),
    frequency_bin_mhz: float | None = 0.25,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract a bounded interval, preserving missing bins and exact overlaps.

    ``frequency_bin_mhz=None`` retains native selected frequency centers and
    midpoint edges.  It prevents a sparse native grid becoming artificial blank
    stripes when displayed on unnecessarily finer frequency bins.

    ``encoded_polarization`` selects RCP or RCP+LCP for eight-bit products only.
    The sum is formed in float64 before binning, without applying BSCALE/BZERO
    or interpreting encoded values as SFU.  Floating products always use RCP+LCP.
    """
    lower, upper = map(float, bounds_unix)
    freq_lower, freq_upper = map(float, frequency_bounds)
    if not paths:
        raise ValueError("At least one CSO source is required")
    if encoded_polarization not in {"RCP", "RCP+LCP"}:
        raise ValueError("encoded_polarization must be 'RCP' or 'RCP+LCP'")
    if not np.isfinite([lower, upper, freq_lower, freq_upper, time_bin_seconds]).all():
        raise ValueError("Bounds and time-bin width must be finite")
    if upper <= lower or freq_upper <= freq_lower or time_bin_seconds <= 0:
        raise ValueError("Bounds must increase and time-bin width must be positive")
    if frequency_bin_mhz is not None and (
        not np.isfinite(frequency_bin_mhz) or frequency_bin_mhz <= 0
    ):
        raise ValueError("Frequency-bin width must be positive or None")

    sources = [_read_index(path, quantized) for path in paths]
    sources.sort(
        key=lambda source: (float(np.min(source["time_unix"])), source["path"])
    )
    time_edges = _edges(lower, upper, time_bin_seconds)
    if frequency_bin_mhz is None:
        selected = [
            s["frequency"][
                (s["frequency"] >= freq_lower) & (s["frequency"] <= freq_upper)
            ]
            for s in sources
        ]
        frequency_centers = np.unique(np.concatenate(selected))
        if not frequency_centers.size:
            raise ValueError(
                "No native frequency coordinates fall inside the requested range"
            )
        frequency_edges = _native_edges(frequency_centers, (freq_lower, freq_upper))
    else:
        frequency_edges = _edges(freq_lower, freq_upper, frequency_bin_mhz)
        frequency_centers = (frequency_edges[:-1] + frequency_edges[1:]) / 2.0

    shape = (frequency_centers.size, time_edges.size - 1)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    seen_times = np.empty(0, dtype=np.float64)
    reports = []
    overlap_removed = 0
    for source in sources:
        all_times = source["time_unix"]
        frequency = source["frequency"]
        positive_steps = np.diff(all_times)
        positive_steps = positive_steps[positive_steps > 0]
        cadence = (
            float(np.median(positive_steps))
            if positive_steps.size
            else time_bin_seconds
        )
        time_indices = np.flatnonzero((all_times >= lower) & (all_times <= upper))
        before_dedup = int(time_indices.size)
        if time_indices.size:
            # unique returns ascending timestamps with the first original
            # column index for duplicates.  The paired image columns are read
            # through these indices, preserving their timestamp association.
            unique_times, first_positions = np.unique(
                all_times[time_indices], return_index=True
            )
            time_indices = time_indices[first_positions]
            selected_reordered = int(
                np.count_nonzero(time_indices != np.sort(time_indices))
            )
            keep = ~np.isin(unique_times, seen_times, assume_unique=True)
            time_indices = time_indices[keep]
            kept_times = unique_times[keep]
            seen_times = np.union1d(seen_times, kept_times)
        else:
            kept_times = np.empty(0, dtype=np.float64)
            selected_reordered = 0
        removed = before_dedup - int(time_indices.size)
        overlap_removed += removed
        frequency_indices = np.flatnonzero(
            (frequency >= freq_lower) & (frequency <= freq_upper)
        )
        public = {
            k: v
            for k, v in source.items()
            if k not in {"time_unix", "frequency", "bscale", "bzero"}
        }
        public.update(
            {
                "native_cadence_seconds": cadence,
                "selected_time_samples": int(time_indices.size),
                "selected_columns_reordered_count": selected_reordered,
                "duplicate_time_samples_removed": removed,
                "selected_frequency_samples": int(frequency_indices.size),
                "selected_actual_coverage": _segments(kept_times, cadence),
            }
        )
        reports.append(public)
        if not time_indices.size or not frequency_indices.size:
            continue
        freq_start, freq_stop = (
            int(frequency_indices[0]),
            int(frequency_indices[-1]) + 1,
        )
        native_frequency = frequency[frequency_indices]
        if frequency_bin_mhz is None:
            frequency_bins = np.searchsorted(frequency_centers, native_frequency)
        else:
            frequency_bins = (
                np.searchsorted(frequency_edges, native_frequency, side="right") - 1
            )
            frequency_bins = np.minimum(frequency_bins, shape[0] - 1)
        freq_group_starts = np.r_[0, np.flatnonzero(np.diff(frequency_bins)) + 1]
        output_frequency_bins = frequency_bins[freq_group_starts]
        # At most about two million frequency/time samples per working chunk;
        # output arrays and small coordinate vectors are the only lasting data.
        chunk_columns = max(1, 2_000_000 // native_frequency.size)
        with fits.open(
            source["path"], memmap=True, do_not_scale_image_data=True
        ) as hdul:
            data = hdul[0].data
            if (
                Path(source["path"]).stat().st_size,
                Path(source["path"]).stat().st_mtime_ns,
            ) != (source["size_bytes"], source["mtime_ns"]):
                raise RuntimeError(f"Source changed while extracting: {source['path']}")
            for offset in range(0, time_indices.size, chunk_columns):
                columns = time_indices[offset : offset + chunk_columns]
                values = np.asarray(
                    data[0, freq_start:freq_stop, :][:, columns], dtype=np.float64
                )
                if quantized:
                    if encoded_polarization == "RCP+LCP":
                        other = np.asarray(
                            data[1, freq_start:freq_stop, :][:, columns],
                            dtype=np.float64,
                        )
                        valid = np.isfinite(values) & np.isfinite(other)
                        values += other
                        valid &= np.isfinite(values)
                        del other
                    else:
                        valid = np.isfinite(values)
                else:
                    values *= source["bscale"]
                    values += source["bzero"]
                    other = np.asarray(
                        data[1, freq_start:freq_stop, :][:, columns], dtype=np.float64
                    )
                    other *= source["bscale"]
                    other += source["bzero"]
                    valid = np.isfinite(values) & np.isfinite(other)
                    with np.errstate(over="ignore", invalid="ignore"):
                        values += other
                    valid &= np.isfinite(values)
                    del other
                values[~valid] = 0.0
                grouped_sums = np.add.reduceat(values, freq_group_starts, axis=0)
                grouped_counts = np.add.reduceat(
                    valid.astype(np.int64), freq_group_starts, axis=0
                )
                time_bins = (
                    np.searchsorted(time_edges, all_times[columns], side="right") - 1
                )
                time_bins = np.minimum(time_bins, shape[1] - 1)
                time_group_starts = np.r_[0, np.flatnonzero(np.diff(time_bins)) + 1]
                output_time_bins = time_bins[time_group_starts]
                target = np.ix_(output_frequency_bins, output_time_bins)
                sums[target] += np.add.reduceat(grouped_sums, time_group_starts, axis=1)
                counts[target] += np.add.reduceat(
                    grouped_counts, time_group_starts, axis=1
                )
            after = Path(source["path"]).stat()
            if (after.st_size, after.st_mtime_ns) != (
                source["size_bytes"],
                source["mtime_ns"],
            ):
                raise RuntimeError(f"Source changed while extracting: {source['path']}")
    intensity = np.full(shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=intensity, where=counts > 0)
    arrays = {
        "time_edges_unix": time_edges,
        "frequency_edges_mhz": frequency_edges,
        "frequency_centers_mhz": frequency_centers,
        "intensity": intensity,
        "sample_count": counts,
    }
    metadata = {
        "sources": reports,
        "unit": "relative encoded intensity" if quantized else "SFU",
        "polarization": encoded_polarization if quantized else "RCP+LCP",
        "unit_caveat": (
            "Native uint8 RCP and LCP codes summed after each channel is converted to float64; the sum is relative encoded intensity and may exceed 255. The header SFU label does not establish an invertible calibration. No BSCALE/BZERO decoding or logarithm applied."
            if quantized and encoded_polarization == "RCP+LCP"
            else (
                "Native uint8 channel-zero RCP codes; the header SFU label does not establish an invertible calibration. No decoding or logarithm applied."
                if quantized
                else "Native floating-point SFU; both polarizations must be finite at the same frequency/time sample."
            )
        ),
        "processing": (
            "convert native RCP and LCP codes separately to float64, sum paired values per native frequency/time sample, then finite sample-weighted mean"
            if quantized and encoded_polarization == "RCP+LCP"
            else (
                "native RCP encoded values, then finite sample mean"
                if quantized
                else "finite paired RCP+LCP per native sample, then sample-weighted mean"
            )
        ),
        "binning": {
            "time_bin_seconds": time_bin_seconds,
            "frequency_bin_mhz": frequency_bin_mhz,
            "frequency_policy": (
                "native centers with clipped midpoint edges"
                if frequency_bin_mhz is None
                else "fixed bins; no interpolation or filling"
            ),
            "requested_frequency_bounds_mhz": [freq_lower, freq_upper],
            "requested_time_bounds_utc": [_iso(lower), _iso(upper)],
            "last_time_bin_includes_upper_endpoint": True,
        },
        "shape_frequency_time": list(shape),
        "empty_bin_count": int(np.count_nonzero(counts == 0)),
        "empty_time_bin_count": int(np.count_nonzero(~np.any(counts > 0, axis=0))),
        "native_samples_aggregated": int(counts.sum()),
        "duplicate_time_samples_removed": overlap_removed,
        "deduplication": "identical absolute timestamps keep first column in earliest-starting source; ties resolve by path",
        "time_order_policy": "stable ascending native timestamps with original paired columns; no timestamp shifts or interpolation",
        "actual_coverage": _segments(
            seen_times,
            min(
                (r["native_cadence_seconds"] for r in reports), default=time_bin_seconds
            ),
        ),
        "source_mutation": "none",
    }
    return arrays, metadata
