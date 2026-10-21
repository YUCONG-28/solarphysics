"""Memory-bounded CSO R+L spectra and native-sample SFU light curves.

Only finite native R+L samples contribute to a bin. Missing samples are never
interpolated, replaced with zero, or connected by compressing the time axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .cso import cso_base_datetime

__all__ = ["CsoComparison", "read_cso_comparison", "normalize_spectrum_db"]


@dataclass
class CsoComparison:
    time_edges: np.ndarray
    frequency_edges: np.ndarray
    spectrum_sfu: np.ndarray
    spectrum_counts: np.ndarray
    lightcurves_sfu: np.ndarray
    lightcurve_counts: np.ndarray
    metadata: dict

    @property
    def time_centers(self):
        return (self.time_edges[:-1] + self.time_edges[1:]) / 2

    @property
    def frequency_centers(self):
        return (self.frequency_edges[:-1] + self.frequency_edges[1:]) / 2


def _inspect(path):
    with fits.open(path, memmap=True) as hd:
        header = hd[0].header
        if header.get("TIMESYS") != "UTC" or header.get("POLARIZA") != "RCP and LCP":
            raise ValueError(f"CSO requires explicit UTC and RCP/LCP channels: {path}")
        unit = str(header.get("BUNIT", ""))
        normalized_unit = "".join(unit.lower().split())
        if normalized_unit not in ("sfu", "solarfluxunit(sfu)"):
            raise ValueError(f"CSO requires linear SFU source units: {path}")
        times = np.asarray(hd[1].data["time"], dtype=float).ravel().copy()
        frequency = np.asarray(hd[1].data["frequency"], dtype=float).ravel().copy()
        if (
            not len(times)
            or not len(frequency)
            or hd[0].shape != (2, len(frequency), len(times))
            or not np.isfinite(times).all()
            or not np.isfinite(frequency).all()
            or not np.all(np.diff(times) > 0)
            or not np.all(np.diff(frequency) > 0)
        ):
            raise ValueError(f"Invalid CSO axes or primary shape: {path}")
        times += Time(cso_base_datetime(header["DATE-OBS"], times)).unix
    stat = path.stat()
    record = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "first_utc": Time(times[0], format="unix").isot,
        "last_utc": Time(times[-1], format="unix").isot,
        "source_unit": unit,
        "source_polarization": "RCP and LCP",
    }
    return path, times, frequency, record


def _means(sums, counts):
    result = np.full(sums.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def read_cso_comparison(
    paths,
    bounds,
    *,
    bands_mhz,
    half_width_mhz=1.0,
    dt=0.1,
    df=0.25,
    max_chunk_bytes=64 * 1024**2,
):
    """Return full positive-frequency CSO spectrum and selected-band curves.

    ``bounds`` is a pair of Unix UTC seconds; the window is half-open [start,
    end). Time bins span that exact window, including uncovered portions.
    Frequency bins include every strictly positive source frequency. Selected
    bands include both edges, ``center +/- half_width_mhz``. Their curves are
    means of *native* finite R+L samples, rather than means of spectrum bins.

    ``max_chunk_bytes`` budgets temporary sample processing, not the returned
    binned arrays. The source image stays memory mapped. Files may be supplied
    in any order, but overlapping native time ranges are rejected to prevent
    accidental double counting. No files is a supported NO DATA result.
    """
    bounds = np.asarray(bounds, dtype=float)
    bands = np.asarray(bands_mhz, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("bounds must be two increasing finite Unix UTC seconds")
    if (
        bands.ndim != 1
        or not len(bands)
        or not np.isfinite(bands).all()
        or np.any(bands <= 0)
        or not np.isfinite(half_width_mhz)
        or half_width_mhz <= 0
        or not np.isfinite(dt)
        or dt <= 0
        or not np.isfinite(df)
        or df <= 0
        or max_chunk_bytes < 64
    ):
        raise ValueError("Invalid CSO frequency, bin, or chunk parameters")
    span = bounds[1] - bounds[0]
    nearest = round(span / dt)
    # Unix double precision has sub-microsecond rounding: a requested .4 s
    # interval must not accidentally acquire a fifth .1 s bin.
    tolerance = 2 * np.spacing(max(abs(bounds)))
    nt = int(
        nearest
        if nearest > 0 and abs(span - nearest * dt) <= tolerance
        else np.ceil(span / dt)
    )
    time_edges = bounds[0] + np.arange(nt + 1) * dt
    time_edges[-1] = bounds[1]
    sources = sorted((_inspect(Path(p)) for p in paths), key=lambda s: s[1][0])
    for previous, current in zip(sources[:-1], sources[1:], strict=True):
        if current[1][0] <= previous[1][-1]:
            raise ValueError("Overlapping CSO files require explicit deduplication")
    selected = [s for s in sources if s[1][-1] >= bounds[0] and s[1][0] < bounds[1]]
    positive = [s[2][s[2] > 0] for s in selected if np.any(s[2] > 0)]
    if positive:
        low = np.floor(min(f[0] for f in positive) / df) * df
        high_bin = int(np.floor((max(f[-1] for f in positive) - low) / df))
        frequency_edges = low + np.arange(high_bin + 2) * df
    else:
        frequency_edges = np.array([], dtype=float)
    nf = max(0, len(frequency_edges) - 1)
    spectrum_sums = np.zeros((nf, nt), dtype=float)
    spectrum_counts = np.zeros((nf, nt), dtype=np.uint64)
    band_sums = np.zeros((len(bands), nt), dtype=float)
    band_counts = np.zeros((len(bands), nt), dtype=np.uint64)
    records = []
    # 64 bytes per native sample conservatively covers the two source casts,
    # R+L, validity masks, band indexing, and one-dimensional reductions.
    sample_budget = max(1, int(max_chunk_bytes) // 64)
    for path, times, frequency, record in selected:
        left, right = np.searchsorted(times, bounds)
        if right == left or not np.any(frequency > 0):
            continue
        bin_indices = np.searchsorted(time_edges, times[left:right], side="right") - 1
        frequency_indices = np.searchsorted(frequency, frequency_edges)
        first_positive = np.searchsorted(frequency, 0.0, side="right")
        band_native = [
            (
                (frequency >= center - half_width_mhz)
                & (frequency <= center + half_width_mhz)
                & (frequency > 0)
            )
            for center in bands
        ]
        record["selected_native_times"] = int(right - left)
        record["bands"] = [
            {
                "center_mhz": float(center),
                "half_width_mhz": float(half_width_mhz),
                "native_channels": int(mask.sum()),
                "native_min_mhz": float(frequency[mask][0]) if mask.any() else None,
                "native_max_mhz": float(frequency[mask][-1]) if mask.any() else None,
            }
            for center, mask in zip(bands, band_native, strict=True)
        ]
        records.append(record)
        with fits.open(path, memmap=True) as hd:
            data = hd[0].data
            for k, (a, b) in enumerate(
                zip(frequency_indices[:-1], frequency_indices[1:], strict=True)
            ):
                a = max(a, first_positive)
                # Usually each .25 MHz bin contains three or four channels.
                # Split unusually large bins as well as long temporal slices.
                frequency_step = max(1, sample_budget // (right - left))
                for fa in range(a, b, frequency_step):
                    fb = min(b, fa + frequency_step)
                    time_step = max(1, sample_budget // (fb - fa))
                    for ta in range(left, right, time_step):
                        tb = min(right, ta + time_step)
                        total = np.array(data[0, fa:fb, ta:tb], dtype=float)
                        with np.errstate(invalid="ignore", over="ignore"):
                            total += np.asarray(data[1, fa:fb, ta:tb], dtype=float)
                        valid = np.isfinite(total)
                        total[~valid] = 0.0
                        bins = bin_indices[ta - left : tb - left]
                        spectrum_sums[k] += np.bincount(
                            bins, weights=total.sum(axis=0), minlength=nt
                        )
                        spectrum_counts[k] += np.bincount(
                            bins, weights=valid.sum(axis=0), minlength=nt
                        ).astype(np.uint64)
                        for j, mask in enumerate(band_native):
                            local_mask = mask[fa:fb]
                            if not local_mask.any():
                                continue
                            band_sums[j] += np.bincount(
                                bins,
                                weights=total[local_mask].sum(axis=0),
                                minlength=nt,
                            )
                            band_counts[j] += np.bincount(
                                bins,
                                weights=valid[local_mask].sum(axis=0),
                                minlength=nt,
                            ).astype(np.uint64)
    spectrum = _means(spectrum_sums, spectrum_counts)
    curves = _means(band_sums, band_counts)
    metadata = {
        "inputs": records,
        "ignored_outside_window": [str(s[0]) for s in sources if s not in selected],
        "window_unix": bounds.tolist(),
        "time_bin_seconds": float(dt),
        "frequency_bin_mhz": float(df),
        "frequency_unit": "MHz",
        "unit": "SFU (source header)",
        "polarization": "RCP+LCP",
        "bands_mhz": bands.tolist(),
        "half_width_mhz": float(half_width_mhz),
        "missing_time_bins": int(np.count_nonzero(~np.any(spectrum_counts, axis=0))),
        "missing_spectrum_bins": int(np.count_nonzero(spectrum_counts == 0)),
        "interpolated_samples": 0,
        "time_compression": False,
        "processing": "finite native RCP+LCP sample means; no background subtraction",
        "lightcurve_processing": "native sample weighted means; no logarithm or smoothing",
    }
    return CsoComparison(
        time_edges,
        frequency_edges,
        spectrum,
        spectrum_counts,
        curves,
        band_counts,
        metadata,
    )


def normalize_spectrum_db(spectrum_sfu, percentile=20.0, upper_percentile=99.5):
    """Per-frequency positive finite P20 background; retain missing data."""
    spectrum = np.asarray(spectrum_sfu, dtype=float)
    if (
        spectrum.ndim != 2
        or not 0 <= percentile <= 100
        or not 0 <= upper_percentile <= 100
    ):
        raise ValueError("Expected a 2-D spectrum and valid percentiles")
    background = np.full(spectrum.shape[0], np.nan)
    db = np.full_like(spectrum, np.nan)
    for i, row in enumerate(spectrum):
        good = np.isfinite(row) & (row > 0)
        if not good.any():
            continue
        background[i] = np.percentile(row[good], percentile)
        # Log subtraction avoids overflowing an intermediate intensity ratio.
        db[i, good] = 10 * (np.log10(row[good]) - np.log10(background[i]))
    valid_values = db[np.isfinite(db)]
    vmax = (
        float(np.percentile(valid_values, upper_percentile))
        if valid_values.size
        else 1.0
    )
    if vmax <= 0:
        vmax = 1.0
    return db, {
        "unit": "dB relative to per-frequency background",
        "formula": "10 log10(I/B(f)); I=RCP+LCP",
        "background_percentile": float(percentile),
        "background_sfu": [float(x) if np.isfinite(x) else None for x in background],
        "upper_percentile": float(upper_percentile),
        "vmin": 0.0,
        "vmax": vmax,
        "interpolated_samples": 0,
    }
