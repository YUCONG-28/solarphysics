"""Lazy FITS-memmap spectrogram container shared with Apps workflows.

Private module (no public-export contract). The reader keeps the opened
HDUList alive because the memmapped planes remain valid only while the file
handle is open; callers must close the returned handle.
"""

from __future__ import annotations

import datetime

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from solar_toolkit._utils.logging import timing_decorator


def _find_range(arr: np.ndarray, lo: float, hi: float):
    """Fast range index lookup using searchsorted with boundary protection."""
    if lo > hi:
        lo, hi = hi, lo
    if len(arr) == 0:
        raise ValueError("Cannot search an empty axis")

    if len(arr) > 1 and arr[0] > arr[-1]:
        arr_rev = arr[::-1]
        r0 = int(np.clip(np.searchsorted(arr_rev, lo, side="left"), 0, len(arr) - 1))
        r1 = int(
            np.clip(np.searchsorted(arr_rev, hi, side="right") - 1, 0, len(arr) - 1)
        )
        i0 = len(arr) - 1 - r1
        i1 = len(arr) - 1 - r0
    else:
        i0 = int(np.clip(np.searchsorted(arr, lo, side="left"), 0, len(arr) - 1))
        i1 = int(np.clip(np.searchsorted(arr, hi, side="right") - 1, 0, len(arr) - 1))
    return i0, max(i0, i1)


class LazySpectrogram:
    """
    Container that holds FITS memmap references and metadata without loading
    the full array into memory. The read_slice_rebinned() method performs
    on-the-fly downsampling with minimal peak memory usage.
    """

    __slots__ = (
        "_raw",
        "time",
        "freq",
        "polar",
        "dateobs",
        "unit",
        "dt_base",
        "source_path",
    )

    def __init__(
        self,
        raw_memmap,
        time_arr,
        freq_arr,
        polar,
        dateobs,
        unit,
        dt_base,
        source_path="",
    ):
        self._raw = raw_memmap
        self.time = time_arr.astype(np.float64)
        self.freq = freq_arr.astype(np.float32)
        self.polar = polar
        self.dateobs = dateobs
        self.unit = unit
        self.dt_base = dt_base
        self.source_path = source_path

    def read_slice_rebinned(
        self,
        t1: datetime.datetime,
        t2: datetime.datetime,
        f1: float,
        f2: float,
        t_bin: int,
        f_bin: int,
        chunk_mem_mb: int = 64,
    ):
        """
        Read from memmap in chunks and immediately apply block-mean downsampling.
        Peak memory usage is approximately chunk_mem_mb.

        Process:
          1. Calculate indices and align to bin multiples
          2. Read chunk_cols_raw columns per iteration (approx chunk_mem_mb / freq rows)
          3. Reshape+mean each chunk for downsampling, write to output array
        """
        t1s = (t1 - self.dt_base).total_seconds()
        t2s = (t2 - self.dt_base).total_seconds()

        ti0, ti1 = _find_range(self.time, t1s, t2s)
        fi0, fi1 = _find_range(self.freq, f1, f2)

        n_freq_raw = fi1 - fi0 + 1
        n_time_raw = ti1 - ti0 + 1

        # Align to bin multiples
        n_freq_trim = (n_freq_raw // f_bin) * f_bin
        n_time_trim = (n_time_raw // t_bin) * t_bin
        n_freq_out = n_freq_trim // f_bin
        n_time_out = n_time_trim // t_bin

        if n_freq_out <= 0 or n_time_out <= 0:
            raise ValueError(
                f"Selected range is too short after binning for {self.polar}: "
                f"raw={n_freq_raw}x{n_time_raw}, bin={f_bin}x{t_bin}, "
                f"file={self.source_path}"
            )

        raw_mb = n_freq_raw * n_time_raw * 4 / 1e6
        out_mb = n_freq_out * n_time_out * 4 / 1e6
        print(
            f"    [{self.polar}] Raw: {n_freq_raw}x{n_time_raw} "
            f"({raw_mb:.0f} MB)  ->  Output: {n_freq_out}x{n_time_out} "
            f"({out_mb:.1f} MB)"
        )

        # Columns per chunk: keep memory approx chunk_mem_mb, must be multiple of t_bin
        cols_per_chunk = max(
            t_bin, (int(chunk_mem_mb * 1e6 / (n_freq_trim * 4)) // t_bin) * t_bin
        )

        Z_out = np.empty((n_freq_out, n_time_out), dtype=np.float32)
        out_col = 0

        for col0 in tqdm(
            range(0, n_time_trim, cols_per_chunk),
            desc=f"    Reading {self.polar}",
            leave=False,
        ):
            col1 = min(col0 + cols_per_chunk, n_time_trim)
            n_cols = ((col1 - col0) // t_bin) * t_bin  # Alignment
            if n_cols == 0:
                continue

            # Trigger actual disk I/O, immediately copy to float32
            chunk = np.array(
                self._raw[fi0 : fi0 + n_freq_trim, ti0 + col0 : ti0 + col0 + n_cols],
                dtype=np.float32,
            )  # (n_freq_trim, n_cols)

            # Perform block-mean for both frequency and time axes
            n_t_chunk = n_cols // t_bin
            chunk_rb = chunk.reshape(n_freq_out, f_bin, n_t_chunk, t_bin).mean(
                axis=(1, 3), dtype=np.float32
            )

            Z_out[:, out_col : out_col + n_t_chunk] = chunk_rb
            out_col += n_t_chunk

        freq_out = (
            self.freq[fi0 : fi0 + n_freq_trim].reshape(n_freq_out, f_bin).mean(axis=1)
        )
        time_out = (
            self.time[ti0 : ti0 + n_time_trim].reshape(n_time_out, t_bin).mean(axis=1)
        )

        return Z_out, time_out, freq_out


@timing_decorator
def read_cso_fits(fn: str):
    """
    Open FITS file, read metadata, and return (list of LazySpectrogram, hdu handle).
    The hdu must remain open until all read_slice_rebinned() calls complete.
    """
    hdu = fits.open(fn, memmap=True)
    try:
        header = hdu[0].header
        raw = hdu[0].data
        time_ = np.ravel(hdu[1].data["time"])
        freq_ = np.ravel(hdu[1].data["frequency"])

        dateobs = header.get("DATE-OBS") or header.get("DATE_OBS")
        dt_base = datetime.datetime.fromisoformat(dateobs[:10])

        if time_[0] < 0:
            dt_base = dt_base + datetime.timedelta(days=1)
            dateobs = dt_base.isoformat()

        polars = header["POLARIZA"]
        if header["NAXIS"] == 3 and polars == "RCP and LCP":
            polars = "RL"

        unit = header.get("BUNIT") or header.get("QUANTITY", "K")

        results = []
        if raw.ndim == 2:
            results.append(
                LazySpectrogram(raw, time_, freq_, polars, dateobs, unit, dt_base, fn)
            )
            print(
                f"  Single polarization: {polars}  Size: {raw.shape}  "
                f"({raw.nbytes/1e9:.2f} GB, not loaded into memory)"
            )
        elif raw.ndim == 3:
            for ii in range(raw.shape[0]):
                polar = polars[ii] * 2
                results.append(
                    LazySpectrogram(
                        raw[ii], time_, freq_, polar, dateobs, unit, dt_base, fn
                    )
                )
            print(
                f"  Dual polarization  Full size: {raw.shape}  "
                f"({raw.nbytes/1e9:.2f} GB, not loaded into memory)"
            )

        return results, hdu

    except Exception:
        hdu.close()
        raise


__all__ = ["LazySpectrogram", "_find_range", "read_cso_fits"]
