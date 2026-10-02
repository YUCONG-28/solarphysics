"""Bounded-memory, finite-sample CSO total-intensity preview binning."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .cso import cso_base_datetime

__all__ = ["read_cso_total_window"]


def read_cso_total_window(paths, bounds, frequency_bounds, dt=0.1, df=0.25):
    """Average native R+L samples into bins; empty bins remain NaN, no shifts."""
    te = np.arange(bounds[0], bounds[1] + dt, dt)
    fe = np.arange(frequency_bounds[0], frequency_bounds[1] + df, df)
    sums = np.zeros((len(fe) - 1, len(te) - 1))
    counts = np.zeros_like(sums)
    records = []
    previous_end = None
    for path in sorted(map(Path, paths)):
        with fits.open(path, memmap=True) as hd:
            h = hd[0].header
            if h.get("TIMESYS") != "UTC" or h.get("POLARIZA") != "RCP and LCP":
                raise ValueError("CSO requires explicit UTC and RCP/LCP channels")
            unit = str(h.get("BUNIT", ""))
            if "SFU" not in unit:
                raise ValueError("CSO requires linear SFU source units")
            t = np.asarray(hd[1].data["time"]).ravel().astype(float)
            f = np.asarray(hd[1].data["frequency"]).ravel().astype(float)
            t += Time(cso_base_datetime(h["DATE-OBS"], t)).unix
            if (
                hd[0].shape != (2, len(f), len(t))
                or not np.all(np.diff(t) > 0)
                or not np.all(np.diff(f) > 0)
            ):
                raise ValueError("Invalid CSO axes or shape")
            if previous_end is not None and t[0] <= previous_end:
                raise ValueError("Overlapping CSO files require explicit deduplication")
            previous_end = t[-1]
            ti = np.searchsorted(t, te)
            fi = np.searchsorted(f, fe)
            if ti[-1] == ti[0]:
                continue
            for k, (a, b) in enumerate(zip(fi[:-1], fi[1:], strict=True)):
                if b == a:
                    continue
                raw = np.asarray(hd[0].data[:, a:b, ti[0] : ti[-1]], dtype=float)
                total = raw[0] + raw[1]
                ok = np.isfinite(total)
                ss = np.r_[0.0, np.cumsum(np.where(ok, total, 0).sum(axis=0))]
                nn = np.r_[0.0, np.cumsum(ok.sum(axis=0))]
                ix = ti - ti[0]
                sums[k] += np.diff(ss[ix])
                counts[k] += np.diff(nn[ix])
            records.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "first_utc": Time(t[0], format="unix").isot,
                    "last_utc": Time(t[-1], format="unix").isot,
                    "unit": unit,
                }
            )
    result = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=result, where=counts > 0)
    return (
        (te[:-1] + te[1:]) / 2,
        (fe[:-1] + fe[1:]) / 2,
        result,
        {
            "inputs": records,
            "time_bin_seconds": dt,
            "frequency_bin_mhz": df,
            "empty_bins": int((counts == 0).sum()),
            "processing": "finite native RCP+LCP mean; no background subtraction, interpolation or time shift",
            "unit": "SFU (source header)",
        },
    )
