"""ASO-S/HXI light-curve loading.

English: Read the four historical HXI quick-look energy channels and crop them
to an inclusive UTC interval.

中文：读取历史 HXI 快视产品的四个能段，并按 UTC 闭区间裁剪。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from solar_toolkit.time import parse_time

DEFAULT_HXI_EPOCH = datetime(2018, 12, 31, 16, 0, 0)
HXI_ENERGY_CHANNELS = (
    "10-20 keV",
    "20-50 keV",
    "50-100 keV",
    "100-300 keV",
)


def load_hxi_lightcurve(
    file_path: str | Path,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    *,
    epoch: datetime = DEFAULT_HXI_EPOCH,
) -> dict:
    """Load the historical four HXI channels within inclusive UTC bounds.

    TIME is interpreted as finite seconds relative to the caller's ``epoch``;
    the historical default epoch and fixed energy labels are retained. Naive
    datetimes mean UTC; aware epoch/bounds are converted to naive UTC. Reversed
    bounds, non-1D/nonfinite TIME, or CTS_THINTHICK other than (N, 4) raise
    ``ValueError``. Input order and count values (including missing counts) are
    preserved, with independent output arrays. This reader does not infer an
    epoch, TIMEUNIT, channel calibration or clock correction from FITS metadata.
    """

    epoch = parse_time(epoch)
    start = None if start_time is None else parse_time(start_time)
    end = None if end_time is None else parse_time(end_time)
    if start is not None and end is not None and start > end:
        raise ValueError("start_time must not be after end_time")

    from astropy.io import fits

    with fits.open(file_path) as hdul:
        offsets = np.asarray(hdul[1].data.TIME, dtype=float)
        counts = np.asarray(hdul[3].data["CTS_THINTHICK"])

    if offsets.ndim != 1 or not np.isfinite(offsets).all():
        raise ValueError("HXI TIME must be a finite one-dimensional seconds array")
    if counts.ndim != 2 or counts.shape != (len(offsets), len(HXI_ENERGY_CHANNELS)):
        raise ValueError("HXI CTS_THINTHICK must have shape (len(TIME), 4)")
    try:
        times = [epoch + timedelta(seconds=float(offset)) for offset in offsets]
    except OverflowError as exc:
        raise ValueError(
            "HXI TIME offsets exceed the supported datetime range"
        ) from exc
    mask = np.ones(len(times), dtype=bool)
    if start is not None:
        mask &= np.asarray([time >= start for time in times], dtype=bool)
    if end is not None:
        mask &= np.asarray([time <= end for time in times], dtype=bool)

    filtered_times = [time for time, keep in zip(times, mask, strict=True) if keep]
    data = {
        label: np.asarray(counts[:, index])[mask]
        for index, label in enumerate(HXI_ENERGY_CHANNELS)
    }
    return {"times": filtered_times, "data": data}


__all__ = [
    "DEFAULT_HXI_EPOCH",
    "HXI_ENERGY_CHANNELS",
    "load_hxi_lightcurve",
]
