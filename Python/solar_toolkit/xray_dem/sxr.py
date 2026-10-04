"""Soft X-ray time-series loading.

English: Load GOES/SXR tables or NetCDF products and optionally crop the
requested time interval.

中文：加载 GOES/SXR 表格或 NetCDF 产品，并可按指定时间区间裁剪。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from solar_toolkit.timeseries import crop_time_range, normalize_time_column
from solar_toolkit.timeseries.tables import _normalize_time_bounds


def load_sxr_data(
    file_path: str | Path,
    start_time=None,
    end_time=None,
    *,
    time_column: str = "time",
) -> Any:
    """Load a GOES/SXR table or NetCDF product and optionally crop by time.

    CSV and text files return a normalized :class:`pandas.DataFrame`. NetCDF
    files return an in-memory :class:`xarray.Dataset`; loading into memory
    ensures the source file can be closed before this function returns.
    Bounds are optional and inclusive: naive timestamps mean UTC, aware ones
    are converted to UTC, and a missing bound leaves that side open.
    """

    path = Path(file_path)
    if path.suffix.casefold() in {".cdf", ".nc", ".nc4", ".netcdf"}:
        return load_goes_sxr_dataset(path, start_time, end_time)
    if path.suffix.casefold() in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_table(path)
    normalized = normalize_time_column(frame, source_column=time_column)
    return crop_time_range(normalized, start_time, end_time)


def load_goes_sxr_dataset(
    file_path: str | Path,
    start_time=None,
    end_time=None,
    *,
    require_data: bool = False,
):
    """Load and detach a NetCDF dataset with optional inclusive UTC bounds.

    Selection preserves input order and duplicate times. Invalid or reversed
    bounds raise ``ValueError``; naive bounds are interpreted as UTC.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"SXR data file does not exist: {path}")
    start, end = _normalize_time_bounds(start_time, end_time)

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading GOES NetCDF data requires the optional 'xarray' dependency"
        ) from exc

    try:
        with xr.open_dataset(path) as source:
            dataset = xr.decode_cf(source)
            if start is not None or end is not None:
                times = pd.to_datetime(dataset.time.values, utc=True).tz_convert(None)
                mask = times.notna()
                if start is not None:
                    mask &= times >= start
                if end is not None:
                    mask &= times <= end
                dataset = dataset.isel(time=mask)
            dataset = dataset.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to read SXR data from {path}: {exc}") from exc

    if require_data and dataset.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No SXR samples found between {start_time!r} and {end_time!r}"
        )
    return dataset


__all__ = ["load_goes_sxr_dataset", "load_sxr_data"]
