"""Shared minimal CSO spectrogram FITS readers.

The helpers here avoid plotting, downloading, GUI work, and data processing.
They only normalize the common FITS-reading behavior duplicated across CSO
scripts.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CSOSpectrogram",
    "cso_base_datetime",
    "cso_unit_from_header",
    "normalize_cso_polarization",
    "read_cso_spectrogram",
    "read_cso_spectrogram_hdul",
    "readcso_spectrofits",
]


@dataclass
class CSOSpectrogram:
    """Container matching the legacy CSO spectrogram attributes."""

    data: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    polar: str
    dateobs: str
    unit: str | None = None
    obsdev: str | None = None
    dt_base: dt.datetime | None = None


def _dateobs_from_header(header: Any) -> str:
    dateobs = header.get("DATE-OBS") or header.get("DATE_OBS")
    if not dateobs:
        raise KeyError("Missing CSO FITS DATE-OBS/DATE_OBS keyword")
    return str(dateobs)


def cso_base_datetime(dateobs: str, time_values: np.ndarray) -> dt.datetime:
    """Return the legacy CSO base date, adjusted when time starts negative."""

    base = dt.datetime.fromisoformat(str(dateobs)[:10])
    if len(time_values) and float(time_values[0]) < 0:
        base = base + dt.timedelta(days=1)
    return base


def normalize_cso_polarization(polars: str, naxis: int) -> str:
    """Normalize CSO polarization labels using the legacy reader rule."""

    polars = str(polars)
    if int(naxis) == 3 and polars == "RCP and LCP":
        return "RL"
    return polars


def cso_unit_from_header(header: Any, default: str | None = None) -> str | None:
    """Return the common CSO unit keyword."""

    return header.get("BUNIT") or header.get("QUANTITY") or default


def read_cso_spectrogram_hdul(hdul) -> list[CSOSpectrogram]:
    """Read CSO spectrogram objects from an already-open HDU-like object."""

    header = hdul[0].header
    data = np.asarray(hdul[0].data)
    time_values = np.ravel(hdul[1].data["time"])
    freq_values = np.ravel(hdul[1].data["frequency"])

    dateobs = _dateobs_from_header(header)
    dt_base = cso_base_datetime(dateobs, time_values)
    if len(time_values) and float(time_values[0]) < 0:
        dateobs = dt_base.isoformat()

    polars = normalize_cso_polarization(header["POLARIZA"], header["NAXIS"])
    unit = cso_unit_from_header(header)

    if data.ndim == 2:
        return [
            CSOSpectrogram(
                data=data,
                time=time_values,
                freq=freq_values,
                polar=polars,
                dateobs=dateobs,
                unit=unit,
                dt_base=dt_base,
            )
        ]

    if data.ndim == 3:
        spectra = []
        for idx in range(data.shape[0]):
            polar = polars[idx] * 2
            spectra.append(
                CSOSpectrogram(
                    data=data[idx, :, :],
                    time=time_values,
                    freq=freq_values,
                    polar=polar,
                    dateobs=dateobs,
                    unit=unit,
                    dt_base=dt_base,
                )
            )
        return spectra

    raise ValueError("CSO spectrogram data must be 2D or 3D")


def read_cso_spectrogram(path: str | Path, *, cache_root=None) -> list[CSOSpectrogram]:
    """Read an entire CSO cube and its coordinates into detached memory.

    This legacy API returns the full cube. Use bounded extraction APIs for
    large compressed observations. An absent path may resolve to path+'.gz'.
    """

    from astropy.io import fits

    from solar_toolkit.io import materialize_fits

    with materialize_fits(path, cache_root=cache_root) as readable:
        # Image scaling can detach the cube while the table coordinates still
        # retain a file mapping. The path-based reader owns both lifetimes.
        with fits.open(readable, memmap=False) as hdul:
            return read_cso_spectrogram_hdul(hdul)


# Legacy spelling used by existing scripts.
readcso_spectrofits = read_cso_spectrogram
