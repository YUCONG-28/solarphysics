"""Select AIA backgrounds through the existing :mod:`solar_toolkit.aia` APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import pandas as pd

from solar_toolkit.aia.background import (
    AiaBackground,
    find_nearest_aia,
    read_aia_background,
    scan_aia_folder,
)

from ..models import CompositeRequest

__all__ = ["AiaSelection", "load_aia_selection", "scan_aia_catalog"]


@dataclass(frozen=True, slots=True)
class AiaSelection:
    """One wavelength-filtered AIA background matched to the requested UTC."""

    background: AiaBackground
    requested_time_utc: datetime
    matched_time_utc: datetime
    delta_seconds: float
    candidate_count: int

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe selection metadata."""

        return {
            "path": str(Path(self.background.path).resolve(strict=False)),
            "wavelength": str(self.background.wavelength),
            "requested_time_utc": self.requested_time_utc.isoformat(),
            "matched_time_utc": self.matched_time_utc.isoformat(),
            "delta_seconds": float(self.delta_seconds),
            "candidate_count": int(self.candidate_count),
            "shape": [int(value) for value in self.background.z.shape],
        }


def load_aia_selection(
    request: CompositeRequest,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
    max_dt_seconds: float = 60.0,
    max_pixels: int = 1024,
    percentile_limits: tuple[float, float] = (1.0, 99.7),
    log_scale: bool = True,
    wcs_mode: str = "header",
    catalog: pd.DataFrame | None = None,
    background_cache: MutableMapping[str, AiaBackground] | None = None,
) -> AiaSelection:
    """Load the nearest AIA background in the request's selected wavelength."""

    table = (
        scan_aia_catalog(
            request,
            pattern=pattern,
            recursive=recursive,
        )
        if catalog is None
        else catalog
    )
    filtered = _filter_wavelength(table, request.aia_wave)
    if filtered.empty:
        raise FileNotFoundError(
            f"No AIA {request.aia_wave} Å FITS files found in "
            f"{request.aia_directory}"
        )

    requested_naive_utc = request.aia_time.astimezone(UTC).replace(tzinfo=None)
    nearest = find_nearest_aia(
        filtered,
        requested_naive_utc,
        max_dt_seconds=max_dt_seconds,
    )
    if nearest.status != "matched" or nearest.path is None:
        detail = nearest.reason or nearest.status
        raise RuntimeError(
            f"Could not match AIA {request.aia_wave} Å at "
            f"{request.aia_time.isoformat()}: {detail}"
        )
    cache_key = str(Path(nearest.path).resolve(strict=False))
    background = (
        background_cache.get(cache_key) if background_cache is not None else None
    )
    if background is None:
        background = read_aia_background(
            nearest.path,
            max_pixels=max_pixels,
            percentile_limits=percentile_limits,
            log_scale=log_scale,
            wcs_mode=wcs_mode,
        )
        if background_cache is not None:
            background_cache[cache_key] = background
    matched = _timestamp_utc(nearest.obs_time, label="matched AIA time")
    return AiaSelection(
        background=background,
        requested_time_utc=request.aia_time,
        matched_time_utc=matched,
        delta_seconds=float(nearest.delta_seconds or 0.0),
        candidate_count=int(len(filtered)),
    )


def scan_aia_catalog(
    request: CompositeRequest,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
) -> pd.DataFrame:
    """Scan the request AIA directory once for repeated time matching."""

    return scan_aia_folder(
        request.aia_directory,
        pattern=pattern,
        recursive=recursive,
    )


def _filter_wavelength(table: pd.DataFrame, wavelength: int) -> pd.DataFrame:
    if table is None or table.empty or "wavelength" not in table:
        return pd.DataFrame(columns=["path", "obs_time", "wavelength"])
    numeric = pd.to_numeric(table["wavelength"], errors="coerce")
    selected = np.isclose(
        numeric.to_numpy(dtype=float),
        float(wavelength),
        rtol=0.0,
        atol=1e-6,
        equal_nan=False,
    )
    return table.loc[selected].reset_index(drop=True)


def _timestamp_utc(value: Any, *, label: str) -> datetime:
    if value is None or pd.isna(value):
        raise ValueError(f"{label} is missing")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.to_pydatetime()
