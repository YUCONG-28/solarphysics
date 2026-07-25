"""Select and fit radio frames through existing radio computation APIs."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from solar_toolkit.map.coordinates import calculate_fits_extent_from_header
from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
    RadioImage,
    filter_radio_images,
    iter_radio_images,
    maybe_make_sum_images,
    parse_time_from_filename,
    select_radio_files,
    to_arcsec,
)
from solar_toolkit.radio.gaussian import (
    GaussianFitResult,
    fit_elliptical_gaussian_on_radio_image,
)
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    extract_radio_roi_lightcurve,
)

from ..models import CompositeRequest, ROI_CURVE_COLUMNS

__all__ = [
    "DEFAULT_ROI_FREQUENCIES_MHZ",
    "RadioGaussianSelection",
    "RadioFrameReference",
    "extract_multi_frequency_roi_curve",
    "fit_radio_gaussian_frame",
    "fit_radio_gaussian_selection",
    "load_radio_candidates",
    "select_radio_frame_from_candidates",
    "select_radio_frame",
]

DEFAULT_ROI_FREQUENCIES_MHZ = (149.0, 164.0, 190.0, 223.0, 238.0)

_REQUEST_TO_TOOLKIT_POLARIZATION = {
    "RR": POL_RCP,
    "LL": POL_LCP,
    "RR+LL": POL_SUM,
}


def extract_multi_frequency_roi_curve(
    request: CompositeRequest,
    roi: RadioRoi,
    *,
    frequencies_mhz: Sequence[float] = DEFAULT_ROI_FREQUENCIES_MHZ,
    pattern: str = "*.fits",
    recursive: bool = True,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
    pair_time_tolerance_sec: float = 0.5,
) -> pd.DataFrame:
    """Extract and standardize a multi-frequency ROI lightcurve.

    Scientific measurements and quality flags come directly from
    :func:`extract_radio_roi_lightcurve`; this adapter only adds the frontend's
    stable ``time`` and ``frequency`` column names and orders the result.
    """

    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    if not isinstance(roi, RadioRoi):
        raise TypeError("roi must be a RadioRoi")
    frequencies = _normalized_frequencies(frequencies_mhz)
    extracted = extract_radio_roi_lightcurve(
        request.radio_directory,
        roi,
        pattern=pattern,
        recursive=recursive,
        freqs=frequencies,
        polarization=request.polarization,
        time_start=time_start,
        time_end=time_end,
        pair_time_tolerance_sec=pair_time_tolerance_sec,
    )
    required_source_columns = {
        "obs_time",
        "freq_mhz",
        "raw_sum",
        "raw_mean",
        "raw_peak",
        "quality_flag",
    }
    missing = sorted(required_source_columns.difference(extracted.columns))
    if missing:
        raise ValueError(
            "ROI lightcurve extractor is missing required columns: " f"{missing}"
        )

    curve = extracted.copy(deep=True)
    curve["time"] = curve["obs_time"]
    curve["frequency"] = curve["freq_mhz"]
    provenance_columns = [
        name for name in curve.columns if name not in ROI_CURVE_COLUMNS
    ]
    curve = curve[[*ROI_CURVE_COLUMNS, *provenance_columns]]
    curve.attrs = {
        **dict(extracted.attrs),
        "frequencies_mhz": list(frequencies),
        "polarization": request.polarization,
        "roi": roi.to_json_dict(),
        "scientific_extractor": (
            "solar_toolkit.radio.roi_lightcurve.extract_radio_roi_lightcurve"
        ),
    }
    return curve


@dataclass(frozen=True, slots=True)
class RadioGaussianSelection:
    """One time-matched radio frame and its existing Gaussian fit result."""

    frame: RadioImage
    fit_result: GaussianFitResult | None
    extent_arcsec: tuple[float, float, float, float]
    image_origin: str
    gaussian_config: Mapping[str, Any]
    requested_time_utc: datetime
    matched_time_utc: datetime
    delta_seconds: float
    candidate_count: int
    failure_diagnostics: Mapping[str, Any]

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe frame and current Gaussian-fit metadata."""

        fit = self.fit_result
        return {
            "path": str(self.frame.path.resolve(strict=False)),
            "hdu_index": int(self.frame.hdu_index),
            "frequency_mhz": float(self.frame.freq_mhz),
            "polarization": str(self.frame.pol),
            "requested_time_utc": self.requested_time_utc.isoformat(),
            "matched_time_utc": self.matched_time_utc.isoformat(),
            "delta_seconds": float(self.delta_seconds),
            "candidate_count": int(self.candidate_count),
            "extent_arcsec": [float(value) for value in self.extent_arcsec],
            "image_origin": self.image_origin,
            "fit": _fit_metadata(fit, self.failure_diagnostics),
        }


@dataclass(frozen=True, slots=True)
class RadioFrameReference:
    """Lightweight time index entry whose image plane is loaded on demand."""

    paths: tuple[Path, ...]
    obs_time: datetime
    freq_mhz: float
    pol: str
    pair_time_tolerance_sec: float = 0.5


def select_radio_frame(
    request: CompositeRequest,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
    max_dt_seconds: float = 60.0,
    pair_time_tolerance_sec: float = 0.5,
) -> tuple[RadioImage, int, float]:
    """Select the nearest matching radio image using existing FITS readers."""

    candidates = load_radio_candidates(
        request,
        pattern=pattern,
        recursive=recursive,
        pair_time_tolerance_sec=pair_time_tolerance_sec,
    )
    return select_radio_frame_from_candidates(
        candidates,
        request.aia_time,
        max_dt_seconds=max_dt_seconds,
    )


def load_radio_candidates(
    request: CompositeRequest,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
    pair_time_tolerance_sec: float = 0.5,
) -> tuple[RadioFrameReference, ...]:
    """Index timed candidates without loading every radio image plane."""

    requested_pol = _REQUEST_TO_TOOLKIT_POLARIZATION[request.polarization]
    path_polarizations = (
        ("RR", "LL") if requested_pol == POL_SUM else (request.polarization,)
    )
    paths = select_radio_files(
        request.radio_directory,
        pattern=pattern,
        recursive=recursive,
        freqs=[request.radio_frequency],
        polarizations=path_polarizations,
    )
    if requested_pol == POL_SUM:
        left = _path_references(
            (
                path
                for path in paths
                if any("LL" in part.upper() for part in path.parts)
            ),
            frequency_mhz=request.radio_frequency,
            polarization=POL_LCP,
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
        right = _path_references(
            (
                path
                for path in paths
                if any("RR" in part.upper() for part in path.parts)
            ),
            frequency_mhz=request.radio_frequency,
            polarization=POL_RCP,
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
        candidates = _paired_references(
            left,
            right,
            tolerance_sec=pair_time_tolerance_sec,
        )
    else:
        candidates = _path_references(
            paths,
            frequency_mhz=request.radio_frequency,
            polarization=requested_pol,
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
    timed = sorted(candidates, key=lambda item: item.obs_time)
    if not timed:
        raise FileNotFoundError(
            f"No timed {request.radio_frequency:g} MHz {request.polarization} "
            f"radio frame found in {request.radio_directory}"
        )

    return tuple(timed)


def select_radio_frame_from_candidates(
    candidates: Sequence[RadioImage | RadioFrameReference],
    target_time_utc: datetime,
    *,
    max_dt_seconds: float,
) -> tuple[RadioImage, int, float]:
    """Select the nearest preloaded radio candidate without rescanning files."""

    timed = tuple(item for item in candidates if item.obs_time is not None)
    if not timed:
        raise FileNotFoundError("No timed radio frame candidates are available")
    target = target_time_utc.astimezone(UTC).replace(tzinfo=None)
    selected = min(
        timed,
        key=lambda item: abs((item.obs_time - target).total_seconds()),
    )
    delta = abs((selected.obs_time - target).total_seconds())
    if delta > float(max_dt_seconds):
        raise RuntimeError(
            f"Nearest radio frame is {delta:.3f} s from the requested time, "
            f"exceeding {float(max_dt_seconds):.3f} s"
        )
    return _materialize_radio_candidate(selected), len(timed), float(delta)


def _path_references(
    paths: Any,
    *,
    frequency_mhz: float,
    polarization: str,
    pair_time_tolerance_sec: float,
) -> tuple[RadioFrameReference, ...]:
    references = []
    for raw_path in paths:
        path = Path(raw_path)
        obs_time = parse_time_from_filename(path)
        if obs_time is None:
            loaded = tuple(iter_radio_images(path, default_pol=polarization))
            obs_times = [item.obs_time for item in loaded if item.obs_time is not None]
            if not obs_times:
                continue
            obs_time = min(obs_times)
        references.append(
            RadioFrameReference(
                paths=(path,),
                obs_time=obs_time,
                freq_mhz=float(frequency_mhz),
                pol=polarization,
                pair_time_tolerance_sec=float(pair_time_tolerance_sec),
            )
        )
    return tuple(sorted(references, key=lambda item: item.obs_time))


def _paired_references(
    left: Sequence[RadioFrameReference],
    right: Sequence[RadioFrameReference],
    *,
    tolerance_sec: float,
) -> tuple[RadioFrameReference, ...]:
    right_times = [item.obs_time for item in right]
    used_right: set[int] = set()
    paired = []
    for left_item in left:
        position = bisect_left(right_times, left_item.obs_time)
        nearby = range(max(0, position - 2), min(len(right), position + 3))
        eligible = [
            (
                abs((right[index].obs_time - left_item.obs_time).total_seconds()),
                index,
            )
            for index in nearby
            if index not in used_right
        ]
        if not eligible:
            continue
        delta, index = min(eligible)
        if delta > float(tolerance_sec):
            continue
        used_right.add(index)
        right_item = right[index]
        midpoint = left_item.obs_time + (right_item.obs_time - left_item.obs_time) / 2
        paired.append(
            RadioFrameReference(
                paths=(left_item.paths[0], right_item.paths[0]),
                obs_time=midpoint,
                freq_mhz=left_item.freq_mhz,
                pol=POL_SUM,
                pair_time_tolerance_sec=float(tolerance_sec),
            )
        )
    return tuple(paired)


def _materialize_radio_candidate(
    candidate: RadioImage | RadioFrameReference,
) -> RadioImage:
    if isinstance(candidate, RadioImage):
        return candidate
    if not isinstance(candidate, RadioFrameReference):
        return candidate
    images = [
        image
        for path in candidate.paths
        for image in iter_radio_images(path, default_pol=candidate.pol)
    ]
    filtered = filter_radio_images(
        images,
        freqs=[candidate.freq_mhz],
        polarizations=(
            [POL_LCP, POL_RCP] if candidate.pol == POL_SUM else [candidate.pol]
        ),
    )
    if candidate.pol == POL_SUM:
        materialized = maybe_make_sum_images(
            filtered,
            tolerance_sec=candidate.pair_time_tolerance_sec,
        )
    else:
        materialized = filtered
    timed = [item for item in materialized if item.obs_time is not None]
    if not timed:
        raise RuntimeError(
            f"Indexed radio candidate could not be loaded: "
            f"{', '.join(str(path) for path in candidate.paths)}"
        )
    return min(
        timed,
        key=lambda item: abs((item.obs_time - candidate.obs_time).total_seconds()),
    )


def fit_radio_gaussian_selection(
    request: CompositeRequest,
    *,
    pattern: str = "*.fits",
    recursive: bool = True,
    max_dt_seconds: float = 60.0,
    pair_time_tolerance_sec: float = 0.5,
    gaussian_overrides: Mapping[str, Any] | None = None,
) -> RadioGaussianSelection:
    """Select one radio frame and run the canonical Gaussian fitting API."""

    frame, candidate_count, delta = select_radio_frame(
        request,
        pattern=pattern,
        recursive=recursive,
        max_dt_seconds=max_dt_seconds,
        pair_time_tolerance_sec=pair_time_tolerance_sec,
    )
    return fit_radio_gaussian_frame(
        frame,
        requested_time_utc=request.aia_time,
        candidate_count=candidate_count,
        delta_seconds=delta,
        gaussian_overrides=gaussian_overrides,
    )


def fit_radio_gaussian_frame(
    frame: RadioImage,
    *,
    requested_time_utc: datetime,
    candidate_count: int,
    delta_seconds: float,
    gaussian_overrides: Mapping[str, Any] | None = None,
) -> RadioGaussianSelection:
    """Fit one already-selected radio frame through the canonical fitter."""

    if frame.obs_time is None:
        raise ValueError("radio frame must contain an observation time")
    extent = _radio_extent_arcsec(frame)
    image_origin = "lower"
    config = _gaussian_config(
        frame.freq_mhz,
        overrides=gaussian_overrides,
    )
    fit_result = fit_elliptical_gaussian_on_radio_image(
        frame.image,
        extent,
        config,
        source_file=str(frame.path),
        fit_input_type="raw",
        image_origin=image_origin,
    )
    failure = dict(config.get("_last_gaussian_failure_diag", {}) or {})
    matched = frame.obs_time.replace(tzinfo=UTC)
    return RadioGaussianSelection(
        frame=frame,
        fit_result=fit_result,
        extent_arcsec=extent,
        image_origin=image_origin,
        gaussian_config=config,
        requested_time_utc=requested_time_utc.astimezone(UTC),
        matched_time_utc=matched,
        delta_seconds=float(delta_seconds),
        candidate_count=candidate_count,
        failure_diagnostics=failure,
    )


def _gaussian_config(
    frequency_mhz: float,
    *,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from solar_apps.workflows.radio import source_map_workflow

    config = dict(source_map_workflow.DEFAULT_CONFIG)
    config.update(
        {
            "enable_gaussian_overlay": True,
            "draw_gaussian_contours": True,
            "draw_gaussian_center": True,
            "draw_gaussian_fwhm_ellipse": True,
            "gaussian_overlay_display_mode": "contours_and_fwhm",
        }
    )
    config = source_map_workflow.config_for_gaussian_band(config, frequency_mhz)
    if overrides:
        config.update(dict(overrides))
    return config


def _radio_extent_arcsec(frame: RadioImage) -> tuple[float, float, float, float]:
    raw_extent = calculate_fits_extent_from_header(
        frame.header,
        frame.image.shape,
        preserve_orientation=True,
    )
    unit_x = str(frame.header.get("CUNIT1", "arcsec"))
    unit_y = str(frame.header.get("CUNIT2", "arcsec"))
    return (
        to_arcsec(raw_extent[0], unit_x),
        to_arcsec(raw_extent[1], unit_x),
        to_arcsec(raw_extent[2], unit_y),
        to_arcsec(raw_extent[3], unit_y),
    )


def _fit_metadata(
    fit: GaussianFitResult | None,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    if fit is None:
        reason = failure.get("quality_flag") or failure.get("reason") or "fit_failed"
        return {
            "available": False,
            "quality_flag": str(reason),
            "quality_flag_detail": str(failure.get("reason") or reason),
            "overlay_valid": False,
        }
    return {
        "available": True,
        "quality_flag": str(fit.quality_flag),
        "quality_flag_detail": str(getattr(fit, "quality_flag_detail", "")),
        "overlay_valid": bool(getattr(fit, "overlay_valid", False)),
        "center_arcsec": [float(value) for value in fit.center_arcsec],
        "center_pixel": [float(value) for value in fit.center_pixel],
        "fwhm_major_arcsec": _finite_or_none(getattr(fit, "fwhm_major_arcsec", None)),
        "fwhm_minor_arcsec": _finite_or_none(getattr(fit, "fwhm_minor_arcsec", None)),
        "snr": _finite_or_none(fit.snr),
        "residual_rms": _finite_or_none(fit.residual_rms),
        "mask_pixel_count": int(fit.mask_pixel_count),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _normalized_frequencies(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("frequencies_mhz must be a sequence of numbers")
    frequencies: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("frequencies_mhz must contain only numbers") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(
                "frequencies_mhz must contain finite values greater than zero"
            )
        if number not in frequencies:
            frequencies.append(number)
    if not frequencies:
        raise ValueError("frequencies_mhz must not be empty")
    return frequencies
