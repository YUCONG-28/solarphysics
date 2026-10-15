"""Select and fit radio frames through existing radio computation APIs."""

from __future__ import annotations

import math
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
from solar_toolkit.radio.time_matching import (
    index_radio_frames,
    materialize_radio_frame,
    pair_radio_frames,
)
from solar_toolkit.time import TimeFrame, TimeStamp, nearest_by_time, normalize_time

from ..models import ROI_CURVE_COLUMNS, CompositeRequest

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
            f"ROI lightcurve extractor is missing required columns: {missing}"
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
    time_stamp: TimeStamp | None = None
    logical_frames: tuple[TimeFrame, ...] = ()


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
    time_start=None,
    time_end=None,
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
            time_start=time_start,
            time_end=time_end,
        )
    else:
        candidates = _path_references(
            paths,
            frequency_mhz=request.radio_frequency,
            polarization=requested_pol,
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
    timed = sorted(
        candidates,
        key=lambda item: (item.time_stamp or normalize_time(item.obs_time)).tai_ns,
    )
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
    target = normalize_time(target_time_utc)

    def key(item):
        return getattr(item, "time_stamp", None) or normalize_time(item.obs_time)

    selected = nearest_by_time(target, timed, key=key, max_diff_seconds=max_dt_seconds)
    if selected is None:
        raise RuntimeError(
            f"No radio frame lies within {float(max_dt_seconds):.9f} s of the requested time"
        )
    delta = abs(key(selected).tai_ns - target.tai_ns) / 1e9
    return _materialize_radio_candidate(selected), len(timed), float(delta)


def _path_references(
    paths: Any,
    *,
    frequency_mhz: float,
    polarization: str,
    pair_time_tolerance_sec: float,
) -> tuple[RadioFrameReference, ...]:
    indexed = index_radio_frames(
        paths, frequency_mhz=frequency_mhz, polarization=polarization, strict=False
    )
    return tuple(
        _reference_from_frames((frame,), polarization, pair_time_tolerance_sec)
        for frame in indexed
    )


def _reference_from_frames(frames, polarization, tolerance, stamp=None):
    stamp = stamp or frames[0].time
    return RadioFrameReference(
        paths=tuple(Path(frame.payload["path"]) for frame in frames),
        obs_time=datetime.fromisoformat(stamp.utc.replace("Z", "+00:00")).replace(
            tzinfo=None
        ),
        freq_mhz=float(frames[0].payload["frequency_mhz"]),
        pol=polarization,
        pair_time_tolerance_sec=float(tolerance),
        time_stamp=stamp,
        logical_frames=tuple(frames),
    )


def _paired_references(
    left: Sequence[RadioFrameReference],
    right: Sequence[RadioFrameReference],
    *,
    tolerance_sec: float,
    time_start=None,
    time_end=None,
) -> tuple[RadioFrameReference, ...]:
    indexed_left = [frame for reference in left for frame in reference.logical_frames]
    indexed_right = [frame for reference in right for frame in reference.logical_frames]
    pairs = pair_radio_frames(
        indexed_left,
        indexed_right,
        tolerance_seconds=tolerance_sec,
        time_start=time_start,
        time_end=time_end,
    )
    lookup = {id(frame.payload): frame for frame in (*indexed_left, *indexed_right)}
    paired = []
    for pair in pairs:
        members = tuple(lookup[id(payload)] for payload in pair.payload["pair_members"])
        paired.append(
            _reference_from_frames(members, POL_SUM, tolerance_sec, pair.time)
        )
    return tuple(paired)


def _materialize_radio_candidate(
    candidate: RadioImage | RadioFrameReference,
) -> RadioImage:
    if isinstance(candidate, RadioImage):
        return candidate
    if not isinstance(candidate, RadioFrameReference):
        return candidate
    if candidate.logical_frames:
        members = candidate.logical_frames
        payload = dict(members[0].payload)
        if len(members) > 1:
            payload.update(
                paths=tuple(str(path) for path in candidate.paths),
                pair_members=tuple(frame.payload for frame in members),
                polarization=POL_SUM,
            )
        stamp = candidate.time_stamp or normalize_time(candidate.obs_time)
        exact = TimeFrame("+".join(frame.frame_id for frame in members), stamp, payload)
        data, header = materialize_radio_frame(exact)
        return RadioImage(
            candidate.paths[0],
            int(payload["hdu_index"]),
            data,
            header,
            candidate.pol,
            candidate.freq_mhz,
            candidate.obs_time,
        )
    images = []
    for index, path in enumerate(candidate.paths):
        exact = candidate.logical_frames[index] if candidate.logical_frames else None
        for image in iter_radio_images(path, default_pol=candidate.pol):
            if exact is not None and image.hdu_index != exact.payload["hdu_index"]:
                continue
            if (
                exact is not None
                and normalize_time(image.obs_time).tai_ns != exact.time.tai_ns
            ):
                # RadioImage's legacy datetime cannot carry sub-microsecond digits.
                if (
                    abs(normalize_time(image.obs_time).tai_ns - exact.time.tai_ns)
                    >= 1000
                ):
                    continue
            images.append(image)
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
