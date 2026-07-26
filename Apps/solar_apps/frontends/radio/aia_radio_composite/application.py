"""Application orchestration for AIA radio composite products."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from solar_apps.workflows.visualization.video_cli import write_video_from_paths
from solar_toolkit.radio.roi_lightcurve import RadioRoi

from . import FRONTEND_ID
from .adapters import (
    DEFAULT_ROI_FREQUENCIES_MHZ,
    extract_spectrum_flux_curve,
    extract_spectrum_flux_curves,
    extract_multi_frequency_roi_curve,
    fit_radio_gaussian_frame,
    load_aia_selection,
    load_radio_candidates,
    load_spectrum_window,
    scan_aia_catalog,
    select_radio_frame_from_candidates,
)
from .models import (
    CompositeRequest,
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumTimeAlignment,
    SpectrumWindow,
    build_spectrum_time_alignment,
)
from .rendering import (
    TopPanelArtifact,
    TriplePanelArtifact,
    render_composite_result,
    render_top_panel,
)

__all__ = [
    "FRONTEND_ID",
    "CompositeVideoArtifact",
    "build_composite",
    "build_composite_result",
    "build_composite_video",
    "build_dynamic_composite_video",
    "build_roi_lightcurve",
    "build_spectrum_flux_curve",
    "build_spectrum_flux_curves",
    "build_spectrum_frequency_time_alignments",
    "build_spectrum_window",
    "build_top_panel",
    "match_reference_radio_time",
]


@dataclass(frozen=True, slots=True)
class CompositeVideoArtifact:
    """Encoded synchronized composite video and reproducibility metadata."""

    video_mp4: bytes
    metadata_json: bytes
    metadata: Mapping[str, Any]


class _FrameMatchError(RuntimeError):
    """Identify the selected source that made one dynamic frame incomplete."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source = source


AIA_VIDEO_MATCH_TOLERANCE_SECONDS = 12.0
RADIO_VIDEO_MATCH_TOLERANCE_SECONDS = 0.1


def match_reference_radio_time(
    request: CompositeRequest,
    *,
    radio_frequencies_mhz: Sequence[float] | None = None,
    max_dt_seconds: float = 60.0,
    pair_time_tolerance_sec: float = 0.5,
) -> datetime:
    """Return the nearest primary-radio UTC without rendering the top panel."""

    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    frequencies = _normalized_radio_frequencies(
        request,
        radio_frequencies_mhz,
    )
    primary_request = replace(
        request,
        radio_frequency=float(frequencies[0]),
    )
    candidates = load_radio_candidates(
        primary_request,
        pair_time_tolerance_sec=pair_time_tolerance_sec,
    )
    primary_frame, _, _ = select_radio_frame_from_candidates(
        candidates,
        request.aia_time,
        max_dt_seconds=max_dt_seconds,
    )
    if primary_frame.obs_time is None:
        raise ValueError("Primary radio frame is missing an observation time")
    matched = pd.Timestamp(primary_frame.obs_time)
    if matched.tzinfo is None:
        matched = matched.tz_localize(UTC)
    else:
        matched = matched.tz_convert(UTC)
    return matched.to_pydatetime()


def build_spectrum_frequency_time_alignments(
    request: CompositeRequest,
    spectrum: SpectrumWindow,
    frequencies_mhz: Sequence[float],
) -> dict[float, SpectrumTimeAlignment]:
    """Build one display-only DART alignment per radio imaging frequency."""

    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    if not isinstance(spectrum, SpectrumWindow):
        raise TypeError("spectrum must be a SpectrumWindow")
    if spectrum.source.upper() != "DART":
        return {}
    frequencies = tuple(dict.fromkeys(float(value) for value in frequencies_mhz))
    if not frequencies:
        raise ValueError("frequencies_mhz must not be empty")
    alignments = {}
    for frequency in frequencies:
        radio_time = match_reference_radio_time(
            request,
            radio_frequencies_mhz=(frequency,),
        )
        alignment = build_spectrum_time_alignment(spectrum, radio_time)
        if alignment is None:
            raise RuntimeError("DART spectrum did not produce a time alignment")
        alignments[frequency] = alignment
    return alignments


def build_composite_result(
    request: CompositeRequest,
    roi: RadioRoi,
    *,
    frequencies_mhz: Sequence[float] = DEFAULT_ROI_FREQUENCIES_MHZ,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    roi_time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    spectrum_time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
    spectrum_band: SpectrumBand | None = None,
    spectrum_bands: Sequence[SpectrumBand] | None = None,
    radio_frequencies_mhz: Sequence[float] | None = None,
    aia_waves: Sequence[int] | None = None,
    display_extent_arcsec: tuple[float, float, float, float] | None = None,
    extended_canvas_color: str = "black",
    spectrum_frequency_range_mhz: tuple[float, float] | None = None,
    spectrum_intensity_range: tuple[float, float] | None = None,
    flux_plot_layout: str = "combined",
    gaussian_overrides: Mapping[str, Any] | None = None,
    dpi: int = 160,
) -> CompositeResult:
    """Run the three adapters and assemble one validated in-memory result."""

    roi_range = roi_time_range_utc or time_range_utc
    spectrum_range = spectrum_time_range_utc or time_range_utc
    top = build_top_panel(
        request,
        dpi=dpi,
        radio_frequencies_mhz=radio_frequencies_mhz,
        aia_waves=aia_waves,
        display_extent_arcsec=display_extent_arcsec,
        extended_canvas_color=extended_canvas_color,
        gaussian_overrides=gaussian_overrides,
    )
    curve = build_roi_lightcurve(
        request,
        roi,
        frequencies_mhz=frequencies_mhz,
        time_start=roi_range[0] if roi_range else None,
        time_end=roi_range[1] if roi_range else None,
    )
    spectrum = build_spectrum_window(
        request,
        time_range_utc=spectrum_range,
        cso_polarization=cso_polarization,
    )
    normalized_bands = _normalized_spectrum_bands(
        spectrum_band=spectrum_band,
        spectrum_bands=spectrum_bands,
    )
    spectrum_flux_curves = (
        build_spectrum_flux_curves(
            request,
            normalized_bands,
            time_range_utc=roi_range,
            cso_polarization=cso_polarization,
        )
        if normalized_bands
        else ()
    )
    map_time = str(
        top.metadata.get(
            "reference_radio_time_utc",
            request.aia_time.isoformat(),
        )
    )
    spectrum_time_alignment = build_spectrum_time_alignment(
        spectrum,
        map_time,
    )
    spectrum_frequency_time_alignments = (
        build_spectrum_frequency_time_alignments(
            request,
            spectrum,
            tuple(curve.requested_band.center_mhz for curve in spectrum_flux_curves),
        )
        if spectrum_flux_curves
        else {}
    )
    return CompositeResult(
        top_image=top.image_png,
        roi_curve=curve,
        spectrum=spectrum,
        metadata={
            "request": request.to_dict(),
            "roi": roi.to_json_dict(),
            "top_panel": dict(top.metadata),
            "map_time_utc": str(map_time),
            "spectrum_time_alignment": (
                spectrum_time_alignment.to_dict()
                if spectrum_time_alignment is not None
                else None
            ),
            "spectrum_flux_time_alignments": {
                f"{frequency:g}": alignment.to_dict()
                for frequency, alignment in spectrum_frequency_time_alignments.items()
            },
            "frequencies_mhz": [float(value) for value in frequencies_mhz],
            "radio_overlay_frequencies_mhz": list(
                _normalized_radio_frequencies(request, radio_frequencies_mhz)
            ),
            "aia_waves": list(_normalized_aia_waves(request, aia_waves)),
            "roi_time_range_utc": (
                [str(roi_range[0]), str(roi_range[1])] if roi_range else None
            ),
            "spectrum_time_range_utc": (
                [str(spectrum_range[0]), str(spectrum_range[1])]
                if spectrum_range
                else None
            ),
            "spectrum_band": (
                normalized_bands[0].to_dict() if normalized_bands else None
            ),
            "spectrum_bands": [band.to_dict() for band in normalized_bands],
            "spectrum_display_frequency_range_mhz": (
                [float(value) for value in spectrum_frequency_range_mhz]
                if spectrum_frequency_range_mhz is not None
                else None
            ),
            "spectrum_display_intensity_range": (
                [float(value) for value in spectrum_intensity_range]
                if spectrum_intensity_range is not None
                else None
            ),
            "extended_canvas_color": str(extended_canvas_color),
            "flux_plot_layout": _normalized_flux_plot_layout(flux_plot_layout),
        },
        spectrum_flux_curve=(spectrum_flux_curves[0] if spectrum_flux_curves else None),
        spectrum_flux_curves=spectrum_flux_curves,
    )


def build_composite(
    request: CompositeRequest,
    roi: RadioRoi,
    *,
    frequencies_mhz: Sequence[float] = DEFAULT_ROI_FREQUENCIES_MHZ,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    roi_time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    spectrum_time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
    spectrum_band: SpectrumBand | None = None,
    spectrum_bands: Sequence[SpectrumBand] | None = None,
    radio_frequencies_mhz: Sequence[float] | None = None,
    aia_waves: Sequence[int] | None = None,
    display_extent_arcsec: tuple[float, float, float, float] | None = None,
    extended_canvas_color: str = "black",
    spectrum_frequency_range_mhz: tuple[float, float] | None = None,
    spectrum_intensity_range: tuple[float, float] | None = None,
    flux_plot_layout: str = "combined",
    gaussian_overrides: Mapping[str, Any] | None = None,
    metric: str = "raw_sum",
    dpi: int = 160,
) -> tuple[CompositeResult, TriplePanelArtifact]:
    """Build and render the complete three-panel scientific product."""

    result = build_composite_result(
        request,
        roi,
        frequencies_mhz=frequencies_mhz,
        time_range_utc=time_range_utc,
        roi_time_range_utc=roi_time_range_utc,
        spectrum_time_range_utc=spectrum_time_range_utc,
        cso_polarization=cso_polarization,
        spectrum_band=spectrum_band,
        spectrum_bands=spectrum_bands,
        radio_frequencies_mhz=radio_frequencies_mhz,
        aia_waves=aia_waves,
        display_extent_arcsec=display_extent_arcsec,
        extended_canvas_color=extended_canvas_color,
        spectrum_frequency_range_mhz=spectrum_frequency_range_mhz,
        spectrum_intensity_range=spectrum_intensity_range,
        flux_plot_layout=flux_plot_layout,
        gaussian_overrides=gaussian_overrides,
        dpi=dpi,
    )
    artifact = render_composite_result(
        result,
        map_time=result.metadata["map_time_utc"],
        metric=metric,
        dpi=dpi,
    )
    return result, artifact


def build_composite_video(
    result: CompositeResult,
    *,
    time_start: datetime | str,
    time_end: datetime | str,
    metric: str = "raw_sum",
    fps: int = 6,
    frame_count: int = 30,
    dpi: int = 120,
    quality: str = "high",
) -> CompositeVideoArtifact:
    """Animate the shared UTC marker and encode the composite as MP4.

    Scientific arrays are never interpolated. The selected AIA/radio top frame
    remains fixed while the common UTC marker advances over the requested
    display interval.
    """

    from .rendering.composite_renderer import _utc_datetime

    if not isinstance(result, CompositeResult):
        raise TypeError("result must be a CompositeResult")
    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    if start >= end:
        raise ValueError("video time_start must be before time_end")
    if int(fps) <= 0:
        raise ValueError("fps must be greater than zero")
    if int(frame_count) < 2:
        raise ValueError("frame_count must be at least 2")
    if quality not in {"high", "low"}:
        raise ValueError("quality must be high or low")
    markers = pd.date_range(start, end, periods=int(frame_count))
    with TemporaryDirectory(prefix="aia-radio-composite-video-") as temp_text:
        temp_dir = Path(temp_text)
        frame_paths: list[Path] = []
        for index, marker in enumerate(markers):
            frame = render_composite_result(
                result,
                map_time=marker.to_pydatetime(),
                metric=metric,
                dpi=int(dpi),
            )
            frame_path = temp_dir / f"frame_{index:05d}.png"
            frame_path.write_bytes(frame.image_png)
            frame_paths.append(frame_path)
        video_path = temp_dir / "aia_radio_composite.mp4"
        encoded = write_video_from_paths(
            [str(path) for path in frame_paths],
            str(video_path),
            fps=float(fps),
            quality=quality,
            workers=1,
            batch_size=2,
        )
        if not encoded or not video_path.is_file():
            raise RuntimeError("MP4 encoder did not produce a video")
        video_mp4 = video_path.read_bytes()
    metadata = {
        "product": "aia-radio-composite-video",
        "time_start_utc": start.isoformat(),
        "time_end_utc": end.isoformat(),
        "fps": int(fps),
        "frame_count": int(frame_count),
        "duration_seconds": float(frame_count) / float(fps),
        "metric": metric,
        "top_panel_mode": "fixed_reference_frame",
        "timeline_mode": "shared_utc_marker_no_data_interpolation",
        "video_sha256": hashlib.sha256(video_mp4).hexdigest(),
        "result": result.to_metadata_dict(),
    }
    metadata_json = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return CompositeVideoArtifact(
        video_mp4=video_mp4,
        metadata_json=metadata_json,
        metadata=metadata,
    )


def build_dynamic_composite_video(
    result: CompositeResult,
    request: CompositeRequest,
    *,
    aia_waves: Sequence[int],
    radio_frequencies_mhz: Sequence[float],
    time_start: datetime | str,
    time_end: datetime | str,
    display_extent_arcsec: tuple[float, float, float, float] | None = None,
    extended_canvas_color: str = "black",
    gaussian_overrides: Mapping[str, Any] | None = None,
    metric: str = "raw_sum",
    fps: int = 6,
    dpi: int = 120,
    quality: str = "high",
    aia_match_tolerance_seconds: float = AIA_VIDEO_MATCH_TOLERANCE_SECONDS,
    radio_match_tolerance_seconds: float = RADIO_VIDEO_MATCH_TOLERANCE_SECONDS,
    pair_time_tolerance_sec: float = 0.5,
    max_aia_pixels: int = 1024,
) -> CompositeVideoArtifact:
    """Encode one frame per real primary-radio observation without interpolation."""

    from .rendering.composite_renderer import _utc_datetime

    if not isinstance(result, CompositeResult):
        raise TypeError("result must be a CompositeResult")
    if not isinstance(request, CompositeRequest):
        raise TypeError("request must be a CompositeRequest")
    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    if start >= end:
        raise ValueError("video time_start must be before time_end")
    if int(fps) <= 0:
        raise ValueError("fps must be greater than zero")
    if quality not in {"high", "low"}:
        raise ValueError("quality must be high or low")
    wavelengths = _normalized_aia_waves(request, aia_waves)
    frequencies = _normalized_radio_frequencies(
        request,
        radio_frequencies_mhz,
    )
    radio_candidates = {
        frequency: load_radio_candidates(
            replace(request, radio_frequency=frequency),
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
        for frequency in frequencies
    }
    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)
    primary_frames = tuple(
        frame
        for frame in radio_candidates[frequencies[0]]
        if frame.obs_time is not None and start_naive <= frame.obs_time <= end_naive
    )
    if len(primary_frames) < 2:
        raise ValueError(
            "Fewer than two primary-radio observations exist in the video window"
        )
    aia_catalog = scan_aia_catalog(request)
    aia_cache: dict[str, Any] = {}
    radio_fit_cache: dict[tuple[Any, ...], Any] = {}
    skipped: list[dict[str, Any]] = []
    frame_metadata: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="aia-radio-dynamic-video-") as temp_text:
        temp_dir = Path(temp_text)
        frame_paths: list[Path] = []
        for primary_candidate in primary_frames:
            anchor = primary_candidate.obs_time.replace(tzinfo=UTC)
            try:
                primary_frame, _, _ = select_radio_frame_from_candidates(
                    (primary_candidate,),
                    anchor,
                    max_dt_seconds=0.0,
                )
                top = _build_matched_top_panel(
                    request,
                    primary_frame=primary_frame,
                    wavelengths=wavelengths,
                    radio_frequencies=frequencies,
                    radio_candidates=radio_candidates,
                    aia_catalog=aia_catalog,
                    aia_cache=aia_cache,
                    radio_fit_cache=radio_fit_cache,
                    aia_max_dt_seconds=float(aia_match_tolerance_seconds),
                    radio_match_tolerance_seconds=float(radio_match_tolerance_seconds),
                    max_aia_pixels=max_aia_pixels,
                    gaussian_overrides=gaussian_overrides,
                    dpi=dpi,
                    display_extent_arcsec=display_extent_arcsec,
                    extended_canvas_color=extended_canvas_color,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "primary_radio_time_utc": anchor.isoformat(),
                        "reason": str(exc),
                        "missing_sources": (
                            [exc.source] if isinstance(exc, _FrameMatchError) else []
                        ),
                    }
                )
                continue
            dynamic_result = replace(result, top_image=top.image_png)
            frame = render_composite_result(
                dynamic_result,
                map_time=anchor,
                metric=metric,
                dpi=int(dpi),
                display_time_range_utc=(start, end),
            )
            frame_path = temp_dir / f"frame_{len(frame_paths):05d}.png"
            frame_path.write_bytes(frame.image_png)
            frame_paths.append(frame_path)
            frame_metadata.append(
                {
                    "primary_radio_time_utc": anchor.isoformat(),
                    "top_panel": dict(top.metadata),
                }
            )
        if len(frame_paths) < 2:
            raise ValueError(
                "Fewer than two fully matched dynamic frames remain after "
                f"skipping {len(skipped)} incomplete observation(s)"
            )
        video_path = temp_dir / "aia_radio_composite.mp4"
        encoded = write_video_from_paths(
            [str(path) for path in frame_paths],
            str(video_path),
            fps=float(fps),
            quality=quality,
            workers=1,
            batch_size=2,
        )
        if not encoded or not video_path.is_file():
            raise RuntimeError("MP4 encoder did not produce a video")
        video_mp4 = video_path.read_bytes()
    metadata = {
        "product": "aia-radio-composite-video",
        "top_panel_mode": "dynamic_radio_first_multi_aia",
        "timeline_mode": "primary_radio_observations_no_interpolation",
        "time_start_utc": start.isoformat(),
        "time_end_utc": end.isoformat(),
        "reference_radio_frequency_mhz": frequencies[0],
        "radio_frequencies_mhz": list(frequencies),
        "aia_waves": list(wavelengths),
        "aia_match_tolerance_seconds": float(aia_match_tolerance_seconds),
        "radio_match_tolerance_seconds": float(radio_match_tolerance_seconds),
        "fps": int(fps),
        "frame_count": len(frame_metadata),
        "duration_seconds": len(frame_metadata) / float(fps),
        "skipped_frame_count": len(skipped),
        "skipped_frames": skipped,
        "frames": frame_metadata,
        "metric": metric,
        "video_sha256": hashlib.sha256(video_mp4).hexdigest(),
        "result": result.to_metadata_dict(),
    }
    metadata_json = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return CompositeVideoArtifact(
        video_mp4=video_mp4,
        metadata_json=metadata_json,
        metadata=metadata,
    )


def build_roi_lightcurve(
    request: CompositeRequest,
    roi: RadioRoi,
    *,
    frequencies_mhz: Sequence[float] = DEFAULT_ROI_FREQUENCIES_MHZ,
    time_start: str | datetime | None = None,
    time_end: str | datetime | None = None,
    pair_time_tolerance_sec: float = 0.5,
) -> pd.DataFrame:
    """Build the standardized multi-frequency ROI lightcurve table."""

    return extract_multi_frequency_roi_curve(
        request,
        roi,
        frequencies_mhz=frequencies_mhz,
        time_start=time_start,
        time_end=time_end,
        pair_time_tolerance_sec=pair_time_tolerance_sec,
    )


def build_spectrum_window(
    request: CompositeRequest,
    *,
    frequency_range_mhz: tuple[float, float] | None = None,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    max_frequency_samples: int = 1600,
    max_time_samples: int = 1600,
    cso_polarization: str | None = None,
) -> SpectrumWindow:
    """Build a normalized DART or CSO display window."""

    return load_spectrum_window(
        request,
        frequency_range_mhz=frequency_range_mhz,
        time_range_utc=time_range_utc,
        max_frequency_samples=max_frequency_samples,
        max_time_samples=max_time_samples,
        cso_polarization=cso_polarization,
    )


def build_spectrum_flux_curve(
    request: CompositeRequest,
    band: SpectrumBand,
    *,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
) -> SpectrumFluxCurve:
    """Build one original-channel CSO or DART spectrum flux curve."""

    return extract_spectrum_flux_curve(
        request,
        band,
        time_range_utc=time_range_utc,
        cso_polarization=cso_polarization,
    )


def build_spectrum_flux_curves(
    request: CompositeRequest,
    bands: Sequence[SpectrumBand],
    *,
    time_range_utc: tuple[datetime | str, datetime | str] | None = None,
    cso_polarization: str | None = None,
) -> tuple[SpectrumFluxCurve, ...]:
    """Build matched original-channel CSO or DART spectrum flux curves."""

    return extract_spectrum_flux_curves(
        request,
        tuple(bands),
        time_range_utc=time_range_utc,
        cso_polarization=cso_polarization,
    )


def _normalized_spectrum_bands(
    *,
    spectrum_band: SpectrumBand | None,
    spectrum_bands: Sequence[SpectrumBand] | None,
) -> tuple[SpectrumBand, ...]:
    if spectrum_bands is None:
        return (spectrum_band,) if spectrum_band is not None else ()
    normalized = tuple(spectrum_bands)
    if any(not isinstance(band, SpectrumBand) for band in normalized):
        raise TypeError("spectrum_bands must contain SpectrumBand values")
    if spectrum_band is not None and (not normalized or normalized[0] != spectrum_band):
        raise ValueError(
            "spectrum_band must match the first spectrum_bands item when both are set"
        )
    return normalized


def _normalized_flux_plot_layout(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in {"combined", "separate"}:
        raise ValueError("flux_plot_layout must be combined or separate")
    return normalized


def _normalized_radio_frequencies(
    request: CompositeRequest,
    values: Sequence[float] | None,
) -> tuple[float, ...]:
    normalized = (
        (float(request.radio_frequency),)
        if values is None
        else tuple(float(value) for value in values)
    )
    if not normalized:
        raise ValueError("radio_frequencies_mhz must not be empty")
    if any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise ValueError(
            "radio_frequencies_mhz must contain finite positive frequencies"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("radio_frequencies_mhz must not contain duplicates")
    return normalized


def _normalized_aia_waves(
    request: CompositeRequest,
    values: Sequence[int] | None,
) -> tuple[int, ...]:
    normalized = (
        (int(request.aia_wave),)
        if values is None
        else tuple(int(value) for value in values)
    )
    if not normalized:
        raise ValueError("aia_waves must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("aia_waves must not contain duplicates")
    unsupported = sorted(
        set(normalized).difference({94, 131, 171, 193, 211, 304, 335, 1600})
    )
    if unsupported:
        raise ValueError(f"Unsupported AIA wavelengths: {unsupported}")
    return normalized


def build_top_panel(
    request: CompositeRequest,
    *,
    aia_max_dt_seconds: float = 60.0,
    radio_max_dt_seconds: float = 60.0,
    pair_time_tolerance_sec: float = 0.5,
    max_aia_pixels: int = 1024,
    radio_frequencies_mhz: Sequence[float] | None = None,
    aia_waves: Sequence[int] | None = None,
    gaussian_overrides: Mapping[str, Any] | None = None,
    display_extent_arcsec: tuple[float, float, float, float] | None = None,
    extended_canvas_color: str = "black",
    dpi: int = 160,
) -> TopPanelArtifact:
    """Build the phase-four AIA background plus radio Gaussian artifact."""

    radio_frequencies = _normalized_radio_frequencies(
        request,
        radio_frequencies_mhz,
    )
    wavelengths = _normalized_aia_waves(request, aia_waves)
    radio_candidates = {
        frequency: load_radio_candidates(
            replace(request, radio_frequency=frequency),
            pair_time_tolerance_sec=pair_time_tolerance_sec,
        )
        for frequency in radio_frequencies
    }
    primary_frame, _, _ = select_radio_frame_from_candidates(
        radio_candidates[radio_frequencies[0]],
        request.aia_time,
        max_dt_seconds=radio_max_dt_seconds,
    )
    return _build_matched_top_panel(
        request,
        primary_frame=primary_frame,
        wavelengths=wavelengths,
        radio_frequencies=radio_frequencies,
        radio_candidates=radio_candidates,
        aia_catalog=scan_aia_catalog(request),
        aia_cache={},
        radio_fit_cache={},
        aia_max_dt_seconds=min(
            float(aia_max_dt_seconds),
            AIA_VIDEO_MATCH_TOLERANCE_SECONDS,
        ),
        radio_match_tolerance_seconds=RADIO_VIDEO_MATCH_TOLERANCE_SECONDS,
        max_aia_pixels=max_aia_pixels,
        gaussian_overrides=gaussian_overrides,
        dpi=dpi,
        display_extent_arcsec=display_extent_arcsec,
        extended_canvas_color=extended_canvas_color,
    )


def _build_matched_top_panel(
    request: CompositeRequest,
    *,
    primary_frame: Any,
    wavelengths: Sequence[int],
    radio_frequencies: Sequence[float],
    radio_candidates: Mapping[float, Sequence[Any]],
    aia_catalog: pd.DataFrame,
    aia_cache: dict[str, Any],
    radio_fit_cache: dict[tuple[Any, ...], Any],
    aia_max_dt_seconds: float,
    radio_match_tolerance_seconds: float,
    max_aia_pixels: int,
    gaussian_overrides: Mapping[str, Any] | None,
    dpi: int,
    display_extent_arcsec: tuple[float, float, float, float] | None,
    extended_canvas_color: str,
) -> TopPanelArtifact:
    """Match all selected inputs to one real primary-radio observation."""

    if primary_frame.obs_time is None:
        raise ValueError("Primary radio frame is missing an observation time")
    anchor = primary_frame.obs_time.replace(tzinfo=UTC)
    primary_frequency = float(radio_frequencies[0])
    radios = []
    for frequency in radio_frequencies:
        candidates = radio_candidates[float(frequency)]
        if float(frequency) == primary_frequency:
            frame = primary_frame
            delta_seconds = 0.0
            candidate_count = len(candidates)
        else:
            try:
                frame, candidate_count, delta_seconds = (
                    select_radio_frame_from_candidates(
                        candidates,
                        anchor,
                        max_dt_seconds=radio_match_tolerance_seconds,
                    )
                )
            except Exception as exc:
                raise _FrameMatchError(
                    f"radio {float(frequency):g} MHz",
                    str(exc),
                ) from exc
        cache_key = (
            str(frame.path.resolve(strict=False)),
            int(frame.hdu_index),
            float(frame.freq_mhz),
            str(frame.pol),
            frame.obs_time.isoformat(),
            json.dumps(dict(gaussian_overrides or {}), sort_keys=True),
        )
        fitted = radio_fit_cache.get(cache_key)
        if fitted is None:
            fitted = fit_radio_gaussian_frame(
                frame,
                requested_time_utc=anchor,
                candidate_count=candidate_count,
                delta_seconds=delta_seconds,
                gaussian_overrides=gaussian_overrides,
            )
            radio_fit_cache[cache_key] = fitted
        elif (
            fitted.requested_time_utc != anchor
            or fitted.delta_seconds != float(delta_seconds)
            or fitted.candidate_count != int(candidate_count)
        ):
            fitted = replace(
                fitted,
                requested_time_utc=anchor,
                delta_seconds=float(delta_seconds),
                candidate_count=int(candidate_count),
            )
        radios.append(fitted)

    aias = []
    for wavelength in wavelengths:
        aia_request = replace(
            request,
            aia_wave=int(wavelength),
            aia_time=anchor,
        )
        try:
            aias.append(
                load_aia_selection(
                    aia_request,
                    max_dt_seconds=aia_max_dt_seconds,
                    max_pixels=max_aia_pixels,
                    catalog=aia_catalog,
                    background_cache=aia_cache,
                )
            )
        except Exception as exc:
            raise _FrameMatchError(
                f"AIA {int(wavelength)} Å",
                str(exc),
            ) from exc
    return render_top_panel(
        tuple(aias),
        tuple(radios),
        dpi=dpi,
        display_extent_arcsec=display_extent_arcsec,
        extended_canvas_color=extended_canvas_color,
    )
