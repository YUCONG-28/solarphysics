"""Multi-frequency preview and sequence export for Radio Composite Figure."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_apps.frontends.radio.composite_figure.composite_figure_application import (
    CompositeArtifactBundle,
    CompositeFrameTemplate,
    annotate_source_map_image,
    build_composite_frame_template,
    build_curve_template_signature,
    render_cached_composite_frame,
)
from solar_apps.frontends.radio.source_map.artifacts import (
    validate_source_map_artifact,
)
from solar_apps.frontends.radio.source_map.worker import FIGURE_RENDER_LOCK, run_job
from solar_apps.workflows.common.image_naming import build_scientific_image_filename
from solar_toolkit.radio.dart_spectrogram import DartNarrowbandResult
from solar_toolkit.radio.roi_lightcurve import RadioRoi
from solar_toolkit.visualization import media

SEQUENCE_SCHEMA_VERSION = "radio-composite-v2"
DEFAULT_SEQUENCE_FPS = 10.0
DEFAULT_SEQUENCE_STRIDE = 1
LARGE_SEQUENCE_WARNING_FRAMES = 2_000

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]
RenderCandidate = Callable[
    [dict[str, Any], dict[str, Any], float, str, Path, int],
    tuple[bytes, dict[str, Any], dict[str, Any]],
]
_FIGURE_RENDER_LOCK = FIGURE_RENDER_LOCK


class CompositeSequenceCancelled(RuntimeError):
    """Raised when a sequence export is canceled before publication."""


@dataclass(frozen=True, slots=True)
class SequenceExportOptions:
    """Frozen media controls for a multi-frequency export."""

    fps: float = DEFAULT_SEQUENCE_FPS
    stride: int = DEFAULT_SEQUENCE_STRIDE
    dpi: int = 160
    quality: str = "high"
    transform: str = "linear"
    save_video: bool = True
    save_frames: bool = True

    def __post_init__(self) -> None:
        fps = float(self.fps)
        stride = int(self.stride)
        dpi = int(self.dpi)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be a positive finite value")
        if stride < 1:
            raise ValueError("Sequence stride must be at least one")
        if dpi < 72 or dpi > 600:
            raise ValueError("Composite DPI must be between 72 and 600")
        transform = str(self.transform).strip().lower()
        if transform not in {"linear", "log10"}:
            raise ValueError("Map transform must be linear or log10")
        if not bool(self.save_video) and not bool(self.save_frames):
            raise ValueError("Enable MP4 video, PNG frames, or both")
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "dpi", dpi)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "save_video", bool(self.save_video))
        object.__setattr__(self, "save_frames", bool(self.save_frames))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "stride": self.stride,
            "dpi": self.dpi,
            "quality": str(self.quality),
            "transform": self.transform,
            "output_format": "mp4",
            "save_video": self.save_video,
            "save_frames": self.save_frames,
        }


@dataclass(frozen=True, slots=True)
class CompositeSequenceBundle:
    """Published, file-backed sequence package."""

    output_directory: Path
    zip_path: Path
    metadata_path: Path
    reference_png_path: Path | None
    videos: Mapping[float, Path]
    frame_directories: Mapping[float, Path]
    radio_csv_paths: Mapping[float, Path]
    dart_csv_path: Path
    roi_json_path: Path
    metadata: Mapping[str, Any]
    dart_csv_paths: Mapping[float, Path] = field(default_factory=dict)
    radio_plot_paths: Mapping[float, Path] = field(default_factory=dict)
    dart_plot_paths: Mapping[float, Path] = field(default_factory=dict)
    curve_templates: Mapping[float, CompositeFrameTemplate] = field(
        default_factory=dict,
        repr=False,
    )


def resolve_single_band_frequency_source(
    source_path: str | Path,
    manifest: pd.DataFrame,
    frequency_mhz: float,
    *,
    polarization: str,
) -> Path:
    """Resolve an event, band, or polarization path to one single-band source."""

    source = Path(source_path).expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"Radio source directory does not exist: {source}")
    if not isinstance(manifest, pd.DataFrame) or manifest.empty:
        raise ValueError("The radio manifest is empty")
    frequency_column = (
        "inferred_freq_mhz" if "inferred_freq_mhz" in manifest else "freq_mhz"
    )
    values = pd.to_numeric(manifest.get(frequency_column), errors="coerce")
    tolerance = max(1e-6, abs(float(frequency_mhz)) * 1e-5)
    rows = manifest.loc[(values - float(frequency_mhz)).abs() <= tolerance].copy()
    if rows.empty:
        raise ValueError(f"No radio files match {float(frequency_mhz):g} MHz")
    wanted = str(polarization).upper()
    inferred = rows.get("inferred_polarization")
    if inferred is not None and wanted in {"RR", "LL"}:
        selected = rows.loc[inferred.astype(str).str.upper() == wanted]
        if not selected.empty:
            rows = selected
    paths = [Path(value).expanduser().resolve() for value in rows["path"].astype(str)]
    paths = [path for path in paths if path == source or source in path.parents]
    if not paths:
        raise ValueError(
            f"No {float(frequency_mhz):g} MHz files are below the selected radio source"
        )
    sample = paths[0]
    parent_name = sample.parent.name.upper()
    if parent_name in {"RR", "LL"}:
        band_directory = sample.parent.parent
        if wanted == "RR+LL":
            if (
                not (band_directory / "RR").is_dir()
                or not (band_directory / "LL").is_dir()
            ):
                raise ValueError(
                    f"{float(frequency_mhz):g} MHz requires matched RR and LL directories"
                )
            return band_directory
        requested_directory = band_directory / wanted
        return requested_directory if requested_directory.is_dir() else sample.parent
    return sample.parent


def group_candidates_by_frequency(
    candidates: Sequence[Mapping[str, Any]],
    frequencies_mhz: Sequence[float],
) -> dict[float, list[dict[str, Any]]]:
    """Return UTC-sorted candidate lists for every requested frequency."""

    grouped: dict[float, list[dict[str, Any]]] = {}
    for requested in frequencies_mhz:
        frequency = float(requested)
        matched = [
            copy.deepcopy(dict(candidate))
            for candidate in candidates
            if candidate.get("observation_time")
            and candidate_contains_frequency(candidate, frequency)
        ]
        matched.sort(key=lambda item: _utc_datetime(item["observation_time"]))
        if not matched:
            raise ValueError(
                f"No timestamped Source Map candidate matches {frequency:g} MHz"
            )
        grouped[frequency] = matched
    return grouped


def common_candidate_time_coverage(
    grouped: Mapping[float, Sequence[Mapping[str, Any]]],
) -> tuple[datetime, datetime]:
    """Return the common UTC coverage of all selected frequency sequences."""

    if not grouped:
        raise ValueError("Select at least one Source Map frequency")
    starts: list[datetime] = []
    ends: list[datetime] = []
    for frequency, candidates in grouped.items():
        values = sorted(
            _utc_datetime(candidate["observation_time"])
            for candidate in candidates
            if candidate.get("observation_time")
        )
        if len(values) < 2 or values[0] == values[-1]:
            raise ValueError(
                f"{float(frequency):g} MHz needs at least two distinct observation times"
            )
        starts.append(values[0])
        ends.append(values[-1])
    start = max(starts)
    end = min(ends)
    if start >= end:
        raise ValueError("Selected radio frequencies have no common UTC coverage")
    return start, end


def select_sequence_candidates(
    candidates: Sequence[Mapping[str, Any]],
    time_start: datetime | str,
    time_end: datetime | str,
    *,
    stride: int = 1,
) -> list[dict[str, Any]]:
    """Filter real observation frames by UTC, then apply the requested stride."""

    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    step = int(stride)
    if start >= end:
        raise ValueError("Shared UTC start must be before the end")
    if step < 1:
        raise ValueError("Sequence stride must be at least one")
    selected = [
        copy.deepcopy(dict(candidate))
        for candidate in candidates
        if candidate.get("observation_time")
        and start <= _utc_datetime(candidate["observation_time"]) <= end
    ]
    selected.sort(key=lambda item: _utc_datetime(item["observation_time"]))
    selected = selected[::step]
    if not selected:
        raise ValueError("No Source Map frame remains in the selected UTC range")
    return selected


def sequence_frame_counts(
    grouped: Mapping[float, Sequence[Mapping[str, Any]]],
    time_start: datetime | str,
    time_end: datetime | str,
    *,
    stride: int,
) -> dict[float, int]:
    return {
        float(frequency): len(
            select_sequence_candidates(candidates, time_start, time_end, stride=stride)
        )
        for frequency, candidates in grouped.items()
    }


def prepare_single_panel_render(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    frequency_mhz: float,
    *,
    transform: str,
    output_directory: str | Path,
    fixed_canvas: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt one discovered frame to a one-panel Source Map render request."""

    cfg = copy.deepcopy(dict(config))
    selected = copy.deepcopy(dict(candidate))
    paths, slot_item = candidate_frequency_paths(selected, float(frequency_mhz))
    if not paths:
        raise ValueError("The selected frame contains no path for this frequency")
    cfg["output_dir"] = str(Path(output_directory).resolve(strict=False))
    cfg["enable_spectrogram_panel"] = False
    cfg["write_source_map_sidecar"] = True
    cfg["show_plot"] = False
    cfg["save_plot"] = True
    cfg["max_workers"] = 1
    if fixed_canvas:
        cfg["_artifact_bbox_inches"] = None
        cfg["_artifact_pad_inches"] = 0.0
        cfg["_artifact_fixed_panel_layout"] = True
    display_payload = dict(cfg.get("spatial_display") or {})
    display_payload["transform"] = str(transform)
    cfg["spatial_display"] = display_payload
    if str(transform).lower() == "linear":
        cfg["mode"] = "single_band"
        cfg["single_file_path"] = paths[0]
        cfg["data_dir"] = str(Path(paths[0]).parent)
        selected = {
            "id": f"{selected['id']}-single-{float(frequency_mhz):g}",
            "mode": "single_band",
            "run_path": paths[0],
            "paths": paths,
            "frequencies_mhz": [float(frequency_mhz)],
            "observation_time": selected.get("observation_time"),
        }
    elif str(transform).lower() == "log10":
        cfg["mode"] = "multi_band"
        cfg["multi_band_freqs"] = [float(frequency_mhz)]
        selected = {
            "id": f"{selected['id']}-log-{float(frequency_mhz):g}",
            "mode": "multi_band",
            "slot_index": 0,
            "slot": [slot_item],
            "paths": paths,
            "frequencies_mhz": [float(frequency_mhz)],
            "observation_time": selected.get("observation_time"),
        }
    else:
        raise ValueError("Map transform must be linear or log10")
    return cfg, selected


def candidate_frequency_paths(
    candidate: Mapping[str, Any], frequency_mhz: float
) -> tuple[list[str], str | list[str]]:
    frequencies = [float(value) for value in candidate.get("frequencies_mhz", [])]
    index = matching_frequency_index(frequencies, frequency_mhz)
    if str(candidate.get("mode")) == "multi_band":
        slot = list(candidate.get("slot") or [])
        if index >= len(slot):
            raise ValueError("Source Map slot metadata does not match its frequencies")
        item = slot[index]
        paths = (
            [str(value) for value in item]
            if isinstance(item, (list, tuple))
            else [str(item)]
        )
        return paths, paths if len(paths) > 1 else paths[0]
    paths = [str(value) for value in candidate.get("paths", [])]
    return paths, paths if len(paths) > 1 else paths[0]


def matching_frequency_index(values: Sequence[float], requested: float) -> int:
    tolerance = max(1e-6, abs(float(requested)) * 1e-5)
    for index, value in enumerate(values):
        if abs(float(value) - float(requested)) <= tolerance:
            return index
    raise ValueError(f"Candidate does not contain {float(requested):g} MHz")


def candidate_contains_frequency(
    candidate: Mapping[str, Any], requested: float
) -> bool:
    try:
        matching_frequency_index(
            [float(value) for value in candidate.get("frequencies_mhz", [])],
            float(requested),
        )
    except ValueError:
        return False
    return True


def roi_intersects_source_map(metadata: Mapping[str, Any], roi: RadioRoi) -> bool:
    """Return whether a world-coordinate ROI overlaps the single map panel."""

    panels = list(metadata.get("panels") or [])
    if len(panels) != 1:
        raise ValueError("Source Map artifact must contain exactly one radio panel")
    panel = panels[0]
    x0, x1 = sorted(float(value) for value in panel["xlim_arcsec"])
    y0, y1 = sorted(float(value) for value in panel["ylim_arcsec"])
    xs = [float(point[0]) for point in roi.vertices_arcsec]
    ys = [float(point[1]) for point in roi.vertices_arcsec]
    return max(xs) >= x0 and min(xs) <= x1 and max(ys) >= y0 and min(ys) <= y1


def render_source_map_candidate(
    config: dict[str, Any],
    candidate: dict[str, Any],
    frequency_mhz: float,
    transform: str,
    output_directory: Path,
    sequence: int,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Render and validate one Source Map candidate using the existing worker."""

    with _FIGURE_RENDER_LOCK:
        render_cfg, render_candidate = prepare_single_panel_render(
            config,
            candidate,
            frequency_mhz,
            transform=transform,
            output_directory=output_directory,
            fixed_canvas=True,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(2):
            result = run_job(
                {
                    "config": render_cfg,
                    "candidate": render_candidate,
                    "sequence": sequence,
                }
            )
            image_path = Path(result["image_path"]).resolve()
            sidecar_path = Path(result["sidecar_path"]).resolve()
            metadata = validate_source_map_artifact(image_path, sidecar_path)
            try:
                _validate_sequence_source_map_pixels(
                    image_path,
                    metadata,
                    render_cfg,
                )
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    continue
                raise RuntimeError(
                    "Source Map rendering produced an invalid sequence frame "
                    f"after retry: {exc}"
                ) from exc
            if len(metadata.get("panels", [])) != 1:
                raise ValueError(
                    "Rendered Source Map did not contain exactly one panel"
                )
            return image_path.read_bytes(), metadata, result
        raise RuntimeError("Source Map sequence render failed") from last_error


def _validate_sequence_source_map_pixels(
    image_path: Path,
    metadata: Mapping[str, Any],
    render_config: Mapping[str, Any],
) -> None:
    """Reject truncated or blank fixed-canvas maps before ROI annotation."""

    from PIL import Image

    image_record = dict(metadata.get("image") or {})
    actual_size = (
        int(image_record.get("width") or 0),
        int(image_record.get("height") or 0),
    )
    figure_size = tuple(float(value) for value in render_config.get("fig_size", ()))
    dpi = int(render_config.get("dpi") or 0)
    if len(figure_size) == 2 and dpi > 0:
        expected_size = (
            int(round(figure_size[0] * dpi)),
            int(round(figure_size[1] * dpi)),
        )
        if actual_size != expected_size:
            raise ValueError(
                "fixed Source Map canvas size mismatch: expected "
                f"{expected_size[0]}x{expected_size[1]}, got "
                f"{actual_size[0]}x{actual_size[1]}"
            )
    with Image.open(image_path) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    if rgb.size == 0 or not np.any(rgb < 250):
        raise RuntimeError("fixed Source Map canvas is blank")


def export_composite_sequences(
    output_directory: str | Path,
    *,
    source_configs: Mapping[float, Mapping[str, Any]],
    candidates_by_frequency: Mapping[float, Sequence[Mapping[str, Any]]],
    radio_curves: Mapping[float, pd.DataFrame],
    dart_result: DartNarrowbandResult,
    dart_results_by_frequency: Mapping[float, DartNarrowbandResult] | None = None,
    roi: RadioRoi,
    reference_frequency_mhz: float,
    reference_time: datetime | str,
    polarization: str,
    time_start: datetime | str,
    time_end: datetime | str,
    request_signature: str,
    source_context: Mapping[str, Any],
    options: SequenceExportOptions,
    reference_bundle: CompositeArtifactBundle | None = None,
    curve_templates_by_frequency: Mapping[float, CompositeFrameTemplate] | None = None,
    render_candidate: RenderCandidate = render_source_map_candidate,
    media_writer: Callable[..., bool] = media.write_media_from_frames,
    media_probe: Callable[..., dict[str, Any]] = media.probe_video,
    cancel_check: CancelCheck | None = None,
    progress: ProgressCallback | None = None,
    generated_at: datetime | None = None,
) -> CompositeSequenceBundle:
    """Atomically publish one full composite video per selected frequency."""

    output_root = Path(output_directory).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    reference_time_utc = _utc_datetime(reference_time)
    generated = _utc_datetime(generated_at or datetime.now(UTC))
    if start >= end:
        raise ValueError("Shared UTC start must be before the end")
    if not start <= reference_time_utc <= end:
        raise ValueError("Reference Source Map time must lie inside the shared range")
    selected: dict[float, list[dict[str, Any]]] = {
        float(frequency): select_sequence_candidates(
            candidates,
            start,
            end,
            stride=options.stride,
        )
        for frequency, candidates in candidates_by_frequency.items()
    }
    if not selected:
        raise ValueError("No selected Source Map frequency is available for export")
    per_frequency_dart: dict[float, DartNarrowbandResult] = {}
    for frequency in selected:
        if dart_results_by_frequency is None:
            per_frequency_dart[frequency] = dart_result
            continue
        frequency_result = _mapping_value_for_frequency(
            dart_results_by_frequency,
            frequency,
        )
        if not isinstance(frequency_result, DartNarrowbandResult):
            raise TypeError(
                f"DART sequence input at {frequency:g} MHz has an invalid type"
            )
        if len(frequency_result.curves) != 1:
            raise ValueError(
                f"DART sequence input at {frequency:g} MHz must contain one curve"
            )
        center = float(frequency_result.curves[0].center_frequency_mhz)
        if not math.isclose(center, frequency, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"DART sequence curve center {center:g} MHz does not match "
                f"radio frequency {frequency:g} MHz"
            )
        per_frequency_dart[frequency] = frequency_result
    for frequency, curve in radio_curves.items():
        if float(frequency) not in selected:
            continue
        valid_pixels = pd.to_numeric(curve.get("valid_pixel_count"), errors="coerce")
        if not np.isfinite(valid_pixels).any() or not (valid_pixels > 0).any():
            raise ValueError(
                f"ROI contains no valid radio pixel at {float(frequency):g} MHz"
            )
    total_frames = sum(len(items) for items in selected.values())
    package_name = Path(
        build_scientific_image_filename(
            sequence=1,
            start_time=start,
            instrument="radio-dart",
            channel="multi",
            polarization=polarization,
            product="radio-composite-sequence",
            qualifiers=(roi.roi_id,),
            generated_at=generated,
            extension=".png",
        )
    ).stem
    destination = _unique_destination(output_root, package_name)
    stage = Path(tempfile.mkdtemp(prefix=".rcseq-", dir=output_root))
    warnings: list[str] = []
    video_paths: dict[float, Path] = {}
    frame_directories: dict[float, Path] = {}
    radio_csv_paths: dict[float, Path] = {}
    dart_csv_paths: dict[float, Path] = {}
    radio_plot_paths: dict[float, Path] = {}
    dart_plot_paths: dict[float, Path] = {}
    curve_templates: dict[float, CompositeFrameTemplate] = {}
    sequence_manifests: list[dict[str, Any]] = []
    completed_frames = 0
    reference_png_relative: str | None = None
    try:
        _raise_if_cancelled(cancel_check)
        roi_path = stage / "roi.json"
        _write_json(roi_path, roi.to_json_dict())
        dart_frame = _dart_curve_frame(dart_result)
        dart_path = stage / "dart-narrowband.csv"
        dart_frame.to_csv(dart_path, index=False, encoding="utf-8-sig")
        if reference_bundle is not None:
            reference_dir = stage / "reference"
            reference_dir.mkdir()
            for key, payload in reference_bundle.files.items():
                filename = reference_bundle.filenames[key]
                (reference_dir / filename).write_bytes(payload)
                if key == "composite_png":
                    reference_png_relative = str(Path("reference") / filename)

        for frequency, frequency_candidates in selected.items():
            _raise_if_cancelled(cancel_check)
            config = _mapping_value_for_frequency(source_configs, frequency)
            radio_curve = _mapping_value_for_frequency(radio_curves, frequency)
            frequency_dart_result = per_frequency_dart[frequency]
            frequency_name = Path(
                build_scientific_image_filename(
                    sequence=1,
                    start_time=start,
                    instrument="radio-dart",
                    channel=f"{frequency:g}mhz",
                    polarization=polarization,
                    product="radio-composite-sequence",
                    qualifiers=(roi.roi_id,),
                    generated_at=generated,
                    extension=".png",
                )
            ).stem
            frequency_dir_name = media.sanitize_filename(f"{frequency:g}mhz")
            concise_stem = media.sanitize_filename(
                f"{start:%Y%m%dT%H%M%SZ}_{frequency:g}mhz_"
                f"{polarization}_radio-composite-sequence"
            )
            frequency_dir = stage / frequency_dir_name
            frames_dir = frequency_dir / "frames"
            source_map_dir = frequency_dir / "source-maps"
            plots_dir = frequency_dir / "plots"
            frames_dir.mkdir(parents=True)
            source_map_dir.mkdir()
            plots_dir.mkdir()
            radio_csv = frequency_dir / f"{concise_stem}_radio-roi.csv"
            radio_curve.to_csv(radio_csv, index=False, encoding="utf-8-sig")
            dart_csv = frequency_dir / f"{concise_stem}_dart-narrowband.csv"
            frequency_dart_frame = _dart_curve_frame(frequency_dart_result)
            frequency_dart_frame.to_csv(
                dart_csv,
                index=False,
                encoding="utf-8-sig",
            )
            video_path = frequency_dir / f"{concise_stem}.mp4"
            radio_plot_path = plots_dir / f"{concise_stem}_radio-roi-lightcurve.png"
            dart_plot_path = (
                plots_dir / f"{concise_stem}_dart-narrowband-lightcurve.png"
            )
            frame_records: dict[int, dict[str, Any]] = {}
            rendered_frames: set[int] = set()
            fixed_template: CompositeFrameTemplate | None = None
            fixed_panel_bounds: Mapping[str, tuple[int, int, int, int]] | None = None
            fixed_source_map_panel_bbox: tuple[float, float, float, float] | None = None
            frequency_timings = {
                "source_map_render": 0.0,
                "curve_template": 0.0,
                "frame_composition": 0.0,
                "media_writer_wall": 0.0,
                "video_probe": 0.0,
            }
            cached_template = (
                _optional_mapping_value_for_frequency(
                    curve_templates_by_frequency,
                    frequency,
                )
                if curve_templates_by_frequency
                else None
            )

            def render_frame(index: int) -> tuple[np.ndarray, tuple[int, int]]:
                nonlocal completed_frames, fixed_template, fixed_panel_bounds
                nonlocal fixed_source_map_panel_bbox
                _raise_if_cancelled(cancel_check)
                frame_number = index + 1
                candidate = frequency_candidates[index]
                frame_path = frames_dir / f"{frame_number:06d}.png"
                source_started = time.perf_counter()
                with _FIGURE_RENDER_LOCK:
                    map_png, map_metadata, map_result = render_candidate(
                        dict(config),
                        dict(candidate),
                        frequency,
                        options.transform,
                        source_map_dir,
                        frame_number,
                    )
                frequency_timings["source_map_render"] += (
                    time.perf_counter() - source_started
                )
                marker = _utc_datetime(candidate["observation_time"])
                intersects = roi_intersects_source_map(map_metadata, roi)
                panel_bbox = tuple(
                    float(value)
                    for value in map_metadata["panels"][0]["bbox_normalized"]
                )
                if fixed_source_map_panel_bbox is None:
                    fixed_source_map_panel_bbox = panel_bbox
                elif not np.allclose(
                    panel_bbox,
                    fixed_source_map_panel_bbox,
                    rtol=0.0,
                    atol=1e-9,
                ):
                    raise ValueError(
                        "Source Map panel geometry changed after the sequence "
                        f"layout was fixed at {frequency:g} MHz: expected "
                        f"{fixed_source_map_panel_bbox}, got {panel_bbox}"
                    )
                frame_warnings: list[str] = []
                if not intersects:
                    frame_warnings.append("ROI does not intersect the Source Map panel")
                annotated = annotate_source_map_image(map_png, map_metadata, roi)
                source_map_input_size = tuple(int(value) for value in annotated.size)
                if fixed_template is None:
                    expected_signature = build_curve_template_signature(
                        radio_curve,
                        frequency_dart_result,
                        map_size_pixels=source_map_input_size,
                        map_frequency_mhz=frequency,
                        polarization=polarization,
                        time_start=start,
                        time_end=end,
                        dpi=options.dpi,
                    )
                    if (
                        isinstance(cached_template, CompositeFrameTemplate)
                        and cached_template.cache_signature == expected_signature
                    ):
                        fixed_template = cached_template
                    else:
                        template_started = time.perf_counter()
                        with _FIGURE_RENDER_LOCK:
                            fixed_template = build_composite_frame_template(
                                annotated,
                                radio_curve,
                                frequency_dart_result,
                                roi=roi,
                                map_frequency_mhz=frequency,
                                polarization=polarization,
                                time_start=start,
                                time_end=end,
                                dpi=options.dpi,
                            )
                        frequency_timings["curve_template"] += (
                            time.perf_counter() - template_started
                        )
                    fixed_panel_bounds = dict(fixed_template.panel_bounds_pixels)
                    radio_plot_path.write_bytes(fixed_template.radio_curve_png)
                    dart_plot_path.write_bytes(fixed_template.dart_curve_png)
                    curve_templates[frequency] = fixed_template
                source_map_template_size = fixed_template.layout.map_size_pixels
                resolution_normalized = (
                    source_map_input_size != source_map_template_size
                )
                if resolution_normalized:
                    frame_warnings.append(
                        "Source Map resolution normalized from "
                        f"{source_map_input_size[0]}x{source_map_input_size[1]} to "
                        f"the fixed {source_map_template_size[0]}x"
                        f"{source_map_template_size[1]} template"
                    )
                composition_started = time.perf_counter()
                rendered = render_cached_composite_frame(
                    fixed_template,
                    annotated,
                    map_time=marker,
                    include_png=options.save_frames,
                )
                frequency_timings["frame_composition"] += (
                    time.perf_counter() - composition_started
                )
                if fixed_panel_bounds is None:
                    fixed_panel_bounds = dict(rendered.panel_bounds_pixels)
                elif dict(rendered.panel_bounds_pixels) != dict(fixed_panel_bounds):
                    raise ValueError(
                        "Composite panel geometry changed after the sequence layout "
                        f"was fixed at {frequency:g} MHz"
                    )
                frame_hash: str | None = None
                frame_filename: str | None = None
                if options.save_frames:
                    if rendered.png_bytes is None:
                        raise RuntimeError(
                            "Composite frame PNG bytes were not produced"
                        )
                    frame_path.write_bytes(rendered.png_bytes)
                    frame_hash = hashlib.sha256(rendered.png_bytes).hexdigest()
                    frame_filename = str(Path("frames") / frame_path.name)
                frame_records[index] = {
                    "index": frame_number,
                    "filename": frame_filename,
                    "sha256": frame_hash,
                    "candidate_id": str(candidate.get("id") or ""),
                    "observation_time_utc": marker.isoformat(),
                    "source_paths": candidate_frequency_paths(candidate, frequency)[0],
                    "source_map_image_sha256": map_metadata["image"]["sha256"],
                    "source_map_input_size_pixels": list(source_map_input_size),
                    "source_map_template_size_pixels": list(source_map_template_size),
                    "source_map_resolution_normalized": resolution_normalized,
                    "source_map_panel_bbox_normalized": list(panel_bbox),
                    "source_map_result": _json_safe(map_result),
                    "roi_intersects_panel": intersects,
                    "roi_valid_pixel_count": _valid_pixel_count_at(radio_curve, marker),
                    "panel_bounds_pixels": {
                        key: list(value)
                        for key, value in rendered.panel_bounds_pixels.items()
                    },
                    "time_marker_x_pixels": {
                        key: int(value)
                        for key, value in (rendered.marker_x_pixels or {}).items()
                    },
                    "warnings": frame_warnings,
                }
                warnings.extend(frame_warnings)
                if index not in rendered_frames:
                    rendered_frames.add(index)
                    completed_frames += 1
                    if progress is not None:
                        progress(
                            completed_frames,
                            total_frames,
                            f"{frequency:g} MHz frame {frame_number}/{len(frequency_candidates)}",
                        )
                rgb = rendered.rgb
                return rgb, (int(rgb.shape[1]), int(rgb.shape[0]))

            first_rgb, first_size = render_frame(0)
            if (
                fixed_template is None
                or fixed_panel_bounds is None
                or fixed_source_map_panel_bbox is None
            ):
                raise RuntimeError("Composite sequence layout was not initialized")
            frame_size = fixed_template.layout.canvas_size_pixels
            if first_size != frame_size:
                raise ValueError(
                    "First composite frame does not match its fixed canvas: "
                    f"expected {frame_size}, got {first_size}"
                )

            def raw_frame_iter():
                for index in range(len(frequency_candidates)):
                    rgb, native_size = (
                        (first_rgb, first_size) if index == 0 else render_frame(index)
                    )
                    if native_size != frame_size:
                        raise ValueError(
                            "Composite frame size changed after the sequence layout "
                            f"was fixed at {frequency:g} MHz: expected {frame_size}, "
                            f"got {native_size}"
                        )
                    yield rgb, frame_size

            def frame_factory():
                return _prefetch_frames(
                    raw_frame_iter(),
                    cancel_check=cancel_check,
                    maxsize=2,
                )

            _raise_if_cancelled(cancel_check)
            probe: dict[str, Any] | None = None
            if options.save_video:
                media_started = time.perf_counter()
                ok = media_writer(
                    frame_factory,
                    video_path,
                    fps=options.fps,
                    quality=options.quality,
                    frame_size=frame_size,
                    output_format="mp4",
                )
                frequency_timings["media_writer_wall"] += (
                    time.perf_counter() - media_started
                )
                if not ok:
                    raise RuntimeError(
                        f"MP4 encoder did not produce a valid {frequency:g} MHz video"
                    )
                _raise_if_cancelled(cancel_check)
                probe_started = time.perf_counter()
                probe = media_probe(
                    video_path,
                    expected_size=frame_size,
                    expected_frame_count=len(frequency_candidates),
                )
                frequency_timings["video_probe"] += time.perf_counter() - probe_started
                reported_fps = probe.get("frame_rate")
                if reported_fps is None or not math.isclose(
                    float(reported_fps), options.fps, rel_tol=0.02, abs_tol=0.05
                ):
                    raise RuntimeError(
                        f"Encoded FPS mismatch at {frequency:g} MHz: {reported_fps}"
                    )
                expected_duration = len(frequency_candidates) / options.fps
                duration = probe.get("duration")
                if duration is None or not math.isclose(
                    float(duration), expected_duration, rel_tol=0.05, abs_tol=0.15
                ):
                    raise RuntimeError(
                        f"Encoded duration mismatch at {frequency:g} MHz: {duration}"
                    )
            else:
                for _frame in raw_frame_iter():
                    pass
            ordered_records = [frame_records[index] for index in sorted(frame_records)]
            layout_record = {
                **fixed_template.layout.to_dict(),
                "source_map_panel_bbox_normalized": list(fixed_source_map_panel_bbox),
                "panel_bounds_pixels": {
                    key: list(value) for key, value in fixed_panel_bounds.items()
                },
                "source_map_bounds_pixels": list(
                    fixed_template.source_map_bounds_pixels
                ),
            }
            curve_plots_record = {
                "cache_signature": fixed_template.cache_signature,
                "radio": {
                    "filename": str(Path("plots") / radio_plot_path.name),
                    "sha256": _sha256_file(radio_plot_path),
                },
                "dart": {
                    "filename": str(Path("plots") / dart_plot_path.name),
                    "sha256": _sha256_file(dart_plot_path),
                },
                "marker_free": True,
            }
            frequency_manifest = {
                "schema_version": SEQUENCE_SCHEMA_VERSION,
                "scientific_stem": frequency_name,
                "frequency_mhz": frequency,
                "polarization": str(polarization),
                "time_start_utc": start.isoformat(),
                "time_end_utc": end.isoformat(),
                "frame_count": len(frequency_candidates),
                "options": options.to_dict(),
                "layout": layout_record,
                "curve_plots": curve_plots_record,
                "timings_seconds": {
                    key: round(float(value), 6)
                    for key, value in frequency_timings.items()
                },
                "dart_curve": _dart_curve_metadata(
                    frequency_dart_result,
                    filename=dart_csv.name,
                    rows=len(frequency_dart_frame),
                ),
                "video": (
                    {
                        "filename": video_path.name,
                        "sha256": _sha256_file(video_path),
                        **_json_safe(probe),
                    }
                    if probe is not None
                    else None
                ),
                "frames": ordered_records,
                "warnings": list(dict.fromkeys(warnings)),
            }
            manifest_path = frequency_dir / "frame-manifest.json"
            _write_json(manifest_path, frequency_manifest)
            sequence_manifests.append(
                {
                    "frequency_mhz": frequency,
                    "scientific_stem": frequency_name,
                    "directory": frequency_dir_name,
                    "video": (
                        str(Path(frequency_dir_name) / video_path.name)
                        if options.save_video
                        else None
                    ),
                    "frames": str(Path(frequency_dir_name) / "frames"),
                    "curve_plots": {
                        "radio": str(
                            Path(frequency_dir_name) / "plots" / radio_plot_path.name
                        ),
                        "dart": str(
                            Path(frequency_dir_name) / "plots" / dart_plot_path.name
                        ),
                        "cache_signature": fixed_template.cache_signature,
                        "marker_free": True,
                    },
                    "radio_curve": str(Path(frequency_dir_name) / radio_csv.name),
                    "dart_curve": {
                        **_dart_curve_metadata(
                            frequency_dart_result,
                            filename=str(Path(frequency_dir_name) / dart_csv.name),
                            rows=len(frequency_dart_frame),
                        ),
                    },
                    "manifest": str(Path(frequency_dir_name) / manifest_path.name),
                    "frame_count": len(frequency_candidates),
                    "video_probe": _json_safe(probe),
                    "layout": layout_record,
                    "timings_seconds": {
                        key: round(float(value), 6)
                        for key, value in frequency_timings.items()
                    },
                }
            )
            if options.save_video:
                video_paths[frequency] = video_path
            frame_directories[frequency] = frames_dir
            radio_csv_paths[frequency] = radio_csv
            dart_csv_paths[frequency] = dart_csv
            radio_plot_paths[frequency] = radio_plot_path
            dart_plot_paths[frequency] = dart_plot_path

        metadata = {
            "schema_version": SEQUENCE_SCHEMA_VERSION,
            "generated_at_utc": generated.isoformat(),
            "request_signature": str(request_signature),
            "reference": {
                "frequency_mhz": float(reference_frequency_mhz),
                "observation_time_utc": reference_time_utc.isoformat(),
                "composite_png": reference_png_relative,
            },
            "roi": roi.to_json_dict(),
            "time_range": {
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "default_policy": "intersection-of-selected-radio-frequencies",
                "resampling": "none",
                "interpolation": "none",
            },
            "options": options.to_dict(),
            "dart_curve": {
                "representation": "source Stokes I dB intensity",
                "filename": dart_path.name,
                "rows": len(dart_frame),
            },
            "dart_curves_by_frequency": {
                f"{float(item['frequency_mhz']):g}": item["dart_curve"]
                for item in sequence_manifests
            },
            "frequencies": sequence_manifests,
            "source": _json_safe(source_context),
            "source_file_identities": _source_file_identities(source_context),
            "warnings": list(dict.fromkeys(warnings)),
        }
        metadata_path = stage / "radio-composite-metadata.json"
        _write_json(metadata_path, metadata)
        zip_path = stage / f"{destination.name}.zip"
        _write_package_zip(stage, zip_path)
        _raise_if_cancelled(cancel_check)
        os.replace(stage, destination)
        return CompositeSequenceBundle(
            output_directory=destination,
            zip_path=destination / zip_path.name,
            metadata_path=destination / metadata_path.name,
            reference_png_path=(
                destination / reference_png_relative
                if reference_png_relative is not None
                else None
            ),
            videos={
                frequency: destination / path.relative_to(stage)
                for frequency, path in video_paths.items()
            },
            frame_directories={
                frequency: destination / path.relative_to(stage)
                for frequency, path in frame_directories.items()
            },
            radio_csv_paths={
                frequency: destination / path.relative_to(stage)
                for frequency, path in radio_csv_paths.items()
            },
            dart_csv_path=destination / dart_path.name,
            roi_json_path=destination / roi_path.name,
            metadata=metadata,
            dart_csv_paths={
                frequency: destination / path.relative_to(stage)
                for frequency, path in dart_csv_paths.items()
            },
            radio_plot_paths={
                frequency: destination / path.relative_to(stage)
                for frequency, path in radio_plot_paths.items()
            },
            dart_plot_paths={
                frequency: destination / path.relative_to(stage)
                for frequency, path in dart_plot_paths.items()
            },
            curve_templates=dict(curve_templates),
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class CompositeSequenceJobRegistry:
    """Thread-backed, cancellable sequence jobs for Streamlit reruns."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self,
        task: Callable[[CancelCheck, ProgressCallback], CompositeSequenceBundle],
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        cancel_event = threading.Event()
        record: dict[str, Any] = {
            "id": job_id,
            "status": "running",
            "completed": 0,
            "total": 0,
            "message": "Preparing sequence export",
            "result": None,
            "error": None,
            "cancel_event": cancel_event,
        }
        with self._lock:
            self._jobs[job_id] = record

        def progress(completed: int, total: int, message: str) -> None:
            with self._lock:
                record["completed"] = int(completed)
                record["total"] = int(total)
                record["message"] = str(message)

        def run() -> None:
            try:
                result = task(cancel_event.is_set, progress)
            except CompositeSequenceCancelled:
                with self._lock:
                    record["status"] = "canceled"
                    record["message"] = "Sequence export canceled"
            except Exception as exc:
                with self._lock:
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    record["message"] = "Sequence export failed"
            else:
                with self._lock:
                    record["status"] = "completed"
                    record["result"] = result
                    record["message"] = "Sequence export completed"

        threading.Thread(target=run, daemon=True).start()
        return self.public(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(job_id)
            if record["status"] == "running":
                record["status"] = "canceling"
                record["message"] = "Cancel requested"
                record["cancel_event"].set()
        return self.public(job_id)

    def public(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(job_id)
            return {
                key: record[key]
                for key in (
                    "id",
                    "status",
                    "completed",
                    "total",
                    "message",
                    "result",
                    "error",
                )
            }

    def stop_all(self) -> None:
        with self._lock:
            ids = [
                job_id
                for job_id, record in self._jobs.items()
                if record["status"] in {"running", "canceling"}
            ]
        for job_id in ids:
            self.cancel(job_id)

    def _require(self, job_id: str) -> dict[str, Any]:
        record = self._jobs.get(str(job_id))
        if record is None:
            raise KeyError("Composite sequence job not found or expired")
        return record


def _mapping_value_for_frequency(mapping: Mapping, frequency: float):
    for key, value in mapping.items():
        try:
            if math.isclose(float(key), frequency, rel_tol=0.0, abs_tol=1e-6):
                return value
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Missing sequence input for {frequency:g} MHz")


def _optional_mapping_value_for_frequency(
    mapping: Mapping | None,
    frequency: float,
) -> Any | None:
    if not mapping:
        return None
    try:
        return _mapping_value_for_frequency(mapping, frequency)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _PrefetchFailure:
    error: BaseException


def _prefetch_frames(
    frames,
    *,
    cancel_check: CancelCheck | None = None,
    maxsize: int = 2,
):
    """Lazily render ahead while keeping at most ``maxsize`` RGB frames."""

    queue_size = max(1, int(maxsize))
    pending: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
    stop = threading.Event()
    finished = object()
    source = iter(frames)

    def put(value: Any) -> bool:
        while not stop.is_set():
            try:
                pending.put(value, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for frame in source:
                _raise_if_cancelled(cancel_check)
                if not put(frame):
                    return
        except BaseException as exc:  # propagate producer failures to the encoder
            put(_PrefetchFailure(exc))
        finally:
            put(finished)

    producer = threading.Thread(
        target=produce,
        name="radio-composite-frame-prefetch",
        daemon=True,
    )
    producer.start()
    try:
        while True:
            item = pending.get()
            if item is finished:
                break
            if isinstance(item, _PrefetchFailure):
                raise item.error
            yield item
    finally:
        stop.set()
        close = getattr(source, "close", None)
        if callable(close):
            try:
                close()
            except (RuntimeError, ValueError):
                # A producer that is unwinding may still own the generator.
                # The stop event remains the authoritative cancellation signal.
                pass
        producer.join(timeout=30.0)


def _valid_pixel_count_at(curve: pd.DataFrame, marker: datetime) -> int | None:
    if "valid_pixel_count" not in curve or "obs_time" not in curve:
        return None
    times = pd.to_datetime(curve["obs_time"], errors="coerce", utc=True)
    rows = curve.loc[times == pd.Timestamp(marker)]
    if rows.empty:
        return None
    values = pd.to_numeric(rows["valid_pixel_count"], errors="coerce")
    finite = values[np.isfinite(values)]
    return int(finite.sum()) if not finite.empty else None


def _dart_curve_frame(result: DartNarrowbandResult) -> pd.DataFrame:
    data: dict[str, Any] = {
        "time_utc": [_utc_datetime(value).isoformat() for value in result.time_utc]
    }
    for index, curve in enumerate(result.curves, start=1):
        name = (
            f"stokes_i_db_{curve.requested_frequency_range_mhz[0]:g}_"
            f"{curve.requested_frequency_range_mhz[1]:g}_mhz"
        )
        if name in data:
            name = f"{name}_{index}"
        data[name] = np.asarray(curve.stokes_i_db, dtype=float)
    return pd.DataFrame(data)


def _dart_curve_metadata(
    result: DartNarrowbandResult,
    *,
    filename: str,
    rows: int,
) -> dict[str, Any]:
    if len(result.curves) != 1:
        raise ValueError("Per-frequency DART metadata requires exactly one curve")
    curve = result.curves[0]
    return {
        "representation": "source Stokes I dB intensity",
        "filename": str(filename),
        "rows": int(rows),
        "center_frequency_mhz": float(curve.center_frequency_mhz),
        "bandwidth_mhz": float(curve.bandwidth_mhz),
        "requested_frequency_range_mhz": [
            float(value) for value in curve.requested_frequency_range_mhz
        ],
        "sampled_frequency_range_mhz": [
            float(value) for value in curve.sampled_frequency_range_mhz
        ],
        "channel_count": int(curve.channel_count),
    }


def _unique_destination(root: Path, stem: str) -> Path:
    destination = root / stem
    suffix = 2
    while destination.exists():
        destination = root / f"{stem}_{suffix:03d}"
        suffix += 1
    return destination


def _write_package_zip(root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, mode="w") as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path == target:
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".media-") for part in relative.parts):
                continue
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.casefold() == ".mp4"
                else zipfile.ZIP_DEFLATED
            )
            archive.write(path, relative.as_posix(), compress_type=compression)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_file_identities(value: Any) -> list[dict[str, Any]]:
    paths: set[Path] = set()

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)
        elif isinstance(item, (str, Path)):
            try:
                path = Path(item).expanduser().resolve()
            except (OSError, TypeError, ValueError):
                return
            if path.is_file():
                paths.add(path)

    collect(value)
    identities = []
    for path in sorted(paths, key=str):
        stat = path.stat()
        identities.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return identities


def _raise_if_cancelled(cancel_check: CancelCheck | None) -> None:
    if cancel_check is not None and cancel_check():
        raise CompositeSequenceCancelled("Sequence export canceled")


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        normalized = text[:-1] + "+00:00" if text.upper().endswith("Z") else text
        parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _utc_datetime(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "CompositeSequenceBundle",
    "CompositeSequenceCancelled",
    "CompositeSequenceJobRegistry",
    "DEFAULT_SEQUENCE_FPS",
    "DEFAULT_SEQUENCE_STRIDE",
    "LARGE_SEQUENCE_WARNING_FRAMES",
    "SEQUENCE_SCHEMA_VERSION",
    "SequenceExportOptions",
    "candidate_contains_frequency",
    "candidate_frequency_paths",
    "common_candidate_time_coverage",
    "export_composite_sequences",
    "group_candidates_by_frequency",
    "prepare_single_panel_render",
    "render_source_map_candidate",
    "resolve_single_band_frequency_source",
    "roi_intersects_source_map",
    "select_sequence_candidates",
    "sequence_frame_counts",
]
