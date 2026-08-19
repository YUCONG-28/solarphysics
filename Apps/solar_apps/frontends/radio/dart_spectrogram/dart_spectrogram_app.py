"""Standalone Streamlit frontend for four-file DART spectrogram analysis."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.patheffects as mpatheffects
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, ScalarFormatter
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import AllowedRootPolicyError
from solar_apps.ui.state import (
    bind_streamlit_fields,
    frontend_path_memory,
    frontend_state_store,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
    render_native_path_input,
    resolve_streamlit_allowed_roots,
)
from solar_apps.ui.theme import render_streamlit_theme
from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
)
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandResult,
    DartSpectrogramFiles,
    DartSpectrogramWindow,
    discover_dart_spectrogram_files,
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)

__all__ = [
    "DYNAMIC_SPECTRUM_FILENAME",
    "DYNAMIC_SPECTRUM_PRODUCT_KEY",
    "LIGHTCURVE_FILENAME",
    "LIGHTCURVE_PRODUCT_KEY",
    "SELECTED_SPECTRUM_FILENAME",
    "SELECTED_SPECTRUM_PRODUCT_KEY",
    "DartDisplayLimits",
    "DartDatasetSummary",
    "build_dart_artifact_filenames",
    "build_dynamic_spectrum_figure",
    "build_dynamic_spectrum_png",
    "build_narrowband_figure",
    "build_narrowband_png",
    "build_parser",
    "build_zip_bytes",
    "inspect_dart_dataset",
    "main",
    "parse_center_frequencies",
    "parse_marked_times",
    "resolve_display_limits",
    "resolve_selected_frequency_range",
    "save_artifacts",
]

DART_UI_FIELD_KEYS = (
    "input_dir",
    "output_dir",
    "analysis_mode",
    "display_mode",
    "limit_mode",
    "stokes_i_percentile_low",
    "stokes_i_percentile_high",
    "stokes_i_direct_low",
    "stokes_i_direct_high",
    "center_frequencies",
    "bandwidth_mhz",
    "limit_frequency",
    "frequency_start",
    "frequency_end",
    "limit_time",
    "time_start",
    "time_end",
    "x_tick_mode",
    "x_tick_interval_seconds",
    "marked_times",
    "max_frequency_samples",
    "max_time_samples",
    "chunk_memory_mb",
    "dpi",
)

# Legacy explicit names remain importable for callers that deliberately use
# them.  Automatic exports use stable product keys plus generated filenames.
DYNAMIC_SPECTRUM_FILENAME = "dart_dynamic_spectrum_stokes_i_vp.png"
SELECTED_SPECTRUM_FILENAME = "dart_selected_spectrum_stokes_i_vp.png"
LIGHTCURVE_FILENAME = "dart_narrowband_stokes_i_lightcurves.png"
DYNAMIC_SPECTRUM_PRODUCT_KEY = "dynamic_spectrum"
SELECTED_SPECTRUM_PRODUCT_KEY = "selected_spectrum"
LIGHTCURVE_PRODUCT_KEY = "narrowband_lightcurve"
ZIP_FILENAME = "dart_spectrogram_results.zip"
FULL_SPECTRUM_MODE = "full_spectrum"
FULL_AND_NARROWBAND_MODE = "full_and_narrowband"
_FULL_SPECTRUM_PRODUCT_KEYS = (DYNAMIC_SPECTRUM_PRODUCT_KEY,)
_FULL_ANALYSIS_PRODUCT_KEYS = (
    DYNAMIC_SPECTRUM_PRODUCT_KEY,
    SELECTED_SPECTRUM_PRODUCT_KEY,
    LIGHTCURVE_PRODUCT_KEY,
)
_MAX_CUSTOM_X_TICKS = 200
_TIME_MARKER_HEIGHT = 0.06
_TIME_MARKER_COLOR = "#facc15"
_TIME_MARKER_EDGE_COLOR = "#111827"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DartDatasetSummary:
    """Validated metadata shown before the large arrays are processed."""

    directory: Path
    files: DartSpectrogramFiles
    matrix_shape: tuple[int, int]
    frequency_range_mhz: tuple[float, float]
    frequency_samples: int
    time_range_utc: tuple[datetime, datetime]
    time_samples: int


@dataclass(frozen=True)
class DartDisplayLimits:
    """Resolved Stokes I limits shared by the full and selected spectra."""

    stokes_i: tuple[float, float]


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for Streamlit script arguments."""

    parser = argparse.ArgumentParser(
        description="Run the standalone DART spectrogram Streamlit app."
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Default directory containing the four DART FITS files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default local directory for optional PNG exports.",
    )
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to this app.",
    )
    return parser


def inspect_dart_dataset(directory: str | Path) -> DartDatasetSummary:
    """Validate the four FITS files and return lightweight dataset metadata."""

    files = discover_dart_spectrogram_files(directory)
    with fits.open(files.frequency, memmap=True) as hdul:
        frequency_mhz = np.asarray(hdul[0].data, dtype=np.float64).reshape(-1)
    with fits.open(files.time, memmap=True) as hdul:
        time_rows = np.asarray(hdul[0].data, dtype=np.float64)
    if frequency_mhz.size == 0:
        raise ValueError("DART frequency axis is empty")
    if time_rows.ndim != 2 or time_rows.shape[1] != 6 or time_rows.shape[0] == 0:
        raise ValueError(
            "DART time FITS must have shape (time, 6); " f"found {time_rows.shape}"
        )

    first_time = _time_row_to_datetime(time_rows[0])
    last_time = _time_row_to_datetime(time_rows[-1])
    probe = read_dart_spectrogram_window(
        files,
        frequency_range_mhz=(float(frequency_mhz[0]), float(frequency_mhz[0])),
        time_range_utc=(first_time, first_time),
        max_frequency_samples=1,
        max_time_samples=1,
        chunk_memory_mb=1,
    )
    if probe.stokes_i_db.shape != (1, 1):
        raise ValueError("DART metadata validation probe returned an invalid shape")
    with fits.open(files.stokes_i_db, memmap=True) as hdul:
        matrix_shape = tuple(int(value) for value in hdul[0].data.shape)

    return DartDatasetSummary(
        directory=Path(directory).expanduser().resolve(),
        files=files,
        matrix_shape=matrix_shape,
        frequency_range_mhz=(
            float(np.min(frequency_mhz)),
            float(np.max(frequency_mhz)),
        ),
        frequency_samples=int(frequency_mhz.size),
        time_range_utc=(first_time, last_time),
        time_samples=int(time_rows.shape[0]),
    )


def parse_center_frequencies(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of one to twenty unique frequencies."""

    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("Enter center frequencies separated by commas")
    try:
        centers = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("Center frequencies must be numeric") from exc
    if len(centers) > 20:
        raise ValueError("At most 20 center frequencies may be entered")
    if not all(np.isfinite(center) for center in centers):
        raise ValueError("Center frequencies must be finite")
    if len(set(centers)) != len(centers):
        raise ValueError("Center frequencies must not contain duplicates")
    return centers


def parse_marked_times(
    value: str | None,
    observation_range_utc: tuple[datetime, datetime],
) -> tuple[datetime, ...]:
    """Parse optional comma-separated UTC marker times inside an observation."""

    text = str(value or "").strip()
    if not text:
        return ()
    start, end = (_as_utc(item) for item in observation_range_utc)
    if start > end:
        raise ValueError("Observation UTC range must start before it ends")
    parts = [part.strip() for part in text.split(",")]
    if any(not part for part in parts):
        raise ValueError("Enter marked UTC times separated by commas")

    resolved: list[datetime] = []
    out_of_range: list[str] = []
    for part in parts:
        time_only = _parse_time_only(part)
        if time_only is None:
            marked = _parse_iso_utc(part)
            matches = [marked] if start <= marked <= end else []
        else:
            matches = []
            day_count = (end.date() - start.date()).days
            for day_offset in range(day_count + 1):
                day = start.date() + timedelta(days=day_offset)
                candidate = datetime.combine(day, time_only, tzinfo=UTC)
                if start <= candidate <= end:
                    matches.append(candidate)
        if len(matches) > 1:
            raise ValueError(
                f"Marked UTC time {part!r} matches more than one observation "
                "date; enter a full ISO-8601 timestamp"
            )
        if not matches:
            out_of_range.append(part)
        else:
            resolved.append(matches[0])
    if out_of_range:
        raise ValueError(
            "Marked UTC times outside the observation range: " + ", ".join(out_of_range)
        )
    return tuple(sorted(set(resolved)))


def resolve_display_limits(
    full_window: DartSpectrogramWindow,
    *,
    display_mode: str = "db",
    limit_mode: str = "percentile",
    stokes_i_bounds: tuple[float, float] = (1.0, 99.5),
) -> DartDisplayLimits:
    """Resolve the Stokes I color scale from the full observation."""

    mode = _normalize_display_mode(display_mode)
    normalized_limit_mode = str(limit_mode).strip().lower()
    stokes_i = _display_stokes_i(full_window.stokes_i_db, mode)
    if normalized_limit_mode == "percentile":
        _validate_percentile_limits(stokes_i_bounds, "Stokes I percentile")
        resolved_stokes_i = _finite_percentile_limits(
            stokes_i,
            *stokes_i_bounds,
        )
    elif normalized_limit_mode == "direct":
        resolved_stokes_i = _validate_direct_limits(
            stokes_i_bounds,
            "Stokes I display",
        )
    else:
        raise ValueError("Color limit mode must be Percentile or Direct")
    return DartDisplayLimits(stokes_i=resolved_stokes_i)


def resolve_selected_frequency_range(
    narrowband: DartNarrowbandResult,
    observation_range_mhz: tuple[float, float],
    override_range_mhz: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return the selected plot range and require it to include every band."""

    if not narrowband.curves:
        raise ValueError("At least one narrowband curve is required")
    requested_low = min(
        curve.requested_frequency_range_mhz[0] for curve in narrowband.curves
    )
    requested_high = max(
        curve.requested_frequency_range_mhz[1] for curve in narrowband.curves
    )
    observation_low, observation_high = _validate_direct_limits(
        observation_range_mhz,
        "Observation frequency",
    )
    if requested_low < observation_low or requested_high > observation_high:
        raise ValueError(
            "Every requested narrowband must be inside the observed frequency range "
            f"{observation_low:g}-{observation_high:g} MHz"
        )
    if override_range_mhz is None:
        return float(requested_low), float(requested_high)
    override_low, override_high = _validate_direct_limits(
        override_range_mhz,
        "Selected-region frequency",
    )
    if override_low < observation_low or override_high > observation_high:
        raise ValueError(
            "Selected-region frequency range must stay inside the observation"
        )
    if override_low > requested_low or override_high < requested_high:
        raise ValueError(
            "Selected-region frequency range must contain every requested "
            f"narrowband ({requested_low:g}-{requested_high:g} MHz)"
        )
    return override_low, override_high


def build_dynamic_spectrum_figure(
    window: DartSpectrogramWindow,
    narrowband: DartNarrowbandResult | None = None,
    *,
    display_mode: str = "db",
    display_limits: DartDisplayLimits | None = None,
    selected_frequency_range_mhz: tuple[float, float] | None = None,
    selected_time_range_utc: tuple[datetime, datetime] | None = None,
    region_label: str = "Full observation",
    x_tick_interval_seconds: float | None = None,
    marked_times_utc: tuple[datetime, ...] = (),
) -> Figure:
    """Build one unit-aware Stokes I and Stokes V/I dynamic spectrum."""

    _validate_plot_inputs(window, narrowband)
    mode = _normalize_display_mode(display_mode)
    stokes_i = _display_stokes_i(window.stokes_i_db, mode)
    stokes_v_over_i = np.asarray(window.stokes_v_over_i, dtype=np.float64)
    if display_limits is None:
        display_limits = resolve_display_limits(
            window,
            display_mode=mode,
            limit_mode="percentile",
            stokes_i_bounds=(1.0, 99.5),
        )
    _validate_direct_limits(display_limits.stokes_i, "Stokes I display")

    figure = Figure(figsize=(12.0, 8.2), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    time_numbers = np.asarray(mdates.date2num(window.time_utc), dtype=np.float64)
    x_limits = _axis_extent(time_numbers, fallback_half_step=0.5 / 86400.0)
    y_limits = _axis_extent(window.frequency_mhz, fallback_half_step=0.5)
    extent = [x_limits[0], x_limits[1], y_limits[0], y_limits[1]]

    image_i = axes[0].imshow(
        stokes_i,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=display_limits.stokes_i[0],
        vmax=display_limits.stokes_i[1],
        interpolation="nearest",
    )
    intensity_label = _stokes_i_label(mode)
    intensity_colorbar = figure.colorbar(image_i, ax=axes[0], pad=0.01)
    intensity_colorbar.set_label(intensity_label)
    if mode == "linear":
        intensity_colorbar.formatter = _scientific_formatter()
        intensity_colorbar.update_ticks()
    axes[0].set_title(intensity_label)

    if narrowband is not None:
        colors = _curve_colors(len(narrowband.curves))
        for curve, color in zip(narrowband.curves, colors, strict=True):
            low, high = curve.requested_frequency_range_mhz
            axes[0].axhspan(low, high, color=color, alpha=0.12, linewidth=0)
            axes[0].axhline(
                curve.center_frequency_mhz,
                color=color,
                linewidth=0.9,
                alpha=0.9,
            )
            axes[0].text(
                0.012,
                curve.center_frequency_mhz,
                f"{low:g}-{high:g} MHz",
                transform=axes[0].get_yaxis_transform(),
                color=color,
                fontsize=8,
                fontweight="bold",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": color,
                    "alpha": 0.78,
                },
            )

    polarization_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image_vp = axes[1].imshow(
        stokes_v_over_i,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        norm=polarization_norm,
        interpolation="nearest",
    )
    polarization_label = "Stokes V/I = (R-L)/(R+L) (dimensionless)"
    figure.colorbar(image_vp, ax=axes[1], pad=0.01).set_label(polarization_label)
    axes[1].set_title(polarization_label)

    if narrowband is not None and selected_frequency_range_mhz is not None:
        frequency_low, frequency_high = selected_frequency_range_mhz
        selected_dates = selected_time_range_utc or (
            window.time_utc[0],
            window.time_utc[-1],
        )
        selected_time_numbers = np.asarray(
            mdates.date2num(selected_dates),
            dtype=np.float64,
        )
        rectangle = Rectangle(
            (float(selected_time_numbers[0]), float(frequency_low)),
            float(selected_time_numbers[1] - selected_time_numbers[0]),
            float(frequency_high - frequency_low),
            fill=False,
            edgecolor="black",
            linewidth=1.8,
            linestyle="--",
            zorder=5,
        )
        axes[0].add_patch(rectangle)
        axes[0].text(
            0.99,
            0.98,
            _selection_annotation(
                selected_frequency_range_mhz,
                selected_dates,
            ),
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.82,
            },
        )

    for axis in axes:
        axis.set_ylabel("Frequency (MHz)")
        axis.set_ylim(y_limits)
        axis.grid(alpha=0.18, linestyle=":", linewidth=0.6)
    axes[1].set_xlabel("Time (UTC)")
    axes[1].set_xlim(x_limits)
    _configure_time_axis(
        axes[1],
        window.time_utc,
        x_tick_interval_seconds=x_tick_interval_seconds,
    )
    visible_markers = _visible_marked_times(marked_times_utc, window.time_utc)
    _add_time_markers(axes, visible_markers, label_axis=axes[1])
    observation_date = window.time_utc[0].strftime("%Y-%m-%d")
    figure.suptitle(
        f"DART Dynamic Spectrum | {region_label} | {observation_date}",
        fontsize=15,
    )
    return figure


def build_dynamic_spectrum_png(
    window: DartSpectrogramWindow,
    narrowband: DartNarrowbandResult | None = None,
    *,
    dpi: int = 160,
    display_mode: str = "db",
    display_limits: DartDisplayLimits | None = None,
    selected_frequency_range_mhz: tuple[float, float] | None = None,
    selected_time_range_utc: tuple[datetime, datetime] | None = None,
    region_label: str = "Full observation",
    x_tick_interval_seconds: float | None = None,
    marked_times_utc: tuple[datetime, ...] = (),
) -> bytes:
    """Render one deterministic dynamic-spectrum PNG payload."""

    figure = build_dynamic_spectrum_figure(
        window,
        narrowband,
        display_mode=display_mode,
        display_limits=display_limits,
        selected_frequency_range_mhz=selected_frequency_range_mhz,
        selected_time_range_utc=selected_time_range_utc,
        region_label=region_label,
        x_tick_interval_seconds=x_tick_interval_seconds,
        marked_times_utc=marked_times_utc,
    )
    return _figure_to_png(figure, dpi=dpi)


def build_narrowband_figure(
    result: DartNarrowbandResult,
    *,
    display_mode: str = "db",
    x_tick_interval_seconds: float | None = None,
    marked_times_utc: tuple[datetime, ...] = (),
) -> Figure:
    """Build narrowband light curves without changing the source calculation."""

    if not result.curves:
        raise ValueError("At least one narrowband curve is required")
    if not result.time_utc:
        raise ValueError("The narrowband UTC time axis is empty")
    mode = _normalize_display_mode(display_mode)
    figure = Figure(figsize=(12.0, 5.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots(1, 1)
    colors = _curve_colors(len(result.curves))
    for curve, color in zip(result.curves, colors, strict=True):
        requested_low, requested_high = curve.requested_frequency_range_mhz
        sampled_low, sampled_high = curve.sampled_frequency_range_mhz
        label = (
            f"{curve.center_frequency_mhz:g} MHz | requested "
            f"{requested_low:g}-{requested_high:g} MHz | sampled "
            f"{sampled_low:.6g}-{sampled_high:.6g} MHz "
            f"({curve.channel_count} ch)"
        )
        axis.plot(
            result.time_utc,
            _display_stokes_i(curve.stokes_i_db, mode),
            color=color,
            linewidth=1.25,
            label=label,
        )

    axis.set_title("DART Narrowband Stokes I Light Curves")
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel(f"Band-averaged {_stokes_i_label(mode)}")
    if mode == "linear":
        axis.yaxis.set_major_formatter(_scientific_formatter())
    axis.grid(alpha=0.28, linestyle=":", linewidth=0.7)
    _configure_time_axis(
        axis,
        result.time_utc,
        x_tick_interval_seconds=x_tick_interval_seconds,
    )
    visible_markers = _visible_marked_times(marked_times_utc, result.time_utc)
    _add_time_markers((axis,), visible_markers, label_axis=axis)
    axis.legend(
        loc="best",
        frameon=False,
        fontsize=8,
        ncols=1 if len(result.curves) < 5 else 2,
    )
    return figure


def build_narrowband_png(
    result: DartNarrowbandResult,
    *,
    dpi: int = 160,
    display_mode: str = "db",
    x_tick_interval_seconds: float | None = None,
    marked_times_utc: tuple[datetime, ...] = (),
) -> bytes:
    """Render one deterministic narrowband light-curve PNG payload."""

    return _figure_to_png(
        build_narrowband_figure(
            result,
            display_mode=display_mode,
            x_tick_interval_seconds=x_tick_interval_seconds,
            marked_times_utc=marked_times_utc,
        ),
        dpi=dpi,
    )


def build_dart_artifact_filenames(
    summary: DartDatasetSummary,
    product_keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return the ordered, observation-time-based DART export filenames."""

    start_time, end_time = summary.time_range_utc
    all_filenames = {
        DYNAMIC_SPECTRUM_PRODUCT_KEY: build_image_filename(
            ImageFilenameSpec(
                sequence=1,
                start_time=start_time,
                end_time=end_time,
                instrument="dart",
                polarization="stokes_i_v_over_i",
                product="dynamic_spectrum",
            )
        ),
        SELECTED_SPECTRUM_PRODUCT_KEY: build_image_filename(
            ImageFilenameSpec(
                sequence=2,
                start_time=start_time,
                end_time=end_time,
                instrument="dart",
                polarization="stokes_i_v_over_i",
                product="selected_spectrum",
            )
        ),
        LIGHTCURVE_PRODUCT_KEY: build_image_filename(
            ImageFilenameSpec(
                sequence=3,
                start_time=start_time,
                end_time=end_time,
                instrument="dart",
                polarization="stokes_i",
                product="narrowband_lightcurve",
            )
        ),
    }
    selected_keys = _normalize_product_keys(product_keys)
    return {key: all_filenames[key] for key in selected_keys}


def build_zip_bytes(
    artifacts: dict[str, bytes],
    *,
    filenames: dict[str, str] | None = None,
) -> bytes:
    """Package prepared artifacts without regenerating either plot."""

    resolved_names = _validate_artifacts(artifacts, filenames=filenames)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for product_key, filename in resolved_names.items():
            archive.writestr(filename, artifacts[product_key])
    return buffer.getvalue()


def save_artifacts(
    artifacts: dict[str, bytes],
    output_directory: str | Path,
    *,
    filenames: dict[str, str] | None = None,
) -> Path:
    """Write prepared bytes into a new run directory without overwriting."""

    resolved_names = _validate_artifacts(artifacts, filenames=filenames)
    base = Path(output_directory).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_directory = _allocate_run_directory(base)
    for product_key, filename in resolved_names.items():
        payload = artifacts[product_key]
        (run_directory / filename).write_bytes(payload)
    return run_directory


def main(argv: list[str] | None = None) -> int:
    """Run the standalone Streamlit application."""

    _run_streamlit_app(argv)
    return 0


def _run_streamlit_app(argv: list[str] | None = None) -> None:
    import streamlit as st

    args, _unknown = build_parser().parse_known_args(argv)
    st.set_page_config(page_title="DART Spectrogram Analysis", layout="wide")
    st.title("DART Spectrogram Analysis")
    st.caption("Direct Stokes I dB and fractional circular polarization analysis")

    layout = RuntimeLayout.discover()
    local_root = layout.local_root
    try:
        allowed_roots = resolve_streamlit_allowed_roots(args.allowed_roots)
    except AllowedRootPolicyError as exc:
        allowed_roots = ()
        st.error(f"Path configuration error: {exc}")
    if not allowed_roots:
        st.error(
            "No valid allowed roots are configured. Path browsing and local path "
            "operations are disabled."
        )
    path_policy = PathAccessPolicy.create(
        allowed_roots,
        protected_output_roots=(layout.outputs_dir / "dart_spectrogram",),
        base_directory=local_root,
    )
    ui_store = frontend_state_store("dart-spectrogram", layout=layout)
    render_streamlit_theme(
        st,
        frontend_id="dart-spectrogram",
        state_store=ui_store,
        path_memory=frontend_path_memory(path_policy.output_roots, layout=layout),
    )
    bind_streamlit_fields(
        st,
        ui_store,
        frontend_id="dart-spectrogram",
        field_keys=DART_UI_FIELD_KEYS,
    )

    default_input = str(args.input_dir or "")
    default_output = str(args.output_dir or layout.outputs_dir / "dart_spectrogram")
    st.subheader("1. Load Dataset")
    input_directory = render_native_path_input(
        st,
        "DART FITS directory",
        key="input_dir",
        initial_value=default_input,
        roots=path_policy.input_roots,
        kind="directory",
        placeholder="Choose an allowed spectrogram folder",
        frontend_id="dart-spectrogram",
        operation="load-dataset",
        state_store=ui_store,
    )
    if st.button("Load Dataset", type="primary", key="load_dataset"):
        try:
            validated_input = path_policy.input_directory(input_directory)
            summary = inspect_dart_dataset(validated_input)
        except Exception as exc:  # Streamlit must keep the page interactive.
            _clear_dataset_state(st)
            st.error(str(exc))
        else:
            _clear_result_state(st)
            st.session_state["dart_dataset_summary"] = summary
            st.session_state["dart_dataset_signature"] = _dataset_signature(summary)
            st.success("Dataset loaded and validated.")

    summary = st.session_state.get("dart_dataset_summary")
    if not _is_dataset_summary(summary):
        st.info("Enter a local directory and load the four DART FITS files.")
        return
    if not _same_directory(input_directory, summary.directory):
        st.warning("The directory changed. Load Dataset again before processing.")
        return
    _render_dataset_summary(st, summary)

    st.subheader("2. Configure")
    controls = _render_configuration(st, summary)
    current_request: dict[str, Any] | None = None
    current_signature: str | None = None
    try:
        current_request = _request_from_controls(controls)
        current_signature = _request_signature(summary, current_request)
    except ValueError:
        pass

    st.subheader("3. Generate Figures")
    if st.button("Generate Figures", type="primary", key="generate_figures"):
        try:
            request = _request_from_controls(controls)
            request_signature = _request_signature(summary, request)
            with st.spinner("Reading FITS windows and rendering PNG files..."):
                full_window = read_dart_spectrogram_window(
                    summary.directory,
                    frequency_range_mhz=None,
                    time_range_utc=None,
                    max_frequency_samples=request["max_frequency_samples"],
                    max_time_samples=request["max_time_samples"],
                    chunk_memory_mb=request["chunk_memory_mb"],
                )
                display_limits = resolve_display_limits(
                    full_window,
                    display_mode=request["display_mode"],
                    limit_mode=request["limit_mode"],
                    stokes_i_bounds=request["stokes_i_bounds"],
                )
                selected_frequency_range: tuple[float, float] | None = None
                selected_time_range: tuple[datetime, datetime] | None = None
                omitted_marked_times = 0
                if request["analysis_mode"] == FULL_SPECTRUM_MODE:
                    artifacts = {
                        DYNAMIC_SPECTRUM_PRODUCT_KEY: build_dynamic_spectrum_png(
                            full_window,
                            dpi=request["dpi"],
                            display_mode=request["display_mode"],
                            display_limits=display_limits,
                            region_label="Full observation",
                            x_tick_interval_seconds=request["x_tick_interval_seconds"],
                            marked_times_utc=request["marked_times_utc"],
                        )
                    }
                else:
                    narrowband = extract_dart_narrowband_lightcurves(
                        summary.directory,
                        request["center_frequencies_mhz"],
                        request["bandwidth_mhz"],
                        time_range_utc=request["time_range_utc"],
                    )
                    selected_frequency_range = resolve_selected_frequency_range(
                        narrowband,
                        summary.frequency_range_mhz,
                        request["frequency_range_mhz"],
                    )
                    selected_window = read_dart_spectrogram_window(
                        summary.directory,
                        frequency_range_mhz=selected_frequency_range,
                        time_range_utc=request["time_range_utc"],
                        max_frequency_samples=request["max_frequency_samples"],
                        max_time_samples=request["max_time_samples"],
                        chunk_memory_mb=request["chunk_memory_mb"],
                    )
                    selected_time_range = (
                        selected_window.time_utc[0],
                        selected_window.time_utc[-1],
                    )
                    omitted_marked_times = _count_marked_times_outside_range(
                        request["marked_times_utc"],
                        selected_time_range,
                    )
                    artifacts = {
                        DYNAMIC_SPECTRUM_PRODUCT_KEY: build_dynamic_spectrum_png(
                            full_window,
                            narrowband,
                            dpi=request["dpi"],
                            display_mode=request["display_mode"],
                            display_limits=display_limits,
                            selected_frequency_range_mhz=selected_frequency_range,
                            selected_time_range_utc=selected_time_range,
                            region_label="Full observation",
                            x_tick_interval_seconds=request["x_tick_interval_seconds"],
                            marked_times_utc=request["marked_times_utc"],
                        ),
                        SELECTED_SPECTRUM_PRODUCT_KEY: build_dynamic_spectrum_png(
                            selected_window,
                            narrowband,
                            dpi=request["dpi"],
                            display_mode=request["display_mode"],
                            display_limits=display_limits,
                            region_label=(
                                "Selected region "
                                f"{selected_frequency_range[0]:g}-"
                                f"{selected_frequency_range[1]:g} MHz"
                            ),
                            x_tick_interval_seconds=request["x_tick_interval_seconds"],
                            marked_times_utc=request["marked_times_utc"],
                        ),
                        LIGHTCURVE_PRODUCT_KEY: build_narrowband_png(
                            narrowband,
                            dpi=request["dpi"],
                            display_mode=request["display_mode"],
                            x_tick_interval_seconds=request["x_tick_interval_seconds"],
                            marked_times_utc=request["marked_times_utc"],
                        ),
                    }
                artifact_filenames = build_dart_artifact_filenames(
                    summary,
                    product_keys=artifacts,
                )
                zip_payload = (
                    None
                    if request["analysis_mode"] == FULL_SPECTRUM_MODE
                    else build_zip_bytes(
                        artifacts,
                        filenames=artifact_filenames,
                    )
                )
            st.session_state["dart_result_signature"] = request_signature
            st.session_state["dart_render_signature"] = _render_signature(
                request_signature,
                request["analysis_mode"],
                tuple(artifacts),
                selected_frequency_range,
                selected_time_range,
                display_limits,
            )
            st.session_state["dart_result_context"] = {
                "analysis_mode": request["analysis_mode"],
                "artifact_keys": tuple(artifacts),
                "selected_frequency_range_mhz": selected_frequency_range,
                "selected_time_range_utc": selected_time_range,
                "display_limits": display_limits,
                "omitted_marked_times": omitted_marked_times,
            }
            st.session_state["dart_artifacts"] = artifacts
            st.session_state["dart_artifact_filenames"] = artifact_filenames
            if zip_payload is None:
                st.session_state.pop("dart_zip_payload", None)
            else:
                st.session_state["dart_zip_payload"] = zip_payload
            st.session_state.pop("dart_saved_directory", None)
            current_request = request
            current_signature = request_signature
            if request["analysis_mode"] == FULL_SPECTRUM_MODE:
                st.success("Full-observation spectrum generated from the FITS data.")
            else:
                st.success("Figures generated from the selected FITS data.")
        except Exception as exc:  # Keep validation failures in the page.
            st.error(str(exc))

    artifacts = st.session_state.get("dart_artifacts")
    artifact_filenames = st.session_state.get("dart_artifact_filenames")
    result_signature = st.session_state.get("dart_result_signature")
    result_context = st.session_state.get("dart_result_context")
    render_signature = st.session_state.get("dart_render_signature")
    if not isinstance(artifacts, dict) or not isinstance(artifact_filenames, dict):
        st.info("Generate figures to open the result and export sections.")
        return
    if current_signature is None or current_signature != result_signature:
        st.warning("Parameters changed. Generate Figures again to refresh results.")
        return
    try:
        expected_render_signature = _render_signature(
            result_signature,
            result_context["analysis_mode"],
            result_context["artifact_keys"],
            result_context["selected_frequency_range_mhz"],
            result_context["selected_time_range_utc"],
            result_context["display_limits"],
        )
    except (KeyError, TypeError, ValueError):
        st.error("Generated figure metadata is incomplete. Generate Figures again.")
        return
    if render_signature != expected_render_signature:
        st.error("Generated figure signature is invalid. Generate Figures again.")
        return
    try:
        _validate_artifacts(artifacts, filenames=artifact_filenames)
    except ValueError as exc:
        st.error(f"Generated figure artifacts are invalid: {exc}")
        return

    result_mode = _normalize_analysis_mode(result_context["analysis_mode"])
    omitted_marked_times = int(result_context.get("omitted_marked_times", 0))
    if omitted_marked_times:
        st.warning(
            f"{omitted_marked_times} marked UTC time(s) are outside the selected "
            "time window and appear only in the full-observation spectrum."
        )

    st.subheader("4. Results")
    if result_mode == FULL_SPECTRUM_MODE:
        st.image(
            artifacts[DYNAMIC_SPECTRUM_PRODUCT_KEY],
            width="stretch",
        )
    else:
        full_spectrum_tab, selected_spectrum_tab, lightcurve_tab = st.tabs(
            ["Full Observation", "Selected Region", "Narrowband Intensity"]
        )
        with full_spectrum_tab:
            st.image(
                artifacts[DYNAMIC_SPECTRUM_PRODUCT_KEY],
                width="stretch",
            )
        with selected_spectrum_tab:
            st.image(
                artifacts[SELECTED_SPECTRUM_PRODUCT_KEY],
                width="stretch",
            )
        with lightcurve_tab:
            st.image(
                artifacts[LIGHTCURVE_PRODUCT_KEY],
                width="stretch",
            )

    st.subheader("5. Export")
    if result_mode == FULL_SPECTRUM_MODE:
        st.download_button(
            "Download Spectrum PNG",
            data=artifacts[DYNAMIC_SPECTRUM_PRODUCT_KEY],
            file_name=artifact_filenames[DYNAMIC_SPECTRUM_PRODUCT_KEY],
            mime="image/png",
            on_click="ignore",
            key="download_spectrum",
            width="stretch",
        )
    else:
        download_columns = st.columns(4)
        with download_columns[0]:
            st.download_button(
                "Download Spectrum PNG",
                data=artifacts[DYNAMIC_SPECTRUM_PRODUCT_KEY],
                file_name=artifact_filenames[DYNAMIC_SPECTRUM_PRODUCT_KEY],
                mime="image/png",
                on_click="ignore",
                key="download_spectrum",
                width="stretch",
            )
        with download_columns[1]:
            st.download_button(
                "Download Selected PNG",
                data=artifacts[SELECTED_SPECTRUM_PRODUCT_KEY],
                file_name=artifact_filenames[SELECTED_SPECTRUM_PRODUCT_KEY],
                mime="image/png",
                on_click="ignore",
                key="download_selected_spectrum",
                width="stretch",
            )
        with download_columns[2]:
            st.download_button(
                "Download Light Curves PNG",
                data=artifacts[LIGHTCURVE_PRODUCT_KEY],
                file_name=artifact_filenames[LIGHTCURVE_PRODUCT_KEY],
                mime="image/png",
                on_click="ignore",
                key="download_lightcurves",
                width="stretch",
            )
        with download_columns[3]:
            st.download_button(
                "Download All as ZIP",
                data=st.session_state["dart_zip_payload"],
                file_name=ZIP_FILENAME,
                mime="application/zip",
                on_click="ignore",
                key="download_zip",
                width="stretch",
            )

    output_directory = render_native_path_input(
        st,
        "Local output directory",
        key="output_dir",
        initial_value=default_output,
        roots=path_policy.output_roots,
        kind="directory",
        frontend_id="dart-spectrogram",
        operation="save-artifacts",
        state_store=ui_store,
    )
    if st.button(
        "Save PNG File" if result_mode == FULL_SPECTRUM_MODE else "Save PNG Files",
        key="save_local",
        width="stretch",
    ):
        try:
            validated_output = path_policy.output_directory(output_directory)
            saved = save_artifacts(
                artifacts,
                validated_output,
                filenames=artifact_filenames,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["dart_saved_directory"] = str(saved)
    saved_directory = st.session_state.get("dart_saved_directory")
    if saved_directory:
        st.success(f"Saved exact generated PNG bytes to {saved_directory}")


def _render_dataset_summary(st: Any, summary: DartDatasetSummary) -> None:
    frequency_low, frequency_high = summary.frequency_range_mhz
    time_start, time_end = summary.time_range_utc
    st.dataframe(
        [
            {
                "Field": "Matrix",
                "Value": f"{summary.matrix_shape[0]} x {summary.matrix_shape[1]}",
            },
            {
                "Field": "Frequency",
                "Value": (
                    f"{frequency_low:.6f}-{frequency_high:.6f} MHz "
                    f"({summary.frequency_samples} samples)"
                ),
            },
            {
                "Field": "UTC time",
                "Value": (
                    f"{time_start.isoformat()} to {time_end.isoformat()} "
                    f"({summary.time_samples} samples)"
                ),
            },
        ],
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        [
            {"Role": "Stokes I dB", "File": summary.files.stokes_i_db.name},
            {"Role": "Stokes V/I", "File": summary.files.stokes_v_over_i.name},
            {"Role": "Frequency", "File": summary.files.frequency.name},
            {"Role": "UTC time", "File": summary.files.time.name},
        ],
        hide_index=True,
        width="stretch",
    )


def _render_configuration(
    st: Any,
    summary: DartDatasetSummary,
) -> dict[str, Any]:
    analysis_choice = st.radio(
        "Generation mode",
        options=(
            "Full spectrum only",
            "Full spectrum + selected region + narrowband intensity",
        ),
        horizontal=True,
        key="analysis_mode",
        help=(
            "Full spectrum only exports the complete frequency and UTC observation "
            "without requiring a selected band. Enable the second mode to also "
            "create a selected-region spectrum and narrowband intensity curves."
        ),
    )
    analysis_mode = (
        FULL_SPECTRUM_MODE
        if analysis_choice == "Full spectrum only"
        else FULL_AND_NARROWBAND_MODE
    )

    display_columns = st.columns(2)
    with display_columns[0]:
        display_choice = st.radio(
            "Stokes I display",
            options=("Log (dB)", "Relative linear"),
            horizontal=True,
            key="display_mode",
            help=(
                "Log shows the file-provided Stokes I dB values. Relative linear "
                "uses 10^(dB/10) only for plotting, with 0 dB = 1."
            ),
        )
    display_mode = "db" if display_choice == "Log (dB)" else "linear"
    with display_columns[1]:
        limit_mode = st.radio(
            "Stokes I color limits",
            options=("Percentile", "Direct"),
            horizontal=True,
            key="limit_mode",
        )
    if display_mode == "linear":
        st.caption(
            "Relative linear Stokes I is dimensionless (0 dB = 1); it is not "
            "calibrated physical flux density."
        )

    limit_columns = st.columns(2)
    if limit_mode == "Percentile":
        with limit_columns[0]:
            stokes_i_low = st.number_input(
                "Stokes I lower percentile",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.1,
                key="stokes_i_percentile_low",
            )
        with limit_columns[1]:
            stokes_i_high = st.number_input(
                "Stokes I upper percentile",
                min_value=0.0,
                max_value=100.0,
                value=99.5,
                step=0.1,
                key="stokes_i_percentile_high",
            )
    else:
        intensity_unit = "dB" if display_mode == "db" else "relative"
        default_stokes_limits = (0.0, 50.0) if display_mode == "db" else (1.0, 1e5)
        with limit_columns[0]:
            stokes_i_low = st.number_input(
                f"Stokes I minimum ({intensity_unit})",
                value=default_stokes_limits[0],
                format="%.6f",
                key="stokes_i_direct_low",
            )
        with limit_columns[1]:
            stokes_i_high = st.number_input(
                f"Stokes I maximum ({intensity_unit})",
                value=default_stokes_limits[1],
                format="%.6f",
                key="stokes_i_direct_high",
            )

    time_axis_columns = st.columns([1, 2])
    with time_axis_columns[0]:
        x_tick_mode = st.radio(
            "X-axis tick spacing",
            options=("Auto", "Custom"),
            horizontal=True,
            key="x_tick_mode",
            help=(
                "Auto uses Matplotlib date ticks. Custom starts at the first "
                "visible sample and advances by the requested number of seconds."
            ),
        )
        x_tick_interval_seconds: float | None = None
        if x_tick_mode == "Custom":
            x_tick_interval_seconds = float(
                st.number_input(
                    "X-axis interval (seconds)",
                    min_value=0.001,
                    value=10.0,
                    step=0.001,
                    format="%.3f",
                    key="x_tick_interval_seconds",
                )
            )
    with time_axis_columns[1]:
        marked_times_text = st.text_input(
            "Marked UTC times (comma separated, optional)",
            value="",
            key="marked_times",
            placeholder="04:45:01.5, 2025-01-24T04:45:02Z",
            help=(
                "Accepts HH:MM:SS[.fraction] or ISO-8601. Time-only values use "
                "the unique matching date inside this observation."
            ),
        )

    centers_text: str | None = None
    bandwidth_mhz: float | None = None
    frequency_range: tuple[float, float] | None = None
    time_range: tuple[str, str] | None = None
    if analysis_mode == FULL_AND_NARROWBAND_MODE:
        frequency_columns = st.columns([3, 1])
        with frequency_columns[0]:
            centers_text = st.text_input(
                "Center frequencies (MHz, comma separated)",
                value="149",
                key="center_frequencies",
            )
        with frequency_columns[1]:
            bandwidth_mhz = st.number_input(
                "Total bandwidth (MHz)",
                min_value=0.000001,
                value=2.0,
                step=0.1,
                format="%.6f",
                key="bandwidth_mhz",
            )

        limit_frequency = st.checkbox(
            "Override selected-region frequency range",
            value=False,
            key="limit_frequency",
            help="The override must contain every requested narrowband.",
        )
        if limit_frequency:
            lower, upper = summary.frequency_range_mhz
            display_columns = st.columns(2)
            with display_columns[0]:
                frequency_start = st.number_input(
                    "Display frequency start (MHz)",
                    min_value=lower,
                    max_value=upper,
                    value=lower,
                    format="%.6f",
                    key="frequency_start",
                )
            with display_columns[1]:
                frequency_end = st.number_input(
                    "Display frequency end (MHz)",
                    min_value=lower,
                    max_value=upper,
                    value=upper,
                    format="%.6f",
                    key="frequency_end",
                )
            frequency_range = (float(frequency_start), float(frequency_end))

        limit_time = st.checkbox(
            "Limit UTC time display",
            value=False,
            key="limit_time",
        )
        if limit_time:
            start, end = summary.time_range_utc
            time_columns = st.columns(2)
            with time_columns[0]:
                time_start = st.text_input(
                    "UTC start",
                    value=start.isoformat(),
                    key="time_start",
                )
            with time_columns[1]:
                time_end = st.text_input(
                    "UTC end",
                    value=end.isoformat(),
                    key="time_end",
                )
            time_range = (time_start.strip(), time_end.strip())
    else:
        st.caption(
            "The exported spectrum covers the complete observed frequency range "
            "and UTC time range."
        )

    with st.expander("Display sampling and PNG settings", expanded=False):
        advanced_columns = st.columns(4)
        with advanced_columns[0]:
            max_frequency_samples = st.number_input(
                "Frequency samples",
                min_value=100,
                max_value=5000,
                value=1600,
                step=100,
                key="max_frequency_samples",
            )
        with advanced_columns[1]:
            max_time_samples = st.number_input(
                "Time samples",
                min_value=100,
                max_value=5000,
                value=1600,
                step=100,
                key="max_time_samples",
            )
        with advanced_columns[2]:
            chunk_memory_mb = st.number_input(
                "Chunk memory (MiB)",
                min_value=8,
                max_value=512,
                value=64,
                step=8,
                key="chunk_memory_mb",
            )
        with advanced_columns[3]:
            dpi = st.number_input(
                "PNG DPI",
                min_value=100,
                max_value=300,
                value=160,
                step=10,
                key="dpi",
            )
    return {
        "analysis_mode": analysis_mode,
        "display_mode": display_mode,
        "limit_mode": str(limit_mode).lower(),
        "stokes_i_bounds": (float(stokes_i_low), float(stokes_i_high)),
        "centers_text": centers_text,
        "bandwidth_mhz": bandwidth_mhz,
        "frequency_range_mhz": frequency_range,
        "time_range_utc": time_range,
        "observation_time_range_utc": summary.time_range_utc,
        "x_tick_mode": x_tick_mode,
        "x_tick_interval_seconds": x_tick_interval_seconds,
        "marked_times_text": marked_times_text,
        "max_frequency_samples": max_frequency_samples,
        "max_time_samples": max_time_samples,
        "chunk_memory_mb": chunk_memory_mb,
        "dpi": dpi,
    }


def _request_from_controls(controls: dict[str, Any]) -> dict[str, Any]:
    analysis_mode = _normalize_analysis_mode(controls["analysis_mode"])
    display_mode = _normalize_display_mode(controls["display_mode"])
    limit_mode = str(controls["limit_mode"]).strip().lower()
    stokes_i_bounds = tuple(float(value) for value in controls["stokes_i_bounds"])
    if limit_mode == "percentile":
        _validate_percentile_limits(stokes_i_bounds, "Stokes I percentile")
    elif limit_mode == "direct":
        _validate_direct_limits(stokes_i_bounds, "Stokes I display")
    else:
        raise ValueError("Color limit mode must be Percentile or Direct")
    x_tick_mode = str(controls["x_tick_mode"]).strip().lower()
    if x_tick_mode == "auto":
        x_tick_interval_seconds = None
    elif x_tick_mode == "custom":
        x_tick_interval_seconds = _validate_x_tick_interval(
            controls["x_tick_interval_seconds"],
            controls["observation_time_range_utc"],
        )
    else:
        raise ValueError("X-axis tick spacing must be Auto or Custom")
    marked_times = parse_marked_times(
        controls["marked_times_text"],
        controls["observation_time_range_utc"],
    )
    centers: tuple[float, ...] = ()
    bandwidth: float | None = None
    frequency_range: tuple[float, float] | None = None
    effective_frequency_range: tuple[float, float] | None = None
    time_range: tuple[str, str] | None = None
    if analysis_mode == FULL_AND_NARROWBAND_MODE:
        centers = parse_center_frequencies(controls["centers_text"])
        bandwidth = float(controls["bandwidth_mhz"])
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Total bandwidth must be greater than zero")
        frequency_range = controls["frequency_range_mhz"]
        if frequency_range is not None:
            frequency_range = _validate_direct_limits(
                frequency_range,
                "Selected-region frequency",
            )
        requested_frequency_range = (
            float(min(centers) - bandwidth / 2.0),
            float(max(centers) + bandwidth / 2.0),
        )
        effective_frequency_range = frequency_range or requested_frequency_range
        time_range = controls["time_range_utc"]
        if time_range is not None and (not time_range[0] or not time_range[1]):
            raise ValueError("Both UTC time limits are required")
    return {
        "analysis_mode": analysis_mode,
        "display_mode": display_mode,
        "limit_mode": limit_mode,
        "stokes_i_bounds": stokes_i_bounds,
        "center_frequencies_mhz": centers,
        "bandwidth_mhz": bandwidth,
        "frequency_range_mhz": frequency_range,
        "effective_selected_frequency_range_mhz": effective_frequency_range,
        "time_range_utc": time_range,
        "x_tick_interval_seconds": x_tick_interval_seconds,
        "marked_times_utc": marked_times,
        "max_frequency_samples": int(controls["max_frequency_samples"]),
        "max_time_samples": int(controls["max_time_samples"]),
        "chunk_memory_mb": int(controls["chunk_memory_mb"]),
        "dpi": int(controls["dpi"]),
    }


def _request_signature(
    summary: DartDatasetSummary,
    request: dict[str, Any],
) -> str:
    payload = {
        "dataset": _dataset_signature(summary),
        "request": request,
        "version": "dart-spectrogram-render-v5",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_signature(
    request_signature: str,
    analysis_mode: str,
    artifact_keys: Iterable[str],
    selected_frequency_range_mhz: tuple[float, float] | None,
    selected_time_range_utc: tuple[datetime, datetime] | None,
    display_limits: DartDisplayLimits,
) -> str:
    payload = {
        "request_signature": request_signature,
        "analysis_mode": _normalize_analysis_mode(analysis_mode),
        "artifact_keys": _normalize_product_keys(artifact_keys),
        "selected_frequency_range_mhz": selected_frequency_range_mhz,
        "selected_time_range_utc": selected_time_range_utc,
        "resolved_stokes_i_limits": display_limits.stokes_i,
        "stokes_v_over_i_limits": (-1.0, 1.0),
        "version": "dart-spectrogram-render-v5",
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_signature(summary: DartDatasetSummary) -> str:
    identities = []
    for path in (
        summary.files.stokes_i_db,
        summary.files.stokes_v_over_i,
        summary.files.frequency,
        summary.files.time,
    ):
        stat = path.stat()
        identities.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    encoded = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clear_dataset_state(st: Any) -> None:
    st.session_state.pop("dart_dataset_summary", None)
    st.session_state.pop("dart_dataset_signature", None)
    _clear_result_state(st)


def _clear_result_state(st: Any) -> None:
    for key in (
        "dart_result_signature",
        "dart_render_signature",
        "dart_result_context",
        "dart_artifacts",
        "dart_artifact_filenames",
        "dart_zip_payload",
        "dart_saved_directory",
    ):
        st.session_state.pop(key, None)


def _same_directory(value: str, expected: Path) -> bool:
    if not str(value).strip():
        return False
    return Path(value).expanduser().resolve() == expected


def _is_dataset_summary(value: Any) -> bool:
    required_fields = (
        "directory",
        "files",
        "matrix_shape",
        "frequency_range_mhz",
        "frequency_samples",
        "time_range_utc",
        "time_samples",
    )
    return value is not None and all(
        hasattr(value, field_name) for field_name in required_fields
    )


def _time_row_to_datetime(row: np.ndarray) -> datetime:
    if len(row) != 6 or not np.all(np.isfinite(row)):
        raise ValueError(f"Invalid DART time row: {np.asarray(row).tolist()}")
    components = [int(round(float(value))) for value in row[:5]]
    if not np.allclose(row[:5], components, rtol=0.0, atol=1e-6):
        raise ValueError(f"Invalid integer DART time fields: {row.tolist()}")
    year, month, day, hour, minute = components
    if 0 <= year < 100:
        year += 2000
    second = float(row[5])
    return datetime(year, month, day, hour, minute, tzinfo=UTC) + timedelta(
        seconds=second
    )


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("UTC time values must be datetime instances")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_time_only(value: str) -> time | None:
    for template in ("%H:%M:%S.%f", "%H:%M:%S"):
        try:
            return datetime.strptime(value, template).time()
        except ValueError:
            continue
    return None


def _parse_iso_utc(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.upper().endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid marked UTC time {value!r}; use HH:MM:SS[.fraction] " "or ISO-8601"
        ) from exc
    return _as_utc(parsed)


def _validate_x_tick_interval(
    value: object,
    time_range_utc: tuple[datetime, datetime],
) -> float:
    try:
        interval = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("X-axis interval seconds must be numeric") from exc
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("X-axis interval seconds must be finite and greater than zero")
    start, end = (_as_utc(item) for item in time_range_utc)
    if start > end:
        raise ValueError("Time axis must start before it ends")
    tick_count = int(math.floor((end - start).total_seconds() / interval)) + 1
    if tick_count > _MAX_CUSTOM_X_TICKS:
        raise ValueError(
            f"X-axis interval would create {tick_count} major ticks; increase the "
            f"interval to keep at most {_MAX_CUSTOM_X_TICKS}"
        )
    return interval


def _configure_time_axis(
    axis: Any,
    time_utc: Sequence[datetime],
    *,
    x_tick_interval_seconds: float | None,
) -> None:
    if not time_utc:
        raise ValueError("The UTC time axis is empty")
    values = tuple(_as_utc(item) for item in time_utc)
    axis.xaxis_date()
    if x_tick_interval_seconds is None:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    else:
        interval = _validate_x_tick_interval(
            x_tick_interval_seconds,
            (values[0], values[-1]),
        )
        span_seconds = (values[-1] - values[0]).total_seconds()
        tick_count = int(math.floor(span_seconds / interval)) + 1
        tick_times = tuple(
            values[0] + timedelta(seconds=index * interval)
            for index in range(tick_count)
        )
        locator = FixedLocator(mdates.date2num(tick_times))
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))


def _visible_marked_times(
    marked_times_utc: Iterable[datetime],
    visible_time_utc: Sequence[datetime],
) -> tuple[datetime, ...]:
    if not visible_time_utc:
        return ()
    start = _as_utc(visible_time_utc[0])
    end = _as_utc(visible_time_utc[-1])
    return tuple(
        marked
        for marked in sorted({_as_utc(item) for item in marked_times_utc})
        if start <= marked <= end
    )


def _count_marked_times_outside_range(
    marked_times_utc: Iterable[datetime],
    visible_range_utc: tuple[datetime, datetime],
) -> int:
    start, end = (_as_utc(item) for item in visible_range_utc)
    return sum(not (start <= _as_utc(marked) <= end) for marked in marked_times_utc)


def _format_marked_time(value: datetime) -> str:
    return _as_utc(value).strftime("%H:%M:%S.%f").rstrip("0").rstrip(".")


def _add_time_markers(
    axes: Sequence[Any],
    marked_times_utc: Iterable[datetime],
    *,
    label_axis: Any,
) -> None:
    path_effects = [
        mpatheffects.Stroke(linewidth=3.2, foreground=_TIME_MARKER_EDGE_COLOR),
        mpatheffects.Normal(),
    ]
    for marked in marked_times_utc:
        for axis in axes:
            line = axis.axvline(
                marked,
                ymin=0.0,
                ymax=_TIME_MARKER_HEIGHT,
                color=_TIME_MARKER_COLOR,
                linewidth=1.6,
                zorder=8,
            )
            line.set_path_effects(path_effects)
        label = label_axis.annotate(
            _format_marked_time(marked),
            xy=(marked, _TIME_MARKER_HEIGHT),
            xycoords=("data", "axes fraction"),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7,
            color=_TIME_MARKER_EDGE_COLOR,
            annotation_clip=True,
            zorder=9,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "#fef3c7",
                "edgecolor": _TIME_MARKER_EDGE_COLOR,
                "alpha": 0.9,
                "linewidth": 0.6,
            },
        )
        label.set_in_layout(True)


def _validate_plot_inputs(
    window: DartSpectrogramWindow,
    narrowband: DartNarrowbandResult | None,
) -> None:
    expected = (window.frequency_mhz.size, len(window.time_utc))
    if window.stokes_i_db.shape != expected:
        raise ValueError(
            f"Stokes I display shape {window.stokes_i_db.shape} != {expected}"
        )
    if window.stokes_v_over_i.shape != expected:
        raise ValueError(
            f"Stokes V/I display shape {window.stokes_v_over_i.shape} != {expected}"
        )
    if narrowband is not None and not narrowband.curves:
        raise ValueError("At least one narrowband curve is required")


def _axis_extent(
    values: np.ndarray,
    *,
    fallback_half_step: float,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return (
            float(values[0] - fallback_half_step),
            float(values[0] + fallback_half_step),
        )
    return (
        float(values[0] - (values[1] - values[0]) / 2.0),
        float(values[-1] + (values[-1] - values[-2]) / 2.0),
    )


def _normalize_display_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"db", "log", "logarithmic"}:
        return "db"
    if normalized in {"linear", "non-log", "relative-linear"}:
        return "linear"
    raise ValueError("Stokes I display mode must be Log or relative linear")


def _normalize_analysis_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "full_spectrum": FULL_SPECTRUM_MODE,
        "full_spectrum_only": FULL_SPECTRUM_MODE,
        "full_and_narrowband": FULL_AND_NARROWBAND_MODE,
        "full_analysis": FULL_AND_NARROWBAND_MODE,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "Generation mode must be full spectrum only or full narrowband analysis"
        ) from exc


def _normalize_product_keys(product_keys: Iterable[str] | None) -> tuple[str, ...]:
    if product_keys is None:
        return _FULL_ANALYSIS_PRODUCT_KEYS
    requested = tuple(str(key) for key in product_keys)
    requested_set = set(requested)
    if len(requested) != len(requested_set):
        raise ValueError("Prepared artifact product keys must be unique")
    if requested_set == set(_FULL_SPECTRUM_PRODUCT_KEYS):
        return _FULL_SPECTRUM_PRODUCT_KEYS
    if requested_set == set(_FULL_ANALYSIS_PRODUCT_KEYS):
        return _FULL_ANALYSIS_PRODUCT_KEYS
    raise ValueError(
        "Prepared artifacts must contain either the full spectrum only or all "
        "three analysis products"
    )


def _display_stokes_i(values: np.ndarray, display_mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mode = _normalize_display_mode(display_mode)
    if mode == "db":
        displayed = values.copy()
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            displayed = np.power(10.0, values / 10.0)
    displayed[~np.isfinite(displayed)] = np.nan
    if not np.any(np.isfinite(displayed)):
        raise ValueError("Stokes I display data contains no finite values")
    return displayed


def _stokes_i_label(display_mode: str) -> str:
    if _normalize_display_mode(display_mode) == "db":
        return "Stokes I (dB; file-provided log scale)"
    return "Relative Stokes I (dimensionless; 0 dB = 1)"


def _validate_percentile_limits(
    values: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    low, high = _coerce_limit_pair(values, label)
    if low < 0.0 or high > 100.0 or low >= high:
        raise ValueError(
            f"{label} limits must be finite with 0 <= lower < upper <= 100"
        )
    return low, high


def _validate_direct_limits(
    values: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    low, high = _coerce_limit_pair(values, label)
    if low >= high:
        raise ValueError(f"{label} limits must have a lower value below the upper")
    return low, high


def _coerce_limit_pair(
    values: tuple[float, float],
    label: str,
) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{label} limits must contain exactly two values")
    low, high = (float(value) for value in values)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"{label} limits must be finite")
    return low, high


def _selection_annotation(
    frequency_range_mhz: tuple[float, float],
    time_range_utc: tuple[datetime, datetime],
) -> str:
    low, high = frequency_range_mhz
    start, end = time_range_utc
    return (
        f"Selected: {low:g}-{high:g} MHz\n"
        f"{start.astimezone(UTC).isoformat()} to "
        f"{end.astimezone(UTC).isoformat()} UTC"
    )


def _scientific_formatter() -> ScalarFormatter:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))
    formatter.set_useOffset(False)
    return formatter


def _finite_percentile_limits(
    values: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Plot data contains no finite values")
    low, high = np.percentile(finite, [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Could not calculate finite plot limits")
    if math.isclose(float(low), float(high), rel_tol=0.0, abs_tol=1e-12):
        padding = max(abs(float(low)) * 0.01, 1e-6)
        return float(low - padding), float(high + padding)
    return float(low), float(high)


def _curve_colors(count: int) -> list[Any]:
    from matplotlib import colormaps

    colormap = colormaps["tab20"]
    return [colormap(index % colormap.N) for index in range(count)]


def _figure_to_png(figure: Figure, *, dpi: int) -> bytes:
    if int(dpi) <= 0:
        raise ValueError("PNG DPI must be greater than zero")
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=int(dpi),
        facecolor="white",
        metadata={"Software": "solarphysics DART spectrogram tool"},
    )
    figure.clear()
    payload = buffer.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Matplotlib did not produce a valid PNG payload")
    return payload


def _validate_artifacts(
    artifacts: dict[str, bytes],
    *,
    filenames: dict[str, str] | None = None,
) -> dict[str, str]:
    if filenames is None:
        legacy_names = {
            DYNAMIC_SPECTRUM_FILENAME: DYNAMIC_SPECTRUM_FILENAME,
            SELECTED_SPECTRUM_FILENAME: SELECTED_SPECTRUM_FILENAME,
            LIGHTCURVE_FILENAME: LIGHTCURVE_FILENAME,
        }
        if set(artifacts) == {DYNAMIC_SPECTRUM_FILENAME}:
            resolved_names = {
                DYNAMIC_SPECTRUM_FILENAME: DYNAMIC_SPECTRUM_FILENAME,
            }
        elif set(artifacts) == set(legacy_names):
            resolved_names = legacy_names
        else:
            raise ValueError(
                "Prepared artifacts must contain either the full spectrum only or "
                "all three analysis products"
            )
    else:
        resolved_names = dict(filenames)
        _normalize_product_keys(resolved_names)
    expected = set(resolved_names)
    if set(artifacts) != expected:
        raise ValueError(f"Prepared artifacts must contain exactly {sorted(expected)}")
    if len(set(resolved_names.values())) != len(resolved_names):
        raise ValueError("Prepared artifact filenames must be unique")
    for product_key, payload in artifacts.items():
        if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG"):
            raise ValueError(f"Prepared artifact is not valid PNG bytes: {product_key}")
    return resolved_names


def _allocate_run_directory(base: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    names = [f"dart_spectrogram_{stamp}"] + [
        f"dart_spectrogram_{stamp}_{index:03d}" for index in range(2, 1000)
    ]
    for name in names:
        candidate = base / name
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        _LOGGER.info("Allocated DART output directory %s", candidate)
        return candidate
    raise RuntimeError(f"Could not allocate a unique run directory under {base}")


if __name__ == "__main__":
    main()
