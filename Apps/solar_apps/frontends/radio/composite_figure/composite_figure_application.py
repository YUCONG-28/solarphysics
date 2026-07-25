"""Pure rendering and export helpers for the radio composite frontend."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from solar_apps.frontends.radio.source_map.artifacts import data_to_image_pixel
from solar_apps.workflows.common.image_naming import build_scientific_image_filename
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandResult,
    DartSpectrogramWindow,
)
from solar_toolkit.radio.roi_lightcurve import RadioRoi

COMPOSITE_SCHEMA_VERSION = "radio-composite-v1"
MAP_TIME_COLOR = "#c2410c"
ROI_COLOR = "#00d4ff"
_DART_BAND_COLORS = (
    "#ff9f1c",
    "#7bd389",
    "#c77dff",
    "#ff5d8f",
    "#ffd166",
    "#4cc9f0",
    "#f28482",
)


@dataclass(frozen=True, slots=True)
class FrequencyBand:
    """One validated DART frequency band in MHz."""

    low_mhz: float
    high_mhz: float

    def __post_init__(self) -> None:
        low = float(self.low_mhz)
        high = float(self.high_mhz)
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("DART frequency bounds must be finite")
        if low >= high:
            raise ValueError("DART frequency lower bound must be below the upper bound")
        object.__setattr__(self, "low_mhz", low)
        object.__setattr__(self, "high_mhz", high)

    @property
    def center_mhz(self) -> float:
        return (self.low_mhz + self.high_mhz) / 2.0

    @property
    def bandwidth_mhz(self) -> float:
        return self.high_mhz - self.low_mhz

    def validate_observed_range(self, observed: Sequence[float]) -> FrequencyBand:
        values = np.asarray(list(observed), dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError("DART frequency axis contains no finite values")
        observed_low = float(np.min(finite))
        observed_high = float(np.max(finite))
        tolerance = max(1e-9, abs(observed_high - observed_low) * 1e-12)
        if (
            self.low_mhz < observed_low - tolerance
            or self.high_mhz > observed_high + tolerance
        ):
            raise ValueError(
                "Selected DART band is outside the observed frequency range: "
                f"{observed_low:g}-{observed_high:g} MHz"
            )
        selected = finite[(finite >= self.low_mhz) & (finite <= self.high_mhz)]
        if not selected.size:
            raise ValueError(
                "Selected DART band contains no original frequency channel"
            )
        return self

    def to_dict(self) -> dict[str, float]:
        return {
            "low_mhz": self.low_mhz,
            "high_mhz": self.high_mhz,
            "center_mhz": self.center_mhz,
            "bandwidth_mhz": self.bandwidth_mhz,
        }


def centered_frequency_band(
    center_frequency_mhz: float,
    bandwidth_mhz: float,
) -> FrequencyBand:
    """Return one symmetric DART band locked to a radio frequency."""

    center = float(center_frequency_mhz)
    bandwidth = float(bandwidth_mhz)
    if not math.isfinite(center):
        raise ValueError("Radio frequency must be finite")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("DART bandwidth must be a finite value greater than zero")
    half_width = bandwidth / 2.0
    return FrequencyBand(center - half_width, center + half_width)


def build_centered_frequency_bands(
    frequencies_mhz: Sequence[float],
    default_bandwidth_mhz: float,
    bandwidth_overrides_mhz: Mapping[float | str, float] | None = None,
) -> dict[float, FrequencyBand]:
    """Build one symmetric DART band per unique selected radio frequency."""

    frequencies = sorted(
        {float(value) for value in frequencies_mhz if math.isfinite(float(value))}
    )
    if not frequencies:
        raise ValueError("Select at least one radio frequency for DART bands")
    default_width = float(default_bandwidth_mhz)
    if not math.isfinite(default_width) or default_width <= 0:
        raise ValueError("Default DART bandwidth must be greater than zero")
    overrides = dict(bandwidth_overrides_mhz or {})
    result: dict[float, FrequencyBand] = {}
    for frequency in frequencies:
        width = default_width
        for raw_key, raw_width in overrides.items():
            try:
                matches = math.isclose(
                    float(raw_key), frequency, rel_tol=0.0, abs_tol=1e-6
                )
            except (TypeError, ValueError):
                continue
            if matches:
                width = float(raw_width)
                break
        result[frequency] = centered_frequency_band(frequency, width)
    return result


@dataclass(frozen=True, slots=True)
class CompositeArtifactBundle:
    """In-memory composite products and their public filenames."""

    files: Mapping[str, bytes]
    filenames: Mapping[str, str]
    zip_bytes: bytes
    zip_filename: str
    metadata: Mapping[str, Any]
    curve_template: CompositeFrameTemplate | None = None


@dataclass(frozen=True, slots=True)
class CompositeFigureLayout:
    """Pixel-stable layout reused by every frame in one frequency sequence."""

    canvas_size_pixels: tuple[int, int]
    map_size_pixels: tuple[int, int]
    height_ratios: tuple[float, float, float]
    radio_ylim: tuple[float, float]
    dart_ylim: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas_size_pixels": list(self.canvas_size_pixels),
            "map_size_pixels": list(self.map_size_pixels),
            "height_ratios": list(self.height_ratios),
            "radio_ylim": list(self.radio_ylim),
            "dart_ylim": list(self.dart_ylim),
        }


@dataclass(frozen=True, slots=True)
class CompositeFrameRender:
    """One RGB frame plus optional PNG bytes and measured panel bounds."""

    rgb: np.ndarray
    png_bytes: bytes | None
    panel_bounds_pixels: Mapping[str, tuple[int, int, int, int]]
    marker_x_pixels: Mapping[str, int] | None = None


@dataclass(frozen=True, slots=True)
class CompositeFrameTemplate:
    """Marker-free composite canvas reused by one frequency sequence."""

    base_rgb: np.ndarray
    layout: CompositeFigureLayout
    source_map_bounds_pixels: tuple[int, int, int, int]
    panel_bounds_pixels: Mapping[str, tuple[int, int, int, int]]
    time_start_utc: datetime
    time_end_utc: datetime
    dpi: int
    cache_signature: str
    radio_curve_png: bytes
    dart_curve_png: bytes

    def to_metadata(self) -> dict[str, Any]:
        return {
            "cache_signature": self.cache_signature,
            "canvas_size_pixels": list(self.layout.canvas_size_pixels),
            "source_map_bounds_pixels": list(self.source_map_bounds_pixels),
            "panel_bounds_pixels": {
                key: list(value) for key, value in self.panel_bounds_pixels.items()
            },
            "time_start_utc": self.time_start_utc.isoformat(),
            "time_end_utc": self.time_end_utc.isoformat(),
            "marker": {
                "color": MAP_TIME_COLOR,
                "line_width_points": 0.9,
                "line_style": "dashed",
            },
        }


def build_request_signature(
    payload: Mapping[str, Any],
    *,
    source_paths: Iterable[str | Path] = (),
) -> str:
    """Hash material controls plus file size and modification identities."""

    identities = []
    for raw_path in sorted((Path(path).resolve() for path in source_paths), key=str):
        stat = raw_path.stat()
        identities.append(
            {
                "path": str(raw_path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    encoded = json.dumps(
        {"schema": COMPOSITE_SCHEMA_VERSION, "request": payload, "files": identities},
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frequency_band_from_selection(event: Any) -> FrequencyBand | None:
    """Read one vertical MHz range from a Streamlit Plotly selection event."""

    selection = _event_get(event, "selection")
    if selection is None:
        return None
    boxes = _event_get(selection, "box") or _event_get(selection, "boxes") or []
    if isinstance(boxes, Mapping):
        boxes = [boxes]
    for box in boxes:
        y0 = _event_get(box, "y0")
        y1 = _event_get(box, "y1")
        if y0 is None or y1 is None:
            values = _event_get(box, "y")
            if isinstance(values, Sequence) and len(values) >= 2:
                y0, y1 = values[0], values[-1]
        if y0 is not None and y1 is not None and float(y0) != float(y1):
            return FrequencyBand(*sorted((float(y0), float(y1))))
    points = _event_get(selection, "points") or []
    ys = [
        float(value)
        for point in points
        if (value := _event_get(point, "y")) is not None and math.isfinite(float(value))
    ]
    if len(ys) >= 2 and min(ys) < max(ys):
        return FrequencyBand(float(min(ys)), float(max(ys)))
    return None


def select_dart_time_overlap(
    time_utc: Sequence[datetime | str],
    time_start: datetime | str,
    time_end: datetime | str,
) -> tuple[datetime, datetime, bool]:
    """Return actual DART sample bounds inside a requested radio time range.

    The final boolean is true when DART does not cover the entire requested
    range.  No interpolation or extrapolation is performed.
    """

    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    if start >= end:
        raise ValueError("Shared UTC start must be before the end")
    samples = sorted(_utc_datetime(value) for value in time_utc)
    selected = [value for value in samples if start <= value <= end]
    if not selected:
        raise ValueError("The radio time range contains no DART sample")
    partial = start < samples[0] or end > samples[-1]
    return selected[0], selected[-1], partial


def build_dart_selection_figure(
    window: DartSpectrogramWindow,
    *,
    band: FrequencyBand | None = None,
    bands: Mapping[float, FrequencyBand] | None = None,
    active_frequency_mhz: float | None = None,
):
    """Build a downsampled DART spectrum with selectable per-radio bands."""

    import plotly.graph_objects as go

    frequencies = np.asarray(window.frequency_mhz, dtype=float)
    values = np.asarray(window.stokes_i_db, dtype=float)
    if values.shape != (frequencies.size, len(window.time_utc)):
        raise ValueError("DART preview arrays do not share frequency and time axes")
    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=values,
            x=list(window.time_utc),
            y=frequencies,
            colorscale="Turbo",
            colorbar={"title": "Stokes I (dB)"},
            hovertemplate=(
                "UTC=%{x|%Y-%m-%d %H:%M:%S.%L}<br>"
                "frequency=%{y:.4f} MHz<br>Stokes I=%{z:.4g} dB<extra></extra>"
            ),
        )
    )
    time_indices = _sample_indices(len(window.time_utc), target=56)
    frequency_indices = _sample_indices(frequencies.size, target=56)
    grid_x: list[datetime] = []
    grid_y: list[float] = []
    for frequency_index in frequency_indices:
        for time_index in time_indices:
            grid_x.append(window.time_utc[time_index])
            grid_y.append(float(frequencies[frequency_index]))
    figure.add_trace(
        go.Scattergl(
            x=grid_x,
            y=grid_y,
            mode="markers",
            marker={"size": 5, "opacity": 0.01, "color": "white"},
            hoverinfo="skip",
            showlegend=False,
            name="Frequency selection grid",
        )
    )
    displayed_bands = dict(bands or {})
    if band is not None and not displayed_bands:
        displayed_bands[band.center_mhz] = band
    for index, (frequency, frequency_band) in enumerate(
        sorted(displayed_bands.items())
    ):
        active = active_frequency_mhz is not None and math.isclose(
            float(active_frequency_mhz),
            float(frequency),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        color = (
            ROI_COLOR if active else _DART_BAND_COLORS[index % len(_DART_BAND_COLORS)]
        )
        figure.add_hrect(
            y0=frequency_band.low_mhz,
            y1=frequency_band.high_mhz,
            line={"color": color, "width": 3 if active else 1.5},
            fillcolor=(
                "rgba(0, 212, 255, 0.22)" if active else "rgba(255, 255, 255, 0.06)"
            ),
            annotation_text=f"{float(frequency):g} MHz",
            annotation_position="top left",
        )
    figure.update_layout(
        title=(
            "DART dynamic spectrum — drag a horizontal width for the active "
            "radio frequency"
        ),
        xaxis_title="Time (UTC)",
        yaxis_title="Frequency (MHz)",
        dragmode="select",
        height=560,
        margin={"l": 70, "r": 30, "t": 60, "b": 55},
    )
    return figure


def build_source_map_selection_figure(
    image_png: bytes,
    metadata: Mapping[str, Any],
    *,
    roi: RadioRoi | None = None,
    roi_mode: str = "box",
):
    """Build a Plotly ROI selector from one Source Map artifact panel."""

    import plotly.graph_objects as go

    panel = _single_panel(metadata)
    with Image.open(io.BytesIO(image_png)) as source:
        image = source.convert("RGBA")
        width, height = image.size
        left, top, right, bottom = [float(value) for value in panel["bbox_normalized"]]
        crop_box = (
            max(0, int(round(left * width))),
            max(0, int(round(top * height))),
            min(width, int(round(right * width))),
            min(height, int(round(bottom * height))),
        )
        crop = image.crop(crop_box)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
    image_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )
    x0, x1 = (float(value) for value in panel["xlim_arcsec"])
    y0, y1 = (float(value) for value in panel["ylim_arcsec"])
    figure = go.Figure()
    figure.add_layout_image(
        source=image_uri,
        xref="x",
        yref="y",
        x=x0,
        y=y1,
        sizex=x1 - x0,
        sizey=y1 - y0,
        sizing="stretch",
        layer="below",
    )
    xs = np.linspace(x0, x1, 80)
    ys = np.linspace(y0, y1, 80)
    grid_x, grid_y = np.meshgrid(xs, ys)
    figure.add_trace(
        go.Scattergl(
            x=grid_x.ravel(),
            y=grid_y.ravel(),
            mode="markers",
            marker={"size": 4, "opacity": 0.01, "color": "white"},
            hoverinfo="skip",
            showlegend=False,
            name="ROI selection grid",
        )
    )
    if roi is not None:
        vertices = list(roi.vertices_arcsec)
        closed = [*vertices, vertices[0]]
        figure.add_trace(
            go.Scatter(
                x=[point[0] for point in closed],
                y=[point[1] for point in closed],
                mode="lines",
                line={"color": ROI_COLOR, "width": 3},
                name=roi.label or "Confirmed ROI",
            )
        )
    figure.update_layout(
        title="Source Map ROI selection",
        xaxis_title="HPLN / arcsec",
        yaxis_title="HPLT / arcsec",
        dragmode="lasso" if str(roi_mode).lower() == "lasso" else "select",
        height=650,
        margin={"l": 70, "r": 30, "t": 60, "b": 60},
    )
    figure.update_xaxes(range=[x0, x1])
    figure.update_yaxes(range=[y0, y1], scaleanchor="x", scaleratio=1)
    return figure


def annotate_source_map_png(
    image_png: bytes,
    metadata: Mapping[str, Any],
    roi: RadioRoi,
    *,
    color: str = ROI_COLOR,
) -> bytes:
    """Draw a confirmed HPLN/HPLT ROI on the exact Source Map PNG bytes."""

    image = annotate_source_map_image(
        image_png,
        metadata,
        roi,
        color=color,
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def annotate_source_map_image(
    image_png: bytes,
    metadata: Mapping[str, Any],
    roi: RadioRoi,
    *,
    color: str = ROI_COLOR,
) -> Image.Image:
    """Return an annotated RGBA map without an intermediate PNG encode."""

    panel = _single_panel(metadata)
    panel_id = str(panel["id"])
    with Image.open(io.BytesIO(image_png)) as source:
        image = source.convert("RGBA")
    points = [
        data_to_image_pixel(metadata, panel_id, float(x), float(y))
        for x, y in roi.vertices_arcsec
    ]
    if len(points) < 3:
        raise ValueError("Confirmed ROI must contain at least three vertices")
    closed = [*points, points[0]]
    width = max(2, int(round(min(image.size) / 450.0 * 2.5)))
    draw = ImageDraw.Draw(image)
    draw.line(closed, fill=color, width=width, joint="curve")
    label = str(roi.label or "ROI").strip() or "ROI"
    anchor_x, anchor_y = points[0]
    font = ImageFont.load_default()
    box = draw.textbbox((anchor_x, anchor_y), label, font=font, stroke_width=1)
    pad = max(2, width)
    draw.rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        fill=(0, 0, 0, 190),
    )
    draw.text(
        (anchor_x, anchor_y),
        label,
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill="black",
    )
    return image


def _padded_limits(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Composite curve contains no finite samples")
    low = float(np.min(finite))
    high = float(np.max(finite))
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        padding = max(1.0, abs(low) * 0.05)
    else:
        padding = (high - low) * 0.05
    return low - padding, high + padding


def _validate_source_map_aspect_ratio(
    expected: tuple[int, int], actual: tuple[int, int]
) -> None:
    """Allow resolution changes while rejecting a real canvas-shape drift."""

    expected_width, expected_height = (int(value) for value in expected)
    actual_width, actual_height = (int(value) for value in actual)
    if (actual_width, actual_height) == (expected_width, expected_height):
        return
    if expected_width * actual_height == actual_width * expected_height:
        return
    raise ValueError(
        "Source Map frame aspect ratio changed after the sequence layout was fixed: "
        f"expected {expected_width}x{expected_height}, got "
        f"{actual_width}x{actual_height}"
    )


def build_composite_layout(
    annotated_map_png: bytes | Image.Image,
    radio_df: pd.DataFrame,
    dart_result: DartNarrowbandResult,
    *,
    map_frequency_mhz: float,
    dpi: int = 160,
) -> CompositeFigureLayout:
    """Freeze the pixel geometry and curve limits for one frequency sequence."""

    requested_dpi = int(dpi)
    if requested_dpi < 1:
        raise ValueError("Composite DPI must be positive")
    if isinstance(annotated_map_png, Image.Image):
        map_size = tuple(int(value) for value in annotated_map_png.size)
    else:
        with Image.open(io.BytesIO(annotated_map_png)) as source:
            map_size = tuple(int(value) for value in source.size)
    aspect = map_size[1] / max(1, map_size[0])
    map_ratio = min(2.5, max(1.45, aspect * 2.25))
    canvas_width = int(round(12.0 * requested_dpi))
    canvas_height = int(round((11.2 + 2.0 * aspect) * requested_dpi))
    canvas_width += canvas_width % 2
    canvas_height += canvas_height % 2
    radio_plot = _radio_plot_frame(radio_df, float(map_frequency_mhz))
    if not dart_result.curves:
        raise ValueError("DART narrowband extraction returned no curve")
    dart_values = np.asarray(dart_result.curves[0].stokes_i_db, dtype=float)
    return CompositeFigureLayout(
        canvas_size_pixels=(canvas_width, canvas_height),
        map_size_pixels=map_size,
        height_ratios=(map_ratio, 1.0, 1.0),
        radio_ylim=_padded_limits(radio_plot["raw_sum"].to_numpy(dtype=float)),
        dart_ylim=_padded_limits(dart_values),
    )


def build_composite_figure(
    annotated_map_png: bytes,
    radio_df: pd.DataFrame,
    dart_result: DartNarrowbandResult,
    *,
    roi: RadioRoi,
    map_time: datetime | str,
    map_frequency_mhz: float,
    polarization: str,
    time_start: datetime | str,
    time_end: datetime | str,
    layout: CompositeFigureLayout | None = None,
    dpi: int = 160,
    show_time_marker: bool = True,
    validate_map_size: bool = True,
):
    """Create the publication-style three-row composite with shared UTC axes."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.dates as mdates

    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    marker = _utc_datetime(map_time)
    if start >= end:
        raise ValueError("Shared time range must have positive duration")
    if not start <= marker <= end:
        raise ValueError("Selected Source Map time is outside the shared radio range")
    frequency = float(map_frequency_mhz)
    if not math.isfinite(frequency):
        raise ValueError("Source Map frequency must be finite")
    radio_plot = _radio_plot_frame(radio_df, frequency)
    if not dart_result.curves:
        raise ValueError("DART narrowband extraction returned no curve")
    dart_curve = dart_result.curves[0]
    dart_times = [_utc_datetime(value) for value in dart_result.time_utc]
    dart_values = np.asarray(dart_curve.stokes_i_db, dtype=float)
    if len(dart_times) != dart_values.size:
        raise ValueError("DART narrowband values do not match the UTC axis")
    finite_dart = np.isfinite(dart_values)
    if not finite_dart.any():
        raise ValueError("DART narrowband curve contains no finite samples")

    with Image.open(io.BytesIO(annotated_map_png)) as source:
        map_image = np.asarray(source.convert("RGBA"))
    map_size = (int(map_image.shape[1]), int(map_image.shape[0]))
    if layout is None:
        aspect = map_image.shape[0] / max(1, map_image.shape[1])
        height_ratios = (min(2.5, max(1.45, aspect * 2.25)), 1.0, 1.0)
        figure_size = (12.0, 11.2 + 2.0 * aspect)
        figure_dpi = 160
    else:
        if validate_map_size:
            _validate_source_map_aspect_ratio(layout.map_size_pixels, map_size)
        figure_dpi = int(dpi)
        figure_size = (
            layout.canvas_size_pixels[0] / figure_dpi,
            layout.canvas_size_pixels[1] / figure_dpi,
        )
        height_ratios = layout.height_ratios
    figure = Figure(figsize=figure_size, dpi=figure_dpi, facecolor="white")
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=height_ratios,
        hspace=0.08,
        left=0.10,
        right=0.97,
        top=0.98,
        bottom=0.08,
    )
    map_axis = figure.add_subplot(grid[0])
    radio_axis = figure.add_subplot(grid[1])
    dart_axis = figure.add_subplot(grid[2], sharex=radio_axis)
    map_axis.imshow(map_image)
    map_axis.set_axis_off()

    for label, group in radio_plot.groupby("polarization", sort=True, dropna=False):
        ordered = group.sort_values("obs_time_dt")
        radio_axis.plot(
            ordered["obs_time_dt"],
            ordered["raw_sum"],
            linewidth=1.0,
            marker=".",
            markersize=2.8,
            label=str(label or polarization),
        )
    radio_axis.set_title(
        f"Confirmed ROI integrated intensity | {frequency:g} MHz | {polarization}",
        fontsize=11,
        loc="left",
    )
    radio_axis.set_ylabel(_radio_axis_label(radio_plot))
    if radio_plot["polarization"].nunique(dropna=False) > 1:
        radio_axis.legend(loc="best", frameon=False, fontsize=8)

    dart_axis.plot(
        np.asarray(dart_times, dtype=object)[finite_dart],
        dart_values[finite_dart],
        color="#2563eb",
        linewidth=1.0,
    )
    dart_axis.set_title(
        "DART narrowband Stokes I intensity | "
        f"{dart_curve.requested_frequency_range_mhz[0]:g}-"
        f"{dart_curve.requested_frequency_range_mhz[1]:g} MHz",
        fontsize=11,
        loc="left",
    )
    dart_axis.set_ylabel("Stokes I intensity (dB)")
    dart_axis.set_xlabel("Time (UTC)")
    if layout is not None:
        radio_axis.set_ylim(*layout.radio_ylim)
        dart_axis.set_ylim(*layout.dart_ylim)

    for axis in (radio_axis, dart_axis):
        if show_time_marker:
            axis.axvline(
                marker,
                color=MAP_TIME_COLOR,
                linewidth=0.9,
                linestyle="--",
                alpha=0.95,
                zorder=4,
            )
        axis.grid(alpha=0.25, linestyle=":", linewidth=0.65)
        axis.set_xlim(start, end)
    radio_axis.tick_params(axis="x", which="both", labelbottom=False)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    dart_axis.xaxis.set_major_locator(locator)
    dart_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=UTC))
    figure.align_ylabels([radio_axis, dart_axis])
    return figure


def render_composite_frame(
    *args: Any,
    dpi: int = 160,
    layout: CompositeFigureLayout | None = None,
    include_png: bool = True,
    **kwargs: Any,
) -> CompositeFrameRender:
    """Render one RGB frame and optional PNG without a disk round trip."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = build_composite_figure(*args, layout=layout, dpi=dpi, **kwargs)
    if layout is None:
        figure.set_dpi(int(dpi))
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    actual_size = (int(rgb.shape[1]), int(rgb.shape[0]))
    if layout is not None and actual_size != layout.canvas_size_pixels:
        figure.clear()
        raise ValueError(
            "Composite canvas size changed after the sequence layout was fixed: "
            f"expected {layout.canvas_size_pixels[0]}x{layout.canvas_size_pixels[1]}, "
            f"got {actual_size[0]}x{actual_size[1]}"
        )
    canvas_height = int(rgb.shape[0])
    renderer = canvas.get_renderer()
    panel_bounds: dict[str, tuple[int, int, int, int]] = {}
    for name, axis in zip(("source_map", "radio_curve", "dart_curve"), figure.axes):
        bounds = axis.get_window_extent(renderer)
        panel_bounds[name] = (
            int(round(bounds.x0)),
            int(round(canvas_height - bounds.y1)),
            int(round(bounds.x1)),
            int(round(canvas_height - bounds.y0)),
        )
    png_bytes: bytes | None = None
    if include_png:
        output = io.BytesIO()
        canvas.print_png(output)
        png_bytes = output.getvalue()
    figure.clear()
    return CompositeFrameRender(
        rgb=rgb,
        png_bytes=png_bytes,
        panel_bounds_pixels=panel_bounds,
    )


def build_curve_template_signature(
    radio_df: pd.DataFrame,
    dart_result: DartNarrowbandResult,
    *,
    map_size_pixels: tuple[int, int],
    map_frequency_mhz: float,
    polarization: str,
    time_start: datetime | str,
    time_end: datetime | str,
    dpi: int,
) -> str:
    """Hash every input that changes a cached curve canvas."""

    radio_plot = _radio_plot_frame(radio_df, float(map_frequency_mhz))
    radio_rows = [
        {
            "obs_time": _utc_datetime(row.obs_time_dt).isoformat(),
            "raw_sum": float(row.raw_sum),
            "polarization": str(row.polarization),
            "bunit": str(getattr(row, "bunit", "") or ""),
        }
        for row in radio_plot.itertuples(index=False)
    ]
    dart_rows = [
        {
            "center_frequency_mhz": float(curve.center_frequency_mhz),
            "requested_frequency_range_mhz": [
                float(value) for value in curve.requested_frequency_range_mhz
            ],
            "values": [float(value) for value in curve.stokes_i_db],
        }
        for curve in dart_result.curves
    ]
    payload = {
        "schema": "radio-composite-curve-template-v1",
        "map_size_pixels": [int(value) for value in map_size_pixels],
        "map_frequency_mhz": float(map_frequency_mhz),
        "polarization": str(polarization),
        "time_start_utc": _utc_datetime(time_start).isoformat(),
        "time_end_utc": _utc_datetime(time_end).isoformat(),
        "dpi": int(dpi),
        "radio": radio_rows,
        "dart_time_utc": [
            _utc_datetime(value).isoformat() for value in dart_result.time_utc
        ],
        "dart": dart_rows,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_composite_frame_template(
    annotated_map_png: bytes | Image.Image,
    radio_df: pd.DataFrame,
    dart_result: DartNarrowbandResult,
    *,
    roi: RadioRoi,
    map_frequency_mhz: float,
    polarization: str,
    time_start: datetime | str,
    time_end: datetime | str,
    dpi: int = 160,
    layout: CompositeFigureLayout | None = None,
) -> CompositeFrameTemplate:
    """Render one marker-free curve canvas for repeated raster composition."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    start = _utc_datetime(time_start)
    end = _utc_datetime(time_end)
    if start >= end:
        raise ValueError("Shared time range must have positive duration")
    if layout is None:
        layout = build_composite_layout(
            annotated_map_png,
            radio_df,
            dart_result,
            map_frequency_mhz=map_frequency_mhz,
            dpi=dpi,
        )
    signature = build_curve_template_signature(
        radio_df,
        dart_result,
        map_size_pixels=layout.map_size_pixels,
        map_frequency_mhz=map_frequency_mhz,
        polarization=polarization,
        time_start=start,
        time_end=end,
        dpi=dpi,
    )
    map_width, map_height = layout.map_size_pixels
    divisor = math.gcd(map_width, map_height)
    placeholder_size = (
        max(1, map_width // divisor),
        max(1, map_height // divisor),
    )
    if max(placeholder_size) > 256:
        scale = 256.0 / max(placeholder_size)
        placeholder_size = (
            max(1, int(round(placeholder_size[0] * scale))),
            max(1, int(round(placeholder_size[1] * scale))),
        )
    placeholder = Image.new("RGB", placeholder_size, "white")
    placeholder_output = io.BytesIO()
    placeholder.save(placeholder_output, format="PNG")
    figure = build_composite_figure(
        placeholder_output.getvalue(),
        radio_df,
        dart_result,
        roi=roi,
        map_time=start,
        map_frequency_mhz=map_frequency_mhz,
        polarization=polarization,
        time_start=start,
        time_end=end,
        layout=layout,
        dpi=dpi,
        show_time_marker=False,
        validate_map_size=False,
    )
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    rgba = np.asarray(canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[:, :, :3])
    canvas_height = int(rgb.shape[0])
    panel_bounds: dict[str, tuple[int, int, int, int]] = {}
    for name, axis in zip(("source_map", "radio_curve", "dart_curve"), figure.axes):
        bounds = axis.get_window_extent(renderer)
        panel_bounds[name] = _display_bbox_to_pixels(bounds, canvas_height)
    map_artist = figure.axes[0].images[0]
    source_map_bounds = _display_bbox_to_pixels(
        map_artist.get_window_extent(renderer), canvas_height
    )
    base_rgb = rgb.copy()
    left, top, right, bottom = source_map_bounds
    base_rgb[top:bottom, left:right, :] = 255
    radio_png = _axis_tight_crop_png(rgb, figure.axes[1], renderer, canvas_height)
    dart_png = _axis_tight_crop_png(rgb, figure.axes[2], renderer, canvas_height)
    figure.clear()
    return CompositeFrameTemplate(
        base_rgb=base_rgb,
        layout=layout,
        source_map_bounds_pixels=source_map_bounds,
        panel_bounds_pixels=panel_bounds,
        time_start_utc=start,
        time_end_utc=end,
        dpi=int(dpi),
        cache_signature=signature,
        radio_curve_png=radio_png,
        dart_curve_png=dart_png,
    )


def render_cached_composite_frame(
    template: CompositeFrameTemplate,
    annotated_map_png: bytes | Image.Image,
    *,
    map_time: datetime | str,
    include_png: bool = True,
) -> CompositeFrameRender:
    """Insert one map and dashed UTC markers into a cached curve canvas."""

    marker = _utc_datetime(map_time)
    start = template.time_start_utc
    end = template.time_end_utc
    if not start <= marker <= end:
        raise ValueError("Selected Source Map time is outside the shared radio range")
    if isinstance(annotated_map_png, Image.Image):
        map_image = annotated_map_png.convert("RGBA")
    else:
        with Image.open(io.BytesIO(annotated_map_png)) as source:
            map_image = source.convert("RGBA")
    _validate_source_map_aspect_ratio(
        template.layout.map_size_pixels,
        tuple(int(value) for value in map_image.size),
    )
    left, top, right, bottom = template.source_map_bounds_pixels
    target_size = (right - left, bottom - top)
    resample = (
        Image.Resampling.BOX
        if map_image.width >= target_size[0] and map_image.height >= target_size[1]
        else Image.Resampling.BICUBIC
    )
    resized = map_image.resize(target_size, resample=resample)
    frame = Image.fromarray(template.base_rgb.copy(), mode="RGB").convert("RGBA")
    frame.alpha_composite(resized, dest=(left, top))
    fraction = (marker - start).total_seconds() / (end - start).total_seconds()
    marker_pixels: dict[str, int] = {}
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    marker_color = (*_hex_rgb(MAP_TIME_COLOR), int(round(255 * 0.95)))
    line_width = max(1, int(round(0.9 * template.dpi / 72.0)))
    dash = max(3, int(round(4.0 * template.dpi / 72.0)))
    gap = max(2, int(round(3.0 * template.dpi / 72.0)))
    for name in ("radio_curve", "dart_curve"):
        axis_left, axis_top, axis_right, axis_bottom = template.panel_bounds_pixels[
            name
        ]
        x = int(round(axis_left + fraction * (axis_right - axis_left)))
        marker_pixels[name] = x
        y = axis_top
        while y < axis_bottom:
            y_end = min(axis_bottom, y + dash)
            draw.line((x, y, x, y_end), fill=marker_color, width=line_width)
            y = y_end + gap
    frame = Image.alpha_composite(frame, overlay).convert("RGB")
    rgb = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    png_bytes: bytes | None = None
    if include_png:
        output = io.BytesIO()
        frame.save(output, format="PNG")
        png_bytes = output.getvalue()
    return CompositeFrameRender(
        rgb=rgb,
        png_bytes=png_bytes,
        panel_bounds_pixels=template.panel_bounds_pixels,
        marker_x_pixels=marker_pixels,
    )


def _display_bbox_to_pixels(
    bounds: Any, canvas_height: int
) -> tuple[int, int, int, int]:
    return (
        int(round(bounds.x0)),
        int(round(canvas_height - bounds.y1)),
        int(round(bounds.x1)),
        int(round(canvas_height - bounds.y0)),
    )


def _axis_tight_crop_png(
    rgb: np.ndarray,
    axis: Any,
    renderer: Any,
    canvas_height: int,
) -> bytes:
    bounds = axis.get_tightbbox(renderer)
    left, top, right, bottom = _display_bbox_to_pixels(bounds, canvas_height)
    padding = 8
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(int(rgb.shape[1]), right + padding)
    bottom = min(int(rgb.shape[0]), bottom + padding)
    crop = Image.fromarray(rgb, mode="RGB").crop((left, top, right, bottom))
    output = io.BytesIO()
    crop.save(output, format="PNG")
    return output.getvalue()


def _hex_rgb(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected a six-digit RGB color, got {value!r}")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def render_composite_png(*args: Any, dpi: int = 160, **kwargs: Any) -> bytes:
    """Render the three-row composite into PNG bytes."""

    rendered = render_composite_frame(*args, dpi=dpi, include_png=True, **kwargs)
    if rendered.png_bytes is None:  # pragma: no cover - defensive contract guard
        raise RuntimeError("Composite PNG rendering returned no bytes")
    return rendered.png_bytes


def build_composite_artifacts(
    source_map_png: bytes,
    source_map_metadata: Mapping[str, Any],
    radio_df: pd.DataFrame,
    dart_result: DartNarrowbandResult,
    *,
    roi: RadioRoi,
    map_time: datetime | str,
    map_frequency_mhz: float,
    polarization: str,
    time_start: datetime | str,
    time_end: datetime | str,
    request_signature: str,
    source_context: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
    dpi: int = 160,
) -> CompositeArtifactBundle:
    """Build PNG, CSV, ROI, metadata, and ZIP products in memory."""

    generated = _utc_datetime(generated_at or datetime.now(UTC))
    marker = _utc_datetime(map_time)
    annotated = annotate_source_map_png(source_map_png, source_map_metadata, roi)
    curve_template = build_composite_frame_template(
        annotated,
        radio_df,
        dart_result,
        roi=roi,
        map_frequency_mhz=map_frequency_mhz,
        polarization=polarization,
        time_start=time_start,
        time_end=time_end,
        dpi=dpi,
    )
    rendered_composite = render_cached_composite_frame(
        curve_template,
        annotated,
        map_time=marker,
        include_png=True,
    )
    if rendered_composite.png_bytes is None:  # pragma: no cover - contract guard
        raise RuntimeError("Composite PNG rendering returned no bytes")
    composite_png = rendered_composite.png_bytes
    image_name = build_scientific_image_filename(
        sequence=1,
        start_time=marker,
        instrument="radio-dart",
        channel=f"{float(map_frequency_mhz):g}mhz",
        polarization=polarization,
        product="radio-composite",
        qualifiers=(roi.roi_id,),
        generated_at=generated,
        extension=".png",
    )
    stem = Path(image_name).stem
    filenames = {
        "composite_png": image_name,
        "radio_curve_png": f"{stem}_radio-roi-lightcurve.png",
        "dart_curve_png": f"{stem}_dart-narrowband-lightcurve.png",
        "radio_csv": f"{stem}_radio-roi.csv",
        "dart_csv": f"{stem}_dart-narrowband.csv",
        "roi_json": f"{stem}_roi.json",
        "metadata_json": f"{stem}_metadata.json",
    }
    dart_frame = _dart_curve_frame(dart_result)
    public_source_map_metadata = {
        key: value
        for key, value in source_map_metadata.items()
        if not str(key).startswith("_")
    }
    metadata = {
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "generated_at_utc": generated.isoformat(),
        "request_signature": str(request_signature),
        "map": {
            "observation_time_utc": marker.isoformat(),
            "frequency_mhz": float(map_frequency_mhz),
            "polarization": str(polarization),
            "source_map": _json_safe(public_source_map_metadata),
        },
        "roi": roi.to_json_dict(),
        "radio_curve": {
            "metric": "raw_sum",
            "time_start_utc": _utc_datetime(time_start).isoformat(),
            "time_end_utc": _utc_datetime(time_end).isoformat(),
            "rows": int(len(radio_df)),
            "image": filenames["radio_curve_png"],
        },
        "dart_curve": {
            "representation": "source Stokes I dB intensity",
            "rows": int(len(dart_frame)),
            "image": filenames["dart_curve_png"],
            "curves": [
                {
                    "center_frequency_mhz": float(curve.center_frequency_mhz),
                    "requested_frequency_range_mhz": list(
                        curve.requested_frequency_range_mhz
                    ),
                    "sampled_frequency_range_mhz": list(
                        curve.sampled_frequency_range_mhz
                    ),
                    "channel_count": int(curve.channel_count),
                }
                for curve in dart_result.curves
            ],
        },
        "source": _json_safe(dict(source_context or {})),
        "curve_template": curve_template.to_metadata(),
        "artifacts": dict(filenames),
    }
    files = {
        "composite_png": composite_png,
        "radio_curve_png": curve_template.radio_curve_png,
        "dart_curve_png": curve_template.dart_curve_png,
        "radio_csv": radio_df.to_csv(index=False).encode("utf-8-sig"),
        "dart_csv": dart_frame.to_csv(index=False).encode("utf-8-sig"),
        "roi_json": (json.dumps(roi.to_json_dict(), indent=2) + "\n").encode("utf-8"),
        "metadata_json": (
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n"
        ).encode("utf-8"),
    }
    zip_output = io.BytesIO()
    with zipfile.ZipFile(
        zip_output, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for key, payload in files.items():
            archive.writestr(filenames[key], payload)
    return CompositeArtifactBundle(
        files=files,
        filenames=filenames,
        zip_bytes=zip_output.getvalue(),
        zip_filename=f"{stem}.zip",
        metadata=metadata,
        curve_template=curve_template,
    )


def save_composite_bundle(
    bundle: CompositeArtifactBundle,
    output_directory: str | Path,
) -> Path:
    """Write a bundle into a new directory without overwriting prior products."""

    output = Path(output_directory).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(bundle.zip_filename).stem
    destination = output / stem
    suffix = 2
    while destination.exists():
        destination = output / f"{stem}_{suffix:03d}"
        suffix += 1
    destination.mkdir(parents=False, exist_ok=False)
    for key, payload in bundle.files.items():
        (destination / bundle.filenames[key]).write_bytes(payload)
    (destination / bundle.zip_filename).write_bytes(bundle.zip_bytes)
    return destination


def _radio_plot_frame(df: pd.DataFrame, frequency_mhz: float) -> pd.DataFrame:
    data = df.copy()
    data["obs_time_dt"] = pd.to_datetime(
        data.get("obs_time"), errors="coerce", utc=True
    )
    data["raw_sum"] = pd.to_numeric(data.get("raw_sum"), errors="coerce")
    frequencies = pd.to_numeric(data.get("freq_mhz"), errors="coerce")
    quality = (
        data.get("quality_flag", pd.Series("ok", index=data.index))
        .astype(str)
        .str.lower()
        .eq("ok")
    )
    tolerance = max(1e-6, abs(float(frequency_mhz)) * 1e-5)
    valid = (
        quality
        & data["obs_time_dt"].notna()
        & np.isfinite(data["raw_sum"].to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(frequencies.to_numpy(dtype=float, na_value=np.nan))
        & (np.abs(frequencies - float(frequency_mhz)) <= tolerance)
    )
    result = data.loc[valid].copy()
    if result.empty:
        raise ValueError("Radio ROI analysis contains no valid raw_sum samples")
    if "polarization" not in result:
        result["polarization"] = ""
    return result


def _radio_axis_label(df: pd.DataFrame) -> str:
    units = []
    if "bunit" in df:
        units = sorted(
            {str(value).strip() for value in df["bunit"].dropna() if str(value).strip()}
        )
    if len(units) == 1:
        return f"ROI raw_sum ({units[0]} × pixel)"
    return "ROI raw_sum (native unit × pixel)"


def _dart_curve_frame(result: DartNarrowbandResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for curve in result.curves:
        for timestamp, value in zip(result.time_utc, curve.stokes_i_db, strict=True):
            rows.append(
                {
                    "time_utc": _utc_datetime(timestamp).isoformat(),
                    "stokes_i_db": float(value),
                    "center_frequency_mhz": float(curve.center_frequency_mhz),
                    "bandwidth_mhz": float(curve.bandwidth_mhz),
                    "requested_low_mhz": float(curve.requested_frequency_range_mhz[0]),
                    "requested_high_mhz": float(curve.requested_frequency_range_mhz[1]),
                    "sampled_low_mhz": float(curve.sampled_frequency_range_mhz[0]),
                    "sampled_high_mhz": float(curve.sampled_frequency_range_mhz[1]),
                    "channel_count": int(curve.channel_count),
                }
            )
    return pd.DataFrame(rows)


def _single_panel(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    panels = metadata.get("panels")
    if not isinstance(panels, list) or len(panels) != 1:
        raise ValueError("Composite Source Map must contain exactly one radio panel")
    return panels[0]


def _sample_indices(size: int, *, target: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("Selection axis must contain at least one sample")
    count = min(int(size), int(target))
    return np.unique(np.linspace(0, size - 1, count).round().astype(int))


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("UTC datetime value must not be blank")
        normalized = text[:-1] + "+00:00" if text.upper().endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid UTC datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _utc_datetime(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


__all__ = [
    "COMPOSITE_SCHEMA_VERSION",
    "CompositeArtifactBundle",
    "CompositeFrameTemplate",
    "CompositeFigureLayout",
    "CompositeFrameRender",
    "FrequencyBand",
    "annotate_source_map_image",
    "annotate_source_map_png",
    "build_centered_frequency_bands",
    "build_composite_artifacts",
    "build_composite_frame_template",
    "build_composite_figure",
    "build_composite_layout",
    "build_curve_template_signature",
    "build_dart_selection_figure",
    "build_request_signature",
    "build_source_map_selection_figure",
    "centered_frequency_band",
    "frequency_band_from_selection",
    "render_composite_png",
    "render_composite_frame",
    "render_cached_composite_frame",
    "save_composite_bundle",
    "select_dart_time_overlap",
]
