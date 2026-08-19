"""Render static AIA/radio composite panels from already-adapted products."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from solar_toolkit.aia.config import AIA_CONFIG
from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d
from solar_toolkit.radio.centers import RadioImage
from solar_toolkit.radio.gaussian import overlay_gaussian_fit_on_axis

from ..adapters import AiaSelection, RadioGaussianSelection
from ..models import (
    AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
    CompositeResult,
    SpectrumFluxCurve,
    SpectrumTimeAlignment,
    parse_roi_curve_times,
)

__all__ = [
    "TriplePanelArtifact",
    "TopPanelArtifact",
    "render_composite_result",
    "render_top_panel",
    "write_composite_artifacts",
]

MAP_TIME_COLOR = "#c2410c"


@dataclass(frozen=True, slots=True)
class TopPanelArtifact:
    """Rendered top-panel PNG bytes and JSON-safe scientific metadata."""

    image_png: bytes
    metadata: Mapping[str, Any]
    radio_frame: RadioImage | None = None
    radio_frames: tuple[RadioImage, ...] = ()


@dataclass(frozen=True, slots=True)
class TriplePanelArtifact:
    """Rendered PNG, metadata, and reproducible flux CSV payloads."""

    image_png: bytes
    metadata_json: bytes
    roi_curve_csv: bytes
    metadata: Mapping[str, Any]
    spectrum_flux_csv: bytes | None = None


def render_composite_result(
    result: CompositeResult,
    *,
    map_time: datetime | str,
    metric: str = "raw_sum",
    dpi: int = 160,
    figure_size: tuple[float, float] = (11.0, 13.0),
    display_time_range_utc: tuple[datetime | str, datetime | str] | None = None,
) -> TriplePanelArtifact:
    """Render AIA/radio, ROI lightcurve, and spectrum on synchronized UTC axes."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure

    if not isinstance(result, CompositeResult):
        raise TypeError("result must be a CompositeResult")
    if metric not in {"raw_sum", "raw_mean", "raw_peak"}:
        raise ValueError("metric must be raw_sum, raw_mean, or raw_peak")
    if int(dpi) <= 0:
        raise ValueError("dpi must be greater than zero")
    marker_time = _utc_datetime(map_time)
    spectrum_time_alignment = _result_spectrum_time_alignment(result)
    spectrum_flux_time_alignments = _result_spectrum_flux_time_alignments(result)
    curve = result.roi_curve.copy(deep=False)
    curve_times = parse_roi_curve_times(curve)
    spectrum_fluxes = result.spectrum_flux_curves
    flux_plot_layout = _normalized_flux_plot_layout(
        result.metadata.get("flux_plot_layout", "combined")
    )
    separate_flux_axes = flux_plot_layout == "separate" and len(spectrum_fluxes) > 1
    flux_axis_count = len(spectrum_fluxes) if separate_flux_axes else 1
    resolved_figure_size = (
        float(figure_size[0]),
        float(figure_size[1]) + 3.2 * max(0, flux_axis_count - 1),
    )

    figure = Figure(
        figsize=resolved_figure_size,
        dpi=int(dpi),
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(
        flux_axis_count + 2,
        2,
        height_ratios=(1.45, *([0.8] * flux_axis_count), 1.0),
        width_ratios=(1.0, 0.035),
        hspace=0.12,
        wspace=0.12,
    )
    top_axis = figure.add_subplot(grid[0, :])
    curve_axes = [figure.add_subplot(grid[1, 0])]
    for row_index in range(1, flux_axis_count):
        curve_axes.append(
            figure.add_subplot(
                grid[row_index + 1, 0],
                sharex=curve_axes[0],
            )
        )
    spectrum_row = flux_axis_count + 1
    spectrum_axis = figure.add_subplot(
        grid[spectrum_row, 0],
        sharex=curve_axes[0],
    )
    colorbar_axis = figure.add_subplot(grid[spectrum_row, 1])

    with Image.open(io.BytesIO(result.top_image)) as top_image:
        top_axis.imshow(top_image.convert("RGB"))
    top_axis.set_title("AIA + Radio Gaussian", fontsize=12)
    top_axis.set_axis_off()

    frequencies = pd.to_numeric(curve["frequency"], errors="raise")
    metric_values = pd.to_numeric(curve[metric], errors="coerce")
    polarizations = (
        curve["polarization"].astype(str)
        if "polarization" in curve
        else pd.Series([""] * len(curve), index=curve.index)
    )
    curve_plot = pd.DataFrame(
        {
            "time": curve_times,
            "frequency": frequencies,
            "value": metric_values,
            "quality": curve["quality_flag"].astype(str),
            "polarization": polarizations,
        }
    )
    untimed_quality_rows = int(curve_plot["time"].isna().sum())
    flagged_count = int((~curve_plot["quality"].str.lower().eq("ok")).sum())
    curve_plot = curve_plot.loc[curve_plot["time"].notna()]
    spectrum_colors = (
        "#f97316",
        "#a855f7",
        "#22c55e",
        "#eab308",
        "#ef4444",
        "#06b6d4",
    )
    if separate_flux_axes:
        flux_axis_inputs = tuple(
            (
                curve_plot.loc[
                    np.isclose(
                        curve_plot["frequency"].to_numpy(dtype=float),
                        float(spectrum_flux.requested_band.center_mhz),
                        rtol=0.0,
                        atol=1e-6,
                    )
                ],
                (spectrum_flux,),
                spectrum_index,
            )
            for spectrum_index, spectrum_flux in enumerate(spectrum_fluxes)
        )
    else:
        flux_axis_inputs = ((curve_plot, spectrum_fluxes, 0),)
    for curve_axis, (
        axis_curve,
        axis_spectrum_fluxes,
        color_offset,
    ) in zip(curve_axes, flux_axis_inputs, strict=True):
        if axis_curve.empty:
            center = axis_spectrum_fluxes[0].requested_band.center_mhz
            raise ValueError(f"ROI curve has no rows matching {center:g} MHz")
        _plot_flux_axis(
            curve_axis,
            axis_curve,
            axis_spectrum_fluxes,
            marker_time=marker_time,
            metric=metric,
            spectrum_colors=spectrum_colors,
            color_offset=color_offset,
            separate=separate_flux_axes,
            time_alignment=spectrum_time_alignment,
            time_alignments=spectrum_flux_time_alignments,
        )

    spectrum = result.spectrum
    spectrum_display_times = (
        spectrum_time_alignment.align_times(spectrum.time_utc)
        if spectrum_time_alignment is not None
        else spectrum.time_utc
    )
    spectrum_flux_display_times = tuple(
        (
            alignment.align_times(item.time_utc)
            if (
                alignment := _spectrum_flux_time_alignment(
                    item,
                    default=spectrum_time_alignment,
                    by_frequency=spectrum_flux_time_alignments,
                )
            )
            is not None
            else item.time_utc
        )
        for item in spectrum_fluxes
    )
    spectrum_intensity_range = _optional_numeric_range(
        result.metadata.get("spectrum_display_intensity_range"),
        label="spectrum display intensity range",
    )
    spectrum_frequency_range = _optional_numeric_range(
        result.metadata.get("spectrum_display_frequency_range_mhz"),
        label="spectrum display frequency range",
    )
    mesh = spectrum_axis.pcolormesh(
        spectrum_display_times,
        spectrum.frequency_mhz,
        spectrum.data,
        shading="auto",
        cmap="turbo",
        vmin=(
            spectrum_intensity_range[0]
            if spectrum_intensity_range is not None
            else None
        ),
        vmax=(
            spectrum_intensity_range[1]
            if spectrum_intensity_range is not None
            else None
        ),
    )
    for index, spectrum_flux in enumerate(spectrum_fluxes):
        band = spectrum_flux.requested_band
        spectrum_axis.axhspan(
            band.low_mhz,
            band.high_mhz,
            facecolor=spectrum_colors[index % len(spectrum_colors)],
            edgecolor=spectrum_colors[index % len(spectrum_colors)],
            alpha=0.12,
            linewidth=0.8,
        )
    spectrum_axis.axvline(
        marker_time,
        color=MAP_TIME_COLOR,
        linewidth=0.9,
        linestyle="--",
    )
    spectrum_axis.set_ylabel("Frequency (MHz)")
    if spectrum_frequency_range is not None:
        spectrum_axis.set_ylim(*spectrum_frequency_range)
    spectrum_axis.set_xlabel("Time (UTC)")
    spectrum_axis.set_title(
        f"{spectrum.source} dynamic spectrum — "
        f"{spectrum.polarization} [{spectrum.unit}]"
    )
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label(f"{spectrum.polarization} ({spectrum.unit})")
    locator = AutoDateLocator()
    spectrum_axis.xaxis.set_major_locator(locator)
    spectrum_axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    if display_time_range_utc is not None:
        time_start, time_end = (
            _utc_datetime(display_time_range_utc[0]),
            _utc_datetime(display_time_range_utc[1]),
        )
        if time_start >= time_end:
            raise ValueError("display_time_range_utc start must be before its end")
    else:
        requested_times = []
        for range_key in ("roi_time_range_utc", "spectrum_time_range_utc"):
            requested_range = result.metadata.get(range_key)
            if isinstance(requested_range, (list, tuple)) and len(requested_range) == 2:
                requested_times.extend(
                    _utc_datetime(value) for value in requested_range
                )
        combined_times = requested_times or [
            *(value.to_pydatetime() for value in curve_times if not pd.isna(value)),
            *spectrum_display_times,
            *(
                time
                for display_times in spectrum_flux_display_times
                for time in display_times
            ),
            marker_time,
        ]
        time_start = min(combined_times)
        time_end = max(combined_times)
    if time_start == time_end:
        from datetime import timedelta

        time_start -= timedelta(seconds=1)
        time_end += timedelta(seconds=1)
    spectrum_axis.set_xlim(time_start, time_end)
    figure.align_ylabels((*curve_axes, spectrum_axis))
    figure.subplots_adjust(left=0.09, right=0.94, top=0.97, bottom=0.06)
    figure.canvas.draw()
    curve_bboxes = [axis.get_position() for axis in curve_axes]
    spectrum_bbox = spectrum_axis.get_position()
    curve_bboxes_normalized = [
        [
            float(bbox.x0),
            float(bbox.y0),
            float(bbox.x1),
            float(bbox.y1),
        ]
        for bbox in curve_bboxes
    ]
    curve_bbox_normalized = curve_bboxes_normalized[0]
    spectrum_bbox_normalized = [
        float(spectrum_bbox.x0),
        float(spectrum_bbox.y0),
        float(spectrum_bbox.x1),
        float(spectrum_bbox.y1),
    ]

    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=int(dpi), facecolor="white")
    image_png = output.getvalue()
    figure.clear()
    roi_csv = curve.to_csv(index=False).encode("utf-8")
    spectrum_flux_csv = None
    if spectrum_fluxes:
        frames = []
        for band_index, spectrum_flux in enumerate(spectrum_fluxes):
            frame = spectrum_flux.to_frame()
            frame.insert(0, "band_index", band_index)
            frames.append(frame)
        spectrum_flux_csv = (
            pd.concat(frames, ignore_index=True).to_csv(index=False).encode("utf-8")
        )
    file_hashes = {
        "png_sha256": hashlib.sha256(image_png).hexdigest(),
        "roi_csv_sha256": hashlib.sha256(roi_csv).hexdigest(),
    }
    if spectrum_flux_csv is not None:
        file_hashes["spectrum_flux_csv_sha256"] = hashlib.sha256(
            spectrum_flux_csv
        ).hexdigest()
    metadata = {
        "schema_version": AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
        "product": "aia-radio-composite",
        "files": file_hashes,
        "render": {
            "dpi": int(dpi),
            "figure_size_inches": [
                float(resolved_figure_size[0]),
                float(resolved_figure_size[1]),
            ],
            "metric": metric,
            "map_time_utc": marker_time.isoformat(),
            "shared_time_range_utc": [
                time_start.isoformat(),
                time_end.isoformat(),
            ],
            "quality_flagged_rows": flagged_count,
            "untimed_quality_rows": untimed_quality_rows,
            "time_alignment": (
                "shared_utc_dart_display_offset_no_interpolation"
                if spectrum_time_alignment is not None
                else "shared_utc_no_interpolation"
            ),
            "spectrum_time_alignment": (
                spectrum_time_alignment.to_dict()
                if spectrum_time_alignment is not None
                else None
            ),
            "spectrum_flux_time_alignments": {
                f"{frequency:g}": alignment.to_dict()
                for frequency, alignment in spectrum_flux_time_alignments.items()
            },
            "main_axis_bbox_normalized": {
                "flux": curve_bbox_normalized,
                "fluxes": curve_bboxes_normalized,
                "spectrum": spectrum_bbox_normalized,
            },
            "main_axis_x_alignment": bool(
                all(
                    np.allclose(
                        [bbox.x0, bbox.x1],
                        [spectrum_bbox.x0, spectrum_bbox.x1],
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for bbox in curve_bboxes
                )
            ),
            "dual_axis_flux": bool(spectrum_fluxes),
            "spectrum_flux_curve_count": len(spectrum_fluxes),
            "flux_plot_layout": ("separate" if separate_flux_axes else "combined"),
            "flux_axis_count": flux_axis_count,
            "spectrum_display_frequency_range_mhz": (
                list(spectrum_frequency_range)
                if spectrum_frequency_range is not None
                else None
            ),
            "spectrum_display_intensity_range": (
                list(spectrum_intensity_range)
                if spectrum_intensity_range is not None
                else None
            ),
        },
        "result": result.to_metadata_dict(),
    }
    metadata_json = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return TriplePanelArtifact(
        image_png=image_png,
        metadata_json=metadata_json,
        roi_curve_csv=roi_csv,
        metadata=metadata,
        spectrum_flux_csv=spectrum_flux_csv,
    )


def _plot_flux_axis(
    curve_axis: Any,
    curve_plot: pd.DataFrame,
    spectrum_fluxes: Sequence[SpectrumFluxCurve],
    *,
    marker_time: datetime,
    metric: str,
    spectrum_colors: Sequence[str],
    color_offset: int,
    separate: bool,
    time_alignment: SpectrumTimeAlignment | None,
    time_alignments: Mapping[float, SpectrumTimeAlignment],
) -> None:
    for (frequency, polarization), frame in curve_plot.groupby(
        ["frequency", "polarization"],
        sort=True,
        dropna=False,
    ):
        frame = frame.sort_values("time", kind="mergesort")
        valid = frame["quality"].str.lower().eq("ok") & frame["value"].notna()
        flagged = ~frame["quality"].str.lower().eq("ok") & frame["value"].notna()
        suffix = f" {polarization}" if polarization else ""
        label = f"{frequency:g} MHz{suffix}"
        if valid.any():
            (line,) = curve_axis.plot(
                frame.loc[valid, "time"],
                frame.loc[valid, "value"],
                marker="o",
                markersize=2.8,
                linewidth=0.9,
                label=label,
            )
        else:
            line = None
        if flagged.any():
            curve_axis.scatter(
                frame.loc[flagged, "time"],
                frame.loc[flagged, "value"],
                marker="x",
                s=26,
                color=line.get_color() if line is not None else None,
                label=f"{label} flagged",
            )
    curve_axis.axvline(
        marker_time,
        color=MAP_TIME_COLOR,
        linewidth=0.9,
        linestyle="--",
        label="Top-panel UTC",
    )
    curve_axis.set_ylabel(metric)
    if separate and spectrum_fluxes:
        center_mhz = spectrum_fluxes[0].requested_band.center_mhz
        curve_axis.set_title(
            f"{center_mhz:g} MHz — radio image ROI + spectrum-band flux"
        )
    else:
        curve_axis.set_title("Radio image ROI + spectrum-band flux")
    curve_axis.grid(alpha=0.2, linestyle=":")
    curve_axis.tick_params(axis="x", labelbottom=False)

    spectrum_flux_axis = None
    if spectrum_fluxes:
        spectrum_flux_axis = curve_axis.twinx()
        for index, spectrum_flux in enumerate(spectrum_fluxes):
            band = spectrum_flux.requested_band
            item_alignment = _spectrum_flux_time_alignment(
                spectrum_flux,
                default=time_alignment,
                by_frequency=time_alignments,
            )
            display_times = (
                item_alignment.align_times(spectrum_flux.time_utc)
                if item_alignment is not None
                else spectrum_flux.time_utc
            )
            spectrum_flux_axis.plot(
                display_times,
                spectrum_flux.values,
                color=spectrum_colors[(color_offset + index) % len(spectrum_colors)],
                linewidth=1.4,
                linestyle="--",
                label=(
                    f"{spectrum_flux.source} {band.center_mhz:g} MHz "
                    f"({band.low_mhz:g}-{band.high_mhz:g}) "
                    f"{spectrum_flux.polarization}"
                ),
            )
        first_flux = spectrum_fluxes[0]
        right_axis_label = (
            f"{first_flux.source} {first_flux.requested_band.center_mhz:g} MHz "
            f"mean ({first_flux.unit})"
            if separate
            else f"{first_flux.source} matched-band mean ({first_flux.unit})"
        )
        spectrum_flux_axis.set_ylabel(right_axis_label, color="#c2410c")
        spectrum_flux_axis.tick_params(axis="y", colors="#c2410c")
    handles, labels = curve_axis.get_legend_handles_labels()
    if spectrum_flux_axis is not None:
        right_handles, right_labels = spectrum_flux_axis.get_legend_handles_labels()
        handles.extend(right_handles)
        labels.extend(right_labels)
    if handles:
        curve_axis.legend(handles, labels, loc="best", fontsize=7, ncols=2)


def _result_spectrum_time_alignment(
    result: CompositeResult,
) -> SpectrumTimeAlignment | None:
    value = result.metadata.get("spectrum_time_alignment")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("spectrum_time_alignment metadata must be a mapping")
    if result.spectrum.source.upper() != "DART":
        raise ValueError("DART time alignment cannot be applied to a CSO spectrum")
    return SpectrumTimeAlignment.from_dict(value)


def _result_spectrum_flux_time_alignments(
    result: CompositeResult,
) -> dict[float, SpectrumTimeAlignment]:
    value = result.metadata.get("spectrum_flux_time_alignments")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("spectrum_flux_time_alignments metadata must be a mapping")
    if result.spectrum.source.upper() != "DART":
        raise ValueError("DART time alignments cannot be applied to a CSO spectrum")
    alignments: dict[float, SpectrumTimeAlignment] = {}
    for frequency, payload in value.items():
        if not isinstance(payload, Mapping):
            raise TypeError("spectrum_flux_time_alignments values must be mappings")
        alignments[float(frequency)] = SpectrumTimeAlignment.from_dict(payload)
    return alignments


def _spectrum_flux_time_alignment(
    spectrum_flux: SpectrumFluxCurve,
    *,
    default: SpectrumTimeAlignment | None,
    by_frequency: Mapping[float, SpectrumTimeAlignment],
) -> SpectrumTimeAlignment | None:
    return by_frequency.get(
        float(spectrum_flux.requested_band.center_mhz),
        default,
    )


def write_composite_artifacts(
    artifact: TriplePanelArtifact,
    output_directory: str | Path,
    *,
    stem: str = "aia_radio_composite",
) -> Mapping[str, Path]:
    """Write PNG, JSON, and CSV into a conflict-safe output directory."""

    if not isinstance(artifact, TriplePanelArtifact):
        raise TypeError("artifact must be a TriplePanelArtifact")
    directory = Path(output_directory).expanduser().resolve(strict=False)
    directory.mkdir(parents=True, exist_ok=True)
    output_stem = _available_stem(directory, stem)
    paths: dict[str, Path] = {
        "png": directory / f"{output_stem}.png",
        "json": directory / f"{output_stem}.json",
        "csv": directory / f"{output_stem}.csv",
    }
    if artifact.spectrum_flux_csv is not None:
        paths["spectrum_csv"] = directory / f"{output_stem}_spectrum_flux.csv"
    paths["png"].write_bytes(artifact.image_png)
    paths["json"].write_bytes(artifact.metadata_json)
    paths["csv"].write_bytes(artifact.roi_curve_csv)
    if artifact.spectrum_flux_csv is not None:
        paths["spectrum_csv"].write_bytes(artifact.spectrum_flux_csv)
    return paths


def render_top_panel(
    aia: AiaSelection | Sequence[AiaSelection],
    radio: RadioGaussianSelection | Sequence[RadioGaussianSelection],
    *,
    dpi: int = 160,
    figure_size: tuple[float, float] = (9.0, 7.5),
    display_extent_arcsec: tuple[float, float, float, float] | None = None,
    extended_canvas_color: str = "black",
) -> TopPanelArtifact:
    """Render one or more AIA bands plus matched radio Gaussian overlays."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if int(dpi) <= 0:
        raise ValueError("dpi must be greater than zero")
    if len(figure_size) != 2 or min(float(value) for value in figure_size) <= 0:
        raise ValueError("figure_size must contain two positive values")

    aias = _normalized_aia_selections(aia)
    radios = _normalized_radio_selections(radio)
    canvas_color = _normalized_canvas_color(extended_canvas_color)
    ncols = min(3, len(aias))
    nrows = int(np.ceil(len(aias) / ncols))
    first_aia_extent = _aia_extent_arcsec(aias[0])
    shared_display_extent = _display_extent(
        display_extent_arcsec,
        first_aia_extent,
    )
    extent_width = shared_display_extent[1] - shared_display_extent[0]
    extent_height = shared_display_extent[3] - shared_display_extent[2]
    panel_aspect = extent_height / extent_width
    if len(aias) > 1:
        panel_width_inches = 4.4
        panel_height_inches = panel_width_inches * panel_aspect
        margins = {
            "left": 0.72,
            "right": 0.14,
            "bottom": 0.58,
            "top": 0.62,
        }
        resolved_size = (
            margins["left"] + panel_width_inches * ncols + margins["right"],
            margins["bottom"] + panel_height_inches * nrows + margins["top"],
        )
    else:
        resolved_size = (float(figure_size[0]), float(figure_size[1]))
        margins = {
            "left": 0.72,
            "right": 0.14,
            "bottom": 0.58,
            "top": 0.62,
        }
    figure = Figure(
        figsize=resolved_size,
        dpi=int(dpi),
        facecolor="white",
    )
    FigureCanvasAgg(figure)
    axes = []
    left = margins["left"] / resolved_size[0]
    right = 1.0 - margins["right"] / resolved_size[0]
    bottom = margins["bottom"] / resolved_size[1]
    top = 1.0 - margins["top"] / resolved_size[1]
    panel_width = (right - left) / ncols
    panel_height = (top - bottom) / nrows
    for index in range(len(aias)):
        row, col = divmod(index, ncols)
        axis = figure.add_axes(
            [
                left + col * panel_width,
                top - (row + 1) * panel_height,
                panel_width,
                panel_height,
            ]
        )
        axes.append(axis)
    reference_frequency = float(radios[0].frame.freq_mhz)
    reference_time = radios[0].matched_time_utc
    figure.suptitle(
        f"{reference_frequency:g} MHz — {_format_utc_z(reference_time)}",
        fontsize=12,
    )
    overlay_colors = ("white", "cyan", "magenta", "lime", "yellow", "orange")
    panels: list[dict[str, Any]] = []
    all_extended_frequencies: set[float] = set()
    aia_extents: list[list[float]] = []
    display_extents: list[list[float]] = []
    canvas_extended = False
    for panel_index, (aia_selection, axis) in enumerate(zip(aias, axes, strict=False)):
        rendered = _draw_aia_radio_axis(
            axis,
            aia_selection,
            radios,
            display_extent_arcsec=shared_display_extent,
            extended_canvas_color=canvas_color,
            overlay_colors=overlay_colors,
            show_legend=panel_index == 0,
        )
        all_extended_frequencies.update(rendered["extended_frequencies_mhz"])
        aia_extents.append(rendered["aia_extent_arcsec"])
        display_extents.append(rendered["display_extent_arcsec"])
        canvas_extended = canvas_extended or rendered["canvas_extended"]
        row, col = divmod(panel_index, ncols)
        is_bottom = row == nrows - 1
        is_left = col == 0
        axis.tick_params(
            labelbottom=is_bottom,
            bottom=is_bottom,
            labelleft=is_left,
            left=is_left,
            top=False,
            right=False,
            labelsize=7,
            length=2,
        )
        axis.set_xlabel("HPLN (arcsec)" if is_bottom else "", fontsize=8)
        axis.set_ylabel("HPLT (arcsec)" if is_left else "", fontsize=8)
        axis.text(
            0.02,
            0.98,
            f"AIA {aia_selection.background.wavelength or 'unknown'} Å",
            transform=axis.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
    figure.canvas.draw()
    for panel_index, (aia_selection, axis) in enumerate(zip(aias, axes, strict=False)):
        position = axis.get_position()
        row, col = divmod(panel_index, ncols)
        panels.append(
            {
                "id": (
                    "aia-radio-top"
                    if panel_index == 0
                    else f"aia-radio-top-{panel_index + 1}"
                ),
                "role": "aia_radio_gaussian",
                "wavelength": str(aia_selection.background.wavelength),
                "bbox_normalized": [
                    float(position.x0),
                    float(1.0 - position.y1),
                    float(position.x1),
                    float(1.0 - position.y0),
                ],
                "xlim_arcsec": [float(value) for value in axis.get_xlim()],
                "ylim_arcsec": [float(value) for value in axis.get_ylim()],
                "show_x_coordinates": row == nrows - 1,
                "show_y_coordinates": col == 0,
            }
        )

    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=int(dpi), facecolor="white")
    image_png = output.getvalue()
    width, height = figure.canvas.get_width_height()
    figure.clear()
    metadata = {
        "schema_version": 2,
        "coordinate_system": "HPLN/HPLT arcsec",
        "image": {
            "width": int(width),
            "height": int(height),
            "sha256": hashlib.sha256(image_png).hexdigest(),
        },
        "panels": panels,
        "aia": aias[0].to_metadata_dict(),
        "aias": [item.to_metadata_dict() for item in aias],
        "radio": radios[0].to_metadata_dict(),
        "radios": [item.to_metadata_dict() for item in radios],
        "reference_radio_frequency_mhz": reference_frequency,
        "reference_radio_time_utc": reference_time.isoformat(),
        "render": {
            "dpi": int(dpi),
            "figure_size_inches": [float(value) for value in resolved_size],
            "grid_rows": nrows,
            "grid_columns": ncols,
            "aia_extents_arcsec": aia_extents,
            "display_extents_arcsec": display_extents,
            "aia_extent_arcsec": aia_extents[0],
            "display_extent_arcsec": display_extents[0],
            "canvas_extended_beyond_observation": canvas_extended,
            "radio_gaussian_evaluated_on_display_canvas": bool(
                all_extended_frequencies
            ),
            "radio_gaussian_extended_frequencies_mhz": sorted(all_extended_frequencies),
            "radio_overlay_frequency_count": len(radios),
            "aia_wavelength_count": len(aias),
            "panel_spacing": {"wspace": 0.0, "hspace": 0.0},
            "outer_coordinate_labels_only": True,
            "extended_canvas_color": canvas_color,
            "draw_gaussian_center": [
                bool(item.gaussian_config.get("draw_gaussian_center", True))
                for item in radios
            ],
            "draw_gaussian_contours": [
                bool(item.gaussian_config.get("draw_gaussian_contours", True))
                for item in radios
            ],
            "draw_gaussian_fwhm_ellipse": [
                bool(
                    item.gaussian_config.get(
                        "draw_gaussian_fwhm_ellipse",
                        True,
                    )
                )
                for item in radios
            ],
        },
    }
    return TopPanelArtifact(
        image_png=image_png,
        metadata=metadata,
        radio_frame=radios[0].frame,
        radio_frames=tuple(item.frame for item in radios),
    )


def _draw_aia_radio_axis(
    axis: Any,
    aia: AiaSelection,
    radios: tuple[RadioGaussianSelection, ...],
    *,
    display_extent_arcsec: tuple[float, float, float, float] | None,
    extended_canvas_color: str,
    overlay_colors: tuple[str, ...],
    show_legend: bool,
) -> dict[str, Any]:
    background = aia.background
    aia_extent = _aia_extent_arcsec(aia)
    display_extent = _display_extent(display_extent_arcsec, aia_extent)
    from matplotlib import colormaps

    aia_colormap = colormaps.get_cmap(_aia_colormap(aia)).with_extremes(
        bad=extended_canvas_color
    )
    axis.set_facecolor(extended_canvas_color)
    axis.imshow(
        np.asarray(background.z, dtype=float),
        extent=aia_extent,
        origin="lower",
        cmap=aia_colormap,
        aspect="equal",
        interpolation="nearest",
    )
    extended_frequencies: list[float] = []
    for radio_index, radio_selection in enumerate(radios):
        if radio_selection.fit_result is not None:
            gaussian_config = dict(radio_selection.gaussian_config)
            if len(radios) > 1:
                overlay_color = overlay_colors[radio_index % len(overlay_colors)]
                gaussian_config.update(
                    {
                        "gaussian_contour_color": overlay_color,
                        "gaussian_center_color": overlay_color,
                        "gaussian_fwhm_color": overlay_color,
                    }
                )
            draw_contours = _should_draw_expanded_gaussian_contours(
                radio_selection.fit_result,
                gaussian_config,
            )
            # Let the canonical overlay helper perform quality validation and
            # draw center/FWHM annotations, but replace its image-bounded
            # contour array with an analytic Gaussian evaluated on the full
            # display canvas.
            canonical_config = dict(gaussian_config)
            canonical_config["draw_gaussian_contours"] = False
            if canonical_config.get("gaussian_overlay_display_mode") != "none":
                overlay_gaussian_fit_on_axis(
                    axis,
                    radio_selection.fit_result,
                    radio_selection.extent_arcsec,
                    radio_selection.frame.image.shape,
                    canonical_config,
                )
            if draw_contours and _gaussian_overlay_is_visible(
                radio_selection.fit_result,
                gaussian_config,
            ):
                grid_x, grid_y, gaussian_model = _expanded_gaussian_model(
                    radio_selection.fit_result,
                    radio_selection.extent_arcsec,
                    radio_selection.frame.image.shape,
                    radio_selection.image_origin,
                    display_extent,
                )
                peak = (
                    float(np.nanmax(gaussian_model))
                    if np.any(np.isfinite(gaussian_model))
                    else np.nan
                )
                levels = sorted(
                    {
                        float(level) * peak
                        for level in gaussian_config.get(
                            "gaussian_contour_levels",
                            [0.5],
                        )
                        if np.isfinite(level) and float(level) > 0
                    }
                )
                if np.isfinite(peak) and peak > 0 and levels:
                    axis.contour(
                        grid_x,
                        grid_y,
                        gaussian_model,
                        levels=levels,
                        colors=gaussian_config.get(
                            "gaussian_contour_color",
                            "white",
                        ),
                        linewidths=gaussian_config.get(
                            "gaussian_contour_linewidth",
                            2.0,
                        ),
                        alpha=gaussian_config.get(
                            "gaussian_contour_alpha",
                            0.9,
                        ),
                    )
                    extended_frequencies.append(float(radio_selection.frame.freq_mhz))
        else:
            reason = (
                radio_selection.failure_diagnostics.get("quality_flag")
                or radio_selection.failure_diagnostics.get("reason")
                or "fit_failed"
            )
            axis.text(
                0.02,
                0.04 + radio_index * 0.045,
                (
                    f"{radio_selection.frame.freq_mhz:g} MHz Gaussian "
                    f"fit unavailable: {reason}"
                ),
                transform=axis.transAxes,
                color="yellow",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
            )
    if len(radios) > 1 and show_legend:
        for radio_index, radio_selection in enumerate(radios):
            overlay_color = overlay_colors[radio_index % len(overlay_colors)]
            axis.plot(
                [],
                [],
                color=overlay_color,
                linewidth=2.0,
                label=f"{radio_selection.frame.freq_mhz:g} MHz Gaussian",
            )
        axis.legend(loc="upper right", fontsize=8, framealpha=0.7)
    axis.set_xlim(display_extent[0], display_extent[1])
    axis.set_ylim(display_extent[2], display_extent[3])
    axis.grid(alpha=0.18, linestyle=":", linewidth=0.6)
    return {
        "aia_extent_arcsec": [float(value) for value in aia_extent],
        "display_extent_arcsec": [float(value) for value in display_extent],
        "canvas_extended": bool(
            display_extent[0] < min(aia_extent[:2])
            or display_extent[1] > max(aia_extent[:2])
            or display_extent[2] < min(aia_extent[2:])
            or display_extent[3] > max(aia_extent[2:])
        ),
        "extended_frequencies_mhz": extended_frequencies,
    }


def _normalized_aia_selections(
    value: AiaSelection | Sequence[AiaSelection],
) -> tuple[AiaSelection, ...]:
    selections = (value,) if isinstance(value, AiaSelection) else tuple(value)
    if not selections:
        raise ValueError("AIA selections must not be empty")
    if any(not isinstance(item, AiaSelection) for item in selections):
        raise TypeError("aia must be an AiaSelection or a sequence of them")
    return selections


def _normalized_canvas_color(value: str) -> str:
    color = str(value).strip().lower()
    if color not in {"black", "white"}:
        raise ValueError("extended_canvas_color must be black or white")
    return color


def _normalized_flux_plot_layout(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in {"combined", "separate"}:
        raise ValueError("flux_plot_layout must be combined or separate")
    return normalized


def _aia_extent_arcsec(
    aia: AiaSelection,
) -> tuple[float, float, float, float]:
    return (
        *_axis_edges(aia.background.x_arcsec),
        *_axis_edges(aia.background.y_arcsec),
    )


def _format_utc_z(value: datetime) -> str:
    utc = value.astimezone(UTC)
    timespec = "seconds" if utc.microsecond == 0 else "milliseconds"
    return utc.isoformat(timespec=timespec).replace("+00:00", "Z")


def _normalized_radio_selections(
    value: RadioGaussianSelection | Sequence[RadioGaussianSelection],
) -> tuple[RadioGaussianSelection, ...]:
    radios = (value,) if isinstance(value, RadioGaussianSelection) else tuple(value)
    if not radios:
        raise ValueError("radio selections must not be empty")
    if any(not isinstance(item, RadioGaussianSelection) for item in radios):
        raise TypeError("radio must be a RadioGaussianSelection or a sequence of them")
    return radios


def _should_draw_expanded_gaussian_contours(
    fit_result: Any,
    config: Mapping[str, Any],
) -> bool:
    display_mode = config.get(
        "gaussian_overlay_display_mode",
        "contours_and_fwhm",
    )
    return bool(
        display_mode in {"contours_and_fwhm", "contours_only"}
        and config.get("draw_gaussian_contours", True)
        and fit_result is not None
    )


def _gaussian_overlay_is_visible(
    fit_result: Any,
    config: Mapping[str, Any],
) -> bool:
    quality_ok = bool(
        getattr(
            fit_result,
            "overlay_valid",
            getattr(fit_result, "quality_flag", "") == "ok",
        )
    )
    if not quality_ok and config.get("gaussian_hide_all_when_fit_invalid", True):
        return False
    return quality_ok or bool(config.get("draw_low_quality_gaussian_contours", False))


def _expanded_gaussian_model(
    fit_result: Any,
    radio_extent: tuple[float, float, float, float],
    image_shape: tuple[int, int],
    image_origin: str,
    display_extent: tuple[float, float, float, float],
    *,
    samples_per_axis: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a fitted radio Gaussian across the complete HPC display canvas."""

    sample_count = max(64, min(int(samples_per_axis), 1024))
    left, right, bottom, top = (float(value) for value in radio_extent)
    display_left, display_right, display_bottom, display_top = (
        float(value) for value in display_extent
    )
    ny, nx = (int(value) for value in image_shape)
    if ny <= 0 or nx <= 0:
        raise ValueError("radio image_shape must contain positive dimensions")
    dx = (right - left) / float(nx)
    dy = (top - bottom) / float(ny)
    if not np.isfinite([dx, dy]).all() or dx == 0 or dy == 0:
        raise ValueError("radio extent must have finite non-zero dimensions")

    x_arcsec = np.linspace(display_left, display_right, sample_count)
    y_arcsec = np.linspace(display_bottom, display_top, sample_count)
    grid_x, grid_y = np.meshgrid(x_arcsec, y_arcsec)
    x_pixel = (grid_x - left) / dx - 0.5
    if str(image_origin).lower() == "upper":
        y_pixel = (top - grid_y) / dy - 0.5
    elif str(image_origin).lower() == "lower":
        y_pixel = (grid_y - bottom) / dy - 0.5
    else:
        raise ValueError(f"Unsupported radio image origin: {image_origin}")

    model = elliptical_gaussian_2d(
        (x_pixel, y_pixel),
        float(fit_result.amplitude),
        float(fit_result.center_pixel[0]),
        float(fit_result.center_pixel[1]),
        float(fit_result.sigma_pixel[0]),
        float(fit_result.sigma_pixel[1]),
        float(fit_result.theta_rad),
    )
    return grid_x, grid_y, np.asarray(model, dtype=float)


def _axis_edges(values: np.ndarray) -> tuple[float, float]:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all():
        raise ValueError("AIA display axes must be finite non-empty vectors")
    if axis.size == 1:
        step = 1.0
        return float(axis[0] - step / 2.0), float(axis[0] + step / 2.0)
    differences = np.diff(axis)
    if np.any(differences == 0) or not (
        np.all(differences > 0) or np.all(differences < 0)
    ):
        raise ValueError("AIA display axes must be strictly monotonic")
    return (
        float(axis[0] - differences[0] / 2.0),
        float(axis[-1] + differences[-1] / 2.0),
    )


def _display_extent(
    requested: tuple[float, float, float, float] | None,
    observed: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if requested is None:
        return observed
    if len(requested) != 4:
        raise ValueError("display_extent_arcsec must contain left, right, bottom, top")
    values = tuple(float(value) for value in requested)
    if not np.isfinite(values).all():
        raise ValueError("display_extent_arcsec must contain finite values")
    left, right, bottom, top = values
    if left >= right or bottom >= top:
        raise ValueError("display_extent_arcsec requires left < right and bottom < top")
    return values


def _optional_numeric_range(
    value: Any,
    *,
    label: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    low, high = (float(item) for item in value)
    if not np.isfinite([low, high]).all():
        raise ValueError(f"{label} must contain finite values")
    if low >= high:
        raise ValueError(f"{label} minimum must be below its maximum")
    return low, high


def _aia_colormap(aia: AiaSelection) -> str:
    try:
        wavelength = int(float(aia.background.wavelength))
    except (TypeError, ValueError):
        return "gray"
    configured = str(AIA_CONFIG.get(wavelength, {}).get("cmap", "gray"))
    try:
        import sunpy.visualization.colormaps  # noqa: F401
        from matplotlib import colormaps

        colormaps[configured]
    except (ImportError, KeyError, RuntimeError, ValueError):
        return "gray"
    return configured


def _utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _available_stem(directory: Path, raw_stem: str) -> str:
    stem = str(raw_stem).strip()
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename stem")
    for index in range(1, 10_000):
        suffix = "" if index == 1 else f"_{index:03d}"
        candidate = f"{stem}{suffix}"
        if not any(
            (directory / f"{candidate}.{extension}").exists()
            for extension in ("png", "json", "csv")
        ):
            return candidate
    raise RuntimeError("Unable to allocate a conflict-safe artifact filename")
