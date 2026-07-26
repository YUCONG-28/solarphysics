"""Interactive ROI components built on the existing radio frontend helpers."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from solar_apps.frontends.radio.composite_figure.composite_figure_application import (
    build_source_map_selection_figure,
    frequency_band_from_selection,
)
from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_app import (
    build_reference_figure,
    selection_to_radio_roi,
)
from solar_apps.ui.theme import apply_plotly_chrome
from solar_toolkit.radio.roi_lightcurve import RadioRoi

from ..models import (
    CompositeRequest,
    ROI_CURVE_COLUMNS,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumTimeAlignment,
    SpectrumWindow,
    parse_roi_curve_times,
)
from .composite_renderer import TopPanelArtifact

__all__ = [
    "apply_radio_roi_to_request",
    "build_roi_lightcurve_figure",
    "build_dual_flux_figure",
    "build_dual_flux_figures",
    "build_spectrum_figure",
    "build_spectrum_selection_figure",
    "build_top_panel_selection_figure",
    "radio_roi_from_selection",
    "radio_roi_json",
    "spectrum_band_from_selection",
]

_ROI_METRICS = ("raw_sum", "raw_mean", "raw_peak")


def build_spectrum_figure(
    spectrum: SpectrumWindow,
    *,
    map_time: Any | None = None,
    time_alignment: SpectrumTimeAlignment | None = None,
    display_time_range_utc: tuple[Any, Any] | None = None,
    display_frequency_range_mhz: tuple[float, float] | None = None,
    display_intensity_range: tuple[float, float] | None = None,
    theme_mode: str = "auto",
):
    """Build an interactive DART/CSO heatmap on a UTC time axis."""

    import plotly.graph_objects as go

    if not isinstance(spectrum, SpectrumWindow):
        raise TypeError("spectrum must be a SpectrumWindow")
    frequency_range = _optional_display_range(
        display_frequency_range_mhz,
        label="display_frequency_range_mhz",
    )
    intensity_range = _optional_display_range(
        display_intensity_range,
        label="display_intensity_range",
    )
    display_times = _aligned_spectrum_times(
        spectrum.time_utc,
        source=spectrum.source,
        time_alignment=time_alignment,
    )
    figure = go.Figure(
        go.Heatmap(
            z=spectrum.data,
            x=list(display_times),
            y=spectrum.frequency_mhz,
            colorscale="Turbo",
            colorbar={
                "title": f"{spectrum.polarization}<br>{spectrum.unit}",
            },
            zmin=intensity_range[0] if intensity_range is not None else None,
            zmax=intensity_range[1] if intensity_range is not None else None,
            hovertemplate=(
                "UTC=%{x|%Y-%m-%d %H:%M:%S.%L}"
                "<br>frequency=%{y:.4f} MHz"
                f"<br>{spectrum.polarization}=%{{z:.6g}} "
                f"{spectrum.unit}<extra></extra>"
            ),
        )
    )
    if map_time is not None:
        figure.add_vline(
            x=pd.Timestamp(map_time).timestamp() * 1000.0,
            line={"color": "#c2410c", "width": 1, "dash": "dash"},
            annotation_text="Top-panel UTC",
        )
    figure.update_layout(
        title=f"{spectrum.source} dynamic spectrum — {spectrum.polarization}",
        xaxis_title="Time (UTC)",
        yaxis_title="Frequency (MHz)",
        margin={"l": 70, "r": 25, "t": 65, "b": 55},
    )
    if display_time_range_utc is not None:
        figure.update_xaxes(range=list(display_time_range_utc))
    if frequency_range is not None:
        figure.update_yaxes(range=list(frequency_range))
    figure.update_layout(
        meta={
            "display_frequency_range_mhz": (
                list(frequency_range) if frequency_range is not None else None
            ),
            "display_intensity_range": (
                list(intensity_range) if intensity_range is not None else None
            ),
            "spectrum_time_alignment": (
                time_alignment.to_dict() if time_alignment is not None else None
            ),
        }
    )
    return apply_plotly_chrome(figure, theme_mode)


def build_spectrum_selection_figure(
    spectrum: SpectrumWindow,
    *,
    band: SpectrumBand | None = None,
    bands: Sequence[SpectrumBand] | None = None,
    map_time: Any | None = None,
    time_alignment: SpectrumTimeAlignment | None = None,
    display_time_range_utc: tuple[Any, Any] | None = None,
    display_frequency_range_mhz: tuple[float, float] | None = None,
    display_intensity_range: tuple[float, float] | None = None,
    theme_mode: str = "auto",
):
    """Build a box-selectable CSO/DART spectrum preview."""

    import plotly.graph_objects as go

    normalized_bands = (
        tuple(bands) if bands is not None else ((band,) if band is not None else ())
    )
    if (
        band is not None
        and bands is not None
        and (not normalized_bands or normalized_bands[0] != band)
    ):
        raise ValueError("band must match the first bands item when both are set")
    for matched_band in normalized_bands:
        if not isinstance(matched_band, SpectrumBand):
            raise TypeError("bands must contain SpectrumBand values")
        matched_band.observed_indices(spectrum.frequency_mhz)
    figure = build_spectrum_figure(
        spectrum,
        map_time=map_time,
        time_alignment=time_alignment,
        display_time_range_utc=display_time_range_utc,
        display_frequency_range_mhz=display_frequency_range_mhz,
        display_intensity_range=display_intensity_range,
        theme_mode=theme_mode,
    )
    time_indices = np.unique(
        np.linspace(
            0,
            len(spectrum.time_utc) - 1,
            min(56, len(spectrum.time_utc)),
            dtype=int,
        )
    )
    frequency_indices = np.unique(
        np.linspace(
            0,
            spectrum.frequency_mhz.size - 1,
            min(56, spectrum.frequency_mhz.size),
            dtype=int,
        )
    )
    display_times = _aligned_spectrum_times(
        spectrum.time_utc,
        source=spectrum.source,
        time_alignment=time_alignment,
    )
    grid_x: list[Any] = []
    grid_y: list[float] = []
    for frequency_index in frequency_indices:
        for time_index in time_indices:
            grid_x.append(display_times[int(time_index)])
            grid_y.append(float(spectrum.frequency_mhz[int(frequency_index)]))
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
    colors = ("#00d4ff", "#f97316", "#a855f7", "#22c55e", "#eab308", "#ef4444")
    for index, matched_band in enumerate(normalized_bands):
        color = colors[index % len(colors)]
        figure.add_hrect(
            y0=matched_band.low_mhz,
            y1=matched_band.high_mhz,
            line={"color": color, "width": 2},
            fillcolor="rgba(0, 212, 255, 0.10)",
            annotation_text=f"{matched_band.center_mhz:g} MHz ROI match",
            annotation_position="top left",
        )
    figure.update_layout(
        title=(f"{spectrum.source} dynamic spectrum — ROI-frequency matched bands"),
        dragmode="select",
        height=560,
    )
    return figure


def _optional_display_range(
    value: tuple[float, float] | None,
    *,
    label: str,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    low, high = (float(item) for item in value)
    if not np.isfinite([low, high]).all():
        raise ValueError(f"{label} must contain finite values")
    if low >= high:
        raise ValueError(f"{label} minimum must be below its maximum")
    return low, high


def build_roi_lightcurve_figure(
    curve: pd.DataFrame,
    *,
    metric: str = "raw_sum",
    theme_mode: str = "auto",
):
    """Build a multi-frequency Plotly curve without changing measurements."""

    import plotly.graph_objects as go
    from plotly.colors import qualitative

    if not isinstance(curve, pd.DataFrame):
        raise TypeError("curve must be a pandas DataFrame")
    selected_metric = str(metric).strip()
    if selected_metric not in _ROI_METRICS:
        raise ValueError(f"metric must be one of: {', '.join(_ROI_METRICS)}")
    missing = [name for name in ROI_CURVE_COLUMNS if name not in curve]
    if missing:
        raise ValueError(f"curve is missing required columns: {missing}")

    display = curve.copy(deep=False)
    display_times = parse_roi_curve_times(display)
    display_frequencies = pd.to_numeric(display["frequency"], errors="coerce")
    if not np.isfinite(display_frequencies.to_numpy(dtype=float)).all():
        raise ValueError("curve frequency must contain finite values")
    display_values = pd.to_numeric(display[selected_metric], errors="coerce")
    polarizations = (
        display["polarization"].astype(str)
        if "polarization" in display
        else pd.Series([""] * len(display), index=display.index)
    )

    figure = go.Figure()
    invalid_without_value = 0
    groups = pd.DataFrame(
        {
            "time": display_times,
            "frequency": display_frequencies.astype(float),
            "value": display_values,
            "quality": display["quality_flag"].astype(str),
            "polarization": polarizations,
        },
        index=display.index,
    )
    untimed_quality_rows = int(groups["time"].isna().sum())
    groups = groups.loc[groups["time"].notna()]
    for group_index, ((frequency, polarization), frame) in enumerate(
        groups.groupby(["frequency", "polarization"], sort=True, dropna=False)
    ):
        frame = frame.sort_values("time", kind="mergesort")
        color = qualitative.Plotly[group_index % len(qualitative.Plotly)]
        suffix = f" | {polarization}" if polarization else ""
        label = f"{frequency:g} MHz{suffix}"
        valid = frame["quality"].str.lower().eq("ok") & frame["value"].notna()
        if valid.any():
            figure.add_trace(
                go.Scattergl(
                    x=frame.loc[valid, "time"],
                    y=frame.loc[valid, "value"],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    line={"color": color},
                    marker={"color": color, "size": 6},
                    customdata=frame.loc[valid, ["quality"]],
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M:%S.%L UTC}"
                        f"<br>{frequency:g} MHz{suffix}"
                        f"<br>{selected_metric}=%{{y:.6g}}"
                        "<br>quality=%{customdata[0]}<extra></extra>"
                    ),
                )
            )
        invalid = ~frame["quality"].str.lower().eq("ok")
        invalid_with_value = invalid & frame["value"].notna()
        invalid_without_value += int((invalid & frame["value"].isna()).sum())
        if invalid_with_value.any():
            figure.add_trace(
                go.Scattergl(
                    x=frame.loc[invalid_with_value, "time"],
                    y=frame.loc[invalid_with_value, "value"],
                    mode="markers",
                    name=f"{label} (quality flagged)",
                    legendgroup=label,
                    marker={
                        "color": color,
                        "size": 9,
                        "symbol": "x",
                        "line": {"width": 1},
                    },
                    customdata=frame.loc[invalid_with_value, ["quality"]],
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M:%S.%L UTC}"
                        f"<br>{frequency:g} MHz{suffix}"
                        f"<br>{selected_metric}=%{{y:.6g}}"
                        "<br>quality=%{customdata[0]}<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        title=f"Radio ROI multi-frequency lightcurve — {selected_metric}",
        xaxis_title="Time (UTC)",
        yaxis_title=selected_metric,
        hovermode="x unified",
        legend={"title": {"text": "Frequency / polarization"}},
        margin={"l": 70, "r": 105, "t": 65, "b": 55},
        meta={
            "metric": selected_metric,
            "row_count": int(len(curve)),
            "plotted_row_count": int(len(groups)),
            "untimed_quality_rows": untimed_quality_rows,
            "quality_flagged_without_numeric_value": invalid_without_value,
        },
    )
    return apply_plotly_chrome(figure, theme_mode)


def build_dual_flux_figure(
    curve: pd.DataFrame,
    spectrum_flux: SpectrumFluxCurve | Sequence[SpectrumFluxCurve],
    *,
    metric: str = "raw_sum",
    theme_mode: str = "auto",
    map_time: Any | None = None,
    time_alignment: SpectrumTimeAlignment | None = None,
    time_alignments: Mapping[float, SpectrumTimeAlignment] | None = None,
    display_time_range_utc: tuple[Any, Any] | None = None,
):
    """Plot image-ROI and spectrum-band fluxes on independent y axes."""

    import plotly.graph_objects as go

    spectrum_fluxes = (
        (spectrum_flux,)
        if isinstance(spectrum_flux, SpectrumFluxCurve)
        else tuple(spectrum_flux)
    )
    if not spectrum_fluxes:
        raise ValueError("spectrum_flux must not be empty")
    if any(not isinstance(item, SpectrumFluxCurve) for item in spectrum_fluxes):
        raise TypeError(
            "spectrum_flux must be a SpectrumFluxCurve or a sequence of them"
        )
    figure = build_roi_lightcurve_figure(
        curve,
        metric=metric,
        theme_mode=theme_mode,
    )
    colors = ("#f97316", "#a855f7", "#22c55e", "#eab308", "#ef4444", "#06b6d4")
    spectrum_display_times: list[tuple[datetime, ...]] = []
    applied_alignments: dict[float, SpectrumTimeAlignment] = {}
    for index, spectrum_item in enumerate(spectrum_fluxes):
        item_alignment = _spectrum_flux_time_alignment(
            spectrum_item,
            default=time_alignment,
            by_frequency=time_alignments,
        )
        if item_alignment is not None:
            applied_alignments[float(spectrum_item.requested_band.center_mhz)] = (
                item_alignment
            )
        display_times = _aligned_spectrum_times(
            spectrum_item.time_utc,
            source=spectrum_item.source,
            time_alignment=item_alignment,
        )
        spectrum_display_times.append(display_times)
        band = spectrum_item.requested_band
        spectrum_label = (
            f"{spectrum_item.source} {band.center_mhz:g} MHz "
            f"({band.low_mhz:g}-{band.high_mhz:g}) {spectrum_item.polarization}"
        )
        figure.add_trace(
            go.Scattergl(
                x=list(display_times),
                y=spectrum_item.values,
                mode="lines",
                name=spectrum_label,
                line={
                    "color": colors[index % len(colors)],
                    "width": 2,
                    "dash": "dash",
                },
                connectgaps=False,
                yaxis="y2",
                customdata=np.column_stack(
                    [
                        np.full(
                            len(spectrum_item.time_utc),
                            spectrum_item.channel_count,
                        ),
                        np.full(
                            len(spectrum_item.time_utc),
                            spectrum_item.sampled_frequency_range_mhz[0],
                        ),
                        np.full(
                            len(spectrum_item.time_utc),
                            spectrum_item.sampled_frequency_range_mhz[1],
                        ),
                    ]
                ),
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M:%S.%L UTC}"
                    f"<br>{spectrum_label}"
                    f"<br>finite-channel mean=%{{y:.6g}} {spectrum_item.unit}"
                    "<br>channels=%{customdata[0]:.0f}"
                    "<br>sampled=%{customdata[1]:.6g}-%{customdata[2]:.6g} MHz"
                    "<extra></extra>"
                ),
            )
        )
    radio_times = parse_roi_curve_times(curve).dropna()
    combined_times = [
        *(value.to_pydatetime() for value in radio_times),
        *(time for times in spectrum_display_times for time in times),
    ]
    x_range = (
        list(display_time_range_utc)
        if display_time_range_utc is not None
        else [min(combined_times), max(combined_times)]
    )
    if map_time is not None:
        figure.add_vline(
            x=pd.Timestamp(map_time).timestamp() * 1000.0,
            line={"color": "#c2410c", "width": 1, "dash": "dash"},
            annotation_text="Reference UTC",
        )
    figure.update_layout(
        title="Radio image ROI + spectrum-band flux",
        yaxis={"title": metric},
        yaxis2={
            "title": (
                (
                    f"{spectrum_fluxes[0].source} "
                    f"{spectrum_fluxes[0].polarization} mean "
                    f"({spectrum_fluxes[0].unit})"
                )
                if len(spectrum_fluxes) == 1
                else (
                    f"{spectrum_fluxes[0].source} matched-band mean "
                    f"({spectrum_fluxes[0].unit})"
                )
            ),
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        margin={"l": 70, "r": 105, "t": 65, "b": 55},
        xaxis={
            "title": "Time (UTC)",
            "range": x_range,
        },
    )
    meta = dict(figure.layout.meta or {})
    meta["spectrum_flux"] = spectrum_fluxes[0].to_metadata_dict()
    meta["spectrum_flux_curves"] = [item.to_metadata_dict() for item in spectrum_fluxes]
    meta["time_alignment"] = (
        "shared_utc_dart_display_offset_no_interpolation"
        if applied_alignments
        else "shared_utc_no_interpolation"
    )
    meta["spectrum_time_alignment"] = (
        time_alignment.to_dict() if time_alignment is not None else None
    )
    meta["spectrum_flux_time_alignments"] = {
        f"{frequency:g}": alignment.to_dict()
        for frequency, alignment in applied_alignments.items()
    }
    figure.update_layout(meta=meta)
    return figure


def build_dual_flux_figures(
    curve: pd.DataFrame,
    spectrum_flux: SpectrumFluxCurve | Sequence[SpectrumFluxCurve],
    *,
    separate_by_frequency: bool = False,
    metric: str = "raw_sum",
    theme_mode: str = "auto",
    map_time: Any | None = None,
    time_alignment: SpectrumTimeAlignment | None = None,
    time_alignments: Mapping[float, SpectrumTimeAlignment] | None = None,
    display_time_range_utc: tuple[Any, Any] | None = None,
) -> tuple[Any, ...]:
    """Build one combined dual-axis chart or one chart per matched frequency."""

    spectrum_fluxes = (
        (spectrum_flux,)
        if isinstance(spectrum_flux, SpectrumFluxCurve)
        else tuple(spectrum_flux)
    )
    if not spectrum_fluxes:
        raise ValueError("spectrum_flux must not be empty")
    if not separate_by_frequency:
        figure = build_dual_flux_figure(
            curve,
            spectrum_fluxes,
            metric=metric,
            theme_mode=theme_mode,
            map_time=map_time,
            time_alignment=time_alignment,
            time_alignments=time_alignments,
            display_time_range_utc=display_time_range_utc,
        )
        meta = dict(figure.layout.meta or {})
        meta["flux_plot_layout"] = "combined"
        figure.update_layout(meta=meta)
        return (figure,)

    frequencies = pd.to_numeric(curve["frequency"], errors="raise").to_numpy(
        dtype=float
    )
    figures: list[Any] = []
    for spectrum_item in spectrum_fluxes:
        center_mhz = float(spectrum_item.requested_band.center_mhz)
        selected_curve = curve.loc[
            np.isclose(frequencies, center_mhz, rtol=0.0, atol=1e-6)
        ]
        if selected_curve.empty:
            raise ValueError(f"ROI curve has no rows matching {center_mhz:g} MHz")
        figure = build_dual_flux_figure(
            selected_curve,
            spectrum_item,
            metric=metric,
            theme_mode=theme_mode,
            map_time=map_time,
            time_alignment=time_alignment,
            time_alignments=time_alignments,
            display_time_range_utc=display_time_range_utc,
        )
        figure.update_layout(
            title=f"{center_mhz:g} MHz — radio image ROI + spectrum-band flux"
        )
        meta = dict(figure.layout.meta or {})
        meta["flux_plot_layout"] = "separate"
        meta["roi_frequency_mhz"] = center_mhz
        figure.update_layout(meta=meta)
        figures.append(figure)
    return tuple(figures)


def _aligned_spectrum_times(
    values: Sequence[datetime],
    *,
    source: str,
    time_alignment: SpectrumTimeAlignment | None,
) -> tuple[datetime, ...]:
    times = tuple(values)
    if time_alignment is None:
        return times
    if str(source).upper() != "DART":
        raise ValueError("DART time alignment cannot be applied to a CSO spectrum")
    return time_alignment.align_times(times)


def _spectrum_flux_time_alignment(
    spectrum_flux: SpectrumFluxCurve,
    *,
    default: SpectrumTimeAlignment | None,
    by_frequency: Mapping[float, SpectrumTimeAlignment] | None,
) -> SpectrumTimeAlignment | None:
    if by_frequency is None:
        return default
    center = float(spectrum_flux.requested_band.center_mhz)
    alignment = by_frequency.get(center, default)
    if alignment is not None and not isinstance(alignment, SpectrumTimeAlignment):
        raise TypeError(
            "time_alignments values must be SpectrumTimeAlignment instances"
        )
    return alignment


def build_top_panel_selection_figure(
    artifact: TopPanelArtifact,
    *,
    roi: RadioRoi | None = None,
    roi_mode: str = "box",
    low_percentile: float = 90.0,
    high_percentile: float = 99.0,
):
    """Build a radio-frame ROI selector using the established ROI frontend."""

    mode = _roi_mode(roi_mode)
    if artifact.radio_frame is not None:
        display_extent = artifact.metadata.get("render", {}).get(
            "display_extent_arcsec"
        )
        display_config = (
            {"fov": tuple(float(value) for value in display_extent)}
            if isinstance(display_extent, (list, tuple)) and len(display_extent) == 4
            else None
        )
        figure = build_reference_figure(
            artifact.radio_frame,
            roi=roi,
            roi_mode=mode,
            display_config=display_config,
            low_percentile=float(low_percentile),
            high_percentile=float(high_percentile),
        )
        figure.update_xaxes(constrain="domain")
        figure.update_yaxes(constrain="domain")
        figure.update_layout(
            title=(
                "Radio source ROI selection — "
                f"{artifact.radio_frame.freq_mhz:g} MHz "
                f"{artifact.radio_frame.pol}"
            )
        )
        meta = dict(figure.layout.meta or {})
        meta.update(
            {
                "roi_coordinate_source": "radio_source_frame",
                "radio_path": str(artifact.radio_frame.path.resolve(strict=False)),
                "radio_hdu_index": int(artifact.radio_frame.hdu_index),
                "display_percentiles": [
                    float(low_percentile),
                    float(high_percentile),
                ],
            }
        )
        figure.update_layout(meta=meta)
        return figure

    # Compatibility fallback for artifacts created before the radio frame was
    # carried alongside the rendered AIA/radio PNG.
    figure = build_source_map_selection_figure(
        artifact.image_png,
        artifact.metadata,
        roi=roi,
        roi_mode=mode,
    )
    # Keep the requested HPC limits exact on responsive/narrow layouts.  With
    # scale-linked axes Plotly otherwise expands one data range to match the
    # viewport aspect ratio, which can expose coordinates beyond the user's
    # selected canvas.  Constraining the domains preserves both equal angular
    # scale and the explicit HPLN/HPLT bounds.
    figure.update_xaxes(constrain="domain")
    figure.update_yaxes(constrain="domain")
    figure.update_layout(title="Legacy composite ROI selection")
    return figure


def radio_roi_from_selection(
    selection_event: dict[str, Any] | None,
    *,
    roi_mode: str,
    label: str = "",
) -> RadioRoi | None:
    """Convert one Plotly box/lasso event with the existing ROI normalizer."""

    return selection_to_radio_roi(
        selection_event,
        mode=_roi_mode(roi_mode),
        label=str(label),
    )


def spectrum_band_from_selection(
    selection_event: dict[str, Any] | None,
) -> SpectrumBand | None:
    """Convert the established Plotly frequency-box event to a generic band."""

    selected = frequency_band_from_selection(selection_event)
    if selected is None:
        return None
    return SpectrumBand(selected.low_mhz, selected.high_mhz)


def apply_radio_roi_to_request(
    request: CompositeRequest,
    roi: RadioRoi,
) -> CompositeRequest:
    """Return a request updated with confirmed HPLN/HPLT arcsec vertices."""

    if not isinstance(roi, RadioRoi):
        raise TypeError("roi must be a RadioRoi")
    roi_type = "box" if roi.kind == "box" else "lasso"
    return replace(
        request,
        roi_type=roi_type,
        roi_vertices_arcsec=tuple(roi.vertices_arcsec),
    )


def radio_roi_json(roi: RadioRoi) -> dict[str, Any]:
    """Return the existing reproducible arcsec-only ``RadioRoi`` JSON schema."""

    if not isinstance(roi, RadioRoi):
        raise TypeError("roi must be a RadioRoi")
    return roi.to_json_dict()


def _roi_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"box", "lasso"}:
        raise ValueError("roi_mode must be box or lasso")
    return mode
