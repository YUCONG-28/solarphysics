"""Streamlit UI for the AIA, radio ROI, and dynamic-spectrum composite."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from solar_apps.platform.layout import RuntimeLayout
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
from solar_toolkit.radio.roi_lightcurve import RadioRoi, radio_roi_from_json

from solar_apps.frontends.radio.aia_radio_composite import FRONTEND_ID
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    DEFAULT_ROI_FREQUENCIES_MHZ,
)
from solar_apps.frontends.radio.aia_radio_composite.application import (
    build_composite,
    build_dynamic_composite_video,
    build_roi_lightcurve,
    build_spectrum_flux_curves,
    build_spectrum_window,
    build_top_panel,
)
from solar_apps.frontends.radio.aia_radio_composite.models import (
    AIA_WAVELENGTHS,
    CompositeRequest,
    SpectrumBand,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    build_dual_flux_figures,
    build_spectrum_selection_figure,
    build_top_panel_selection_figure,
    radio_roi_from_selection,
    write_composite_artifacts,
)

__all__ = ["build_parser", "main"]

UI_FIELD_KEYS = (
    "aia_directory",
    "radio_directory",
    "spectrum_path",
    "output_directory",
    "aia_wave",
    "aia_waves",
    "aia_time",
    "radio_frequency",
    "radio_frequencies",
    "polarization",
    "roi_mode",
    "use_custom_fov",
    "extended_canvas_color",
    "hpln_min",
    "hpln_max",
    "hplt_min",
    "hplt_max",
    "spectrum_type",
    "cso_path_kind",
    "curve_frequencies",
    "flux_plot_layout",
    "spectrum_match_bandwidth_mhz",
    "use_custom_spectrum_frequency_range",
    "spectrum_frequency_min",
    "spectrum_frequency_max",
    "use_custom_spectrum_intensity_range",
    "spectrum_intensity_min",
    "spectrum_intensity_max",
    "metric",
    "gaussian_show_center",
    "gaussian_show_contours",
    "gaussian_contour_percent",
    "radio_display_low_percentile",
    "radio_display_high_percentile",
    "flux_time_start",
    "flux_time_end",
    "spectrum_time_start",
    "spectrum_time_end",
    "video_fps",
    "confirmed_roi_json",
    "confirmed_roi_coordinate_source",
    "spectrum_band_low",
    "spectrum_band_high",
    "confirmed_spectrum_band_json",
    "confirmed_spectrum_source_signature",
)

ROI_COORDINATE_SOURCE = "radio_source_frame"


def build_parser() -> argparse.ArgumentParser:
    """Build the direct Streamlit application parser."""

    parser = argparse.ArgumentParser(
        prog="solar-apps frontend aia-radio-composite",
        description="Build an AIA/radio/ROI/dynamic-spectrum composite.",
    )
    parser.add_argument("--aia-dir", default="")
    parser.add_argument("--radio-dir", default="")
    parser.add_argument("--spectrum-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allowed-roots", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Streamlit application body."""

    _run_streamlit_app(argv)
    return 0


def _run_streamlit_app(argv: list[str] | None = None) -> None:
    import streamlit as st

    args, _unknown = build_parser().parse_known_args(argv)
    st.set_page_config(page_title="AIA Radio Composite", layout="wide")
    layout = RuntimeLayout.discover()
    state_store = frontend_state_store(FRONTEND_ID, layout=layout)
    try:
        allowed_roots = resolve_streamlit_allowed_roots(args.allowed_roots)
    except Exception as exc:
        allowed_roots = ()
        st.error(f"Path configuration error: {exc}")
    protected_output = layout.outputs_dir / "aia_radio_composite"
    path_policy = PathAccessPolicy.create(
        allowed_roots,
        protected_output_roots=(protected_output,),
        base_directory=layout.repo_root,
    )
    path_memory = frontend_path_memory(path_policy.output_roots, layout=layout)
    theme_mode = render_streamlit_theme(
        st,
        frontend_id=FRONTEND_ID,
        state_store=state_store,
        path_memory=path_memory,
    )
    bind_streamlit_fields(
        st,
        state_store,
        frontend_id=FRONTEND_ID,
        field_keys=UI_FIELD_KEYS,
    )
    st.session_state.setdefault("aia_wave", 171)
    if not isinstance(st.session_state.get("aia_waves"), (list, tuple)):
        st.session_state["aia_waves"] = [int(st.session_state.get("aia_wave", 171))]
    st.session_state.setdefault(
        "radio_frequencies",
        [int(st.session_state.get("radio_frequency", 149))],
    )
    _restore_confirmed_roi(st)
    st.title("AIA + Radio Composite")
    st.caption(
        "Quality-controlled Gaussian source map, multi-frequency ROI "
        "lightcurve, and synchronized DART/CSO dynamic spectrum."
    )

    with st.sidebar:
        st.header("Data")
        spectrum_type = st.selectbox(
            "Spectrum type",
            options=["DART", "CSO"],
            key="spectrum_type",
        ).lower()
        cso_path_kind = (
            st.radio(
                "CSO source",
                options=["FITS file", "Directory"],
                horizontal=True,
                key="cso_path_kind",
            )
            if spectrum_type == "cso"
            else "Directory"
        )
        aia_directory = render_native_path_input(
            st,
            "AIA directory",
            key="aia_directory",
            initial_value=args.aia_dir,
            roots=path_policy.input_roots,
            kind="directory",
            frontend_id=FRONTEND_ID,
            operation="aia-input",
            state_store=state_store,
            stacked=True,
        )
        radio_directory = render_native_path_input(
            st,
            "Radio directory",
            key="radio_directory",
            initial_value=args.radio_dir,
            roots=path_policy.input_roots,
            kind="directory",
            frontend_id=FRONTEND_ID,
            operation="radio-input",
            state_store=state_store,
            stacked=True,
        )
        spectrum_path = render_native_path_input(
            st,
            "Spectrum directory or FITS file",
            key="spectrum_path",
            initial_value=args.spectrum_path,
            roots=path_policy.input_roots,
            kind=(
                "file"
                if spectrum_type == "cso" and cso_path_kind == "FITS file"
                else "directory"
            ),
            extensions=(".fits", ".fit", ".fts"),
            frontend_id=FRONTEND_ID,
            operation=f"{spectrum_type}-input",
            state_store=state_store,
            stacked=True,
        )
        output_directory = render_native_path_input(
            st,
            "Output directory",
            key="output_directory",
            initial_value=args.output_dir or str(protected_output),
            roots=path_policy.output_roots,
            kind="directory",
            frontend_id=FRONTEND_ID,
            operation="composite-output",
            state_store=state_store,
            stacked=True,
        )
        if not allowed_roots:
            st.error(
                "No allowed roots are configured. Add data directories to "
                "Local/configs/paths.local.yaml to enable Browse."
            )
        st.header("Controls")
        aia_waves = st.multiselect(
            "AIA waves (Å)",
            options=sorted(AIA_WAVELENGTHS),
            key="aia_waves",
            help=(
                "Each selected wavelength is rendered in its own panel, in "
                "selection order, with the same matched radio overlays."
            ),
        )
        aia_time = st.text_input(
            "Reference UTC",
            value="2025-01-24T04:48:32Z",
            key="aia_time",
        )
        radio_frequencies = st.multiselect(
            "AIA overlay radio frequencies (MHz)",
            options=[149, 164, 190, 205, 223, 238],
            default=[149],
            key="radio_frequencies",
            help=(
                "One Gaussian source overlay is fitted and drawn for every "
                "selected radio frequency."
            ),
        )
        polarization = st.selectbox(
            "Polarization",
            options=["RR", "LL", "RR+LL"],
            key="polarization",
        )
        roi_mode = st.radio(
            "ROI mode",
            options=["box", "lasso"],
            horizontal=True,
            key="roi_mode",
        )
        use_custom_fov = st.checkbox(
            "Use custom HPC display range",
            value=False,
            key="use_custom_fov",
            help=(
                "Set the displayed HPLN/HPLT bounds in arcsec. Bounds outside "
                "the AIA observation expand the canvas without extrapolating data."
            ),
        )
        extended_canvas_color = st.selectbox(
            "Extended canvas color",
            options=["black", "white"],
            format_func=lambda value: value.capitalize(),
            key="extended_canvas_color",
            help="Color used only where the selected HPC canvas extends beyond AIA data.",
        )
        fov_columns = st.columns(2)
        with fov_columns[0]:
            hpln_min = float(
                st.number_input(
                    "HPLN min (arcsec)",
                    value=-1200.0,
                    key="hpln_min",
                )
            )
            hplt_min = float(
                st.number_input(
                    "HPLT min (arcsec)",
                    value=-1200.0,
                    key="hplt_min",
                )
            )
        with fov_columns[1]:
            hpln_max = float(
                st.number_input(
                    "HPLN max (arcsec)",
                    value=1200.0,
                    key="hpln_max",
                )
            )
            hplt_max = float(
                st.number_input(
                    "HPLT max (arcsec)",
                    value=1200.0,
                    key="hplt_max",
                )
            )
        curve_frequencies = st.multiselect(
            "ROI frequencies (MHz)",
            options=[149, 164, 190, 205, 223, 238],
            default=[int(value) for value in DEFAULT_ROI_FREQUENCIES_MHZ],
            key="curve_frequencies",
        )
        spectrum_match_bandwidth_mhz = float(
            st.number_input(
                "Matched spectrum bandwidth (MHz)",
                min_value=0.001,
                value=2.0,
                step=0.25,
                key="spectrum_match_bandwidth_mhz",
                help=(
                    "Each selected ROI imaging frequency is used as the center "
                    "of one spectrum band with this full width."
                ),
            )
        )
        with st.expander("Spectrum display", expanded=False):
            use_custom_spectrum_frequency_range = st.checkbox(
                "Use custom spectrum frequency range",
                value=False,
                key="use_custom_spectrum_frequency_range",
            )
            spectrum_frequency_columns = st.columns(2)
            with spectrum_frequency_columns[0]:
                spectrum_frequency_min = float(
                    st.number_input(
                        "Spectrum frequency min (MHz)",
                        value=140.0,
                        key="spectrum_frequency_min",
                        disabled=not use_custom_spectrum_frequency_range,
                    )
                )
            with spectrum_frequency_columns[1]:
                spectrum_frequency_max = float(
                    st.number_input(
                        "Spectrum frequency max (MHz)",
                        value=500.0,
                        key="spectrum_frequency_max",
                        disabled=not use_custom_spectrum_frequency_range,
                    )
                )
            use_custom_spectrum_intensity_range = st.checkbox(
                "Use custom spectrum intensity range",
                value=False,
                key="use_custom_spectrum_intensity_range",
            )
            spectrum_intensity_columns = st.columns(2)
            with spectrum_intensity_columns[0]:
                spectrum_intensity_min = float(
                    st.number_input(
                        "Spectrum intensity min",
                        value=0.0,
                        key="spectrum_intensity_min",
                        disabled=not use_custom_spectrum_intensity_range,
                    )
                )
            with spectrum_intensity_columns[1]:
                spectrum_intensity_max = float(
                    st.number_input(
                        "Spectrum intensity max",
                        value=10.0,
                        key="spectrum_intensity_max",
                        disabled=not use_custom_spectrum_intensity_range,
                    )
                )
        metric = st.selectbox(
            "ROI metric",
            options=["raw_sum", "raw_mean", "raw_peak"],
            key="metric",
        )
        flux_plot_layout = st.radio(
            "Flux plot layout",
            options=["Combined", "One chart per frequency"],
            horizontal=True,
            key="flux_plot_layout",
            help=(
                "Combine all selected frequencies in one dual-axis chart, or "
                "draw one chart per frequency with its matching image-ROI and "
                "spectrum-band flux."
            ),
        )
        with st.expander("Gaussian display", expanded=False):
            gaussian_show_center = st.checkbox(
                "Show Gaussian fitted center",
                value=True,
                key="gaussian_show_center",
            )
            gaussian_show_contours = st.checkbox(
                "Show Gaussian fitted contour",
                value=True,
                key="gaussian_show_contours",
            )
            gaussian_contour_percent = float(
                st.number_input(
                    "Gaussian contour (% of fitted peak)",
                    min_value=1.0,
                    max_value=99.99,
                    value=95.0,
                    step=1.0,
                    key="gaussian_contour_percent",
                )
            )
        with st.expander("Radio source intensity", expanded=False):
            radio_display_low_percentile = float(
                st.number_input(
                    "Radio display low percentile",
                    min_value=0.0,
                    max_value=100.0,
                    value=90.0,
                    step=0.5,
                    key="radio_display_low_percentile",
                )
            )
            radio_display_high_percentile = float(
                st.number_input(
                    "Radio display high percentile",
                    min_value=0.0,
                    max_value=100.0,
                    value=99.0,
                    step=0.5,
                    key="radio_display_high_percentile",
                )
            )
        with st.expander("UTC display windows", expanded=True):
            flux_time_start = st.text_input(
                "Flux UTC start",
                value="2025-01-24T04:48:30Z",
                key="flux_time_start",
            )
            flux_time_end = st.text_input(
                "Flux UTC end",
                value="2025-01-24T04:49:00Z",
                key="flux_time_end",
            )
            spectrum_time_start = st.text_input(
                "Spectrum UTC start",
                value="2025-01-24T04:48:30Z",
                key="spectrum_time_start",
            )
            spectrum_time_end = st.text_input(
                "Spectrum UTC end",
                value="2025-01-24T04:49:00Z",
                key="spectrum_time_end",
            )
        with st.expander("Video export", expanded=False):
            video_fps = int(
                st.number_input(
                    "Video FPS",
                    min_value=1,
                    max_value=30,
                    value=6,
                    step=1,
                    key="video_fps",
                )
            )
            st.caption(
                "Dynamic mode creates one frame per real observation of the "
                "first selected radio frequency. FPS controls playback only."
            )
        display_extent_arcsec = (
            (hpln_min, hpln_max, hplt_min, hplt_max) if use_custom_fov else None
        )
        spectrum_display_frequency_range = (
            (spectrum_frequency_min, spectrum_frequency_max)
            if use_custom_spectrum_frequency_range
            else None
        )
        spectrum_display_intensity_range = (
            (spectrum_intensity_min, spectrum_intensity_max)
            if use_custom_spectrum_intensity_range
            else None
        )
        gaussian_overrides = _gaussian_display_overrides(
            show_center=gaussian_show_center,
            show_contours=gaussian_show_contours,
            contour_percent=gaussian_contour_percent,
        )

    try:
        if not aia_waves:
            raise ValueError("Select at least one AIA wavelength")
        aia_wave = int(aia_waves[0])
        st.session_state["aia_wave"] = aia_wave
        if not radio_frequencies:
            raise ValueError("Select at least one AIA overlay radio frequency")
        radio_frequency = float(radio_frequencies[0])
        st.session_state["radio_frequency"] = radio_frequency
        request = _request_from_controls(
            aia_directory=aia_directory,
            aia_wave=aia_wave,
            aia_time=aia_time,
            radio_directory=radio_directory,
            radio_frequency=radio_frequency,
            polarization=polarization,
            roi=_session_roi(st) or _placeholder_roi(),
            spectrum_type=spectrum_type,
            spectrum_path=spectrum_path,
            path_policy=path_policy,
        )
    except Exception as exc:
        st.warning(f"Complete or correct the inputs to continue: {exc}")
        request = None
    try:
        flux_time_range = _validated_utc_range(
            flux_time_start,
            flux_time_end,
            label="Flux UTC window",
        )
        spectrum_time_range = _validated_utc_range(
            spectrum_time_start,
            spectrum_time_end,
            label="Spectrum UTC window",
        )
        aligned_time_range = _combined_utc_range(
            flux_time_range,
            spectrum_time_range,
        )
        if radio_display_low_percentile >= radio_display_high_percentile:
            raise ValueError(
                "Radio display low percentile must be below the high percentile"
            )
        if spectrum_display_frequency_range is not None and (
            spectrum_display_frequency_range[0] < 0
            or spectrum_display_frequency_range[0]
            >= spectrum_display_frequency_range[1]
        ):
            raise ValueError(
                "Spectrum frequency minimum must be non-negative and below the maximum"
            )
        if (
            spectrum_display_intensity_range is not None
            and spectrum_display_intensity_range[0]
            >= spectrum_display_intensity_range[1]
        ):
            raise ValueError("Spectrum intensity minimum must be below the maximum")
        if request is not None and not (
            aligned_time_range[0] <= request.aia_time <= aligned_time_range[1]
        ):
            raise ValueError(
                "Reference UTC must lie inside the combined flux/spectrum "
                "display range so its dashed line is visible"
            )
    except Exception as exc:
        st.warning(f"Correct the display controls to continue: {exc}")
        flux_time_range = None
        spectrum_time_range = None
        aligned_time_range = None

    spectrum_render_signature = json.dumps(
        {
            "frequency_range_mhz": spectrum_display_frequency_range,
            "intensity_range": spectrum_display_intensity_range,
            "flux_plot_layout": flux_plot_layout,
        },
        sort_keys=True,
    )
    previous_spectrum_render_signature = st.session_state.get(
        "_spectrum_render_signature"
    )
    if (
        previous_spectrum_render_signature is not None
        and previous_spectrum_render_signature != spectrum_render_signature
    ):
        for key in (
            "aia_radio_result",
            "aia_radio_artifact",
            "aia_radio_video_artifact",
        ):
            st.session_state.pop(key, None)
    st.session_state["_spectrum_render_signature"] = spectrum_render_signature

    top_signature = json.dumps(
        {
            "request": (
                {
                    "aia_directory": str(request.aia_directory),
                    "aia_waves": [int(value) for value in aia_waves],
                    "aia_time": request.aia_time.isoformat(),
                    "radio_directory": str(request.radio_directory),
                    "radio_frequencies": [float(value) for value in radio_frequencies],
                    "polarization": request.polarization,
                }
                if request is not None
                else None
            ),
            "display_extent_arcsec": display_extent_arcsec,
            "extended_canvas_color": extended_canvas_color,
            "gaussian_overrides": gaussian_overrides,
        },
        sort_keys=True,
    )
    previous_top_signature = st.session_state.get("_top_controls_signature")
    if previous_top_signature is not None and previous_top_signature != top_signature:
        for key in (
            "aia_radio_top",
            "aia_radio_result",
            "aia_radio_artifact",
            "aia_radio_video_artifact",
        ):
            st.session_state.pop(key, None)
    st.session_state["_top_controls_signature"] = top_signature

    st.subheader("Panel 1 — AIA + Radio Gaussian")
    if st.button(
        "Build top panel",
        disabled=request is None,
        type="primary",
    ):
        try:
            st.session_state["aia_radio_top"] = build_top_panel(
                request,
                radio_frequencies_mhz=radio_frequencies,
                aia_waves=aia_waves,
                display_extent_arcsec=display_extent_arcsec,
                extended_canvas_color=extended_canvas_color,
                gaussian_overrides=gaussian_overrides,
            )
        except Exception as exc:
            st.error(f"Top-panel generation failed: {exc}")
    top = st.session_state.get("aia_radio_top")
    reference_display_time = request.aia_time if request is not None else None
    if top is not None:
        reference_value = top.metadata.get("reference_radio_time_utc")
        if reference_value:
            reference_display_time = pd.Timestamp(reference_value).to_pydatetime()
    if top is not None:
        st.image(
            top.image_png,
            caption="AIA background + radio Gaussian preview",
            width="stretch",
        )
        st.markdown("#### Radio source ROI selection")
        st.caption(
            "Draw the ROI on the first selected frequency's matched radio FITS "
            "image. "
            "The AIA background above is preview-only."
        )
        selection_figure = build_top_panel_selection_figure(
            top,
            roi=_session_roi(st),
            roi_mode=roi_mode,
            low_percentile=radio_display_low_percentile,
            high_percentile=radio_display_high_percentile,
        )
        event = st.plotly_chart(
            selection_figure,
            width="stretch",
            on_select="rerun",
            selection_mode=(roi_mode,),
            key="aia_radio_roi_selector",
        )
        candidate = radio_roi_from_selection(
            event,
            roi_mode=roi_mode,
            label="Radio source ROI",
        )
        if candidate is not None:
            st.session_state["aia_radio_candidate_roi"] = candidate.to_json_dict()
        controls = st.columns(2)
        if controls[0].button(
            "Confirm ROI",
            disabled=_candidate_roi(st) is None,
        ):
            st.session_state["aia_radio_confirmed_roi"] = _candidate_roi(
                st
            ).to_json_dict()
            st.session_state["confirmed_roi_json"] = json.dumps(
                st.session_state["aia_radio_confirmed_roi"],
                sort_keys=True,
            )
            st.session_state["confirmed_roi_coordinate_source"] = ROI_COORDINATE_SOURCE
            st.rerun()
        if controls[1].button(
            "Clear ROI",
            disabled=_session_roi(st) is None and _candidate_roi(st) is None,
        ):
            st.session_state.pop("aia_radio_candidate_roi", None)
            st.session_state.pop("aia_radio_confirmed_roi", None)
            st.session_state["confirmed_roi_json"] = ""
            st.session_state["confirmed_roi_coordinate_source"] = ""
            st.rerun()
        if _session_roi(st) is not None:
            st.success("Radio-source ROI confirmed in HPLN/HPLT arcsec.")
            st.json(_session_roi(st).to_json_dict(), expanded=False)
    else:
        st.info(
            "Build the top panel, then draw and confirm a box or lasso ROI "
            "on the matched radio source image."
        )

    confirmed_roi = _session_roi(st)
    source_signature = (
        json.dumps(
            {
                "type": request.spectrum_type,
                "path": str(request.spectrum_path),
                "polarization": request.polarization,
                "time_range": (
                    [value.isoformat() for value in spectrum_time_range]
                    if spectrum_time_range
                    else None
                ),
            },
            sort_keys=True,
        )
        if request is not None
        else ""
    )
    previous_source_signature = st.session_state.get("_spectrum_source_signature")
    if (
        previous_source_signature is not None
        and previous_source_signature != source_signature
    ):
        _clear_spectrum_state(st, clear_confirmed=True)
    st.session_state["_spectrum_source_signature"] = source_signature
    if (
        st.session_state.get("confirmed_spectrum_source_signature")
        and st.session_state["confirmed_spectrum_source_signature"] != source_signature
    ):
        _clear_spectrum_state(st, clear_confirmed=True)

    st.subheader("Panel 2 — ROI-matched spectrum bands + dual-axis flux")
    if st.button(
        "Load CSO / DART spectrum",
        disabled=request is None or spectrum_time_range is None,
    ):
        try:
            st.session_state["aia_radio_spectrum"] = build_spectrum_window(
                request,
                time_range_utc=spectrum_time_range,
            )
            st.session_state["spectrum_band_revision"] = (
                int(st.session_state.get("spectrum_band_revision", 0)) + 1
            )
        except Exception as exc:
            st.error(f"Spectrum loading failed: {exc}")
    spectrum = st.session_state.get("aia_radio_spectrum")
    matched_bands: tuple[SpectrumBand, ...] = ()
    if spectrum is not None:
        try:
            matched_bands = _matched_spectrum_bands(
                curve_frequencies,
                spectrum_match_bandwidth_mhz,
                spectrum,
            )
        except ValueError as exc:
            st.error(str(exc))
        selection_figure = build_spectrum_selection_figure(
            spectrum,
            bands=matched_bands,
            map_time=reference_display_time,
            display_time_range_utc=aligned_time_range,
            display_frequency_range_mhz=spectrum_display_frequency_range,
            display_intensity_range=spectrum_display_intensity_range,
            theme_mode=theme_mode,
        )
        st.plotly_chart(
            selection_figure,
            width="stretch",
            key="aia_radio_spectrum_matched_bands",
        )
        if matched_bands:
            st.success(
                f"Automatically matched {len(matched_bands)} spectrum band(s) "
                "to the selected ROI imaging frequencies."
            )
            st.dataframe(
                [
                    {
                        "ROI frequency (MHz)": float(frequency),
                        "spectrum low (MHz)": band.low_mhz,
                        "spectrum high (MHz)": band.high_mhz,
                        "bandwidth (MHz)": band.bandwidth_mhz,
                        "original channels": int(
                            band.observed_indices(spectrum.frequency_mhz).size
                        ),
                    }
                    for frequency, band in zip(
                        curve_frequencies,
                        matched_bands,
                        strict=True,
                    )
                ],
                width="stretch",
            )
        st.json(spectrum.to_metadata_dict(), expanded=False)
    else:
        st.info(
            "Load a CSO or DART spectrum. Panel 2 will automatically center one "
            "spectrum band on every selected ROI imaging frequency."
        )

    dual_signature = json.dumps(
        {
            "roi": confirmed_roi.to_json_dict() if confirmed_roi is not None else None,
            "frequencies": [float(value) for value in curve_frequencies],
            "metric": metric,
            "radio_directory": str(request.radio_directory) if request else "",
            "polarization": request.polarization if request else "",
            "spectrum_source": source_signature,
            "bands": [band.to_dict() for band in matched_bands],
            "bandwidth_mhz": spectrum_match_bandwidth_mhz,
            "flux_time_range": (
                [value.isoformat() for value in flux_time_range]
                if flux_time_range
                else None
            ),
            "spectrum_time_range": (
                [value.isoformat() for value in spectrum_time_range]
                if spectrum_time_range
                else None
            ),
        },
        sort_keys=True,
    )
    previous_dual_signature = st.session_state.get("_dual_flux_signature")
    if (
        previous_dual_signature is not None
        and previous_dual_signature != dual_signature
    ):
        _clear_dual_flux_state(st)
    st.session_state["_dual_flux_signature"] = dual_signature

    if st.button(
        "Extract dual-axis flux",
        disabled=(
            request is None
            or confirmed_roi is None
            or not matched_bands
            or spectrum is None
            or not curve_frequencies
            or flux_time_range is None
        ),
    ):
        try:
            st.session_state["aia_radio_curve"] = build_roi_lightcurve(
                request,
                confirmed_roi,
                frequencies_mhz=curve_frequencies,
                time_start=flux_time_range[0],
                time_end=flux_time_range[1],
            )
            st.session_state["aia_radio_spectrum_flux_curves"] = (
                build_spectrum_flux_curves(
                    request,
                    matched_bands,
                    time_range_utc=flux_time_range,
                )
            )
        except Exception as exc:
            st.error(f"Dual-axis flux extraction failed: {exc}")
    curve = st.session_state.get("aia_radio_curve")
    spectrum_fluxes = st.session_state.get("aia_radio_spectrum_flux_curves")
    if curve is not None and spectrum_fluxes:
        try:
            dual_figures = build_dual_flux_figures(
                curve,
                spectrum_fluxes,
                separate_by_frequency=flux_plot_layout == "One chart per frequency",
                metric=metric,
                theme_mode=theme_mode,
                map_time=reference_display_time,
                display_time_range_utc=aligned_time_range,
            )
            for dual_figure in dual_figures:
                st.plotly_chart(dual_figure, width="stretch")
            untimed = sum(
                int(figure.layout.meta.get("untimed_quality_rows", 0))
                for figure in dual_figures
            )
            if untimed:
                st.warning(
                    f"{untimed} quality-flagged radio row(s) have no valid UTC "
                    "and are retained in the ROI CSV but omitted from the plot."
                )
        except Exception as exc:
            st.error(f"Dual-axis flux display failed: {exc}")
        st.dataframe(curve, width="stretch")
        st.dataframe(
            pd.concat(
                [
                    item.to_frame().assign(band_index=index)
                    for index, item in enumerate(spectrum_fluxes)
                ],
                ignore_index=True,
            ),
            width="stretch",
        )
    else:
        st.info(
            "Confirm the image ROI, select one or more ROI frequencies, and load "
            "a spectrum before extracting matched flux."
        )

    st.subheader("Panel 3 — Composite export")
    if st.button(
        "Generate synchronized composite",
        disabled=(
            request is None
            or confirmed_roi is None
            or not matched_bands
            or not curve_frequencies
            or flux_time_range is None
            or spectrum_time_range is None
        ),
        type="primary",
    ):
        try:
            result, artifact = build_composite(
                request,
                confirmed_roi,
                frequencies_mhz=curve_frequencies,
                radio_frequencies_mhz=radio_frequencies,
                aia_waves=aia_waves,
                spectrum_bands=matched_bands,
                roi_time_range_utc=flux_time_range,
                spectrum_time_range_utc=spectrum_time_range,
                display_extent_arcsec=display_extent_arcsec,
                extended_canvas_color=extended_canvas_color,
                spectrum_frequency_range_mhz=spectrum_display_frequency_range,
                spectrum_intensity_range=spectrum_display_intensity_range,
                flux_plot_layout=(
                    "separate"
                    if flux_plot_layout == "One chart per frequency"
                    else "combined"
                ),
                gaussian_overrides=gaussian_overrides,
                metric=metric,
            )
            st.session_state["aia_radio_result"] = result
            st.session_state["aia_radio_artifact"] = artifact
            st.session_state.pop("aia_radio_video_artifact", None)
        except Exception as exc:
            st.error(f"Composite generation failed: {exc}")
    artifact = st.session_state.get("aia_radio_artifact")
    if artifact is not None:
        st.image(artifact.image_png, width="stretch")
        downloads = st.columns(4)
        downloads[0].download_button(
            "Download PNG",
            artifact.image_png,
            file_name="aia_radio_composite.png",
            mime="image/png",
        )
        downloads[1].download_button(
            "Download metadata JSON",
            artifact.metadata_json,
            file_name="aia_radio_composite.json",
            mime="application/json",
        )
        downloads[2].download_button(
            "Download ROI CSV",
            artifact.roi_curve_csv,
            file_name="aia_radio_composite.csv",
            mime="text/csv",
        )
        if artifact.spectrum_flux_csv is not None:
            downloads[3].download_button(
                "Download spectrum flux CSV",
                artifact.spectrum_flux_csv,
                file_name="aia_radio_composite_spectrum_flux.csv",
                mime="text/csv",
            )
        if st.button(
            "Save PNG / JSON / ROI CSV / spectrum CSV",
            disabled=not output_directory.strip(),
        ):
            try:
                paths = write_composite_artifacts(
                    artifact,
                    path_policy.output_directory(output_directory),
                )
                st.success(f"Saved to {paths['png'].parent}")
                st.json(
                    {name: str(path) for name, path in paths.items()},
                    expanded=False,
                )
            except Exception as exc:
                st.error(f"Artifact save failed: {exc}")

    result = st.session_state.get("aia_radio_result")
    if result is not None and aligned_time_range is not None:
        video_signature = json.dumps(
            {
                "result": result.to_metadata_dict(),
                "time_range": [value.isoformat() for value in aligned_time_range],
                "fps": video_fps,
                "metric": metric,
                "aia_waves": [int(value) for value in aia_waves],
                "radio_frequencies": [float(value) for value in radio_frequencies],
                "display_extent_arcsec": display_extent_arcsec,
                "extended_canvas_color": extended_canvas_color,
                "spectrum_display_frequency_range": spectrum_display_frequency_range,
                "spectrum_display_intensity_range": spectrum_display_intensity_range,
                "gaussian_overrides": gaussian_overrides,
            },
            sort_keys=True,
        )
        previous_video_signature = st.session_state.get("_video_signature")
        if (
            previous_video_signature is not None
            and previous_video_signature != video_signature
        ):
            st.session_state.pop("aia_radio_video_artifact", None)
        st.session_state["_video_signature"] = video_signature
        st.markdown("#### Synchronized MP4 video")
        st.caption(
            "The first selected radio frequency supplies the real observation "
            "timeline. Every frame rematches all selected radio frequencies "
            "and AIA wavelengths; incomplete times are skipped. FPS changes "
            "playback speed only, with no interpolation or repeated frames."
        )
        if st.button("Generate MP4 video", type="primary"):
            try:
                with st.spinner("Rendering synchronized video frames..."):
                    st.session_state["aia_radio_video_artifact"] = (
                        build_dynamic_composite_video(
                            result,
                            request,
                            aia_waves=aia_waves,
                            radio_frequencies_mhz=radio_frequencies,
                            time_start=aligned_time_range[0],
                            time_end=aligned_time_range[1],
                            display_extent_arcsec=display_extent_arcsec,
                            extended_canvas_color=extended_canvas_color,
                            gaussian_overrides=gaussian_overrides,
                            metric=metric,
                            fps=video_fps,
                        )
                    )
            except Exception as exc:
                st.error(f"Video generation failed: {exc}")
        video_artifact = st.session_state.get("aia_radio_video_artifact")
        if video_artifact is not None:
            st.video(video_artifact.video_mp4, format="video/mp4")
            video_downloads = st.columns(2)
            video_downloads[0].download_button(
                "Download MP4 video",
                video_artifact.video_mp4,
                file_name="aia_radio_composite.mp4",
                mime="video/mp4",
            )
            video_downloads[1].download_button(
                "Download video metadata JSON",
                video_artifact.metadata_json,
                file_name="aia_radio_composite_video.json",
                mime="application/json",
            )


def _request_from_controls(
    *,
    aia_directory: str,
    aia_wave: int,
    aia_time: str,
    radio_directory: str,
    radio_frequency: float,
    polarization: str,
    roi: RadioRoi,
    spectrum_type: str,
    spectrum_path: str,
    path_policy: PathAccessPolicy | None = None,
) -> CompositeRequest:
    """Create one validated request from sidebar values."""

    aia_path = (
        path_policy.input_directory(aia_directory)
        if path_policy is not None
        else Path(aia_directory)
    )
    radio_path = (
        path_policy.input_directory(radio_directory)
        if path_policy is not None
        else Path(radio_directory)
    )
    if path_policy is None:
        normalized_spectrum_path = Path(spectrum_path)
    elif spectrum_type == "dart" or Path(spectrum_path).is_dir():
        normalized_spectrum_path = path_policy.input_directory(spectrum_path)
    else:
        normalized_spectrum_path = path_policy.input_file(spectrum_path)
    return CompositeRequest(
        aia_directory=aia_path,
        aia_wave=aia_wave,
        aia_time=_utc_datetime(aia_time),
        radio_directory=radio_path,
        radio_frequency=radio_frequency,
        polarization=polarization,
        roi_type="box" if roi.kind == "box" else "lasso",
        roi_vertices_arcsec=roi.vertices_arcsec,
        spectrum_type=spectrum_type,
        spectrum_path=normalized_spectrum_path,
    )


def _placeholder_roi() -> RadioRoi:
    return RadioRoi.from_box(-1.0, -1.0, 1.0, 1.0, label="placeholder")


def _candidate_roi(st: Any) -> RadioRoi | None:
    payload = st.session_state.get("aia_radio_candidate_roi")
    return radio_roi_from_json(payload) if payload else None


def _session_roi(st: Any) -> RadioRoi | None:
    payload = st.session_state.get("aia_radio_confirmed_roi")
    return radio_roi_from_json(payload) if payload else None


def _restore_confirmed_roi(st: Any) -> None:
    """Restore only ROIs confirmed on the radio-source coordinate image."""

    if st.session_state.get("aia_radio_confirmed_roi"):
        return
    payload = st.session_state.get("confirmed_roi_json")
    if not payload:
        return
    if st.session_state.get("confirmed_roi_coordinate_source") != ROI_COORDINATE_SOURCE:
        st.session_state["confirmed_roi_json"] = ""
        st.session_state["confirmed_roi_coordinate_source"] = ""
        return
    try:
        decoded = json.loads(str(payload))
        roi = radio_roi_from_json(decoded)
    except TypeError, ValueError, json.JSONDecodeError:
        st.session_state["confirmed_roi_json"] = ""
        return
    st.session_state["aia_radio_confirmed_roi"] = roi.to_json_dict()


def _session_spectrum_band(st: Any) -> SpectrumBand | None:
    payload = st.session_state.get("aia_radio_confirmed_spectrum_band")
    if not payload:
        return None
    try:
        return SpectrumBand(
            float(payload["low_mhz"]),
            float(payload["high_mhz"]),
        )
    except KeyError, TypeError, ValueError:
        return None


def _matched_spectrum_bands(
    roi_frequencies_mhz: list[float] | tuple[float, ...],
    bandwidth_mhz: float,
    spectrum: SpectrumWindow,
) -> tuple[SpectrumBand, ...]:
    """Match every selected ROI frequency to an observed spectrum band."""

    width = float(bandwidth_mhz)
    if width <= 0:
        raise ValueError("Matched spectrum bandwidth must be greater than zero")
    if not roi_frequencies_mhz:
        return ()
    half_width = width / 2.0
    bands = tuple(
        SpectrumBand(float(frequency) - half_width, float(frequency) + half_width)
        for frequency in roi_frequencies_mhz
    )
    for frequency, band in zip(roi_frequencies_mhz, bands, strict=True):
        try:
            band.observed_indices(spectrum.frequency_mhz)
        except ValueError as exc:
            raise ValueError(
                f"ROI frequency {float(frequency):g} MHz cannot be matched "
                f"with a {width:g} MHz spectrum band: {exc}"
            ) from exc
    return bands


def _restore_confirmed_spectrum_band(st: Any) -> None:
    """Restore the last confirmed frequency band from primitive UI state."""

    if st.session_state.get("aia_radio_confirmed_spectrum_band"):
        return
    payload = st.session_state.get("confirmed_spectrum_band_json")
    if not payload:
        return
    try:
        decoded = json.loads(str(payload))
        band = SpectrumBand(
            float(decoded["low_mhz"]),
            float(decoded["high_mhz"]),
        )
    except KeyError, TypeError, ValueError, json.JSONDecodeError:
        st.session_state["confirmed_spectrum_band_json"] = ""
        return
    st.session_state["aia_radio_confirmed_spectrum_band"] = band.to_dict()


def _ensure_spectrum_band_defaults(st: Any, spectrum: SpectrumWindow) -> None:
    """Keep persisted numeric bounds valid for the currently loaded source."""

    current_low = st.session_state.get("spectrum_band_low")
    current_high = st.session_state.get("spectrum_band_high")
    try:
        current = SpectrumBand(float(current_low), float(current_high))
        current.observed_indices(spectrum.frequency_mhz)
        return
    except TypeError, ValueError:
        pass
    frequencies = spectrum.frequency_mhz
    observed_low = float(frequencies[0])
    observed_high = float(frequencies[-1])
    if observed_low >= observed_high:
        raise ValueError(
            "Spectrum needs at least two distinct frequencies for selection"
        )
    center_index = int(len(frequencies) // 2)
    center = float(frequencies[center_index])
    if len(frequencies) > 1:
        spacing = float(min(abs(frequencies[1:] - frequencies[:-1])))
    else:
        spacing = observed_high - observed_low
    half_width = max((observed_high - observed_low) * 0.025, spacing * 0.51)
    low = max(observed_low, center - half_width)
    high = min(observed_high, center + half_width)
    if low >= high:
        low, high = observed_low, observed_high
    st.session_state["spectrum_band_low"] = low
    st.session_state["spectrum_band_high"] = high


def _clear_dual_flux_state(st: Any) -> None:
    for key in (
        "aia_radio_curve",
        "aia_radio_spectrum_flux",
        "aia_radio_spectrum_flux_curves",
        "aia_radio_result",
        "aia_radio_artifact",
        "aia_radio_video_artifact",
    ):
        st.session_state.pop(key, None)


def _clear_spectrum_state(
    st: Any,
    *,
    clear_confirmed: bool,
    keep_window: bool = False,
) -> None:
    if not keep_window:
        st.session_state.pop("aia_radio_spectrum", None)
    st.session_state.pop("_pending_spectrum_band", None)
    st.session_state["spectrum_band_revision"] = (
        int(st.session_state.get("spectrum_band_revision", 0)) + 1
    )
    if clear_confirmed:
        st.session_state.pop("aia_radio_confirmed_spectrum_band", None)
        st.session_state["confirmed_spectrum_band_json"] = ""
        st.session_state["confirmed_spectrum_source_signature"] = ""
    _clear_dual_flux_state(st)


def _utc_datetime(value: str | datetime) -> datetime:
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


def _gaussian_display_overrides(
    *,
    show_center: bool,
    show_contours: bool,
    contour_percent: float,
) -> dict[str, Any]:
    return {
        "draw_gaussian_center": bool(show_center),
        "draw_gaussian_contours": bool(show_contours),
        # This frontend exposes contours, not a separate FWHM ellipse control.
        # Keep the implicit ellipse disabled so unchecked means no outline.
        "draw_gaussian_fwhm_ellipse": False,
        "gaussian_contour_levels": [float(contour_percent) / 100.0],
    }


def _validated_utc_range(
    start: str | datetime,
    end: str | datetime,
    *,
    label: str,
) -> tuple[datetime, datetime]:
    """Return one strict UTC display/extraction window."""

    parsed_start = _utc_datetime(start)
    parsed_end = _utc_datetime(end)
    if parsed_start >= parsed_end:
        raise ValueError(f"{label} start must be before end")
    return parsed_start, parsed_end


def _combined_utc_range(
    first: tuple[datetime, datetime],
    second: tuple[datetime, datetime],
) -> tuple[datetime, datetime]:
    """Return a common axis range without changing either data window."""

    return min(first[0], second[0]), max(first[1], second[1])


if __name__ == "__main__":
    main()
