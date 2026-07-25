"""All-in-one Streamlit workflow for radio/DART composite figures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_apps.frontends.radio.composite_figure.composite_figure_application import (
    CompositeArtifactBundle,
    CompositeFrameTemplate,
    FrequencyBand,
    annotate_source_map_png,
    build_centered_frequency_bands,
    build_composite_artifacts,
    build_dart_selection_figure,
    build_request_signature,
    build_source_map_selection_figure,
    frequency_band_from_selection,
    save_composite_bundle,
    select_dart_time_overlap,
)
from solar_apps.frontends.radio.composite_figure.composite_sequence import (
    CompositeSequenceBundle,
    CompositeSequenceJobRegistry,
    DEFAULT_SEQUENCE_FPS,
    DEFAULT_SEQUENCE_STRIDE,
    LARGE_SEQUENCE_WARNING_FRAMES,
    SequenceExportOptions,
    candidate_contains_frequency,
    candidate_frequency_paths,
    common_candidate_time_coverage,
    export_composite_sequences,
    group_candidates_by_frequency,
    prepare_single_panel_render as _prepare_sequence_single_panel_render,
    render_source_map_candidate,
    resolve_single_band_frequency_source,
    roi_intersects_source_map,
    sequence_frame_counts,
)
from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_app import (
    _RoiImportChoice,
    _RoiImportDocument,
    _roi_import_document_from_uploaded_or_path,
    build_file_manifest,
    discover_frequency_options,
    selection_to_radio_roi,
)
from solar_apps.frontends.radio.source_map.artifacts import (
    validate_source_map_artifact,
)
from solar_apps.frontends.radio.source_map.service import (
    PathPolicy,
    discover_candidates,
    parse_request_config,
)
from solar_apps.frontends.radio.source_map.worker import run_job
from solar_apps.platform.layout import RuntimeLayout
from solar_apps.ui.state import (
    bind_streamlit_fields,
    frontend_path_memory,
    frontend_state_store,
    save_streamlit_fields,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
    render_native_path_input,
    resolve_streamlit_allowed_roots,
)
from solar_apps.ui.theme import apply_plotly_chrome, render_streamlit_theme
from solar_apps.workflows.radio.configs import DEFAULT_CONFIG_NAME
from solar_apps.workflows.radio.spatial_display import SpatialRadioDisplay
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandCurve,
    DartNarrowbandResult,
    DartSpectrogramWindow,
    discover_dart_spectrogram_files,
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    extract_radio_roi_lightcurve,
    radio_roi_from_json,
)

FRONTEND_ID = "radio-composite"
DEFAULT_OUTPUT_RELATIVE = "outputs/radio_composite"
DEFAULT_PATTERN = "*.fits"
DEFAULT_DPI = 160

UI_FIELD_KEYS = (
    "radio_dir",
    "dart_dir",
    "output_dir",
    "radio_pattern",
    "radio_recursive",
    "source_mode",
    "source_frequencies",
    "source_polarization",
    "source_config",
    "map_transform",
    "map_cmap",
    "map_bad_color",
    "map_range_mode",
    "map_low_percentile",
    "map_high_percentile",
    "map_vmin",
    "map_vmax",
    "map_unit",
    "map_use_fov",
    "map_fov_xmin",
    "map_fov_xmax",
    "map_fov_ymin",
    "map_fov_ymax",
    "gaussian_overlay",
    "background_mode",
    "background_display",
    "background_fit",
    "advanced_source_map_json",
    "selected_map_frequency",
    "selected_candidate_id",
    "roi_json_path",
    "roi_mode",
    "roi_label",
    "dart_default_bandwidth_mhz",
    "dart_band_overrides_json",
    "dart_active_frequency_mhz",
    "dart_band_low",
    "dart_band_high",
    "shared_time_start",
    "shared_time_end",
    "composite_dpi",
    "sequence_stride",
    "sequence_fps",
    "sequence_quality",
    "sequence_save_video",
    "sequence_save_frames",
    "sequence_preview_frequency",
    "sequence_preview_index",
)

TRANSIENT_KEYS = (
    "radio_manifest",
    "radio_frequency_options",
    "dart_window",
    "dart_files",
    "inspection_signature",
    "source_map_config",
    "source_map_configs_by_frequency",
    "source_map_candidates",
    "source_map_candidates_by_frequency",
    "source_selected_frequencies",
    "source_common_coverage",
    "source_map_discovery_signature",
    "source_map_image_bytes",
    "source_map_metadata",
    "source_map_result",
    "source_map_candidate",
    "source_map_observation_time",
    "source_map_frequency_mhz",
    "candidate_roi",
    "confirmed_roi",
    "roi_import_document",
    "roi_import_source_kind",
    "roi_import_source_label",
    "roi_import_upload_signature",
    "roi_import_selected_key",
    "sequence_preview_cache",
    "radio_curves_by_frequency",
    "dart_narrowband_result",
    "dart_narrowband_results_by_frequency",
    "composite_bundle",
    "composite_signature",
    "composite_saved_directory",
    "curve_plot_cache_by_frequency",
    "sequence_signature",
    "sequence_job_id",
    "sequence_bundle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="solar-apps frontend radio-composite",
        description=(
            "Build multi-frequency Source Map ROI composites and sequence videos."
        ),
    )
    parser.add_argument("--radio-dir", default=None)
    parser.add_argument("--dart-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allowed-roots", default=None)
    parser.add_argument("--pattern", default=None)
    recursive = parser.add_mutually_exclusive_group()
    recursive.add_argument("--recursive", dest="recursive", action="store_true")
    recursive.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    _run_streamlit_app(argv)
    return 0


def _run_streamlit_app(argv: list[str] | None = None) -> None:
    import streamlit as st

    args, _unknown = build_parser().parse_known_args(argv)
    layout = RuntimeLayout.discover()
    state_store = frontend_state_store(FRONTEND_ID, layout=layout)
    try:
        allowed_roots = resolve_streamlit_allowed_roots(args.allowed_roots)
    except Exception as exc:
        st.error(f"Path configuration error: {exc}")
        return
    protected_output = layout.outputs_dir / "radio_composite"
    path_policy = PathAccessPolicy.create(
        allowed_roots,
        protected_output_roots=(protected_output,),
        base_directory=layout.repo_root,
    )
    path_memory = frontend_path_memory(path_policy.output_roots, layout=layout)

    st.set_page_config(page_title="Radio Composite Figure", layout="wide")
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
    _apply_cli_defaults_once(st, args)
    _apply_pending_band(st)
    st.title("Radio Composite Figure")
    st.caption(
        "Confirm one world-coordinate ROI, preview it across frequencies and "
        "times, then export one reproducible three-row sequence per frequency."
    )
    if not allowed_roots:
        st.error(
            "No allowed roots are configured. Add the radio, DART, and output "
            "directories to Local/configs/paths.local.yaml."
        )
        return

    default_output = str(Path(args.output_dir) if args.output_dir else protected_output)
    radio_dir_text, dart_dir_text, output_dir_text = _render_path_controls(
        st,
        path_policy,
        state_store,
        radio_default=str(args.radio_dir or ""),
        dart_default=str(args.dart_dir or ""),
        output_default=default_output,
    )
    _render_inspection_step(
        st,
        path_policy,
        radio_dir_text=radio_dir_text,
        dart_dir_text=dart_dir_text,
        pattern_default=str(args.pattern or DEFAULT_PATTERN),
        recursive_default=True if args.recursive is None else bool(args.recursive),
    )

    manifest = st.session_state.get("radio_manifest")
    dart_window = st.session_state.get("dart_window")
    frequency_options = st.session_state.get("radio_frequency_options")
    if not isinstance(manifest, pd.DataFrame) or not isinstance(
        dart_window, DartSpectrogramWindow
    ):
        st.info("Inspect both datasets to unlock Source Map configuration.")
        save_streamlit_fields(st, state_store, UI_FIELD_KEYS)
        return

    _render_source_map_configuration(
        st,
        path_policy,
        radio_dir_text=radio_dir_text,
        frequency_options=frequency_options,
    )
    candidates = st.session_state.get("source_map_candidates")
    source_configs = st.session_state.get("source_map_configs_by_frequency")
    grouped_candidates = st.session_state.get("source_map_candidates_by_frequency")
    if (
        not isinstance(candidates, list)
        or not isinstance(source_configs, dict)
        or not isinstance(grouped_candidates, dict)
    ):
        st.info("Discover Source Map frames to choose the map time and frequency.")
        _render_dart_band_step(st, dart_window, theme_mode, frequencies=())
        save_streamlit_fields(st, state_store, UI_FIELD_KEYS)
        return

    candidate, map_frequency, coverage = _render_map_candidate_step(
        st,
        candidates,
        source_configs,
        grouped_candidates,
        path_policy,
    )
    if candidate is None or map_frequency is None:
        st.warning("Select a valid Source Map frequency and frame.")
        _render_dart_band_step(
            st,
            dart_window,
            theme_mode,
            frequencies=sorted(float(value) for value in grouped_candidates),
        )
        save_streamlit_fields(st, state_store, UI_FIELD_KEYS)
        return

    _render_roi_step(st, theme_mode, path_policy, state_store)
    _render_sequence_preview_step(
        st,
        path_policy,
        radio_dir_text=radio_dir_text,
        source_configs=source_configs,
        grouped_candidates=grouped_candidates,
    )
    _render_dart_band_step(
        st,
        dart_window,
        theme_mode,
        frequencies=sorted(float(value) for value in grouped_candidates),
    )
    sequence_jobs = st.cache_resource(_create_sequence_job_registry)()
    _render_analysis_step(
        st,
        path_policy,
        radio_dir_text=radio_dir_text,
        dart_dir_text=dart_dir_text,
        output_dir_text=output_dir_text,
        map_frequency=map_frequency,
        candidate=candidate,
        radio_coverage=coverage,
        dart_window=dart_window,
        source_configs=source_configs,
        grouped_candidates=grouped_candidates,
        sequence_jobs=sequence_jobs,
    )
    save_streamlit_fields(st, state_store, UI_FIELD_KEYS)


def _render_path_controls(
    st: Any,
    path_policy: PathAccessPolicy,
    state_store: Any,
    *,
    radio_default: str,
    dart_default: str,
    output_default: str,
) -> tuple[str, str, str]:
    st.subheader("1. Data locations")
    radio_dir = render_native_path_input(
        st,
        "Radio FITS sequence directory",
        key="radio_dir",
        initial_value=radio_default,
        roots=path_policy.input_roots,
        kind="directory",
        frontend_id=FRONTEND_ID,
        operation="radio-input",
        state_store=state_store,
        help_text="The same sequence supplies the selected Source Map frame and ROI curve.",
    )
    dart_dir = render_native_path_input(
        st,
        "DART four-FITS directory",
        key="dart_dir",
        initial_value=dart_default,
        roots=path_policy.input_roots,
        kind="directory",
        frontend_id=FRONTEND_ID,
        operation="dart-input",
        state_store=state_store,
    )
    output_dir = render_native_path_input(
        st,
        "Composite output directory",
        key="output_dir",
        initial_value=output_default,
        roots=path_policy.output_roots,
        kind="directory",
        frontend_id=FRONTEND_ID,
        operation="composite-output",
        state_store=state_store,
    )
    return radio_dir, dart_dir, output_dir


def _render_inspection_step(
    st: Any,
    path_policy: PathAccessPolicy,
    *,
    radio_dir_text: str,
    dart_dir_text: str,
    pattern_default: str,
    recursive_default: bool,
) -> None:
    columns = st.columns([2, 1, 1])
    with columns[0]:
        pattern = st.text_input(
            "Radio FITS glob",
            value=pattern_default,
            key="radio_pattern",
        )
    with columns[1]:
        recursive = st.checkbox(
            "Recursive radio scan",
            value=recursive_default,
            key="radio_recursive",
        )
    with columns[2]:
        st.write("")
        inspect_clicked = st.button("Inspect datasets", width="stretch", type="primary")
    _invalidate_if_controls_changed(
        st,
        "inspection_controls_signature",
        {
            "radio_directory": radio_dir_text,
            "dart_directory": dart_dir_text,
            "pattern": pattern,
            "recursive": bool(recursive),
        },
        _invalidate_after_inspection,
    )
    if inspect_clicked:
        try:
            radio_dir = path_policy.input_directory(radio_dir_text)
            dart_dir = path_policy.input_directory(dart_dir_text)
            manifest = build_file_manifest(
                radio_dir,
                pattern=pattern,
                recursive=bool(recursive),
            )
            if manifest.empty:
                raise ValueError("No radio FITS files matched the requested pattern")
            frequencies = discover_frequency_options(
                radio_dir,
                pattern=pattern,
                recursive=bool(recursive),
            )
            if frequencies.empty:
                raise ValueError("No finite radio frequencies were discovered")
            dart_files = discover_dart_spectrogram_files(dart_dir)
            dart_window = read_dart_spectrogram_window(
                dart_files,
                max_frequency_samples=700,
                max_time_samples=1400,
                chunk_memory_mb=64,
            )
            radio_paths = [Path(value) for value in manifest["path"].astype(str)]
            signature = build_request_signature(
                {
                    "radio_dir": str(radio_dir),
                    "dart_dir": str(dart_dir),
                    "pattern": pattern,
                    "recursive": bool(recursive),
                },
                source_paths=[
                    *radio_paths,
                    dart_files.stokes_i_db,
                    dart_files.stokes_v_over_i,
                    dart_files.frequency,
                    dart_files.time,
                ],
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            _invalidate_after_inspection(st)
            st.session_state["radio_manifest"] = manifest
            st.session_state["radio_frequency_options"] = frequencies
            st.session_state["dart_window"] = dart_window
            st.session_state["dart_files"] = dart_files
            st.session_state["inspection_signature"] = signature
            radio_values = _frequency_values(frequencies)
            default_band = _default_band(dart_window.frequency_mhz, radio_values[0])
            st.session_state["dart_default_bandwidth_mhz"] = default_band.bandwidth_mhz
            st.session_state["dart_band_overrides_json"] = "{}"
            st.session_state["dart_active_frequency_mhz"] = radio_values[0]
            st.session_state["dart_band_low"] = default_band.low_mhz
            st.session_state["dart_band_high"] = default_band.high_mhz
            st.success(
                f"Loaded {len(manifest):,} radio file(s), {len(radio_values)} "
                f"frequency option(s), and DART matrix {dart_window.stokes_i_db.shape}."
            )
    if isinstance(st.session_state.get("radio_manifest"), pd.DataFrame):
        manifest = st.session_state["radio_manifest"]
        frequencies = st.session_state["radio_frequency_options"]
        st.dataframe(frequencies, hide_index=True, width="stretch")
        st.caption(f"Radio manifest: {len(manifest):,} file records.")


def _render_source_map_configuration(
    st: Any,
    path_policy: PathAccessPolicy,
    *,
    radio_dir_text: str,
    frequency_options: Any,
) -> None:
    st.subheader("2. Source Map settings")
    frequencies = _frequency_values(frequency_options)
    main_columns = st.columns([1, 2, 1, 1])
    with main_columns[0]:
        mode = st.selectbox(
            "Input organization",
            options=("single_band", "multi_band"),
            key="source_mode",
            format_func=lambda value: (
                "Single-band sequence"
                if value == "single_band"
                else "Synchronized multi-band"
            ),
        )
    with main_columns[1]:
        selected_frequencies = st.multiselect(
            "Source frequencies (MHz)",
            options=frequencies,
            default=frequencies[:1],
            key="source_frequencies",
            format_func=lambda value: f"{value:g}",
        )
    with main_columns[2]:
        polarization = st.selectbox(
            "Polarization",
            options=("RR+LL", "RR", "LL"),
            key="source_polarization",
        )
    with main_columns[3]:
        config_name = st.text_input(
            "Event config",
            value=DEFAULT_CONFIG_NAME,
            key="source_config",
        )

    display_columns = st.columns(4)
    with display_columns[0]:
        transform = st.selectbox(
            "Map transform",
            options=("linear", "log10"),
            key="map_transform",
        )
        cmap = st.selectbox(
            "Color map",
            options=("hot", "inferno", "magma", "viridis", "plasma", "jet", "cividis"),
            key="map_cmap",
        )
        bad_color = st.color_picker(
            "Bad-value color", value="#000080", key="map_bad_color"
        )
    with display_columns[1]:
        range_mode = st.selectbox(
            "Color range",
            options=("auto", "fixed"),
            key="map_range_mode",
        )
        low_percentile = st.number_input(
            "Lower percentile",
            min_value=0.0,
            max_value=99.999,
            value=99.7,
            step=0.1,
            key="map_low_percentile",
        )
        high_percentile = st.number_input(
            "Upper percentile",
            min_value=0.001,
            max_value=100.0,
            value=99.99,
            step=0.01,
            key="map_high_percentile",
        )
    with display_columns[2]:
        vmin = st.number_input("Fixed minimum", value=0.0, key="map_vmin")
        vmax = st.number_input("Fixed maximum", value=1.0, key="map_vmax")
        unit = st.text_input(
            "Display unit override",
            value="",
            key="map_unit",
            placeholder="Use FITS BUNIT",
        )
    with display_columns[3]:
        gaussian_overlay = st.checkbox(
            "Gaussian overlay", value=True, key="gaussian_overlay"
        )
        background_mode = st.selectbox(
            "Background",
            options=("off", "noise_map_only", "local_mesh", "local_median"),
            key="background_mode",
        )
        background_display = st.checkbox(
            "Apply background to display", value=False, key="background_display"
        )
        background_fit = st.checkbox(
            "Apply background to fit", value=False, key="background_fit"
        )

    use_fov = st.checkbox("Custom field of view", value=False, key="map_use_fov")
    fov_columns = st.columns(4)
    fov_values = []
    for column, label, key, default in zip(
        fov_columns,
        ("HPLN min", "HPLN max", "HPLT min", "HPLT max"),
        ("map_fov_xmin", "map_fov_xmax", "map_fov_ymin", "map_fov_ymax"),
        (-1000.0, 1000.0, -1000.0, 1000.0),
        strict=True,
    ):
        with column:
            fov_values.append(
                float(
                    st.number_input(label, value=default, key=key, disabled=not use_fov)
                )
            )
    advanced = st.text_area(
        "Advanced Source Map JSON",
        value="{}",
        key="advanced_source_map_json",
        help="Only established non-path Source Map options are accepted.",
    )
    _invalidate_if_controls_changed(
        st,
        "source_controls_signature",
        {
            "inspection": st.session_state.get("inspection_signature"),
            "radio_directory": radio_dir_text,
            "mode": mode,
            "frequencies": list(selected_frequencies),
            "polarization": polarization,
            "config": config_name,
            "transform": transform,
            "cmap": cmap,
            "bad_color": bad_color,
            "range_mode": range_mode,
            "percentiles": [low_percentile, high_percentile],
            "fixed_range": [vmin, vmax],
            "unit": unit,
            "gaussian_overlay": gaussian_overlay,
            "background": [background_mode, background_display, background_fit],
            "fov": fov_values if use_fov else None,
            "advanced": advanced,
        },
        _invalidate_after_source_controls,
    )
    if st.button("Discover Source Map frames", type="primary", width="stretch"):
        try:
            if not selected_frequencies:
                raise ValueError("Select at least one radio frequency")
            preview_dir = (
                RuntimeLayout.discover().outputs_dir / "radio_composite" / "source_map"
            )
            policy = PathPolicy(path_policy.output_roots)
            base_request = {
                "config": config_name,
                "mode": mode,
                "source_path": str(path_policy.input_directory(radio_dir_text)),
                "output_dir": str(preview_dir),
                "frequencies": list(selected_frequencies),
                "polarization": polarization,
                "gaussian_overlay": gaussian_overlay,
                "cmap": cmap,
                "color_range_mode": range_mode,
                "fixed_vmin": vmin if range_mode == "fixed" else None,
                "fixed_vmax": vmax if range_mode == "fixed" else None,
                "radio_unit": unit,
                "background_mode": background_mode,
                "background_display": background_display,
                "background_fit": background_fit,
                "spectrogram_panel": False,
                "advanced": advanced,
            }
            display = SpatialRadioDisplay(
                cmap=str(cmap),
                bad_color=str(bad_color),
                transform=str(transform),
                range_mode=str(range_mode),
                range_scope="frame",
                auto_method="fixed_percentile",
                percentiles=(float(low_percentile), float(high_percentile)),
                vmin=float(vmin) if range_mode == "fixed" else None,
                vmax=float(vmax) if range_mode == "fixed" else None,
                unit=str(unit).strip() or None,
                fov=tuple(fov_values) if use_fov else None,
                render_profile="preview",
            )
            selected = [float(value) for value in selected_frequencies]
            configs_by_frequency: dict[float, dict[str, Any]] = {}
            candidates: list[dict[str, Any]] = []
            if mode == "single_band":
                manifest = st.session_state.get("radio_manifest")
                if not isinstance(manifest, pd.DataFrame):
                    raise ValueError("Inspect the radio dataset before discovery")
                for frequency in selected:
                    request = dict(base_request)
                    request["frequencies"] = [frequency]
                    request["source_path"] = str(
                        resolve_single_band_frequency_source(
                            base_request["source_path"],
                            manifest,
                            frequency,
                            polarization=polarization,
                        )
                    )
                    config = parse_request_config(request, policy=policy)
                    config = display.apply_to_legacy_config(config)
                    config["spatial_display"] = display.to_dict()
                    config["enable_spectrogram_panel"] = False
                    discovered = discover_candidates(config, policy=policy)
                    for candidate in discovered:
                        candidate["id"] = f"{frequency:g}mhz-{candidate['id']}"
                        if not candidate.get("frequencies_mhz"):
                            candidate["frequencies_mhz"] = [frequency]
                    configs_by_frequency[frequency] = config
                    candidates.extend(discovered)
            else:
                request = dict(base_request)
                config = parse_request_config(request, policy=policy)
                config = display.apply_to_legacy_config(config)
                config["spatial_display"] = display.to_dict()
                config["enable_spectrogram_panel"] = False
                candidates = discover_candidates(config, policy=policy)
                configs_by_frequency = {
                    frequency: copy.deepcopy(config) for frequency in selected
                }
            candidates = [
                candidate
                for candidate in candidates
                if any(
                    candidate_contains_frequency(candidate, value) for value in selected
                )
            ]
            if not candidates:
                raise ValueError(
                    "No Source Map candidates match the selected frequencies"
                )
            grouped = group_candidates_by_frequency(candidates, selected)
            common_coverage = common_candidate_time_coverage(grouped)
            signature = build_request_signature(
                {
                    "inspection": st.session_state.get("inspection_signature"),
                    "request": base_request,
                    "display": display.to_dict(),
                    "resolved_sources": {
                        f"{frequency:g}": config.get("data_dir")
                        or config.get("multi_band_root")
                        for frequency, config in configs_by_frequency.items()
                    },
                }
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            _invalidate_after_discovery(st)
            st.session_state["source_map_config"] = copy.deepcopy(
                configs_by_frequency[selected[0]]
            )
            st.session_state["source_map_configs_by_frequency"] = configs_by_frequency
            st.session_state["source_map_candidates"] = candidates
            st.session_state["source_map_candidates_by_frequency"] = grouped
            st.session_state["source_selected_frequencies"] = selected
            st.session_state["source_common_coverage"] = (
                common_coverage[0].isoformat(),
                common_coverage[1].isoformat(),
            )
            st.session_state["source_map_discovery_signature"] = signature
            st.success(
                f"Discovered {len(candidates):,} Source Map candidate record(s) "
                f"across {len(grouped)} frequency sequence(s)."
            )


def _render_map_candidate_step(
    st: Any,
    candidates: list[dict[str, Any]],
    source_configs: dict[float, dict[str, Any]],
    grouped_candidates: dict[float, list[dict[str, Any]]],
    path_policy: PathAccessPolicy,
) -> tuple[dict[str, Any] | None, float | None, tuple[datetime, datetime] | None]:
    st.subheader("3. Select and render the reference Source Map")
    frequencies = sorted(float(value) for value in grouped_candidates)
    if not frequencies:
        st.error("Source Map candidates contain no finite frequency metadata.")
        return None, None, None
    map_frequency = float(
        st.selectbox(
            "Top map frequency (MHz)",
            options=frequencies,
            key="selected_map_frequency",
            format_func=lambda value: f"{value:g}",
        )
    )
    common_raw = st.session_state.get("source_common_coverage")
    coverage = (
        (_utc_datetime(common_raw[0]), _utc_datetime(common_raw[1]))
        if isinstance(common_raw, (list, tuple)) and len(common_raw) == 2
        else common_candidate_time_coverage(grouped_candidates)
    )
    matching = [
        candidate
        for candidate in _frequency_mapping_value(grouped_candidates, map_frequency)
        if coverage[0] <= _utc_datetime(candidate["observation_time"]) <= coverage[1]
    ]
    if not matching:
        st.error("No timestamped Source Map frame matches the selected frequency.")
        return None, map_frequency, None
    candidate_by_id = {str(candidate["id"]): candidate for candidate in matching}
    selected_id = st.selectbox(
        "Source Map frame",
        options=list(candidate_by_id),
        key="selected_candidate_id",
        format_func=lambda value: _candidate_label(candidate_by_id[value]),
    )
    candidate = candidate_by_id[str(selected_id)]
    _invalidate_if_controls_changed(
        st,
        "map_selection_controls_signature",
        {
            "discovery": st.session_state.get("source_map_discovery_signature"),
            "frequency_mhz": map_frequency,
            "candidate_id": str(selected_id),
        },
        _invalidate_after_map_selection,
    )
    st.caption(
        f"Common selected-frequency coverage: "
        f"{coverage[0].isoformat()} to {coverage[1].isoformat()} UTC"
    )
    if st.button("Render selected Source Map", type="primary", width="stretch"):
        try:
            preview_dir = (
                RuntimeLayout.discover().outputs_dir / "radio_composite" / "source_map"
            )
            source_config = _frequency_mapping_value(source_configs, map_frequency)
            render_cfg, render_candidate = _prepare_sequence_single_panel_render(
                source_config,
                candidate,
                map_frequency,
                transform=str(st.session_state.get("map_transform", "linear")),
                output_directory=preview_dir,
            )
            policy = PathPolicy(path_policy.output_roots)
            policy.resolve(preview_dir, must_exist=False).mkdir(
                parents=True, exist_ok=True
            )
            with st.spinner("Rendering the selected Source Map and sidecar..."):
                result = run_job(
                    {"config": render_cfg, "candidate": render_candidate, "sequence": 1}
                )
            image_path = policy.resolve(
                result["image_path"], must_exist=True, kind="file"
            )
            sidecar_path = policy.resolve(
                result["sidecar_path"], must_exist=True, kind="file"
            )
            metadata = validate_source_map_artifact(image_path, sidecar_path)
            if len(metadata.get("panels", [])) != 1:
                raise ValueError(
                    "Rendered Source Map did not contain exactly one panel"
                )
            observed = _utc_datetime(candidate["observation_time"])
        except Exception as exc:
            st.error(str(exc))
        else:
            _invalidate_after_map(st)
            st.session_state["source_map_image_bytes"] = image_path.read_bytes()
            st.session_state["source_map_metadata"] = metadata
            st.session_state["source_map_result"] = result
            st.session_state["source_map_candidate"] = copy.deepcopy(candidate)
            st.session_state["source_map_observation_time"] = observed.isoformat()
            st.session_state["source_map_frequency_mhz"] = map_frequency
            st.session_state["shared_time_start"] = coverage[0].isoformat()
            st.session_state["shared_time_end"] = coverage[1].isoformat()
            st.success("Source Map rendered. Draw and confirm one ROI below.")
    return candidate, map_frequency, coverage


def _render_roi_step(
    st: Any,
    theme_mode: str,
    path_policy: PathAccessPolicy,
    state_store: Any,
) -> None:
    image_bytes = st.session_state.get("source_map_image_bytes")
    metadata = st.session_state.get("source_map_metadata")
    if not isinstance(image_bytes, bytes) or not isinstance(metadata, dict):
        st.info("Render a Source Map to unlock spatial ROI selection.")
        return
    st.subheader("4. Draw and confirm one ROI")
    _render_roi_import_controls(st, path_policy, state_store)
    controls = st.columns([1, 2, 1, 1])
    with controls[0]:
        roi_mode = st.radio(
            "ROI tool",
            options=("box", "lasso"),
            horizontal=True,
            key="roi_mode",
            format_func=lambda value: "Rectangle" if value == "box" else "Lasso",
        )
    with controls[1]:
        roi_label = st.text_input("ROI label", value="ROI 1", key="roi_label")
    _invalidate_if_controls_changed(
        st,
        "roi_controls_signature",
        {
            "source_map": st.session_state.get("source_map_result"),
            "mode": roi_mode,
            "label": roi_label,
        },
        _invalidate_after_roi_controls,
    )
    candidate = _session_roi(st, "candidate_roi")
    confirmed = _session_roi(st, "confirmed_roi")
    active = candidate or confirmed
    figure = build_source_map_selection_figure(
        image_bytes,
        metadata,
        roi=active,
        roi_mode=roi_mode,
    )
    apply_plotly_chrome(figure, theme_mode)
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode=(roi_mode,),
        key=f"radio_composite_roi_{roi_mode}_{active.roi_id if active else 'empty'}",
    )
    selected = selection_to_radio_roi(event, mode=roi_mode, label=roi_label)
    if selected is not None and (
        candidate is None or selected.to_json_dict() != candidate.to_json_dict()
    ):
        st.session_state["candidate_roi"] = selected.to_json_dict()
        st.session_state.pop("confirmed_roi", None)
        _invalidate_composite(st)
        candidate = selected
        confirmed = None
    with controls[2]:
        if st.button(
            "Confirm ROI",
            disabled=candidate is None,
            type="primary",
            width="stretch",
        ):
            st.session_state["confirmed_roi"] = candidate.to_json_dict()
            _invalidate_composite(st)
            st.rerun()
    with controls[3]:
        if st.button("Clear ROI", disabled=active is None, width="stretch"):
            st.session_state.pop("candidate_roi", None)
            st.session_state.pop("confirmed_roi", None)
            _invalidate_composite(st)
            st.rerun()
    confirmed = _session_roi(st, "confirmed_roi")
    if confirmed is not None:
        st.success(f"Confirmed {confirmed.label or confirmed.roi_id}.")
        st.json(confirmed.to_json_dict(), expanded=False)
    elif candidate is not None:
        st.warning("ROI is staged. Confirm it before analysis.")


def _render_roi_import_controls(
    st: Any,
    path_policy: PathAccessPolicy,
    state_store: Any,
) -> None:
    """Load a saved world-coordinate ROI without bypassing confirmation."""

    st.markdown("**Import saved ROI JSON**")
    uploaded = st.file_uploader(
        "Load ROI JSON",
        type=["json"],
        key="radio_composite_roi_json_upload",
        help=(
            "Load a Radio ROI JSON previously exported by this page, or a "
            "Source Map ROI JSON containing one or more HPLN/HPLT regions."
        ),
    )
    if uploaded is not None:
        payload = uploaded.getvalue()
        signature = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("roi_import_upload_signature") != signature:
            try:
                document = _roi_import_document_from_uploaded_or_path(
                    uploaded_payload=payload,
                    path_text="",
                    path_policy=path_policy,
                )
            except Exception as exc:  # noqa: BLE001 - visible frontend error.
                st.error(str(exc))
            else:
                _store_roi_import_document(
                    st,
                    document,
                    source_kind="upload",
                    source_label=str(getattr(uploaded, "name", "uploaded JSON")),
                    upload_signature=signature,
                )

    roi_json_path = render_native_path_input(
        st,
        "ROI JSON path",
        key="roi_json_path",
        initial_value="",
        roots=path_policy.input_roots,
        kind="file",
        extensions=(".json",),
        placeholder="Choose an allowed ROI JSON file",
        help_text=(
            "Choose an exported ROI JSON on this computer. An uploaded JSON "
            "takes priority while it remains selected."
        ),
        frontend_id=FRONTEND_ID,
        operation="import-roi",
        state_store=state_store,
    )
    if st.button(
        "Load ROI JSON Path",
        disabled=uploaded is not None,
        help="Load the local JSON path. Upload takes priority while present.",
    ):
        try:
            document = _roi_import_document_from_uploaded_or_path(
                uploaded_payload=None,
                path_text=roi_json_path,
                path_policy=path_policy,
            )
        except Exception as exc:  # noqa: BLE001 - visible frontend error.
            st.error(str(exc))
        else:
            _store_roi_import_document(
                st,
                document,
                source_kind="path",
                source_label=roi_json_path,
            )

    document = st.session_state.get("roi_import_document")
    if not isinstance(document, _RoiImportDocument):
        return
    source_label = str(st.session_state.get("roi_import_source_label") or "ROI JSON")
    if document.source_format == "source_map":
        st.caption(
            f"Loaded {len(document.choices)} Source Map region(s) from "
            f"{source_label}. The saved image SHA-256 is retained as provenance; "
            "the HPLN/HPLT coordinates are mapped onto the current Source Map."
        )
        st.caption(f"Source image SHA-256: {document.source_image_sha256}")
    else:
        st.caption(f"Loaded one Radio ROI from {source_label}.")

    choice_by_key = {choice.key: choice for choice in document.choices}
    if len(document.choices) == 1:
        choice = document.choices[0]
        st.caption(f"Imported region: {choice.display_label}")
        if st.button(
            "Use Imported ROI",
            disabled=_imported_roi_is_staged(st, choice),
            help="Stage the imported region, then use Confirm ROI below.",
        ):
            _stage_imported_roi(st, choice)
        return

    selected_key = st.selectbox(
        "Imported region",
        options=list(choice_by_key),
        key="roi_import_selected_key",
        format_func=lambda key: choice_by_key[key].display_label,
        help="Choose one region from the Source Map ROI set for this analysis.",
    )
    if st.button(
        "Use Selected Imported Region",
        type="primary",
        help="Stage this region, then use Confirm ROI below.",
    ):
        changed = _stage_imported_roi(st, choice_by_key[selected_key])
        if changed:
            st.success("Selected imported region staged. Confirm it below.")
        else:
            st.info("The selected imported region is already staged.")


def _store_roi_import_document(
    st: Any,
    document: _RoiImportDocument,
    *,
    source_kind: str,
    source_label: str,
    upload_signature: str | None = None,
) -> None:
    st.session_state["roi_import_document"] = document
    st.session_state["roi_import_source_kind"] = source_kind
    st.session_state["roi_import_source_label"] = source_label
    if upload_signature is not None:
        st.session_state["roi_import_upload_signature"] = upload_signature
    else:
        st.session_state.pop("roi_import_upload_signature", None)
    st.session_state["roi_import_selected_key"] = document.default_choice_key
    if len(document.choices) == 1:
        _stage_imported_roi(st, document.choices[0])


def _imported_roi_is_staged(st: Any, choice: _RoiImportChoice) -> bool:
    candidate = _session_roi(st, "candidate_roi")
    return (
        candidate is not None and candidate.to_json_dict() == choice.roi.to_json_dict()
    )


def _stage_imported_roi(st: Any, choice: _RoiImportChoice) -> bool:
    if _imported_roi_is_staged(st, choice):
        return False
    st.session_state["candidate_roi"] = choice.roi.to_json_dict()
    st.session_state.pop("confirmed_roi", None)
    _invalidate_composite(st)
    return True


def _render_sequence_preview_step(
    st: Any,
    path_policy: PathAccessPolicy,
    *,
    radio_dir_text: str,
    source_configs: dict[float, dict[str, Any]],
    grouped_candidates: dict[float, list[dict[str, Any]]],
) -> None:
    confirmed = _session_roi(st, "confirmed_roi")
    if confirmed is None:
        st.info("Confirm the ROI to preview it across frequencies and times.")
        return
    st.subheader("5. Preview the confirmed ROI across the sequence")
    if isinstance(st.session_state.get("sequence_job_id"), str):
        _pause_sequence_preview(st)
        st.info("Sequence preview is paused while the video export is running.")
        return
    heading_columns = st.columns([3, 1])
    with heading_columns[0]:
        st.caption(
            "The ROI remains in HPLN/HPLT coordinates while Source Maps are "
            "rendered on demand and cached."
        )
    with heading_columns[1]:
        if st.button("Return to edit ROI", width="stretch"):
            st.session_state.pop("confirmed_roi", None)
            _invalidate_composite(st)
            st.session_state.pop("sequence_preview_cache", None)
            st.rerun()

    preview_interval = (
        0.8 if bool(st.session_state.get("sequence_preview_playing")) else None
    )

    @st.fragment(run_every=preview_interval)
    def preview_fragment() -> None:
        if st.session_state.pop("sequence_preview_pause_requested", False):
            _pause_sequence_preview(st)
        frequencies = sorted(float(value) for value in grouped_candidates)
        selected_frequency = float(
            st.selectbox(
                "Preview frequency (MHz)",
                options=frequencies,
                key="sequence_preview_frequency",
                format_func=lambda value: f"{value:g}",
            )
        )
        candidates = list(
            _frequency_mapping_value(grouped_candidates, selected_frequency)
        )
        range_start = _utc_datetime(st.session_state.get("shared_time_start"))
        range_end = _utc_datetime(st.session_state.get("shared_time_end"))
        candidates = [
            candidate
            for candidate in candidates
            if range_start <= _utc_datetime(candidate["observation_time"]) <= range_end
        ]
        if not candidates:
            st.error("No preview frame lies inside the shared UTC range.")
            return
        frequency_signature = f"{selected_frequency:.9g}|{len(candidates)}"
        if (
            st.session_state.get("sequence_preview_frequency_signature")
            != frequency_signature
        ):
            st.session_state["sequence_preview_frequency_signature"] = (
                frequency_signature
            )
            st.session_state["sequence_preview_index"] = 0
            st.session_state["sequence_preview_slider"] = 0
            st.session_state["sequence_preview_playing"] = False
        current = min(
            max(0, int(st.session_state.get("sequence_preview_index", 0))),
            len(candidates) - 1,
        )
        if st.session_state.get("sequence_preview_playing"):
            current = (current + 1) % len(candidates)
            st.session_state["sequence_preview_index"] = current
            st.session_state["sequence_preview_slider"] = current
        if "sequence_preview_slider" not in st.session_state:
            st.session_state["sequence_preview_slider"] = current

        def move_preview(delta: int) -> None:
            value = (
                int(st.session_state.get("sequence_preview_index", 0)) + delta
            ) % len(candidates)
            st.session_state["sequence_preview_index"] = value
            st.session_state["sequence_preview_slider"] = value

        def select_preview_frame() -> None:
            st.session_state["sequence_preview_index"] = int(
                st.session_state["sequence_preview_slider"]
            )

        controls = st.columns([1, 4, 1, 1])
        with controls[0]:
            st.button(
                "Previous",
                key="sequence_preview_previous",
                width="stretch",
                on_click=move_preview,
                args=(-1,),
            )
        with controls[1]:
            if len(candidates) == 1:
                st.caption("Frame 0 of 0")
                current = 0
            else:
                current = int(
                    st.slider(
                        "Frame",
                        min_value=0,
                        max_value=len(candidates) - 1,
                        value=current,
                        key="sequence_preview_slider",
                        format="%d",
                        on_change=select_preview_frame,
                    )
                )
        with controls[2]:
            st.button(
                "Next",
                key="sequence_preview_next",
                width="stretch",
                on_click=move_preview,
                args=(1,),
            )
        with controls[3]:
            st.toggle("Play", key="sequence_preview_playing")
        candidate = candidates[current]
        try:
            preview = _preview_source_map(
                st,
                path_policy,
                radio_dir_text=radio_dir_text,
                source_config=_frequency_mapping_value(
                    source_configs, selected_frequency
                ),
                candidate=candidate,
                frequency_mhz=selected_frequency,
                roi=confirmed,
            )
        except Exception as exc:
            st.error(str(exc))
            st.session_state["sequence_preview_pause_requested"] = True
            return
        annotated = preview["annotated_png"]
        st.image(annotated, width="stretch")
        metrics = st.columns(4)
        metrics[0].metric("Frequency", f"{selected_frequency:g} MHz")
        metrics[1].metric(
            "UTC", _utc_datetime(candidate["observation_time"]).isoformat()
        )
        metrics[2].metric("ROI intersection", "yes" if preview["intersects"] else "no")
        valid_pixels = preview.get("valid_pixel_count")
        metrics[3].metric(
            "Valid ROI pixels",
            "unknown" if valid_pixels is None else f"{valid_pixels:,}",
        )
        if not preview["intersects"]:
            st.warning("The confirmed ROI does not intersect this Source Map panel.")

    preview_fragment()


def _preview_source_map(
    st: Any,
    path_policy: PathAccessPolicy,
    *,
    radio_dir_text: str,
    source_config: dict[str, Any],
    candidate: dict[str, Any],
    frequency_mhz: float,
    roi: RadioRoi,
) -> dict[str, Any]:
    paths, _slot = candidate_frequency_paths(candidate, frequency_mhz)
    signature = build_request_signature(
        {
            "discovery": st.session_state.get("source_map_discovery_signature"),
            "candidate_id": candidate.get("id"),
            "frequency_mhz": frequency_mhz,
            "transform": st.session_state.get("map_transform", "linear"),
            "roi": roi.to_json_dict(),
        },
        source_paths=paths,
    )
    cache = st.session_state.setdefault("sequence_preview_cache", {})
    cached = cache.get(signature)
    if isinstance(cached, dict):
        return cached
    preview_dir = RuntimeLayout.discover().outputs_dir / "radio_composite" / "preview"
    policy = PathPolicy(path_policy.output_roots)
    policy.resolve(preview_dir, must_exist=False).mkdir(parents=True, exist_ok=True)
    map_png, metadata, result = render_source_map_candidate(
        source_config,
        candidate,
        frequency_mhz,
        str(st.session_state.get("map_transform", "linear")),
        preview_dir,
        len(cache) + 1,
    )
    intersects = roi_intersects_source_map(metadata, roi)
    annotated = annotate_source_map_png(map_png, metadata, roi)
    valid_pixel_count: int | None = None
    try:
        preview_curve = extract_radio_roi_lightcurve(
            path_policy.input_directory(radio_dir_text),
            roi,
            files=[Path(path) for path in paths],
            freqs=[frequency_mhz],
            polarization=str(st.session_state.get("source_polarization", "RR+LL")),
        )
        values = pd.to_numeric(preview_curve.get("valid_pixel_count"), errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            valid_pixel_count = int(finite.sum())
    except Exception:
        valid_pixel_count = None
    value = {
        "annotated_png": annotated,
        "source_map_png": map_png,
        "metadata": metadata,
        "result": result,
        "intersects": intersects,
        "valid_pixel_count": valid_pixel_count,
    }
    cache[signature] = value
    return value


def _render_dart_band_step(
    st: Any,
    window: DartSpectrogramWindow,
    theme_mode: str,
    *,
    frequencies: list[float] | tuple[float, ...],
) -> None:
    st.subheader("6. Select one DART frequency band per radio frequency")
    selected_frequencies = sorted(
        {float(value) for value in frequencies if math.isfinite(float(value))}
    )
    if not selected_frequencies:
        st.info(
            "Discover the selected Source Map frequencies first. Their radio "
            "frequencies become the locked centers of the DART bands."
        )
        return
    observed_low = float(np.nanmin(window.frequency_mhz))
    observed_high = float(np.nanmax(window.frequency_mhz))
    fallback_band = _default_band(window.frequency_mhz, selected_frequencies[0])
    default_width = _valid_bandwidth_or_default(
        st.session_state.get("dart_default_bandwidth_mhz"),
        fallback_band.bandwidth_mhz,
        maximum=observed_high - observed_low,
    )
    if "dart_default_bandwidth_mhz" not in st.session_state:
        legacy_low = st.session_state.get("dart_band_low")
        legacy_high = st.session_state.get("dart_band_high")
        try:
            legacy_width = float(legacy_high) - float(legacy_low)
        except TypeError, ValueError:
            legacy_width = default_width
        default_width = _valid_bandwidth_or_default(
            legacy_width,
            default_width,
            maximum=observed_high - observed_low,
        )
        st.session_state["dart_default_bandwidth_mhz"] = default_width
    elif not math.isclose(
        float(st.session_state["dart_default_bandwidth_mhz"]),
        default_width,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        st.session_state["dart_default_bandwidth_mhz"] = default_width

    active_raw = st.session_state.get(
        "dart_active_frequency_mhz", selected_frequencies[0]
    )
    try:
        active_raw_value = float(active_raw)
        active_frequency = _nearest_frequency(selected_frequencies, active_raw_value)
    except TypeError, ValueError:
        active_raw_value = selected_frequencies[0]
        active_frequency = selected_frequencies[0]
    if not math.isclose(active_raw_value, active_frequency, rel_tol=0.0, abs_tol=1e-6):
        st.session_state["dart_active_frequency_mhz"] = active_frequency

    overrides = _dart_band_overrides(
        st.session_state.get("dart_band_overrides_json"),
        selected_frequencies,
    )
    _store_dart_band_overrides(st, overrides)
    selected_suffixes = {
        _frequency_widget_suffix(frequency) for frequency in selected_frequencies
    }
    for key in list(st.session_state):
        text_key = str(key)
        for prefix in ("dart_override_enabled_", "dart_override_width_"):
            if text_key.startswith(prefix) and text_key.removeprefix(prefix) not in (
                selected_suffixes
            ):
                st.session_state.pop(key, None)
            if text_key.startswith(prefix):
                break

    columns = st.columns([1.2, 1.0, 1.4])
    with columns[0]:
        default_width = float(
            st.number_input(
                "Default total bandwidth (MHz)",
                min_value=0.000001,
                max_value=max(0.000001, observed_high - observed_low),
                value=default_width,
                key="dart_default_bandwidth_mhz",
                format="%.6f",
            )
        )
    with columns[1]:
        active_frequency = float(
            st.selectbox(
                "Active radio frequency (MHz)",
                options=selected_frequencies,
                key="dart_active_frequency_mhz",
                format_func=lambda value: f"{float(value):g}",
            )
        )
    widget_suffix = _frequency_widget_suffix(active_frequency)
    override_enabled_key = f"dart_override_enabled_{widget_suffix}"
    override_width_key = f"dart_override_width_{widget_suffix}"
    if override_enabled_key not in st.session_state:
        st.session_state[override_enabled_key] = active_frequency in overrides
    override_enabled = bool(
        st.checkbox(
            f"Override bandwidth for {active_frequency:g} MHz",
            key=override_enabled_key,
        )
    )
    effective_width = overrides.get(active_frequency, default_width)
    if not override_enabled:
        st.session_state[override_width_key] = default_width
        effective_width = default_width
    elif override_width_key not in st.session_state:
        st.session_state[override_width_key] = effective_width
    with columns[2]:
        active_width = float(
            st.number_input(
                f"{active_frequency:g} MHz total bandwidth",
                min_value=0.000001,
                max_value=max(0.000001, observed_high - observed_low),
                value=float(effective_width),
                key=override_width_key,
                format="%.6f",
                disabled=not override_enabled,
            )
        )
    if override_enabled:
        overrides[active_frequency] = active_width
    else:
        overrides.pop(active_frequency, None)
    _store_dart_band_overrides(st, overrides)
    if st.button(
        "Clear all per-frequency bandwidth overrides",
        disabled=not overrides,
        width="stretch",
    ):
        st.session_state["_pending_dart_band_reset_overrides"] = True
        st.rerun()

    bands = build_centered_frequency_bands(
        selected_frequencies,
        default_width,
        overrides,
    )
    active_band = bands[active_frequency]
    st.session_state["dart_band_low"] = active_band.low_mhz
    st.session_state["dart_band_high"] = active_band.high_mhz
    _invalidate_if_controls_changed(
        st,
        "dart_band_controls_signature",
        {
            "inspection": st.session_state.get("inspection_signature"),
            "bands": {
                f"{frequency:g}": frequency_band.to_dict()
                for frequency, frequency_band in bands.items()
            },
        },
        _invalidate_composite,
    )
    valid_bands: dict[float, FrequencyBand] = {}
    errors: dict[float, str] = {}
    for frequency, frequency_band in bands.items():
        try:
            frequency_band.validate_observed_range(window.frequency_mhz)
        except ValueError as exc:
            errors[frequency] = str(exc)
        else:
            valid_bands[frequency] = frequency_band
    for frequency, message in errors.items():
        st.error(f"{frequency:g} MHz: {message}")

    figure = build_dart_selection_figure(
        window,
        bands=bands,
        active_frequency_mhz=active_frequency,
    )
    apply_plotly_chrome(figure, theme_mode)
    revision = int(st.session_state.get("dart_band_revision", 0))
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode=("box",),
        key=f"radio_composite_dart_band_{revision}",
    )
    selected = frequency_band_from_selection(event)
    if selected is not None:
        recentered = build_centered_frequency_bands(
            [active_frequency],
            selected.bandwidth_mhz,
        )[active_frequency]
        try:
            recentered.validate_observed_range(window.frequency_mhz)
        except ValueError as exc:
            st.error(f"{active_frequency:g} MHz: {exc}")
        else:
            if not math.isclose(
                recentered.bandwidth_mhz,
                active_band.bandwidth_mhz,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                st.session_state["_pending_dart_bandwidth"] = {
                    "frequency_mhz": active_frequency,
                    "bandwidth_mhz": recentered.bandwidth_mhz,
                }
                st.session_state["dart_band_revision"] = revision + 1
                _invalidate_composite(st)
                st.rerun()

    rows = []
    for frequency, frequency_band in bands.items():
        rows.append(
            {
                "radio_frequency_mhz": frequency,
                "dart_center_mhz": frequency_band.center_mhz,
                "band_low_mhz": frequency_band.low_mhz,
                "band_high_mhz": frequency_band.high_mhz,
                "bandwidth_mhz": frequency_band.bandwidth_mhz,
                "width_source": "override" if frequency in overrides else "default",
                "status": "ready" if frequency in valid_bands else "invalid",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Each DART center is locked to its radio Source Map frequency. Dragging "
        "the spectrum changes only the active frequency's width and recenters it."
    )


def _render_analysis_step(
    st: Any,
    path_policy: PathAccessPolicy,
    *,
    radio_dir_text: str,
    dart_dir_text: str,
    output_dir_text: str,
    map_frequency: float,
    candidate: dict[str, Any],
    radio_coverage: tuple[datetime, datetime] | None,
    dart_window: DartSpectrogramWindow,
    source_configs: dict[float, dict[str, Any]],
    grouped_candidates: dict[float, list[dict[str, Any]]],
    sequence_jobs: CompositeSequenceJobRegistry,
) -> None:
    st.subheader("7. Analyze and export static and sequence products")
    if radio_coverage is None:
        st.error("Radio time coverage is unavailable.")
        return
    output_mode_columns = st.columns(2)
    with output_mode_columns[0]:
        save_video = bool(
            st.checkbox(
                "Generate MP4 video",
                value=True,
                key="sequence_save_video",
            )
        )
    with output_mode_columns[1]:
        save_frames = bool(
            st.checkbox(
                "Save every composite frame as PNG",
                value=True,
                key="sequence_save_frames",
            )
        )
    if not save_video and not save_frames:
        st.error("Enable MP4 video, PNG frames, or both before sequence export.")
    image_bytes = st.session_state.get("source_map_image_bytes")
    metadata = st.session_state.get("source_map_metadata")
    map_time_text = st.session_state.get("source_map_observation_time")
    if (
        not isinstance(image_bytes, bytes)
        or not isinstance(metadata, dict)
        or not map_time_text
    ):
        st.info("Render the reference Source Map before analysis and export.")
        return
    start_default, end_default = radio_coverage
    time_columns = st.columns([2, 2, 1, 1, 1])
    with time_columns[0]:
        start_text = st.text_input(
            "Shared UTC start",
            value=start_default.isoformat(),
            key="shared_time_start",
        )
    with time_columns[1]:
        end_text = st.text_input(
            "Shared UTC end",
            value=end_default.isoformat(),
            key="shared_time_end",
        )
    with time_columns[2]:
        dpi = int(
            st.number_input(
                "Export DPI",
                min_value=100,
                max_value=400,
                value=DEFAULT_DPI,
                step=10,
                key="composite_dpi",
            )
        )
    with time_columns[3]:
        stride = int(
            st.number_input(
                "Frame stride",
                min_value=1,
                value=DEFAULT_SEQUENCE_STRIDE,
                step=1,
                key="sequence_stride",
            )
        )
    with time_columns[4]:
        fps = float(
            st.number_input(
                "Video FPS",
                min_value=0.2,
                max_value=60.0,
                value=DEFAULT_SEQUENCE_FPS,
                step=1.0,
                key="sequence_fps",
                disabled=not save_video,
            )
        )
    quality = st.selectbox(
        "Video quality",
        options=("high", "medium", "low"),
        key="sequence_quality",
        disabled=not save_video,
    )
    frequencies = sorted(float(value) for value in grouped_candidates)
    try:
        bands = _session_dart_bands(st, frequencies, dart_window)
    except Exception as exc:
        st.error(str(exc))
        return
    _invalidate_if_controls_changed(
        st,
        "analysis_controls_signature",
        {
            "map": st.session_state.get("source_map_result"),
            "roi": st.session_state.get("confirmed_roi"),
            "dart_bands_by_frequency": {
                f"{frequency:g}": band.to_dict() for frequency, band in bands.items()
            },
            "time_start": start_text,
            "time_end": end_text,
            "dpi": dpi,
        },
        _invalidate_composite,
    )
    _invalidate_if_controls_changed(
        st,
        "sequence_controls_signature",
        {
            "stride": stride,
            "fps": fps if save_video else None,
            "quality": quality if save_video else None,
            "save_video": save_video,
            "save_frames": save_frames,
        },
        _invalidate_sequence,
    )
    confirmed_roi = _session_roi(st, "confirmed_roi")
    try:
        start = _utc_datetime(start_text)
        end = _utc_datetime(end_text)
        map_time = _utc_datetime(map_time_text)
        if start < radio_coverage[0] or end > radio_coverage[1]:
            raise ValueError(
                "Shared time range must stay inside the radio sequence coverage"
            )
        if start >= end:
            raise ValueError("Shared UTC start must be before the end")
        if not start <= map_time <= end:
            raise ValueError(
                "Selected Source Map time must lie inside the shared range"
            )
        overlap_start, overlap_end, partial_dart_coverage = select_dart_time_overlap(
            dart_window.time_utc,
            start,
            end,
        )
    except Exception as exc:
        st.error(str(exc))
        return
    if partial_dart_coverage:
        st.warning(
            "DART covers only part of the shared radio time range. The lower "
            "panel will remain empty outside DART coverage; no values are extrapolated."
        )
    if confirmed_roi is None:
        st.warning("Confirm one ROI before generating the composite.")
    generate_disabled = confirmed_roi is None
    try:
        frame_counts = sequence_frame_counts(
            grouped_candidates,
            start,
            end,
            stride=stride,
        )
    except Exception as exc:
        st.error(str(exc))
        return
    count_frame = pd.DataFrame(
        [
            {
                "frequency_mhz": frequency,
                "output_frames": count,
                "video_seconds": count / fps,
                "estimated_png_gib": round(
                    count
                    * max(
                        500_000,
                        len(image_bytes) * 2 if isinstance(image_bytes, bytes) else 0,
                    )
                    / 1024**3,
                    3,
                ),
            }
            for frequency, count in frame_counts.items()
        ]
    )
    st.dataframe(count_frame, hide_index=True, width="stretch")
    total_frames = sum(frame_counts.values())
    large_confirmed = True
    if total_frames > LARGE_SEQUENCE_WARNING_FRAMES:
        st.warning(
            f"This export contains {total_frames:,} composite frames. Rendering and "
            "encoding may take substantial time and disk space."
        )
        large_confirmed = bool(
            st.checkbox(
                "I understand the size and want to enable this sequence export",
                key="confirm_large_sequence_export",
            )
        )
    if st.button(
        "Analyze all selected frequencies and generate the reference composite",
        type="primary",
        width="stretch",
        disabled=generate_disabled,
    ):
        try:
            radio_dir = path_policy.input_directory(radio_dir_text)
            dart_dir = path_policy.input_directory(dart_dir_text)
            manifest = st.session_state["radio_manifest"]
            radio_paths_by_frequency = {
                frequency: _manifest_paths_for_request(
                    manifest,
                    frequency_mhz=frequency,
                    start=start,
                    end=end,
                )
                for frequency in frequencies
            }
            dart_files = st.session_state["dart_files"]
            signature = build_request_signature(
                {
                    "inspection": st.session_state.get("inspection_signature"),
                    "source_map": st.session_state.get("source_map_result"),
                    "candidate": candidate,
                    "frequencies_mhz": frequencies,
                    "reference_frequency_mhz": map_frequency,
                    "map_time": map_time,
                    "polarization": st.session_state.get("source_polarization"),
                    "display": metadata.get("display"),
                    "roi": confirmed_roi.to_json_dict(),
                    "time_start": start,
                    "time_end": end,
                    "dart_bands_by_frequency": {
                        f"{frequency:g}": frequency_band.to_dict()
                        for frequency, frequency_band in bands.items()
                    },
                    "metric": "raw_sum",
                    "dart_representation": "stokes_i_db",
                    "dpi": dpi,
                },
                source_paths=[
                    *[
                        path
                        for paths in radio_paths_by_frequency.values()
                        for path in paths
                    ],
                    dart_files.stokes_i_db,
                    dart_files.stokes_v_over_i,
                    dart_files.frequency,
                    dart_files.time,
                ],
            )
            cached = st.session_state.get("composite_bundle")
            if (
                isinstance(cached, CompositeArtifactBundle)
                and st.session_state.get("composite_signature") == signature
                and isinstance(st.session_state.get("radio_curves_by_frequency"), dict)
                and isinstance(
                    st.session_state.get("dart_narrowband_results_by_frequency"),
                    dict,
                )
            ):
                bundle = cached
                radio_curves = st.session_state["radio_curves_by_frequency"]
                dart_result = st.session_state["dart_narrowband_result"]
                dart_results_by_frequency = st.session_state[
                    "dart_narrowband_results_by_frequency"
                ]
                source_context = st.session_state.get("composite_source_context", {})
                st.success(
                    "Reused the current composite; no FITS files were read again."
                )
            else:
                with st.spinner(
                    "Extracting the radio ROI and DART narrowband curves..."
                ):
                    radio_curves: dict[float, pd.DataFrame] = {}
                    for frequency in frequencies:
                        radio_df = extract_radio_roi_lightcurve(
                            radio_dir,
                            confirmed_roi,
                            pattern=str(
                                st.session_state.get("radio_pattern", DEFAULT_PATTERN)
                            ),
                            recursive=bool(
                                st.session_state.get("radio_recursive", True)
                            ),
                            files=radio_paths_by_frequency[frequency],
                            freqs=[frequency],
                            polarization=str(
                                st.session_state.get("source_polarization", "RR+LL")
                            ),
                        )
                        radio_times = pd.to_datetime(
                            radio_df.get("obs_time"), errors="coerce", utc=True
                        )
                        in_range = (radio_times >= pd.Timestamp(start)) & (
                            radio_times <= pd.Timestamp(end)
                        )
                        radio_df = radio_df.loc[in_range].reset_index(drop=True)
                        if radio_df.empty:
                            raise ValueError(
                                f"{frequency:g} MHz contains no radio sample in the shared UTC range"
                            )
                        valid_pixels = pd.to_numeric(
                            radio_df.get("valid_pixel_count"), errors="coerce"
                        )
                        if not (valid_pixels > 0).any():
                            raise ValueError(
                                f"ROI contains no valid pixel at {frequency:g} MHz"
                            )
                        radio_curves[frequency] = radio_df
                    dart_results_by_frequency, dart_result = (
                        _extract_dart_results_by_frequency(
                            dart_dir,
                            bands,
                            time_range_utc=(overlap_start, overlap_end),
                        )
                    )
                    reference_dart_result = _frequency_mapping_value(
                        dart_results_by_frequency,
                        map_frequency,
                    )
                    bundle = build_composite_artifacts(
                        image_bytes,
                        metadata,
                        radio_curves[map_frequency],
                        reference_dart_result,
                        roi=confirmed_roi,
                        map_time=map_time,
                        map_frequency_mhz=map_frequency,
                        polarization=str(
                            st.session_state.get("source_polarization", "RR+LL")
                        ),
                        time_start=start,
                        time_end=end,
                        request_signature=signature,
                        source_context={
                            "radio_directory": str(radio_dir),
                            "radio_files_by_frequency": {
                                f"{frequency:g}": [str(path) for path in paths]
                                for frequency, paths in radio_paths_by_frequency.items()
                            },
                            "dart_directory": str(dart_dir),
                            "dart_files": {
                                "stokes_i_db": str(dart_files.stokes_i_db),
                                "stokes_v_over_i": str(dart_files.stokes_v_over_i),
                                "frequency": str(dart_files.frequency),
                                "time": str(dart_files.time),
                            },
                        },
                        dpi=dpi,
                    )
                    source_context = {
                        "radio_directory": str(radio_dir),
                        "radio_files_by_frequency": {
                            f"{frequency:g}": [str(path) for path in paths]
                            for frequency, paths in radio_paths_by_frequency.items()
                        },
                        "dart_directory": str(dart_dir),
                        "dart_band": bands[map_frequency].to_dict(),
                        "dart_bands_by_frequency": {
                            f"{frequency:g}": frequency_band.to_dict()
                            for frequency, frequency_band in bands.items()
                        },
                        "dart_partial_coverage": partial_dart_coverage,
                        "dart_files": {
                            "stokes_i_db": str(dart_files.stokes_i_db),
                            "stokes_v_over_i": str(dart_files.stokes_v_over_i),
                            "frequency": str(dart_files.frequency),
                            "time": str(dart_files.time),
                        },
                    }
                st.session_state["composite_bundle"] = bundle
                st.session_state["composite_signature"] = signature
                st.session_state["radio_curves_by_frequency"] = radio_curves
                st.session_state["dart_narrowband_result"] = dart_result
                st.session_state["dart_narrowband_results_by_frequency"] = (
                    dart_results_by_frequency
                )
                st.session_state["composite_source_context"] = source_context
                if isinstance(bundle.curve_template, CompositeFrameTemplate):
                    st.session_state["curve_plot_cache_by_frequency"] = {
                        float(map_frequency): bundle.curve_template
                    }
                st.session_state.pop("composite_saved_directory", None)
                _invalidate_sequence(st)
                st.success("Composite and reproducibility products generated.")
        except Exception as exc:
            st.error(str(exc))

    bundle = st.session_state.get("composite_bundle")
    if not isinstance(bundle, CompositeArtifactBundle):
        return
    st.image(bundle.files["composite_png"], width="stretch")
    download_columns = st.columns(3)
    _download(
        download_columns[0],
        "Download composite PNG",
        bundle.files["composite_png"],
        bundle.filenames["composite_png"],
        "image/png",
        "download_composite_png",
    )
    _download(
        download_columns[1],
        "Download radio ROI CSV",
        bundle.files["radio_csv"],
        bundle.filenames["radio_csv"],
        "text/csv",
        "download_radio_csv",
    )
    _download(
        download_columns[2],
        "Download DART CSV",
        bundle.files["dart_csv"],
        bundle.filenames["dart_csv"],
        "text/csv",
        "download_dart_csv",
    )
    curve_columns = st.columns(2)
    _download(
        curve_columns[0],
        "Download radio curve PNG (no marker)",
        bundle.files["radio_curve_png"],
        bundle.filenames["radio_curve_png"],
        "image/png",
        "download_radio_curve_png",
    )
    _download(
        curve_columns[1],
        "Download DART curve PNG (no marker)",
        bundle.files["dart_curve_png"],
        bundle.filenames["dart_curve_png"],
        "image/png",
        "download_dart_curve_png",
    )
    more_columns = st.columns(3)
    _download(
        more_columns[0],
        "Download ROI JSON",
        bundle.files["roi_json"],
        bundle.filenames["roi_json"],
        "application/json",
        "download_roi_json",
    )
    _download(
        more_columns[1],
        "Download metadata JSON",
        bundle.files["metadata_json"],
        bundle.filenames["metadata_json"],
        "application/json",
        "download_metadata_json",
    )
    _download(
        more_columns[2],
        "Download complete ZIP",
        bundle.zip_bytes,
        bundle.zip_filename,
        "application/zip",
        "download_composite_zip",
    )
    if st.button("Save complete package to output directory", width="stretch"):
        try:
            output_directory = path_policy.output_directory(output_dir_text)
            saved = save_composite_bundle(bundle, output_directory)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["composite_saved_directory"] = str(saved)
    if saved := st.session_state.get("composite_saved_directory"):
        st.success(f"Saved the complete package to {saved}")

    radio_curves = st.session_state.get("radio_curves_by_frequency")
    dart_result = st.session_state.get("dart_narrowband_result")
    dart_results_by_frequency = st.session_state.get(
        "dart_narrowband_results_by_frequency"
    )
    source_context = st.session_state.get("composite_source_context")
    if (
        not isinstance(radio_curves, dict)
        or not isinstance(dart_result, DartNarrowbandResult)
        or not isinstance(dart_results_by_frequency, dict)
    ):
        return
    sequence_signature = build_request_signature(
        {
            "analysis_signature": st.session_state.get("composite_signature"),
            "source_map_discovery": st.session_state.get(
                "source_map_discovery_signature"
            ),
            "time_start": start,
            "time_end": end,
            "stride": stride,
            "fps": fps if save_video else None,
            "quality": quality if save_video else None,
            "save_video": save_video,
            "save_frames": save_frames,
            "dpi": dpi,
            "transform": st.session_state.get("map_transform", "linear"),
        }
    )
    active_job_id = st.session_state.get("sequence_job_id")
    active_running = False
    if isinstance(active_job_id, str):
        try:
            active_running = sequence_jobs.public(active_job_id)["status"] in {
                "running",
                "canceling",
            }
        except KeyError:
            st.session_state.pop("sequence_job_id", None)
    sequence_button_label = (
        "Generate one MP4 and PNG sequence per frequency"
        if save_video and save_frames
        else (
            "Generate one MP4 sequence per frequency"
            if save_video
            else "Generate one PNG sequence per frequency"
        )
    )
    if st.button(
        sequence_button_label,
        type="primary",
        width="stretch",
        disabled=(
            active_running
            or not large_confirmed
            or (not save_video and not save_frames)
        ),
        on_click=_pause_sequence_preview,
        args=(st,),
    ):
        try:
            output_directory = path_policy.output_directory(output_dir_text)
            options = SequenceExportOptions(
                fps=fps,
                stride=stride,
                dpi=dpi,
                quality=quality,
                transform=str(st.session_state.get("map_transform", "linear")),
                save_video=save_video,
                save_frames=save_frames,
            )
            frozen_configs = copy.deepcopy(source_configs)
            frozen_candidates = copy.deepcopy(grouped_candidates)
            frozen_curves = {
                float(frequency): curve.copy(deep=True)
                for frequency, curve in radio_curves.items()
            }
            frozen_dart_results = copy.deepcopy(dart_results_by_frequency)
            frozen_context = copy.deepcopy(dict(source_context or {}))
            session_curve_templates = st.session_state.get(
                "curve_plot_cache_by_frequency"
            )
            frozen_curve_templates = (
                {
                    float(frequency): template
                    for frequency, template in session_curve_templates.items()
                    if isinstance(template, CompositeFrameTemplate)
                }
                if isinstance(session_curve_templates, Mapping)
                else {}
            )
            polarization = str(st.session_state.get("source_polarization", "RR+LL"))

            def task(cancel_check, progress):
                return export_composite_sequences(
                    output_directory,
                    source_configs=frozen_configs,
                    candidates_by_frequency=frozen_candidates,
                    radio_curves=frozen_curves,
                    dart_result=dart_result,
                    dart_results_by_frequency=frozen_dart_results,
                    roi=confirmed_roi,
                    reference_frequency_mhz=map_frequency,
                    reference_time=map_time,
                    polarization=polarization,
                    time_start=start,
                    time_end=end,
                    request_signature=sequence_signature,
                    source_context=frozen_context,
                    options=options,
                    reference_bundle=bundle,
                    curve_templates_by_frequency=frozen_curve_templates,
                    cancel_check=cancel_check,
                    progress=progress,
                )

            job = sequence_jobs.start(task)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["sequence_job_id"] = job["id"]
            st.session_state["sequence_signature"] = sequence_signature
            st.success("Sequence export started in the background.")

    _render_sequence_job_status(st, sequence_jobs)
    sequence_bundle = st.session_state.get("sequence_bundle")
    if isinstance(sequence_bundle, CompositeSequenceBundle):
        _render_sequence_downloads(st, sequence_bundle)


def _render_sequence_job_status(
    st: Any, sequence_jobs: CompositeSequenceJobRegistry
) -> None:
    job_id = st.session_state.get("sequence_job_id")
    notice = st.session_state.pop("sequence_job_notice", None)
    if isinstance(notice, dict):
        if notice.get("status") == "failed":
            st.error(str(notice.get("message") or "Sequence export failed"))
        elif notice.get("status") == "canceled":
            st.warning("Sequence export canceled; the previous valid package was kept.")
    if not isinstance(job_id, str):
        return

    @st.fragment(run_every=1.0)
    def status_fragment() -> None:
        try:
            job = sequence_jobs.public(job_id)
        except KeyError:
            st.session_state.pop("sequence_job_id", None)
            st.session_state["sequence_job_notice"] = {
                "status": "failed",
                "message": "Sequence export job expired before completion.",
            }
            st.rerun()
            return
        total = int(job.get("total") or 0)
        completed = int(job.get("completed") or 0)
        ratio = completed / total if total > 0 else 0.0
        st.progress(
            min(1.0, max(0.0, ratio)),
            text=str(job.get("message") or "Sequence export running"),
        )
        status = str(job.get("status"))
        if status in {"running", "canceling"}:
            if st.button(
                "Cancel sequence export",
                key=f"cancel_sequence_{job_id}",
                disabled=status == "canceling",
            ):
                sequence_jobs.cancel(job_id)
            return
        st.session_state.pop("sequence_job_id", None)
        if status == "completed" and isinstance(
            job.get("result"), CompositeSequenceBundle
        ):
            completed_bundle = job["result"]
            st.session_state["sequence_bundle"] = completed_bundle
            cached_templates = st.session_state.get("curve_plot_cache_by_frequency")
            merged_templates = (
                dict(cached_templates) if isinstance(cached_templates, Mapping) else {}
            )
            merged_templates.update(completed_bundle.curve_templates)
            st.session_state["curve_plot_cache_by_frequency"] = merged_templates
            st.session_state["sequence_job_notice"] = {
                "status": "completed",
                "message": "Sequence export completed.",
            }
        elif status == "failed":
            st.session_state["sequence_job_notice"] = {
                "status": "failed",
                "message": job.get("error") or "Sequence export failed",
            }
        else:
            st.session_state["sequence_job_notice"] = {
                "status": "canceled",
                "message": "Sequence export canceled",
            }
        st.rerun()

    status_fragment()


def _pause_sequence_preview(st: Any) -> None:
    """Pause the Play widget from a callback or before it is instantiated."""

    st.session_state["sequence_preview_playing"] = False


def _render_sequence_downloads(st: Any, bundle: CompositeSequenceBundle) -> None:
    st.success(f"Sequence package saved to {bundle.output_directory}")
    dart_csv_paths = getattr(bundle, "dart_csv_paths", {})
    radio_plot_paths = getattr(bundle, "radio_plot_paths", {})
    dart_plot_paths = getattr(bundle, "dart_plot_paths", {})
    frequencies = sorted(
        {
            *[float(value) for value in bundle.videos],
            *[float(value) for value in bundle.frame_directories],
            *[float(value) for value in bundle.radio_csv_paths],
            *[float(value) for value in dart_csv_paths],
            *[float(value) for value in radio_plot_paths],
            *[float(value) for value in dart_plot_paths],
        }
    )
    if not frequencies:
        st.warning("The sequence package contains no per-frequency products.")
        return
    selected = float(
        st.selectbox(
            "Sequence product frequency (MHz)",
            options=frequencies,
            key="sequence_product_frequency",
            format_func=lambda value: f"{value:g}",
        )
    )
    radio_csv_path = _frequency_mapping_value(bundle.radio_csv_paths, selected)
    dart_csv_path = (
        _frequency_mapping_value(dart_csv_paths, selected)
        if dart_csv_paths
        else bundle.dart_csv_path
    )
    video_path = (
        _frequency_mapping_value(bundle.videos, selected) if bundle.videos else None
    )
    if video_path is not None:
        st.video(str(video_path))
    columns = st.columns(5 if video_path is not None else 4)
    _download_file(
        columns[0],
        "Download complete sequence ZIP",
        bundle.zip_path,
        "application/zip",
        "download_sequence_zip",
    )
    next_column = 1
    if video_path is not None:
        _download_file(
            columns[next_column],
            f"Download {selected:g} MHz MP4",
            video_path,
            "video/mp4",
            "download_sequence_video",
        )
        next_column += 1
    _download_file(
        columns[next_column],
        f"Download {selected:g} MHz ROI CSV",
        radio_csv_path,
        "text/csv",
        "download_sequence_radio_csv",
    )
    next_column += 1
    _download_file(
        columns[next_column],
        f"Download {selected:g} MHz DART CSV",
        dart_csv_path,
        "text/csv",
        "download_sequence_frequency_dart_csv",
    )
    next_column += 1
    _download_file(
        columns[next_column],
        "Download sequence metadata",
        bundle.metadata_path,
        "application/json",
        "download_sequence_metadata",
    )
    shared = st.columns(2)
    _download_file(
        shared[0],
        "Download shared DART CSV",
        bundle.dart_csv_path,
        "text/csv",
        "download_sequence_dart_csv",
    )
    _download_file(
        shared[1],
        "Download shared ROI JSON",
        bundle.roi_json_path,
        "application/json",
        "download_sequence_roi_json",
    )
    if radio_plot_paths and dart_plot_paths:
        curve_columns = st.columns(2)
        _download_file(
            curve_columns[0],
            f"Download {selected:g} MHz radio curve PNG (no marker)",
            _frequency_mapping_value(radio_plot_paths, selected),
            "image/png",
            "download_sequence_radio_curve_png",
        )
        _download_file(
            curve_columns[1],
            f"Download {selected:g} MHz DART curve PNG (no marker)",
            _frequency_mapping_value(dart_plot_paths, selected),
            "image/png",
            "download_sequence_dart_curve_png",
        )


def _download_file(
    column: Any,
    label: str,
    path: Path,
    mime: str,
    key: str,
) -> None:
    with column, path.open("rb") as handle:
        import streamlit as st

        st.download_button(
            label,
            data=handle,
            file_name=path.name,
            mime=mime,
            on_click="ignore",
            key=key,
            width="stretch",
        )


def _create_sequence_job_registry() -> CompositeSequenceJobRegistry:
    return CompositeSequenceJobRegistry()


def _frequency_mapping_value(mapping: dict | Mapping, frequency: float):
    for key, value in mapping.items():
        try:
            if math.isclose(float(key), float(frequency), rel_tol=0.0, abs_tol=1e-6):
                return value
        except TypeError, ValueError:
            continue
    raise ValueError(f"Missing data for {float(frequency):g} MHz")


def prepare_single_panel_render(
    config: dict[str, Any],
    candidate: dict[str, Any],
    frequency_mhz: float,
    *,
    transform: str,
    output_directory: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility wrapper for the shared sequence render adapter."""

    return _prepare_sequence_single_panel_render(
        config,
        candidate,
        frequency_mhz,
        transform=transform,
        output_directory=output_directory,
    )


def _candidate_frequency_paths(
    candidate: dict[str, Any], frequency_mhz: float
) -> tuple[list[str], str | list[str]]:
    return candidate_frequency_paths(candidate, frequency_mhz)


def _matching_frequency_index(values: list[float], requested: float) -> int:
    from solar_apps.frontends.radio.composite_figure.composite_sequence import (
        matching_frequency_index,
    )

    return matching_frequency_index(values, requested)


def _candidate_has_frequency(candidate: dict[str, Any], requested: float) -> bool:
    return candidate_contains_frequency(candidate, requested)


def _candidate_label(candidate: dict[str, Any]) -> str:
    frequencies = ", ".join(
        f"{float(value):g}" for value in candidate.get("frequencies_mhz", [])
    )
    return (
        f"{candidate.get('observation_time') or 'unknown time'} | "
        f"{frequencies or 'unknown'} MHz | {candidate.get('title') or candidate['id']}"
    )


def _candidate_time_coverage(
    candidates: list[dict[str, Any]],
) -> tuple[datetime, datetime]:
    values = sorted(
        _utc_datetime(candidate["observation_time"])
        for candidate in candidates
        if candidate.get("observation_time")
    )
    if not values:
        raise ValueError("Selected radio sequence contains no observation times")
    if values[0] == values[-1]:
        raise ValueError("Selected radio sequence needs at least two distinct times")
    return values[0], values[-1]


def _manifest_paths_for_request(
    manifest: pd.DataFrame,
    *,
    frequency_mhz: float,
    start: datetime,
    end: datetime,
) -> list[Path]:
    data = manifest.copy()
    frequency_column = (
        "inferred_freq_mhz" if "inferred_freq_mhz" in data else "freq_mhz"
    )
    frequencies = pd.to_numeric(data.get(frequency_column), errors="coerce")
    times = pd.to_datetime(data.get("inferred_obs_time"), errors="coerce", utc=True)
    tolerance = max(1e-6, abs(float(frequency_mhz)) * 1e-5)
    frequency_mask = np.isfinite(frequencies.to_numpy(dtype=float, na_value=np.nan)) & (
        np.abs(frequencies - float(frequency_mhz)) <= tolerance
    )
    if times.notna().any():
        time_mask = times.isna() | (
            (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        )
        mask = frequency_mask & time_mask
    else:
        mask = frequency_mask
    paths = [Path(value) for value in data.loc[mask, "path"].astype(str)]
    if not paths:
        raise ValueError("No radio files match the selected frequency and time range")
    return paths


def _frequency_values(frame: Any) -> list[float]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    column = "freq_mhz" if "freq_mhz" in frame else frame.columns[0]
    values = pd.to_numeric(frame[column], errors="coerce")
    return sorted({float(value) for value in values if np.isfinite(value)})


def _default_band(frequencies: Any, preferred: float | None) -> FrequencyBand:
    values = np.sort(np.unique(np.asarray(frequencies, dtype=float)))
    values = values[np.isfinite(values)]
    if values.size < 2:
        center = float(values[0]) if values.size else float(preferred or 1.0)
        return FrequencyBand(center - 0.5, center + 0.5)
    low_observed = float(values[0])
    high_observed = float(values[-1])
    spacing = float(np.median(np.diff(values)))
    width = max(abs(spacing) * 3.0, (high_observed - low_observed) / 100.0)
    center = (
        float(preferred)
        if preferred is not None and low_observed <= float(preferred) <= high_observed
        else (low_observed + high_observed) / 2.0
    )
    low = max(low_observed, center - width / 2.0)
    high = min(high_observed, center + width / 2.0)
    if low >= high:
        low, high = low_observed, high_observed
    return FrequencyBand(low, high)


def _valid_bandwidth_or_default(
    value: Any,
    default: float,
    *,
    maximum: float,
) -> float:
    try:
        width = float(value)
    except TypeError, ValueError:
        width = float(default)
    if not math.isfinite(width) or width <= 0 or width > float(maximum):
        width = float(default)
    return width


def _nearest_frequency(frequencies: list[float], requested: float) -> float:
    if not frequencies:
        raise ValueError("No radio frequency is available")
    return min(frequencies, key=lambda value: abs(float(value) - float(requested)))


def _frequency_widget_suffix(frequency: float) -> str:
    return f"{float(frequency):.9g}".replace("-", "m").replace(".", "p")


def _dart_band_overrides(
    raw: Any,
    frequencies: list[float] | tuple[float, ...],
) -> dict[float, float]:
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = {}
    elif isinstance(raw, Mapping):
        decoded = dict(raw)
    else:
        decoded = {}
    selected = [float(value) for value in frequencies]
    overrides: dict[float, float] = {}
    if not isinstance(decoded, Mapping):
        return overrides
    for raw_frequency, raw_width in decoded.items():
        try:
            frequency = _nearest_frequency(selected, float(raw_frequency))
            width = float(raw_width)
        except TypeError, ValueError:
            continue
        if not math.isclose(frequency, float(raw_frequency), rel_tol=0.0, abs_tol=1e-6):
            continue
        if math.isfinite(width) and width > 0:
            overrides[frequency] = width
    return overrides


def _store_dart_band_overrides(st: Any, overrides: Mapping[float, float]) -> None:
    st.session_state["dart_band_overrides_json"] = json.dumps(
        {
            f"{float(frequency):.9g}": float(width)
            for frequency, width in sorted(overrides.items())
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _session_dart_bands(
    st: Any,
    frequencies: list[float],
    window: DartSpectrogramWindow,
) -> dict[float, FrequencyBand]:
    if not frequencies:
        raise ValueError("Select at least one radio frequency")
    fallback = _default_band(window.frequency_mhz, frequencies[0]).bandwidth_mhz
    default_width = _valid_bandwidth_or_default(
        st.session_state.get("dart_default_bandwidth_mhz"),
        fallback,
        maximum=float(
            np.nanmax(window.frequency_mhz) - np.nanmin(window.frequency_mhz)
        ),
    )
    overrides = _dart_band_overrides(
        st.session_state.get("dart_band_overrides_json"),
        frequencies,
    )
    bands = build_centered_frequency_bands(frequencies, default_width, overrides)
    errors: list[str] = []
    for frequency, band in bands.items():
        try:
            band.validate_observed_range(window.frequency_mhz)
        except ValueError as exc:
            errors.append(f"{frequency:g} MHz: {exc}")
    if errors:
        raise ValueError("Invalid DART frequency band(s): " + "; ".join(errors))
    return bands


def _extract_dart_results_by_frequency(
    dart_directory: Path,
    bands: Mapping[float, FrequencyBand],
    *,
    time_range_utc: tuple[datetime, datetime],
    extractor: Any = extract_dart_narrowband_lightcurves,
) -> tuple[dict[float, DartNarrowbandResult], DartNarrowbandResult]:
    """Extract matching DART curves while batching equal bandwidths."""

    grouped: list[tuple[float, list[float]]] = []
    for frequency, band in sorted(bands.items()):
        for width, centers in grouped:
            if math.isclose(width, band.bandwidth_mhz, rel_tol=0.0, abs_tol=1e-9):
                centers.append(float(frequency))
                break
        else:
            grouped.append((band.bandwidth_mhz, [float(frequency)]))

    results: dict[float, DartNarrowbandResult] = {}
    shared_times: tuple[datetime, ...] | None = None
    curves_by_frequency: dict[float, DartNarrowbandCurve] = {}
    for bandwidth, centers in grouped:
        extracted = extractor(
            dart_directory,
            centers,
            bandwidth,
            time_range_utc=time_range_utc,
        )
        if shared_times is None:
            shared_times = tuple(extracted.time_utc)
        elif tuple(extracted.time_utc) != shared_times:
            raise ValueError("DART narrowband results do not share one UTC time axis")
        for center in centers:
            matching = [
                curve
                for curve in extracted.curves
                if math.isclose(
                    float(curve.center_frequency_mhz),
                    center,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"DART extraction did not return exactly one {center:g} MHz curve"
                )
            curves_by_frequency[center] = matching[0]

    if shared_times is None:
        raise ValueError("No DART frequency band is available for extraction")
    for frequency in sorted(bands):
        curve = curves_by_frequency.get(float(frequency))
        if curve is None:
            raise ValueError(f"Missing DART curve for {frequency:g} MHz")
        results[float(frequency)] = DartNarrowbandResult(
            time_utc=shared_times,
            curves=(curve,),
        )
    combined = DartNarrowbandResult(
        time_utc=shared_times,
        curves=tuple(
            curves_by_frequency[float(frequency)] for frequency in sorted(bands)
        ),
    )
    return results, combined


def _session_roi(st: Any, key: str) -> RadioRoi | None:
    value = st.session_state.get(key)
    if not isinstance(value, dict):
        return None
    try:
        return radio_roi_from_json(value)
    except Exception:
        return None


def _apply_pending_band(st: Any) -> None:
    pending = st.session_state.pop("_pending_dart_band", None)
    if isinstance(pending, dict):
        st.session_state["dart_band_low"] = float(pending["low_mhz"])
        st.session_state["dart_band_high"] = float(pending["high_mhz"])
        st.session_state["dart_default_bandwidth_mhz"] = float(
            pending["high_mhz"]
        ) - float(pending["low_mhz"])
    if st.session_state.pop("_pending_dart_band_reset_overrides", False):
        st.session_state["dart_band_overrides_json"] = "{}"
        for key in list(st.session_state):
            if str(key).startswith("dart_override_enabled_") or str(key).startswith(
                "dart_override_width_"
            ):
                st.session_state.pop(key, None)
    pending_width = st.session_state.pop("_pending_dart_bandwidth", None)
    if isinstance(pending_width, Mapping):
        frequency = float(pending_width["frequency_mhz"])
        bandwidth = float(pending_width["bandwidth_mhz"])
        raw = st.session_state.get("dart_band_overrides_json", "{}")
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError, TypeError, ValueError:
            decoded = {}
        decoded[f"{frequency:.9g}"] = bandwidth
        st.session_state["dart_band_overrides_json"] = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
        )
        suffix = _frequency_widget_suffix(frequency)
        st.session_state[f"dart_override_enabled_{suffix}"] = True
        st.session_state[f"dart_override_width_{suffix}"] = bandwidth


def _apply_cli_defaults_once(st: Any, args: argparse.Namespace) -> None:
    """Let explicit launcher paths win over older private UI state once."""

    if st.session_state.get("_radio_composite_cli_defaults_applied"):
        return
    overrides = {
        "radio_dir": args.radio_dir,
        "dart_dir": args.dart_dir,
        "output_dir": args.output_dir,
        "radio_pattern": args.pattern,
        "radio_recursive": args.recursive,
    }
    for key, value in overrides.items():
        if value is not None:
            st.session_state[key] = value
    st.session_state["_radio_composite_cli_defaults_applied"] = True


def _invalidate_after_inspection(st: Any) -> None:
    for key in TRANSIENT_KEYS:
        st.session_state.pop(key, None)


def _invalidate_after_discovery(st: Any) -> None:
    for key in (
        "source_map_image_bytes",
        "source_map_metadata",
        "source_map_result",
        "source_map_candidate",
        "source_map_observation_time",
        "source_map_frequency_mhz",
        "candidate_roi",
        "confirmed_roi",
        "sequence_preview_cache",
        "sequence_preview_index",
        "sequence_preview_slider",
        "sequence_preview_playing",
        "radio_curves_by_frequency",
        "dart_narrowband_result",
        "dart_narrowband_results_by_frequency",
        "composite_source_context",
        "composite_bundle",
        "composite_signature",
        "composite_saved_directory",
        "sequence_signature",
        "sequence_job_id",
        "sequence_bundle",
    ):
        st.session_state.pop(key, None)


def _invalidate_after_source_controls(st: Any) -> None:
    for key in (
        "source_map_config",
        "source_map_configs_by_frequency",
        "source_map_candidates",
        "source_map_candidates_by_frequency",
        "source_selected_frequencies",
        "source_common_coverage",
        "source_map_discovery_signature",
    ):
        st.session_state.pop(key, None)
    _invalidate_after_discovery(st)


def _invalidate_after_map_selection(st: Any) -> None:
    for key in (
        "source_map_image_bytes",
        "source_map_metadata",
        "source_map_result",
        "source_map_candidate",
        "source_map_observation_time",
        "source_map_frequency_mhz",
    ):
        st.session_state.pop(key, None)
    _invalidate_after_map(st)


def _invalidate_after_roi_controls(st: Any) -> None:
    st.session_state.pop("candidate_roi", None)
    st.session_state.pop("confirmed_roi", None)
    _invalidate_composite(st)


def _invalidate_after_map(st: Any) -> None:
    for key in (
        "candidate_roi",
        "confirmed_roi",
        "sequence_preview_cache",
    ):
        st.session_state.pop(key, None)
    _invalidate_composite(st)


def _invalidate_composite(st: Any) -> None:
    for key in (
        "composite_bundle",
        "composite_signature",
        "composite_saved_directory",
        "radio_curves_by_frequency",
        "dart_narrowband_result",
        "dart_narrowband_results_by_frequency",
        "composite_source_context",
        "curve_plot_cache_by_frequency",
    ):
        st.session_state.pop(key, None)
    _invalidate_sequence(st)


def _invalidate_sequence(st: Any) -> None:
    job_id = st.session_state.get("sequence_job_id")
    if isinstance(job_id, str) and hasattr(st, "cache_resource"):
        try:
            st.cache_resource(_create_sequence_job_registry)().cancel(job_id)
        except KeyError, RuntimeError:
            pass
    for key in (
        "sequence_signature",
        "sequence_job_id",
        "sequence_bundle",
        "sequence_job_notice",
    ):
        st.session_state.pop(key, None)


def _invalidate_if_controls_changed(
    st: Any,
    signature_key: str,
    payload: dict[str, Any],
    invalidator: Any,
) -> str:
    """Invalidate dependent state only when a material control value changes."""

    signature = build_request_signature(payload)
    previous = st.session_state.get(signature_key)
    if isinstance(previous, str) and previous != signature:
        invalidator(st)
    st.session_state[signature_key] = signature
    return signature


def _download(
    column: Any,
    label: str,
    data: bytes,
    filename: str,
    mime: str,
    key: str,
) -> None:
    with column:
        import streamlit as st

        st.download_button(
            label,
            data=data,
            file_name=filename,
            mime=mime,
            on_click="ignore",
            key=key,
            width="stretch",
        )


def _utc_datetime(value: datetime | str | None) -> datetime:
    if value is None:
        raise ValueError("UTC datetime value is missing")
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        normalized = text[:-1] + "+00:00" if text.upper().endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid UTC datetime: {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    main()


__all__ = [
    "FRONTEND_ID",
    "UI_FIELD_KEYS",
    "build_parser",
    "main",
    "prepare_single_panel_render",
]
