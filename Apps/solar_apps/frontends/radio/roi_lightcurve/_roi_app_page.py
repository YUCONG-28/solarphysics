"""Streamlit page composition and controls for the ROI workflow.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import AllowedRootPolicyError
from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
    RadioImage,
)
from solar_toolkit.radio.roi_lightcurve import (
    RadioRoi,
    extract_radio_roi_lightcurve,
)
from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_application import (
    PRODUCT_FILENAMES,
)
from solar_apps.ui.state import (
    bind_streamlit_fields,
    frontend_path_memory,
    frontend_state_store,
    save_streamlit_fields,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
    resolve_streamlit_allowed_roots,
)
from solar_apps.ui.theme import apply_plotly_chrome, render_streamlit_theme
from solar_apps.workflows.radio.spatial_display import spatial_display_for_reference
from ._roi_app_helpers import (
    _frequency_state_key,
    _option_index,
)
from ._roi_app_helpers import _roi_app_facade


from ._roi_app_state import (
    PRODUCT_LABELS,
    PRODUCT_MIME_TYPES,
    DISPLAY_COLORMAPS,
    DISPLAY_TRANSFORMS,
    DISPLAY_RANGE_MODES,
    DISPLAY_RANGE_SCOPES,
    ROI_UI_FIELD_KEYS,
    SELECTION_ACTIONS,
    ROI_KEYS,
    ANALYSIS_KEYS,
    EXPORT_KEYS,
    REFERENCE_KEYS,
    _LIGHTCURVE_Y_AXIS_MODES,
    _LIGHTCURVE_PRODUCT_KEYS,
    _LIGHTCURVE_DEFAULT_MARKER_SIZE,
    _LIGHTCURVE_PLOT_STYLE_LABELS,
    _LIGHTCURVE_Y_TRANSFORM_LABELS,
    _RoiImportDocument,
    _DisplayReferencePreview,
)


def main(argv: list[str] | None = None) -> int:
    """Run the Streamlit app."""

    _run_streamlit_app(argv)
    return 0


def _run_streamlit_app(argv: list[str] | None = None) -> None:
    import streamlit as st

    args = _roi_app_facade().build_parser().parse_args(argv)
    settings_file = (
        Path(args.settings_file).expanduser()
        if args.settings_file
        else _roi_app_facade().default_settings_path()
    )
    layout = RuntimeLayout.discover()
    ui_store = frontend_state_store("roi-lightcurve", layout=layout)
    ui_snapshot = ui_store.load(default={})
    import_legacy = not bool(ui_snapshot.get("legacy_imported"))
    stored = (
        _roi_app_facade().load_app_settings(
            settings_file, reset=bool(args.reset_settings)
        )
        if import_legacy
        else {}
    )
    settings = _roi_app_facade().resolve_app_settings(args, stored)

    st.set_page_config(page_title="Radio ROI Light Curve", layout="wide")
    _roi_app_facade()._init_session_state(st)
    st.title("Radio ROI Light Curve")
    st.caption(
        "Load radio FITS files, select a time series, draw one ROI, preview the curve, then export."
    )

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
        protected_output_roots=(layout.outputs_dir / "radio_roi_lightcurve",),
        base_directory=local_root,
    )
    render_streamlit_theme(
        st,
        frontend_id="roi-lightcurve",
        state_store=ui_store,
        path_memory=frontend_path_memory(path_policy.output_roots, layout=layout),
    )
    bind_streamlit_fields(
        st,
        ui_store,
        frontend_id="roi-lightcurve",
        field_keys=ROI_UI_FIELD_KEYS,
    )
    ui_store.update({"legacy_imported": True})

    current_settings = _render_load_step(
        st, settings, settings_file, path_policy, ui_store
    )
    manifest = st.session_state.get("loaded_manifest")
    if manifest is None:
        st.info("Enter a radio FITS folder and click Load Data.")
        return
    if manifest.empty:
        st.warning("No matching FITS files were found.")
        return

    selected_paths = _render_file_selection_step(st, manifest, current_settings)
    if not selected_paths:
        st.warning("Select one or more FITS files to continue.")
        return

    reference_images = _render_reference_step(
        st, selected_paths, manifest, current_settings
    )
    if not reference_images:
        st.warning("Render selected frequencies as reference images.")
        return

    display_config, reference_previews = _render_display_settings_step(
        st,
        reference_images,
        current_settings,
    )
    roi = _render_roi_step(
        st,
        reference_images,
        reference_previews,
        display_config,
        path_policy,
        ui_store,
    )
    if roi is None:
        st.warning("Draw and confirm an ROI on the reference image.")
        return

    _render_analysis_and_export_steps(
        selected_paths,
        reference_images,
        roi,
        current_settings,
        display_config,
        path_policy,
    )


def _render_load_step(
    st: Any,
    settings: dict[str, Any],
    settings_file: Path,
    path_policy: PathAccessPolicy,
    ui_store: Any,
) -> dict[str, Any]:
    del settings_file  # legacy settings are imported read-only into Local/state
    st.subheader("Step 1. Load Data")
    c1, c2 = st.columns([3, 1])
    with c1:
        radio_dir = _roi_app_facade().render_native_path_input(
            st,
            "Radio FITS folder",
            key="radio_dir",
            initial_value=str(settings["radio_dir"]),
            roots=path_policy.input_roots,
            kind="directory",
            placeholder="data/radio",
            help_text="Folder that contains the radio source FITS files. Subfolders such as 149MHz/LL are supported.",
            frontend_id="roi-lightcurve",
            operation="load-data",
            state_store=ui_store,
        )
    with c2:
        pattern = st.text_input(
            "FITS pattern",
            key="pattern",
            value=str(settings["pattern"]),
            placeholder="*.fits",
            help="Glob pattern used to find FITS files before frequency and time filtering.",
        )
    c3, c4 = st.columns([1, 3])
    with c3:
        recursive = st.checkbox(
            "Search subfolders",
            key="recursive",
            value=bool(settings["recursive"]),
            help="Enable this when frequency and polarization files live in nested folders.",
        )
    source_settings = {
        "radio_dir": radio_dir,
        "pattern": pattern,
        "recursive": recursive,
    }
    source_signature = _roi_app_facade()._source_signature(source_settings)
    with c4:
        if st.button(
            "Discover Frequencies",
            type="primary",
            help="Scan matching FITS paths and list available observing frequencies without loading image arrays.",
        ):
            _roi_app_facade()._discover_frequencies_into_state(
                st, source_settings, path_policy
            )

    frequency_options = st.session_state.get("frequency_options")
    if frequency_options is None:
        st.info("Click Discover Frequencies to list the bands available in the folder.")
    elif st.session_state.get("frequency_source_signature") != source_signature:
        st.warning(
            "The folder, pattern, or recursive setting changed. Discover frequencies again before loading."
        )
    elif frequency_options.empty:
        st.warning(
            "No path-inferred frequencies were found. You can still load all matching FITS files."
        )
    else:
        st.dataframe(frequency_options, hide_index=True, width="stretch")

    freq_values = _roi_app_facade()._frequency_options_list(frequency_options)
    default_freqs = _roi_app_facade()._default_selected_frequencies(
        freq_values, settings.get("selected_freqs_mhz", [])
    )
    selected_freqs = st.multiselect(
        "Frequencies to load (MHz)",
        key="selected_freqs_mhz",
        options=freq_values,
        default=default_freqs,
        format_func=lambda value: f"{value:g} MHz",
        placeholder="Select one or more frequencies",
        help="Choose one or more radio bands. Empty means load all matching frequencies.",
    )

    range_hint = _roi_app_facade()._manifest_time_range_hint(
        st.session_state.get("loaded_full_manifest")
    )
    c5, c6 = st.columns([1, 1])
    with c5:
        time_start = st.text_input(
            "Start time",
            key="time_start",
            value=str(settings["time_start"]),
            placeholder=range_hint.get("start", "YYYY-MM-DDTHH:MM:SS"),
            help="Inclusive UTC start time. Leave blank to use the first available time shown in the data range hint.",
        )
    with c6:
        time_end = st.text_input(
            "End time",
            key="time_end",
            value=str(settings["time_end"]),
            placeholder=range_hint.get("end", "YYYY-MM-DDTHH:MM:SS"),
            help="Inclusive UTC end time. Leave blank to use the last available time shown in the data range hint.",
        )
    if range_hint:
        st.caption(
            "Available selected-data time range: "
            f"{range_hint['start']} to {range_hint['end']} UTC "
            f"({range_hint['timed_count']:,} timed files; {range_hint['untimed_count']:,} without filename time)."
        )

    c6, c7, c8 = st.columns([1, 1, 2])
    with c6:
        pair_tolerance = st.number_input(
            "LCP/RCP pair tolerance (s)",
            key="pair_time_tolerance_sec",
            min_value=0.0,
            value=float(settings["pair_time_tolerance_sec"]),
            step=0.1,
            help="Maximum time difference allowed when pairing LCP and RCP files for L+R analysis.",
        )
    with c7:
        polarization = st.selectbox(
            "Polarization",
            [POL_SUM, POL_LCP, POL_RCP, "all"],
            key="polarization",
            index=[POL_SUM, POL_LCP, POL_RCP, "all"].index(
                str(settings["polarization"])
            ),
            help="Analysis polarization mode. L+R pairs matching LCP/RCP files; all keeps individual planes.",
        )
    with c8:
        metric = st.selectbox(
            "Curve metric",
            ["raw_sum", "raw_mean", "raw_peak"],
            key="metric",
            index=["raw_sum", "raw_mean", "raw_peak"].index(str(settings["metric"])),
            help="Statistic plotted in the preview light curve and exported PNG.",
        )
    output_dir = _roi_app_facade().render_native_path_input(
        st,
        "Output folder",
        key="output_dir",
        initial_value=str(settings["output_dir"]),
        roots=path_policy.output_roots,
        kind="directory",
        placeholder="Choose an allowed output folder",
        help_text="Folder used by Save Selected Products. Browser downloads are available without saving locally.",
        frontend_id="roi-lightcurve",
        operation="save-products",
        state_store=ui_store,
    )

    current_settings = {
        "radio_dir": radio_dir,
        "pattern": pattern,
        "recursive": recursive,
        "selected_freqs_mhz": [float(item) for item in selected_freqs],
        "time_start": time_start,
        "time_end": time_end,
        "output_dir": output_dir,
        "pair_time_tolerance_sec": pair_tolerance,
        "polarization": polarization,
        "metric": metric,
        "display_colormap": settings["display_colormap"],
        "display_transform": settings["display_transform"],
        "display_range_mode": settings["display_range_mode"],
        "display_range_scope": settings["display_range_scope"],
        "display_low_percentile": float(settings["display_low_percentile"]),
        "display_high_percentile": float(settings["display_high_percentile"]),
        "display_manual_min": float(settings["display_manual_min"]),
        "display_manual_max": float(settings["display_manual_max"]),
        "display_bad_color": settings["display_bad_color"],
        "display_use_custom_fov": bool(settings["display_use_custom_fov"]),
        "display_x_min_arcsec": float(settings["display_x_min_arcsec"]),
        "display_x_max_arcsec": float(settings["display_x_max_arcsec"]),
        "display_y_min_arcsec": float(settings["display_y_min_arcsec"]),
        "display_y_max_arcsec": float(settings["display_y_max_arcsec"]),
        "preview_max_side": int(settings["preview_max_side"]),
        "page_size": int(settings["page_size"]),
    }
    c9, c10, c11 = st.columns([1, 1, 3])
    with c9:
        if st.button(
            "Load Selected Frequencies",
            type="primary",
            help="Build a lightweight file index for the chosen frequencies, then apply the optional time range.",
        ):
            _roi_app_facade()._load_manifest_into_state(
                st, current_settings, path_policy
            )
    with c10:
        if st.button(
            "Apply Time Range",
            help="Reuse the already loaded frequency index and apply the start/end time fields without rescanning.",
        ):
            _roi_app_facade()._apply_loaded_time_filter(st, current_settings)
    with c11:
        if st.button(
            "Save Defaults",
            help="Save the current form values to the private Local UI state.",
        ):
            save_streamlit_fields(st, ui_store, ROI_UI_FIELD_KEYS)
            st.success("Saved the latest UI settings locally.")
    save_streamlit_fields(st, ui_store, ROI_UI_FIELD_KEYS)
    return current_settings


def _render_file_selection_step(
    st: Any,
    manifest: pd.DataFrame,
    settings: dict[str, Any],
) -> list[str]:
    st.subheader("Step 2. Choose Data Files")
    query = st.text_input(
        "Filter files",
        value="",
        placeholder="Search relative path, time, frequency, or polarization",
        help="Narrows the table before selecting by page, all filtered files, or File # expression.",
    )
    filtered = _roi_app_facade()._filter_manifest(manifest, query)
    page_size = int(settings.get("page_size", 200))
    page_count = max(1, int(np.ceil(len(filtered) / max(1, page_size))))
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        page = int(
            st.number_input(
                "Page",
                min_value=1,
                max_value=page_count,
                value=1,
                step=1,
                help="Table page number. File # values remain stable across pages.",
            )
        )
    with c2:
        if st.button(
            "Select Page", help="Add every file visible on the current table page."
        ):
            page_paths = _roi_app_facade()._page_paths(
                filtered, page=page, page_size=page_size
            )
            _roi_app_facade()._set_selected_paths(
                st,
                set(st.session_state.get("selected_paths", [])) | set(page_paths),
                manifest,
            )
    with c3:
        if st.button(
            "Select All Filtered", help="Add all rows matching the current filter."
        ):
            _roi_app_facade()._set_selected_paths(
                st,
                set(st.session_state.get("selected_paths", []))
                | set(filtered["path"].astype(str)),
                manifest,
            )
    with c4:
        if st.button(
            "Invert Filtered",
            help="Toggle selection for all rows matching the current filter.",
        ):
            filtered_paths = set(filtered["path"].astype(str))
            current = set(st.session_state.get("selected_paths", []))
            _roi_app_facade()._set_selected_paths(
                st, (current - filtered_paths) | (filtered_paths - current), manifest
            )
    with c5:
        if st.button("Clear Selection", help="Remove every selected file."):
            _roi_app_facade()._set_selected_paths(st, set(), manifest)

    q1, q2, q3 = st.columns([2, 1, 1])
    with q1:
        number_expression = st.text_input(
            "Quick select by File #",
            value="",
            placeholder="1, 3, 8-20",
            help="Use comma-separated File # values or inclusive ranges. Example: 1,3,8-20.",
        )
    with q2:
        selection_action = st.selectbox(
            "Quick action",
            SELECTION_ACTIONS,
            help="Replace uses only the entered numbers; Add and Remove modify the current selection.",
        )
    with q3:
        if st.button(
            "Apply Numbers",
            help="Apply the File # expression to rows that are still present after filtering.",
        ):
            _roi_app_facade()._apply_number_selection(
                st, manifest, filtered, number_expression, selection_action
            )

    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end].copy()
    selected = set(st.session_state.get("selected_paths", []))
    page_df.insert(0, "selected", page_df["path"].astype(str).isin(selected))
    edited = st.data_editor(
        page_df[
            [
                "selected",
                "row",
                "relative_path",
                "inferred_obs_time",
                "inferred_freq_mhz",
                "inferred_polarization",
                "size_mib",
                "path",
            ]
        ],
        key=f"radio_roi_file_editor_{st.session_state.get('dataset_signature', '')}_{page}",
        disabled=[
            "row",
            "relative_path",
            "inferred_obs_time",
            "inferred_freq_mhz",
            "inferred_polarization",
            "size_bytes",
            "mtime_ns",
            "size_mib",
            "path",
        ],
        column_config={
            "selected": st.column_config.CheckboxColumn(
                "Select", help="Toggle this file in the selected set."
            ),
            "row": st.column_config.NumberColumn(
                "File #", help="Stable sequence number for quick selection."
            ),
            "relative_path": st.column_config.TextColumn(
                "Relative path", help="Path relative to the radio FITS folder."
            ),
            "inferred_obs_time": st.column_config.TextColumn(
                "Time", help="UTC time inferred from the filename."
            ),
            "inferred_freq_mhz": st.column_config.NumberColumn(
                "MHz", help="Frequency inferred from path or FITS-style naming."
            ),
            "inferred_polarization": st.column_config.TextColumn(
                "Pol",
                help="Polarization inferred from folder, header text, or filename.",
            ),
            "size_bytes": None,
            "mtime_ns": None,
            "size_mib": st.column_config.NumberColumn("MiB", help="File size in MiB."),
            "path": st.column_config.TextColumn(
                "Absolute path", help="Full FITS file path used during analysis."
            ),
        },
        hide_index=True,
        width="stretch",
    )
    current_page_paths = set(page_df["path"].astype(str))
    edited_selected = set(
        edited.loc[edited["selected"].astype(bool), "path"].astype(str)
    )
    _roi_app_facade()._set_selected_paths(
        st,
        (set(st.session_state.get("selected_paths", [])) - current_page_paths)
        | edited_selected,
        manifest,
    )
    selected_paths = _roi_app_facade()._order_paths_by_manifest(
        st.session_state.get("selected_paths", []), manifest
    )
    st.caption(
        f"Showing {len(page_df):,} of {len(filtered):,} filtered files. Selected {len(selected_paths):,} files."
    )
    return selected_paths


def _render_reference_step(
    st: Any,
    selected_paths: list[str],
    manifest: pd.DataFrame,
    settings: dict[str, Any],
) -> list[RadioImage]:
    st.subheader("Step 3. Render Reference Images")
    selected_set = set(selected_paths)
    available = manifest.loc[manifest["path"].astype(str).isin(selected_set)].copy()
    if available.empty:
        return []
    frequencies = _roi_app_facade()._manifest_frequency_values(available)
    current_primary = st.session_state.get("primary_reference_freq_mhz")
    primary_default = (
        current_primary
        if current_primary in frequencies
        else (frequencies[0] if frequencies else math.nan)
    )
    selected_rows = set(available["row"].astype(int).tolist())
    current_anchor = st.session_state.get("reference_file_number")
    anchor_default = (
        int(current_anchor)
        if current_anchor in selected_rows
        else int(available.iloc[0]["row"])
    )
    preview_pol_options = [POL_SUM, POL_LCP, POL_RCP]
    stored_preview_pol = st.session_state.get("preview_polarization")
    default_preview_pol = (
        stored_preview_pol
        if stored_preview_pol in preview_pol_options
        else (
            settings["polarization"]
            if settings["polarization"] in preview_pol_options
            else POL_SUM
        )
    )
    with st.form("radio_roi_reference_grid_form"):
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            primary_frequency = st.selectbox(
                "Primary ROI frequency",
                frequencies,
                index=(
                    frequencies.index(primary_default)
                    if primary_default in frequencies
                    else 0
                ),
                format_func=lambda value: f"{value:g} MHz",
                help="The first rendered panel is interactive. Draw the ROI there; it is stored in HPLN/HPLT arcsec and applied to all frequencies.",
            )
        with c2:
            anchor_number = int(
                st.number_input(
                    "Anchor File #",
                    min_value=int(manifest["row"].min()),
                    max_value=int(manifest["row"].max()),
                    value=anchor_default,
                    step=1,
                    help="Representative images are chosen nearest to this selected file's time in each frequency.",
                )
            )
        with c3:
            preview_polarization = st.selectbox(
                "Preview polarization",
                preview_pol_options,
                index=preview_pol_options.index(default_preview_pol),
                help="Display-only polarization used for reference images. Analysis uses the Step 1 polarization setting.",
            )
        submitted = st.form_submit_button(
            "Render Reference Grid",
            type="primary",
            help="Submit these controls once, then load or reuse one representative image for each selected frequency.",
        )

    if submitted and anchor_number not in selected_rows:
        st.error("Anchor File # must be one of the selected rows.")
    elif submitted:
        try:
            pair_tolerance_sec = float(settings["pair_time_tolerance_sec"])
            references = list(st.session_state.get("reference_images") or [])
            reuse_signature = _roi_app_facade()._reference_reuse_signature(
                st,
                primary_frequency=float(primary_frequency),
                anchor_number=anchor_number,
                preview_polarization=str(preview_polarization),
                pair_tolerance_sec=pair_tolerance_sec,
            )
            if references and reuse_signature == st.session_state.get(
                "reference_reuse_signature"
            ):
                st.info("Reused the cached reference grid; no FITS files were reread.")
            else:
                plans = _roi_app_facade()._plan_reference_grid(
                    available,
                    primary_frequency=float(primary_frequency),
                    anchor_number=anchor_number,
                    preview_polarization=str(preview_polarization),
                    pair_tolerance_sec=pair_tolerance_sec,
                )
                signature = _roi_app_facade()._reference_grid_signature(
                    available,
                    plans,
                    primary_frequency=float(primary_frequency),
                    anchor_number=anchor_number,
                    preview_polarization=str(preview_polarization),
                    pair_tolerance_sec=pair_tolerance_sec,
                )
                if (
                    signature == st.session_state.get("reference_grid_signature")
                    and references
                ):
                    st.info(
                        "Reused the cached reference grid; no FITS files were reread."
                    )
                else:
                    references, reference_meta = (
                        _roi_app_facade()._materialize_reference_grid(plans)
                    )
                    if not references:
                        raise RuntimeError(
                            "No usable representative image could be loaded."
                        )
                    _roi_app_facade()._clear_keys(
                        st,
                        ("reference_preview_cache_key", "reference_preview_cache"),
                    )
                    st.session_state["reference_images"] = references
                    st.session_state["reference_metadata"] = reference_meta
                    st.session_state["reference_path"] = str(references[0].path)
                    st.session_state["reference_grid_signature"] = signature
                    _roi_app_facade()._clear_keys(
                        st, (*ROI_KEYS, *ANALYSIS_KEYS, *EXPORT_KEYS)
                    )
                    st.session_state["roi_chart_generation"] = (
                        int(st.session_state.get("roi_chart_generation", 0)) + 1
                    )
            if not references:
                raise RuntimeError("No usable representative image could be loaded.")
            st.session_state["reference_file_number"] = anchor_number
            st.session_state["primary_reference_freq_mhz"] = float(primary_frequency)
            st.session_state["preview_polarization"] = str(preview_polarization)
            st.session_state["reference_pair_tolerance_sec"] = pair_tolerance_sec
            st.session_state[
                "reference_reuse_signature"
            ] = _roi_app_facade()._reference_reuse_signature(
                st,
                primary_frequency=float(primary_frequency),
                anchor_number=anchor_number,
                preview_polarization=str(preview_polarization),
                pair_tolerance_sec=pair_tolerance_sec,
            )
        except Exception as exc:  # noqa: BLE001 - visible app error.
            st.error(str(exc))
            return []
    references = list(st.session_state.get("reference_images") or [])
    reference_meta = list(st.session_state.get("reference_metadata") or [])
    referenced_paths = {
        str(path)
        for meta in reference_meta
        for path in (meta.get("path"), meta.get("paired_path"))
        if path
    }
    if not references or not referenced_paths.issubset(selected_set):
        _roi_app_facade()._clear_keys(
            st, (*REFERENCE_KEYS, *ROI_KEYS, *ANALYSIS_KEYS, *EXPORT_KEYS)
        )
        return []
    return references


def _render_display_settings_step(
    st: Any,
    reference_images: list[RadioImage],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], list[_DisplayReferencePreview]]:
    st.subheader("Step 4. Display Settings")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        colormap = st.selectbox(
            "Colormap",
            DISPLAY_COLORMAPS,
            key="display_colormap",
            index=_option_index(DISPLAY_COLORMAPS, str(settings["display_colormap"])),
            help="Common radio image color map. Hot matches the source-visibility preset from the existing radio plots.",
        )
    with c2:
        transform = st.selectbox(
            "Intensity transform",
            DISPLAY_TRANSFORMS,
            key="display_transform",
            index=_option_index(DISPLAY_TRANSFORMS, str(settings["display_transform"])),
            help="Linear shows raw values; Log10 positive shows log10 only for positive pixels.",
        )
    with c3:
        range_mode = st.selectbox(
            "Intensity range",
            DISPLAY_RANGE_MODES,
            key="display_range_mode",
            index=_option_index(
                DISPLAY_RANGE_MODES, str(settings["display_range_mode"])
            ),
            help="Auto percentile derives limits from the displayed representative images; manual uses entered raw FITS values.",
        )
    with c4:
        range_scope = st.selectbox(
            "Range scope",
            DISPLAY_RANGE_SCOPES,
            key="display_range_scope",
            index=_option_index(
                DISPLAY_RANGE_SCOPES, str(settings["display_range_scope"])
            ),
            help="Per frequency computes limits per panel; shared/global uses one scale for every displayed frequency.",
        )

    config: dict[str, Any] = {
        "colormap": colormap,
        "transform": transform,
        "range_mode": range_mode,
        "range_scope": range_scope,
        "bad_color": str(settings["display_bad_color"]),
        "preview_max_side": int(settings["preview_max_side"]),
    }
    if range_mode == "Auto percentile":
        c5, c6 = st.columns([1, 1])
        with c5:
            low = st.number_input(
                "Low percentile",
                key="display_low_percentile",
                min_value=0.0,
                max_value=100.0,
                value=float(settings["display_low_percentile"]),
                step=0.1,
                help="Lower auto display percentile. The source-visibility preset uses 99.7.",
            )
        with c6:
            high = st.number_input(
                "High percentile",
                key="display_high_percentile",
                min_value=0.0,
                max_value=100.0,
                value=float(settings["display_high_percentile"]),
                step=0.01,
                help="Upper auto display percentile. The source-visibility preset uses 99.99.",
            )
        config["low_percentile"] = min(float(low), float(high))
        config["high_percentile"] = max(float(low), float(high))
    elif range_scope == "Per frequency":
        display_limits, raw_limits = _manual_range_editor(
            st, reference_images, settings, transform=transform
        )
        config["limits_by_frequency"] = display_limits
        config["manual_ranges_raw"] = raw_limits
        config["range_mode"] = "Manual min/max"
    else:
        c7, c8 = st.columns([1, 1])
        with c7:
            manual_min = st.number_input(
                "Manual min",
                key="display_manual_min",
                value=float(settings["display_manual_min"]),
                help="Raw FITS-unit lower display limit. In log mode this value must be positive.",
            )
        with c8:
            manual_max = st.number_input(
                "Manual max",
                key="display_manual_max",
                value=float(settings["display_manual_max"]),
                help="Raw FITS-unit upper display limit. In log mode this value must be positive.",
            )
        config["manual_min"] = float(manual_min)
        config["manual_max"] = float(manual_max)

    f1, f2, f3, f4, f5 = st.columns([1, 1, 1, 1, 1])
    with f1:
        use_custom_fov = st.checkbox(
            "Use custom FOV",
            key="display_use_custom_fov",
            value=bool(settings["display_use_custom_fov"]),
            help="Limit the displayed HPLN/HPLT range. It does not crop the scientific ROI extraction.",
        )
    with f2:
        x_min = st.number_input(
            "HPLN min",
            key="display_x_min_arcsec",
            value=float(settings["display_x_min_arcsec"]),
            help="Left display bound in arcsec.",
        )
    with f3:
        x_max = st.number_input(
            "HPLN max",
            key="display_x_max_arcsec",
            value=float(settings["display_x_max_arcsec"]),
            help="Right display bound in arcsec.",
        )
    with f4:
        y_min = st.number_input(
            "HPLT min",
            key="display_y_min_arcsec",
            value=float(settings["display_y_min_arcsec"]),
            help="Bottom display bound in arcsec.",
        )
    with f5:
        y_max = st.number_input(
            "HPLT max",
            key="display_y_max_arcsec",
            value=float(settings["display_y_max_arcsec"]),
            help="Top display bound in arcsec.",
        )
    config.update(
        {
            "use_custom_fov": bool(use_custom_fov),
            "x_min_arcsec": float(x_min),
            "x_max_arcsec": float(x_max),
            "y_min_arcsec": float(y_min),
            "y_max_arcsec": float(y_max),
        }
    )
    raw_previews = _roi_app_facade()._reference_previews_from_state(
        st,
        reference_images,
        max_side=int(settings["preview_max_side"]),
    )
    display_previews = [
        _DisplayReferencePreview(
            display_view=_roi_app_facade()._display_array(preview.raw_view, config),
            x_arcsec=preview.x_arcsec,
            y_arcsec=preview.y_arcsec,
        )
        for preview in raw_previews
    ]
    _roi_app_facade()._attach_auto_display_limits(
        reference_images,
        config,
        previews=display_previews,
    )
    display_contract = spatial_display_for_reference(config)
    config["display_contract"] = display_contract.to_dict()
    config["display_cache_signature"] = display_contract.cache_signature()
    return config, display_previews


def _render_roi_import_controls(
    st: Any, path_policy: PathAccessPolicy, ui_store: Any
) -> None:
    st.markdown("**Import ROI JSON**")
    uploaded = st.file_uploader(
        "Load ROI JSON",
        type=["json"],
        help=(
            "Load a Radio ROI JSON or a Source Map ROI set. Source Map files may "
            "contain multiple HPLN/HPLT regions."
        ),
    )
    if uploaded is not None:
        payload = uploaded.getvalue()
        signature = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("roi_import_upload_signature") != signature:
            try:
                document = _roi_app_facade()._roi_import_document_from_uploaded_or_path(
                    uploaded_payload=payload,
                    path_text="",
                    path_policy=path_policy,
                )
            except Exception as exc:  # noqa: BLE001 - visible app error.
                st.error(str(exc))
            else:
                _roi_app_facade()._store_roi_import_document(
                    st,
                    document,
                    source_kind="upload",
                    source_label=str(getattr(uploaded, "name", "uploaded JSON")),
                    upload_signature=signature,
                )

    roi_json_path = _roi_app_facade().render_native_path_input(
        st,
        "ROI JSON path",
        key="roi_json_path",
        initial_value="",
        roots=path_policy.input_roots,
        kind="file",
        extensions=(".json",),
        placeholder="Choose an allowed ROI JSON file",
        help_text=(
            "Optional local ROI JSON path. An uploaded ROI JSON remains the "
            "immediate source when one is present."
        ),
        frontend_id="roi-lightcurve",
        operation="import-roi",
        state_store=ui_store,
    )
    if st.button(
        "Load ROI JSON Path",
        disabled=uploaded is not None,
        help="Load the editable local ROI JSON path. Upload takes priority while present.",
    ):
        try:
            document = _roi_app_facade()._roi_import_document_from_uploaded_or_path(
                uploaded_payload=None,
                path_text=roi_json_path,
                path_policy=path_policy,
            )
        except Exception as exc:  # noqa: BLE001 - visible app error.
            st.error(str(exc))
        else:
            _roi_app_facade()._store_roi_import_document(
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
            f"Loaded {len(document.choices)} Source Map region(s) from {source_label}. "
            "The image SHA-256 is retained as provenance and is not compared with "
            "the analysis FITS files."
        )
        st.caption(f"Source image SHA-256: {document.source_image_sha256}")
    else:
        st.caption(f"Loaded one Radio ROI from {source_label}.")

    if len(document.choices) == 1:
        choice = document.choices[0]
        st.caption(f"Active import: {choice.display_label}")
        return

    choice_by_key = {choice.key: choice for choice in document.choices}
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
        help="Stage this region as the candidate ROI. Confirm it before analysis.",
    ):
        changed = _roi_app_facade()._stage_imported_roi(st, choice_by_key[selected_key])
        if changed:
            st.success(
                "Selected imported region staged. Confirm the ROI to analyze it."
            )
        else:
            st.info("The selected imported region is already active.")


def _render_roi_step(
    st: Any,
    references: list[RadioImage],
    previews: list[_DisplayReferencePreview],
    display_config: dict[str, Any],
    path_policy: PathAccessPolicy,
    ui_store: Any,
) -> RadioRoi | None:
    st.subheader("Step 5. Draw and Confirm ROI")
    _roi_app_facade()._render_roi_import_controls(st, path_policy, ui_store)
    c1, c2 = st.columns([1, 1])
    with c1:
        roi_mode = st.radio(
            "ROI mode",
            ["box", "lasso"],
            horizontal=True,
            help="Box creates a rectangular HPLN/HPLT ROI; lasso creates a polygon ROI.",
        )
    candidate = _roi_app_facade()._session_roi(st, "candidate_roi")
    confirmed = _roi_app_facade()._session_roi(st, "confirmed_roi")
    active_roi = candidate or confirmed
    roi_geometry_key = active_roi.roi_id if active_roi is not None else "empty"
    chart_key = (
        f"radio_roi_selection_chart_"
        f"{st.session_state.get('roi_chart_generation', 0)}_"
        f"{roi_mode}_{roi_geometry_key}"
    )
    st.caption(
        "Draw once on the first panel. The confirmed HPLN/HPLT ROI is overlaid on every loaded frequency."
    )
    columns = st.columns(min(3, max(1, len(references))))
    event = None
    for index, (reference, preview) in enumerate(
        zip(references, previews, strict=True)
    ):
        with columns[index % len(columns)]:
            selection_enabled = index == 0 and confirmed is None
            figure = _roi_app_facade()._build_reference_figure_from_preview(
                reference,
                preview,
                roi=active_roi,
                roi_mode=roi_mode,
                display_config=display_config,
                selection_enabled=selection_enabled,
            )
            apply_plotly_chrome(
                figure,
                st.session_state.get("roi-lightcurve_theme_mode", "auto"),
            )
            if selection_enabled:
                event = st.plotly_chart(
                    figure,
                    width="stretch",
                    on_select="rerun",
                    selection_mode=(roi_mode,),
                    key=chart_key,
                )
            else:
                st.plotly_chart(
                    figure,
                    width="stretch",
                    key=f"{chart_key}_{index}",
                )
    selected_roi = _roi_app_facade().selection_to_radio_roi(
        event, mode=roi_mode, label="active"
    )
    if selected_roi is not None:
        st.session_state["candidate_roi"] = selected_roi.to_json_dict()
        candidate = selected_roi
    with c2:
        if st.button(
            "Confirm ROI",
            type="primary",
            disabled=candidate is None,
            help="Lock the staged ROI for analysis and export.",
        ):
            st.session_state["confirmed_roi"] = candidate.to_json_dict()
            _roi_app_facade()._clear_keys(st, ANALYSIS_KEYS + EXPORT_KEYS)
            confirmed = candidate
    if st.button("Clear ROI", help="Remove both staged and confirmed ROI selections."):
        _roi_app_facade()._clear_keys(st, (*ROI_KEYS, *ANALYSIS_KEYS, *EXPORT_KEYS))
        st.session_state["roi_chart_generation"] = (
            int(st.session_state.get("roi_chart_generation", 0)) + 1
        )
        return None
    if confirmed is not None:
        st.success("ROI confirmed.")
        st.json(confirmed.to_json_dict(), expanded=False)
    elif candidate is not None:
        st.info("ROI is staged. Click Confirm ROI before analysis.")
        st.json(candidate.to_json_dict(), expanded=False)
    return confirmed


def _render_analysis_step(
    st: Any,
    selected_paths: list[str],
    references: list[RadioImage],
    roi: RadioRoi,
    settings: dict[str, Any],
    display_config: dict[str, Any],
) -> pd.DataFrame | None:
    st.subheader("Step 6. Analyze and Preview")
    context_signature = _roi_app_facade()._analysis_context_signature(
        selected_paths,
        roi,
        settings,
        selection_token={
            "dataset_signature": st.session_state.get("dataset_signature", ""),
            "selection_revision": int(st.session_state.get("selection_revision", 0)),
        },
    )
    (
        input_bytes,
        unknown_size_count,
    ) = _roi_app_facade()._selected_input_size_from_manifest(
        st,
        selected_paths,
    )
    st.caption(
        f"Selected input: {len(selected_paths):,} files, "
        f"approximately {input_bytes / (1024**3):.3f} GiB."
    )
    if unknown_size_count:
        st.caption(
            f"The loaded manifest has no size for {unknown_size_count:,} selected "
            "files; they are excluded from the estimate."
        )
    analyze_clicked = st.button(
        "Analyze Selected Files",
        type="primary",
        key="radio_roi_analyze_selected_files_v2",
        help="Extract full-resolution ROI statistics from every selected file using the confirmed HPLN/HPLT ROI.",
    )
    if analyze_clicked:
        file_identities = _roi_app_facade()._selected_file_identities(selected_paths)
        signature = _roi_app_facade()._analysis_signature(
            selected_paths,
            roi,
            settings,
            file_identities=file_identities,
        )
        cached_df = st.session_state.get("analysis_df")
        if (
            isinstance(cached_df, pd.DataFrame)
            and st.session_state.get("analysis_context_signature") == context_signature
            and st.session_state.get("analysis_signature") == signature
        ):
            st.success("Reused the cached analysis; no FITS files were read again.")
        else:
            with st.spinner(
                "Extracting full-resolution ROI statistics from selected files..."
            ):
                try:
                    df = extract_radio_roi_lightcurve(
                        settings["radio_dir"],
                        roi,
                        pattern=settings["pattern"],
                        recursive=bool(settings["recursive"]),
                        files=selected_paths,
                        polarization=settings["polarization"],
                        pair_time_tolerance_sec=float(
                            settings["pair_time_tolerance_sec"]
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - visible app error.
                    st.error(str(exc))
                    return None
            st.session_state["analysis_df"] = df
            st.session_state["analysis_context_signature"] = context_signature
            st.session_state["analysis_signature"] = signature
            st.session_state[
                "analysis_result_signature"
            ] = _roi_app_facade()._stable_sha256(
                {
                    "analysis_signature": signature,
                    "dataframe": _roi_app_facade()._dataframe_content_signature(df),
                }
            )
            st.session_state["lightcurve_png_cache"] = {}
            _roi_app_facade()._clear_keys(st, EXPORT_KEYS)
    df = st.session_state.get("analysis_df")
    if (
        df is None
        or st.session_state.get("analysis_context_signature") != context_signature
    ):
        st.info("Click Analyze Selected Files to compute the light curve.")
        return None
    if not st.session_state.get("analysis_result_signature"):
        st.session_state[
            "analysis_result_signature"
        ] = _roi_app_facade()._stable_sha256(
            {
                "analysis_signature": st.session_state.get("analysis_signature", ""),
                "dataframe": _roi_app_facade()._dataframe_content_signature(df),
            }
        )
    preview_rows = 500
    st.dataframe(df.head(preview_rows), width="stretch")
    if len(df) > preview_rows:
        st.caption(
            f"Showing the first {preview_rows:,} of {len(df):,} rows. "
            "The complete table remains available in the Statistics CSV export."
        )
    y_axis_config = _render_lightcurve_y_axis_controls(
        st,
        df,
        metric=str(settings["metric"]),
    )
    if not bool(y_axis_config.get("plot_ready", True)):
        return df
    preview_kwargs = {
        "analysis_result_signature": str(st.session_state["analysis_result_signature"]),
        "metric": str(settings["metric"]),
        "y_axis_mode": str(y_axis_config["mode"]),
        "lightcurve_y_limits": y_axis_config.get("limits"),
        "lightcurve_frequency_y_limits": _roi_app_facade()._frequency_limit_mapping(
            y_axis_config
        ),
        "lightcurve_frequency_config": _roi_app_facade()._canonical_frequency_configs(
            y_axis_config
        ),
        "lightcurve_marker_size": float(y_axis_config["marker_size"]),
        "lightcurve_detail_frequency_mhz": y_axis_config.get("detail_frequency_mhz"),
        "lightcurve_plot_style": str(y_axis_config["plot_style"]),
        "lightcurve_y_transform": str(y_axis_config["y_transform"]),
    }
    overview_tab, detail_tab, normalized_tab = st.tabs(
        [
            "Frequency Overview",
            "Frequency Detail",
            "Normalized Comparison",
        ]
    )
    with overview_tab:
        st.image(
            _roi_app_facade()._cached_lightcurve_png(
                st,
                df,
                roi,
                product_key="lightcurve_png",
                **preview_kwargs,
            )
        )
    with detail_tab:
        st.image(
            _roi_app_facade()._cached_lightcurve_png(
                st,
                df,
                roi,
                product_key="lightcurve_detail_png",
                **preview_kwargs,
            )
        )
    with normalized_tab:
        st.caption(
            "Each frequency is mapped from its current displayed Y range after "
            "the selected transform to 0-1. Samples beyond that range are clipped "
            "only in this view."
        )
        st.image(
            _roi_app_facade()._cached_lightcurve_png(
                st,
                df,
                roi,
                product_key="lightcurve_normalized_png",
                **preview_kwargs,
            )
        )
    return df


def _default_detail_frequency(st: Any, frequencies: list[float]) -> float:
    if not frequencies:
        raise ValueError("No valid frequencies are available for the light curve.")
    try:
        primary = float(st.session_state.get("primary_reference_freq_mhz"))
    except TypeError, ValueError:
        primary = math.nan
    if np.isfinite(primary):
        for frequency in frequencies:
            if np.isclose(frequency, primary, rtol=0.0, atol=1e-6):
                return frequency
    return frequencies[0]


def _render_lightcurve_y_axis_controls(
    st: Any,
    df: pd.DataFrame,
    *,
    metric: str,
) -> dict[str, Any]:
    result_signature = str(
        st.session_state.get("analysis_result_signature")
        or _roi_app_facade()._dataframe_content_signature(df)
    )
    base_identity = _roi_app_facade()._stable_sha256(
        {"analysis_result_signature": result_signature, "metric": metric}
    )[:16]
    style_column, transform_column = st.columns(2)
    with style_column:
        plot_style_label = st.selectbox(
            "Curve style",
            list(_LIGHTCURVE_PLOT_STYLE_LABELS),
            key=f"lightcurve_plot_style_{base_identity}",
            help="Choose independent scatter points or time-ordered connected lines.",
        )
    with transform_column:
        y_transform_label = st.selectbox(
            "Y transform",
            list(_LIGHTCURVE_Y_TRANSFORM_LABELS),
            key=f"lightcurve_y_transform_{base_identity}",
            help=(
                "Log10 positive plots log10 of positive samples and leaves raw "
                "values unchanged in the Statistics CSV."
            ),
        )
    plot_style = _roi_app_facade()._normalize_lightcurve_plot_style(plot_style_label)
    y_transform = _roi_app_facade()._normalize_lightcurve_y_transform(y_transform_label)
    st.caption(
        f"Y-axis label: {_roi_app_facade()._lightcurve_axis_label(df, metric, y_transform)}"
    )

    raw_data = _roi_app_facade()._lightcurve_metric_frame(
        df, metric, y_transform="linear"
    )
    data = _roi_app_facade()._lightcurve_metric_frame(
        df, metric, y_transform=y_transform
    )
    frequencies = _roi_app_facade()._lightcurve_frequencies(data)
    excluded_nonpositive = 0
    if y_transform == "log10":
        excluded_nonpositive = int(
            np.count_nonzero(raw_data[metric].to_numpy(dtype=float) <= 0.0)
        )
        if excluded_nonpositive:
            st.warning(
                f"Log10 omitted {excluded_nonpositive:,} non-positive samples from "
                "the plots only; raw values remain in the table and Statistics CSV."
            )
    if not frequencies:
        st.error(
            "No samples can be plotted with the selected Y transform. "
            "Select Linear when the analysis contains no positive values."
        )
        stored = {
            "mode": "Per frequency",
            "limits": None,
            "display_limits": None,
            "valid": False,
            "plot_ready": False,
            "outside_count": 0,
            "plot_style": plot_style,
            "y_transform": y_transform,
            "metric_unit": _roi_app_facade()._lightcurve_metric_unit(df, metric),
            "marker_size": _LIGHTCURVE_DEFAULT_MARKER_SIZE,
            "detail_frequency_mhz": None,
            "normalization": "per-frequency displayed Y limits mapped to 0-1",
            "frequencies": [],
        }
        st.session_state["lightcurve_y_axis_config"] = stored
        return stored

    identity = _roi_app_facade()._stable_sha256(
        {
            "analysis_result_signature": result_signature,
            "metric": metric,
            "y_transform": y_transform,
        }
    )[:16]
    state_key = "lightcurve_frequency_y_axis_state"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or state.get("identity") != identity:
        state = {"identity": identity, "configs": {}}
    configs = state.setdefault("configs", {})

    marker_size = st.slider(
        "Marker size",
        min_value=1.0,
        max_value=12.0,
        value=_LIGHTCURVE_DEFAULT_MARKER_SIZE,
        step=0.5,
        key=f"lightcurve_marker_size_{identity}",
        help="Point diameter in typographic points for every light-curve view.",
    )
    default_detail = _default_detail_frequency(st, frequencies)
    detail_key = f"lightcurve_detail_frequency_{identity}"
    if st.session_state.get(detail_key) not in frequencies:
        st.session_state[detail_key] = default_detail
    detail_frequency = st.selectbox(
        "Detail frequency",
        frequencies,
        key=detail_key,
        format_func=lambda value: f"{value:g} MHz",
        help="Frequency shown in the enlarged Frequency Detail view.",
    )

    selected_key = f"lightcurve_frequency_to_configure_{identity}"
    if st.session_state.get(selected_key) not in frequencies:
        st.session_state[selected_key] = default_detail
    selected_frequency = st.selectbox(
        "Frequency to configure",
        frequencies,
        key=selected_key,
        format_func=lambda value: f"{value:g} MHz",
        help="Choose one frequency, then set only that panel's Y-axis range.",
    )
    selected_state_key = _frequency_state_key(selected_frequency)
    editor_prefix = f"lightcurve_frequency_editor_{identity}_"
    mode_key = f"{editor_prefix}{selected_state_key}_mode"
    min_key = f"{editor_prefix}{selected_state_key}_min"
    max_key = f"{editor_prefix}{selected_state_key}_max"
    reset_selected, reset_all = st.columns(2)
    with reset_selected:
        reset_selected_clicked = st.button(
            "Reset selected frequency",
            key=f"lightcurve_reset_selected_{identity}",
            help="Restore Robust auto for only the configured frequency.",
        )
    with reset_all:
        reset_all_clicked = st.button(
            "Reset all to Robust auto",
            key=f"lightcurve_reset_all_{identity}",
            help="Discard every manual/full range in this analysis session.",
        )
    if reset_all_clicked:
        configs.clear()
        for key in list(st.session_state):
            if str(key).startswith(editor_prefix):
                del st.session_state[key]
    elif reset_selected_clicked:
        configs.pop(selected_state_key, None)
        for key in (mode_key, min_key, max_key):
            st.session_state.pop(key, None)

    selected_data = _roi_app_facade()._frequency_rows(data, selected_frequency)
    selected_values = selected_data[metric].to_numpy(dtype=float)
    selected_stored = configs.get(selected_state_key, {})
    selected_mode = str(selected_stored.get("mode", "Robust auto"))
    if selected_mode not in _LIGHTCURVE_Y_AXIS_MODES:
        selected_mode = "Robust auto"
    st.session_state.setdefault(mode_key, selected_mode)
    mode = st.selectbox(
        "Y-axis range",
        list(_LIGHTCURVE_Y_AXIS_MODES),
        key=mode_key,
        help=(
            "Robust auto uses the selected frequency's finite distribution. "
            "Full data includes all of its finite samples. Manual accepts "
            "scientific notation."
        ),
    )
    manual_limits = None
    previous_limits = _roi_app_facade()._coerce_lightcurve_y_limits(
        selected_stored.get("last_valid_limits")
    )
    seed_limits = (
        _roi_app_facade()._robust_lightcurve_y_limits(selected_values)
        or _roi_app_facade()._full_lightcurve_y_limits(selected_values)
        or (-1.0, 1.0)
    )
    if mode == "Manual":
        stored_manual_min = selected_stored.get("manual_min")
        stored_manual_max = selected_stored.get("manual_max")
        st.session_state.setdefault(
            min_key,
            float(seed_limits[0] if stored_manual_min is None else stored_manual_min),
        )
        st.session_state.setdefault(
            max_key,
            float(seed_limits[1] if stored_manual_max is None else stored_manual_max),
        )
        c1, c2 = st.columns(2)
        with c1:
            manual_min = st.number_input(
                "Y minimum",
                key=min_key,
                format="%.6e",
                help="Lower displayed Y limit. Scientific notation is accepted.",
            )
        with c2:
            manual_max = st.number_input(
                "Y maximum",
                key=max_key,
                format="%.6e",
                help="Upper displayed Y limit. It must be greater than Y minimum.",
            )
        manual_limits = (float(manual_min), float(manual_max))
    selected_config = _roi_app_facade()._resolve_lightcurve_y_limits(
        selected_values,
        mode,
        manual_limits=manual_limits,
        previous_limits=previous_limits,
    )
    configs[selected_state_key] = {
        "mode": str(mode),
        "limits": selected_config["limits"],
        "display_limits": selected_config["display_limits"],
        "valid": bool(selected_config["valid"]),
        "last_valid_limits": (
            selected_config["limits"]
            if mode == "Manual" and selected_config["valid"]
            else previous_limits
        ),
        "manual_min": (
            manual_limits[0]
            if manual_limits is not None
            else selected_stored.get("manual_min")
        ),
        "manual_max": (
            manual_limits[1]
            if manual_limits is not None
            else selected_stored.get("manual_max")
        ),
    }
    if mode == "Manual" and not selected_config["valid"]:
        st.error(
            "Y minimum and Y maximum must be finite, and Y minimum must be "
            "less than Y maximum. This frequency still uses its last valid range."
        )

    frequency_entries: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    selected_diagnostics: dict[str, Any] | None = None
    any_extreme_span = False
    for frequency in frequencies:
        frequency_key = _frequency_state_key(frequency)
        frequency_data = _roi_app_facade()._frequency_rows(data, frequency)
        raw_frequency_data = _roi_app_facade()._frequency_rows(raw_data, frequency)
        nonpositive_omitted = (
            int(
                np.count_nonzero(
                    raw_frequency_data[metric].to_numpy(dtype=float) <= 0.0
                )
            )
            if y_transform == "log10"
            else 0
        )
        values = frequency_data[metric].to_numpy(dtype=float)
        saved = configs.get(frequency_key, {})
        saved_mode = str(saved.get("mode", "Robust auto"))
        if saved_mode not in _LIGHTCURVE_Y_AXIS_MODES:
            saved_mode = "Robust auto"
        resolved = _roi_app_facade()._resolve_lightcurve_y_limits(
            values,
            saved_mode,
            manual_limits=(
                (saved.get("manual_min"), saved.get("manual_max"))
                if saved_mode == "Manual"
                else None
            ),
            previous_limits=_roi_app_facade()._coerce_lightcurve_y_limits(
                saved.get("last_valid_limits")
            ),
        )
        diagnostics = _roi_app_facade()._lightcurve_diagnostics(
            frequency_data,
            metric,
            resolved["display_limits"],
        )
        any_extreme_span = any_extreme_span or diagnostics["span_ratio"] >= 100.0
        entry = {
            "freq_mhz": float(frequency),
            "mode": saved_mode,
            "limits": resolved["limits"],
            "display_limits": resolved["display_limits"],
            "valid": bool(resolved["valid"]),
            "outside_count": int(diagnostics["outside_count"]),
            "nonpositive_omitted": nonpositive_omitted,
        }
        frequency_entries.append(entry)
        full_limits = diagnostics["full_limits"]
        display_limits = diagnostics["display_limits"]
        diagnostic_rows.append(
            {
                "Frequency (MHz)": float(frequency),
                "Mode": saved_mode,
                "Valid": int(diagnostics["valid_count"]),
                "Plot values < 0": int(diagnostics["negative_count"]),
                "Nonpositive omitted": nonpositive_omitted,
                "Outside": int(diagnostics["outside_count"]),
                "Full min": full_limits[0] if full_limits is not None else np.nan,
                "Full max": full_limits[1] if full_limits is not None else np.nan,
                "Display min": (
                    display_limits[0] if display_limits is not None else np.nan
                ),
                "Display max": (
                    display_limits[1] if display_limits is not None else np.nan
                ),
            }
        )
        if frequency == selected_frequency:
            selected_diagnostics = diagnostics

    valid = all(entry["valid"] for entry in frequency_entries)
    outside_count = sum(entry["outside_count"] for entry in frequency_entries)
    if any_extreme_span:
        st.warning(
            "At least one frequency has finite extreme samples that expand its "
            "full span by more than 100x. The samples remain unchanged in the "
            "analysis and Statistics CSV."
        )
    st.caption(
        f"Across {len(frequencies):,} frequencies, {outside_count:,} valid samples "
        "are outside their displayed ranges. Display settings never change the "
        "DataFrame, quality flags, or Statistics CSV."
    )
    st.dataframe(pd.DataFrame(diagnostic_rows), width="stretch", hide_index=True)
    with st.expander("Selected frequency diagnostics", expanded=False):
        if y_transform == "log10":
            st.write(
                "Non-positive raw values are omitted from log10 plots but retained "
                "in the analysis table and Statistics CSV. At most 20 transformed "
                "samples farthest beyond the displayed range are listed."
            )
        else:
            st.write(
                "Negative and out-of-range values are retained. At most 20 samples "
                "farthest beyond the selected frequency's displayed range are listed."
            )
        if selected_diagnostics is None or selected_diagnostics["outside_rows"].empty:
            st.caption("No valid samples are outside the displayed Y range.")
        else:
            st.dataframe(
                selected_diagnostics["outside_rows"],
                width="stretch",
            )

    stored = {
        "mode": "Per frequency",
        "limits": None,
        "display_limits": None,
        "valid": valid,
        "plot_ready": True,
        "outside_count": int(outside_count),
        "plot_style": plot_style,
        "y_transform": y_transform,
        "metric_unit": _roi_app_facade()._lightcurve_metric_unit(df, metric),
        "excluded_nonpositive_count": excluded_nonpositive,
        "marker_size": float(marker_size),
        "detail_frequency_mhz": float(detail_frequency),
        "normalization": "per-frequency displayed Y limits mapped to 0-1",
        "frequencies": frequency_entries,
    }
    state["configs"] = configs
    st.session_state[state_key] = state
    st.session_state["lightcurve_y_axis_config"] = stored
    return stored


def _streamlit_fragment(function: Any) -> Any:
    """Decorate at module load while keeping Streamlit an optional dependency."""

    try:
        import streamlit as st
    except ModuleNotFoundError:
        return function
    return st.fragment(function)


@_streamlit_fragment
def _render_analysis_and_export_steps(
    selected_paths: list[str],
    references: list[RadioImage],
    roi: RadioRoi,
    settings: dict[str, Any],
    display_config: dict[str, Any],
    path_policy: PathAccessPolicy,
) -> None:
    """Render Steps 6-7 as one fragment isolated from the reference grid."""

    import streamlit as st

    df = _roi_app_facade()._render_analysis_step(
        st,
        selected_paths,
        references,
        roi,
        settings,
        display_config,
    )
    if df is None:
        return
    _roi_app_facade()._render_export_step(
        st,
        df,
        selected_paths,
        references,
        roi,
        settings,
        display_config,
        path_policy,
    )


def _render_export_step(
    st: Any,
    df: pd.DataFrame,
    selected_paths: list[str],
    references: list[RadioImage],
    roi: RadioRoi,
    settings: dict[str, Any],
    display_config: dict[str, Any],
    path_policy: PathAccessPolicy,
) -> None:
    st.subheader("Step 7. Export and Download")
    available_products = tuple(PRODUCT_FILENAMES)
    columns = st.columns(min(3, len(available_products)))
    product_keys = []
    for index, key in enumerate(available_products):
        column = columns[index % len(columns)]
        with column:
            if st.checkbox(
                PRODUCT_LABELS[key],
                value=key
                not in {
                    "lightcurve_detail_png",
                    "lightcurve_normalized_png",
                },
                key=f"export_{key}",
                help=f"Include {PRODUCT_FILENAMES[key]} in browser downloads and local save.",
            ):
                product_keys.append(key)
    if not product_keys:
        st.warning("Select at least one export product.")
        return
    analysis_result_signature = st.session_state.get(
        "analysis_result_signature"
    ) or _roi_app_facade()._dataframe_content_signature(df)
    reference_identities = _roi_app_facade()._reference_file_identities(st, references)
    y_axis_config = dict(st.session_state.get("lightcurve_y_axis_config") or {})
    y_axis_valid = bool(y_axis_config.get("valid", False))
    y_axis_mode = str(y_axis_config.get("mode", "Robust auto"))
    lightcurve_y_limits = _roi_app_facade()._coerce_lightcurve_y_limits(
        y_axis_config.get("limits")
    )
    frequency_y_limits = _roi_app_facade()._frequency_limit_mapping(y_axis_config)
    marker_size = float(
        y_axis_config.get("marker_size", _LIGHTCURVE_DEFAULT_MARKER_SIZE)
    )
    detail_frequency = y_axis_config.get("detail_frequency_mhz")
    plot_style = _roi_app_facade()._normalize_lightcurve_plot_style(
        y_axis_config.get("plot_style", "scatter")
    )
    y_transform = _roi_app_facade()._normalize_lightcurve_y_transform(
        y_axis_config.get("y_transform", "linear")
    )
    lightcurve_products_selected = bool(
        set(product_keys).intersection(_LIGHTCURVE_PRODUCT_KEYS)
    )
    export_config_valid = y_axis_valid or not lightcurve_products_selected
    signature = _roi_app_facade()._export_signature(
        analysis_result_signature=str(analysis_result_signature),
        product_keys=tuple(product_keys),
        metric=str(settings["metric"]),
        reference_identities=reference_identities,
        display_config=display_config,
        y_axis_mode=y_axis_mode,
        lightcurve_y_limits=lightcurve_y_limits,
        lightcurve_frequency_y_limits=frequency_y_limits,
        lightcurve_frequency_config=_roi_app_facade()._canonical_frequency_configs(
            y_axis_config
        ),
        lightcurve_marker_size=marker_size,
        lightcurve_detail_frequency_mhz=detail_frequency,
        lightcurve_plot_style=plot_style,
        lightcurve_y_transform=y_transform,
    )
    if not export_config_valid:
        if not bool(y_axis_config.get("plot_ready", True)):
            st.error(
                "The selected Y transform has no plottable samples. Select Linear "
                "or deselect all light-curve PNG products before preparing downloads."
            )
        else:
            st.error(
                "At least one manual frequency range is invalid. Enter a finite Y "
                "minimum below its Y maximum before preparing downloads. The preview "
                "continues to use that frequency's last valid range."
            )
    if st.button(
        "Prepare Downloads",
        type="primary",
        key="radio_roi_prepare_downloads_v2",
        disabled=not export_config_valid,
        help="Generate the selected export products once and cache their exact bytes for download or local save.",
    ):
        cached_artifacts = st.session_state.get("export_artifacts")
        if (
            isinstance(cached_artifacts, dict)
            and st.session_state.get("export_signature") == signature
        ):
            st.success("Reused the prepared export products.")
        else:
            with st.spinner("Preparing the selected export products..."):
                try:
                    artifacts = _roi_app_facade()._build_cached_export_artifacts(
                        st,
                        df,
                        roi,
                        selected_paths=selected_paths,
                        references=references,
                        settings=settings,
                        display_config=display_config,
                        product_keys=tuple(product_keys),
                        lightcurve_y_axis=y_axis_config,
                    )
                except Exception as exc:  # noqa: BLE001 - visible app error.
                    st.error(str(exc))
                    return
            st.session_state["export_artifacts"] = artifacts
            st.session_state["export_signature"] = signature
    if not export_config_valid:
        return
    artifacts = st.session_state.get("export_artifacts")
    artifact_filenames = st.session_state.get("export_artifact_filenames")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(artifact_filenames, dict)
        or st.session_state.get("export_signature") != signature
    ):
        st.info("Click Prepare Downloads to generate the currently selected products.")
        return
    if len(artifacts) > 1:
        st.download_button(
            "Download ZIP",
            data=_roi_app_facade()._zip_artifacts(artifacts, artifact_filenames),
            file_name="radio_roi_lightcurve_exports.zip",
            mime="application/zip",
            on_click="ignore",
            help="Download all selected products as one ZIP archive.",
        )
    for key, payload in artifacts.items():
        st.download_button(
            f"Download {PRODUCT_LABELS[key]}",
            data=payload,
            file_name=artifact_filenames[key],
            mime=PRODUCT_MIME_TYPES[key],
            on_click="ignore",
            help=f"Download only {artifact_filenames[key]}.",
        )
    if st.button(
        "Save Selected Products",
        help="Write the exact prepared artifact bytes to a new run folder under the output folder.",
    ):
        try:
            output_dir = path_policy.output_directory(settings["output_dir"])
            products = _roi_app_facade()._write_prepared_artifacts(
                artifacts,
                output_dir,
                filenames=artifact_filenames,
            )
        except Exception as exc:  # noqa: BLE001 - visible app error.
            st.error(str(exc))
        else:
            st.success(f"Saved products to {products['output_dir']}")


def _manual_range_editor(
    st: Any,
    reference_images: list[RadioImage],
    settings: dict[str, Any],
    *,
    transform: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    rows = []
    for item in reference_images:
        key = _roi_app_facade()._display_frequency_key(item.freq_mhz)
        rows.append(
            {
                "freq_mhz": float(item.freq_mhz),
                "vmin": float(settings["display_manual_min"]),
                "vmax": float(settings["display_manual_max"]),
                "key": key,
            }
        )
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        disabled=["freq_mhz", "key"],
        column_config={
            "freq_mhz": st.column_config.NumberColumn(
                "MHz", help="Frequency for this display limit row."
            ),
            "vmin": st.column_config.NumberColumn(
                "Manual min",
                help="Raw FITS-unit lower display limit for this frequency.",
            ),
            "vmax": st.column_config.NumberColumn(
                "Manual max",
                help="Raw FITS-unit upper display limit for this frequency.",
            ),
            "key": st.column_config.TextColumn(
                "Key", help="Internal frequency key written to exported metadata."
            ),
        },
        width="stretch",
        key="radio_roi_manual_display_ranges",
    )
    raw_limits: dict[str, list[float]] = {}
    display_limits: dict[str, list[float]] = {}
    transform_config = {"transform": transform}
    for _, row in edited.iterrows():
        key = str(row["key"])
        raw_min = float(row["vmin"])
        raw_max = float(row["vmax"])
        raw_limits[key] = [raw_min, raw_max]
        display_limits[key] = [
            _roi_app_facade()._transform_display_limit(raw_min, transform_config),
            _roi_app_facade()._transform_display_limit(raw_max, transform_config),
        ]
    return display_limits, raw_limits


__all__ = [
    "main",
    "_run_streamlit_app",
    "_render_load_step",
    "_render_file_selection_step",
    "_render_reference_step",
    "_render_display_settings_step",
    "_render_roi_import_controls",
    "_render_roi_step",
    "_render_analysis_step",
    "_default_detail_frequency",
    "_render_lightcurve_y_axis_controls",
    "_streamlit_fragment",
    "_render_analysis_and_export_steps",
    "_render_export_step",
    "_manual_range_editor",
]
