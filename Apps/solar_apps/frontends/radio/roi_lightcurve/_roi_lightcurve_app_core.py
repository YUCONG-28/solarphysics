"""Compatibility exports for the split ROI Streamlit implementation.

Each cache and model retains one owner; the public facade keeps its existing
exports and monkeypatch anchors.
"""

# ruff: noqa: F401

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import AllowedRootPolicyError
from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
    RadioImage,
    infer_polarization,
    iter_radio_images,
    parse_datetime_value,
    parse_frequency_mhz,
    parse_time_from_filename,
    select_radio_files,
)
from solar_toolkit.radio.roi_lightcurve import (
    DEFAULT_PAIR_TOLERANCE_SEC,
    RadioRoi,
    extract_radio_roi_lightcurve,
    radio_roi_from_json,
)
from solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_application import (
    PRODUCT_FILENAMES,
    build_radio_roi_artifacts,
    build_radio_roi_product_filenames,
)
from solar_apps.ui.state import (
    bind_streamlit_fields,
    frontend_path_memory,
    frontend_state_store,
    save_streamlit_fields,
)
from solar_apps.ui.streamlit_paths import (
    PathAccessPolicy,
    render_native_path_input,  # noqa: F401 - re-exported via the facade
    resolve_streamlit_allowed_roots,
)
from solar_apps.ui.theme import apply_plotly_chrome, render_streamlit_theme
from solar_apps.workflows.radio.spatial_display import spatial_display_for_reference
from ._roi_app_helpers import (
    _expanded_lightcurve_limits,
    _frequency_state_key,
    _option_index,
)


from ._roi_app_state import (
    DEFAULT_APP_SETTINGS,
    APP_CLI_SETTING_MAP,
    PRODUCT_LABELS,
    PRODUCT_MIME_TYPES,
    DISPLAY_COLORMAPS,
    DISPLAY_TRANSFORMS,
    DISPLAY_RANGE_MODES,
    DISPLAY_RANGE_SCOPES,
    ROI_UI_FIELD_KEYS,
    SELECTION_ACTIONS,
    ROI_SELECTION_KEYS,
    ROI_IMPORT_KEYS,
    ROI_KEYS,
    ANALYSIS_KEYS,
    EXPORT_KEYS,
    REFERENCE_KEYS,
    _REFERENCE_DECODER_VERSION,
    _REFERENCE_PLANE_CACHE_SIZE,
    _ANALYSIS_REQUEST_VERSION,
    _LIGHTCURVE_CACHE_SIZE,
    _LIGHTCURVE_Y_AXIS_MODES,
    _LIGHTCURVE_ROBUST_MIN_SAMPLES,
    _LIGHTCURVE_PRODUCT_KEYS,
    _LIGHTCURVE_DEFAULT_MARKER_SIZE,
    _LIGHTCURVE_PLOT_STYLE_LABELS,
    _LIGHTCURVE_Y_TRANSFORM_LABELS,
    _NAT_INT64,
    _RoiImportChoice,
    _RoiImportDocument,
    _ReferencePlan,
    _ReferencePreview,
    _DisplayReferencePreview,
    build_parser,
    default_settings_path,
    load_app_settings,
    save_app_settings,
    resolve_app_settings,
    _init_session_state,
    _clear_keys,
)

from ._roi_app_files import (
    build_file_manifest,
    discover_frequency_options,
    _load_manifest_into_state,
    _apply_loaded_time_filter,
    _set_selected_paths,
    _discover_frequencies_into_state,
    _source_signature,
    _frequency_options_list,
    _default_selected_frequencies,
    _normalize_frequency_selection,
    _frequency_matches_any,
    _parse_frequency_hint_mhz,
    _manifest_time_range_hint,
    _filter_manifest_by_time,
    _parse_optional_time,
    parse_row_selection_expression,
    _apply_number_selection,
    _order_paths_by_manifest,
    _filter_manifest,
    _page_paths,
    _load_first_radio_image,
    _cached_first_radio_image,
    _cached_reference_image,
    _clone_reference_image,
    _manifest_frequency_values,
    _load_reference_grid,
    _plan_reference_grid,
    _select_paired_rows,
    _materialize_reference_grid,
    _reference_grid_signature,
    _reference_reuse_signature,
    _nearest_manifest_row,
    _manifest_time_ns,
    _row_time,
    _relative_path_text,
)

from ._roi_app_regions import (
    selection_to_radio_roi,
    _store_roi_import_document,
    _stage_imported_roi,
    _session_roi,
    _roi_from_uploaded_or_path,
    _roi_import_document_from_uploaded_or_path,
    _parse_roi_import_document,
    _selection_xy,
    _float_list,
    _box_roi_from_xy,
    _event_get,
)

from ._roi_app_analysis import (
    _dataset_signature,
    _analysis_signature,
    _analysis_context_signature,
    _selected_input_size_from_manifest,
    _selected_file_identities,
    _file_identity,
    _dataframe_content_signature,
    _reference_file_identities,
    _export_signature,
    _stable_sha256,
    _canonical_frequency_limit_items,
    _cached_lightcurve_png,
    _build_cached_export_artifacts,
)

from ._roi_app_display import (
    build_reference_figure,
    _build_reference_figure_from_preview,
    _normalize_lightcurve_plot_style,
    _normalize_lightcurve_y_transform,
    _lightcurve_metric_unit,
    _lightcurve_axis_label,
    _lightcurve_metric_frame,
    _full_lightcurve_y_limits,
    _robust_lightcurve_y_limits,
    _coerce_lightcurve_y_limits,
    _resolve_lightcurve_y_limits,
    _lightcurve_diagnostics,
    _lightcurve_frequencies,
    _frequency_rows,
    _canonical_frequency_configs,
    _frequency_limit_mapping,
    _attach_auto_display_limits,
    _display_array,
    _display_limits_for_item,
    _transform_display_limit,
    _clean_display_limits,
    _display_frequency_key,
    _display_colorbar_title,
    _apply_plotly_fov,
    _write_prepared_artifacts,
    _allocate_unique_run_directory,
    _run_metadata,
    _settings_with_reference,
    _zip_artifacts,
    _downsample_for_preview,
    _preview_coordinate_grid,
    _prepare_reference_preview,
    _reference_previews_from_state,
    _add_roi_shape,
    _reference_title,
)

from ._roi_app_page import (
    main,
    _run_streamlit_app,
    _render_load_step,
    _render_file_selection_step,
    _render_reference_step,
    _render_display_settings_step,
    _render_roi_import_controls,
    _render_roi_step,
    _render_analysis_step,
    _default_detail_frequency,
    _render_lightcurve_y_axis_controls,
    _streamlit_fragment,
    _render_analysis_and_export_steps,
    _render_export_step,
    _manual_range_editor,
)

__all__ = [
    "DEFAULT_APP_SETTINGS",
    "build_file_manifest",
    "build_parser",
    "build_reference_figure",
    "default_settings_path",
    "discover_frequency_options",
    "load_app_settings",
    "parse_row_selection_expression",
    "main",
    "resolve_app_settings",
    "save_app_settings",
    "selection_to_radio_roi",
]
