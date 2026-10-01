"""Shared ROI settings and session models; no page rendering.

Cross-module calls resolve through the compatibility facade at call time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from solar_toolkit.radio.centers import (
    POL_LCP,
    POL_RCP,
    POL_SUM,
)
from solar_toolkit.radio.roi_lightcurve import (
    DEFAULT_PAIR_TOLERANCE_SEC,
    RadioRoi,
)


DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "radio_dir": "",
    "pattern": "*.fits",
    "recursive": True,
    "time_start": "",
    "time_end": "",
    "selected_freqs_mhz": [],
    "output_dir": "outputs/radio_roi_lightcurve",
    "pair_time_tolerance_sec": DEFAULT_PAIR_TOLERANCE_SEC,
    "polarization": POL_SUM,
    "metric": "raw_sum",
    "display_colormap": "Hot",
    "display_transform": "Linear",
    "display_range_mode": "Auto percentile",
    "display_range_scope": "Per frequency",
    "display_low_percentile": 99.7,
    "display_high_percentile": 99.99,
    "display_manual_min": 0.0,
    "display_manual_max": 1.0,
    "display_bad_color": "#000080",
    "display_use_custom_fov": True,
    "display_x_min_arcsec": -3000.0,
    "display_x_max_arcsec": 3000.0,
    "display_y_min_arcsec": -3000.0,
    "display_y_max_arcsec": 3000.0,
    "preview_max_side": 256,
    "page_size": 200,
}

APP_CLI_SETTING_MAP = {
    "radio_dir": "radio_dir",
    "pattern": "pattern",
    "recursive": "recursive",
    "time_start": "time_start",
    "time_end": "time_end",
    "output_dir": "output_dir",
    "pair_time_tolerance_sec": "pair_time_tolerance_sec",
    "polarization": "polarization",
    "metric": "metric",
}

PRODUCT_LABELS = {
    "csv": "Statistics CSV",
    "coordinates_csv": "ROI Coordinates CSV (HPLN/HPLT arcsec)",
    "json": "ROI JSON",
    "reference_png": "Reference PNG",
    "lightcurve_png": "Frequency Overview PNG",
    "lightcurve_detail_png": "Frequency Detail PNG",
    "lightcurve_normalized_png": "Normalized Comparison PNG",
}

PRODUCT_MIME_TYPES = {
    "csv": "text/csv",
    "coordinates_csv": "text/csv",
    "json": "application/json",
    "reference_png": "image/png",
    "lightcurve_png": "image/png",
    "lightcurve_detail_png": "image/png",
    "lightcurve_normalized_png": "image/png",
}

DISPLAY_COLORMAPS = [
    "Hot",
    "Viridis",
    "Cividis",
    "Plasma",
    "Inferno",
    "Magma",
    "Turbo",
    "Greys",
    "Jet",
]

DISPLAY_TRANSFORMS = ["Linear", "Log10 positive"]

DISPLAY_RANGE_MODES = ["Auto percentile", "Manual min/max"]

DISPLAY_RANGE_SCOPES = ["Per frequency", "Shared/global"]

ROI_UI_FIELD_KEYS = (
    "radio_dir",
    "pattern",
    "recursive",
    "selected_freqs_mhz",
    "time_start",
    "time_end",
    "pair_time_tolerance_sec",
    "polarization",
    "metric",
    "output_dir",
    "display_colormap",
    "display_transform",
    "display_range_mode",
    "display_range_scope",
    "display_low_percentile",
    "display_high_percentile",
    "display_manual_min",
    "display_manual_max",
    "display_use_custom_fov",
    "display_x_min_arcsec",
    "display_x_max_arcsec",
    "display_y_min_arcsec",
    "display_y_max_arcsec",
)

SELECTION_ACTIONS = ["Replace", "Add", "Remove"]

ROI_SELECTION_KEYS = ("candidate_roi", "confirmed_roi")

ROI_IMPORT_KEYS = (
    "roi_import_document",
    "roi_import_source_kind",
    "roi_import_source_label",
    "roi_import_upload_signature",
    "roi_import_selected_key",
)

ROI_KEYS = (*ROI_SELECTION_KEYS, *ROI_IMPORT_KEYS)

ANALYSIS_KEYS = (
    "analysis_df",
    "analysis_context_signature",
    "analysis_signature",
    "analysis_result_signature",
    "analysis_input_summary",
    "lightcurve_png_cache",
)

EXPORT_KEYS = (
    "export_artifact_filenames",
    "export_artifacts",
    "export_signature",
)

REFERENCE_KEYS = (
    "reference_path",
    "reference_images",
    "reference_image",
    "reference_metadata",
    "reference_grid_signature",
    "reference_reuse_signature",
    "reference_preview_cache_key",
    "reference_preview_cache",
)

_REFERENCE_DECODER_VERSION = "first-2d-v1"

_REFERENCE_PLANE_CACHE_SIZE = 64

_ANALYSIS_REQUEST_VERSION = "roi-extraction-request-v2"

_LIGHTCURVE_CACHE_SIZE = 8

_LIGHTCURVE_Y_AXIS_MODES = ("Robust auto", "Full data", "Manual")

_LIGHTCURVE_ROBUST_MIN_SAMPLES = 100

_LIGHTCURVE_PRODUCT_KEYS = (
    "lightcurve_png",
    "lightcurve_detail_png",
    "lightcurve_normalized_png",
)

_LIGHTCURVE_DEFAULT_MARKER_SIZE = 3.0

_LIGHTCURVE_PLOT_STYLE_LABELS = ("Scatter", "Line")

_LIGHTCURVE_Y_TRANSFORM_LABELS = ("Log10 positive", "Linear")

_NAT_INT64 = np.datetime64("NaT", "ns").astype("int64")


@dataclass(frozen=True)
class _RoiImportChoice:
    key: str
    source_id: str
    name: str
    source_type: str
    visible: bool
    color: str
    roi: RadioRoi

    @property
    def display_label(self) -> str:
        visibility = "visible" if self.visible else "hidden"
        return f"{self.name} — {self.source_type} — {visibility} — {self.color}"


@dataclass(frozen=True)
class _RoiImportDocument:
    source_format: str
    choices: tuple[_RoiImportChoice, ...]
    source_image_sha256: str = ""
    provenance: Mapping[str, Any] | None = None

    @property
    def default_choice_key(self) -> str:
        for choice in self.choices:
            if choice.visible:
                return choice.key
        return self.choices[0].key


@dataclass(frozen=True)
class _ReferencePlan:
    freq_mhz: float
    row: int
    path: str
    paired_row: int | None
    paired_path: str | None
    obs_time: str
    delta_from_anchor_sec: float
    polarization: str


@dataclass(frozen=True)
class _ReferencePreview:
    raw_view: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray


@dataclass(frozen=True)
class _DisplayReferencePreview:
    display_view: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """Build a help parser for the Streamlit app."""

    parser = argparse.ArgumentParser(
        description=(
            "Launch the Streamlit radio ROI light-curve app. "
            "Run with: streamlit run solar_toolkit/radio/roi_lightcurve_app.py"
        )
    )
    parser.add_argument("--radio-dir", default=None, help="Default radio FITS folder.")
    parser.add_argument("--pattern", default=None, help="Default FITS glob pattern.")
    recursive = parser.add_mutually_exclusive_group()
    recursive.add_argument("--recursive", dest="recursive", action="store_true")
    recursive.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.set_defaults(recursive=None)
    parser.add_argument(
        "--time-start", default=None, help="Default inclusive time start."
    )
    parser.add_argument("--time-end", default=None, help="Default inclusive time end.")
    parser.add_argument("--output-dir", default=None, help="Default output folder.")
    parser.add_argument(
        "--allowed-roots",
        default=None,
        help="Semicolon-separated local filesystem roots available to this app.",
    )
    parser.add_argument(
        "--pair-time-tolerance-sec",
        type=float,
        default=None,
        help="Default LCP/RCP pairing tolerance in seconds.",
    )
    parser.add_argument(
        "--polarization",
        choices=[POL_SUM, POL_LCP, POL_RCP, "all"],
        default=None,
        help="Default polarization mode.",
    )
    parser.add_argument(
        "--metric",
        choices=["raw_sum", "raw_mean", "raw_peak"],
        default=None,
        help="Default plotted metric.",
    )
    parser.add_argument(
        "--settings-file", default=None, help="Local JSON settings file."
    )
    parser.add_argument("--reset-settings", action="store_true")
    return parser


def default_settings_path() -> Path:
    """Return the per-user default app settings path."""

    return Path.home() / ".solar_toolkit" / "radio_roi_lightcurve_app_settings.json"


def load_app_settings(path: str | Path, *, reset: bool = False) -> dict[str, Any]:
    """Load app settings, falling back to defaults."""

    defaults = dict(DEFAULT_APP_SETTINGS)
    if reset:
        return defaults
    settings_path = Path(path).expanduser()
    if not settings_path.exists():
        return defaults
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return defaults
    settings = dict(defaults)
    for key in defaults:
        if key in data:
            settings[key] = data[key]
    return settings


def save_app_settings(path: str | Path, settings: dict[str, Any]) -> Path:
    """Persist app settings to JSON."""

    settings_path = Path(path).expanduser()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: settings.get(key, DEFAULT_APP_SETTINGS[key])
        for key in DEFAULT_APP_SETTINGS
    }
    settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return settings_path


def resolve_app_settings(
    args: argparse.Namespace, stored: dict[str, Any]
) -> dict[str, Any]:
    """Apply CLI overrides to stored settings."""

    resolved = dict(DEFAULT_APP_SETTINGS)
    resolved.update(stored or {})
    for arg_name, setting_name in APP_CLI_SETTING_MAP.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            resolved[setting_name] = value
    return resolved


def _init_session_state(st: Any) -> None:
    st.session_state.setdefault("selected_paths", [])
    st.session_state.setdefault("selection_revision", 0)
    st.session_state.setdefault("roi_chart_generation", 0)


def _clear_keys(st: Any, keys: tuple[str, ...]) -> None:
    for key in keys:
        st.session_state.pop(key, None)


__all__ = [
    "DEFAULT_APP_SETTINGS",
    "APP_CLI_SETTING_MAP",
    "PRODUCT_LABELS",
    "PRODUCT_MIME_TYPES",
    "DISPLAY_COLORMAPS",
    "DISPLAY_TRANSFORMS",
    "DISPLAY_RANGE_MODES",
    "DISPLAY_RANGE_SCOPES",
    "ROI_UI_FIELD_KEYS",
    "SELECTION_ACTIONS",
    "ROI_SELECTION_KEYS",
    "ROI_IMPORT_KEYS",
    "ROI_KEYS",
    "ANALYSIS_KEYS",
    "EXPORT_KEYS",
    "REFERENCE_KEYS",
    "_REFERENCE_DECODER_VERSION",
    "_REFERENCE_PLANE_CACHE_SIZE",
    "_ANALYSIS_REQUEST_VERSION",
    "_LIGHTCURVE_CACHE_SIZE",
    "_LIGHTCURVE_Y_AXIS_MODES",
    "_LIGHTCURVE_ROBUST_MIN_SAMPLES",
    "_LIGHTCURVE_PRODUCT_KEYS",
    "_LIGHTCURVE_DEFAULT_MARKER_SIZE",
    "_LIGHTCURVE_PLOT_STYLE_LABELS",
    "_LIGHTCURVE_Y_TRANSFORM_LABELS",
    "_NAT_INT64",
    "_RoiImportChoice",
    "_RoiImportDocument",
    "_ReferencePlan",
    "_ReferencePreview",
    "_DisplayReferencePreview",
    "build_parser",
    "default_settings_path",
    "load_app_settings",
    "save_app_settings",
    "resolve_app_settings",
    "_init_session_state",
    "_clear_keys",
]
