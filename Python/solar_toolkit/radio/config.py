"""Validated configuration loading helpers for radio workflows.

The canonical loader lives in the installable package. Event configuration is
always supplied explicitly as a mapping, module, object, or qualified module
name; the public library never discovers a workstation event automatically.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, TypeAlias

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_DRIFT_SELECTION_PRODUCT_CONFIG",
    "DEFAULT_NEWKIRK_CONFIG",
    "DEFAULT_NEWKIRK_HEIGHT_COMPARISON_CONFIG",
    "DEFAULT_OUTPUT_CONFIG",
    "DEFAULT_RADIO_DIAGNOSTIC_PRESENTATION_CONFIG",
    "ConfigSource",
    "RadioEventConfig",
    "load_aia_radio_hmi_user_config",
    "load_aia_radio_overlay_user_config",
    "load_drift_selection_product_config",
    "load_newkirk_height_comparison_config",
    "load_radio_config_module",
    "load_radio_diagnostic_presentation_config",
    "load_radio_event_config",
    "load_radio_output_config",
    "load_radio_user_config",
]

_SECTION_LEGACY_NAMES = {
    "user": "USER_CONFIG",
    "output": "OUTPUT_CONFIG",
    "aia_radio_hmi": "AIA_RADIO_HMI_CONFIG",
    "aia_raw_radio_spectrogram": "AIA_RAW_RADIO_SPECTROGRAM_CONFIG",
    "aia_multi_wave_raw_radio_spectrogram": (
        "AIA_MULTI_WAVE_RAW_RADIO_SPECTROGRAM_CONFIG"
    ),
    "aia_multi_wave_gaussian_spectrogram": (
        "AIA_MULTI_WAVE_GAUSSIAN_SPECTROGRAM_CONFIG"
    ),
    "newkirk": "NEWKIRK_CONFIG",
    "newkirk_height_comparison": "NEWKIRK_HEIGHT_COMPARISON_CONFIG",
    "drift_selection_products": "DRIFT_SELECTION_PRODUCT_CONFIG",
    "diagnostic_presentation": "RADIO_DIAGNOSTIC_PRESENTATION_CONFIG",
}


@dataclass(frozen=True)
class RadioEventConfig:
    """Validated, explicit collection of radio event configuration sections."""

    sections: dict[str, dict[str, Any]]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RadioEventConfig:
        unknown = set(value) - set(_SECTION_LEGACY_NAMES)
        if unknown:
            raise KeyError(f"Unknown radio event config sections: {sorted(unknown)}")
        sections: dict[str, dict[str, Any]] = {}
        for name, section in value.items():
            if section is None:
                sections[name] = {}
            elif isinstance(section, Mapping):
                sections[name] = copy.deepcopy(dict(section))
            else:
                raise TypeError(
                    f"Radio event config section {name!r} must be a mapping"
                )
        return cls(sections)

    def section(self, name: str) -> dict[str, Any]:
        """Return a defensive copy of one validated section."""

        if name not in _SECTION_LEGACY_NAMES:
            raise KeyError(f"Unknown radio event config section: {name}")
        return copy.deepcopy(self.sections.get(name, {}))


ConfigSource: TypeAlias = str | ModuleType | Mapping[str, Any] | RadioEventConfig

DEFAULT_CONFIG_NAME: None = None
DEFAULT_NEWKIRK_CONFIG = {
    "enabled": True,
    "multipliers": [1, 2, 4],
    "harmonics": [1, 2],
    "solar_radius_arcsec": 959.63,
}
DEFAULT_NEWKIRK_HEIGHT_COMPARISON_CONFIG = {
    "enable": True,
    "selected_models": [
        {"multiplier": 1.0, "harmonic": 1},
        {"multiplier": 1.0, "harmonic": 2},
        {"multiplier": 2.0, "harmonic": 1},
        {"multiplier": 2.0, "harmonic": 2},
        {"multiplier": 4.0, "harmonic": 1},
        {"multiplier": 4.0, "harmonic": 2},
    ],
    "solar_radius_arcsec": None,
    "plot_height_frequency": True,
    "plot_height_time": True,
    "plot_residual_frequency": True,
    "output_table_name": "gaussian_newkirk_height_comparison_table.csv",
    "raw_output_table_name": "gaussian_newkirk_height_rows.csv",
    "height_frequency_plot_name": "newkirk_height_frequency",
    "height_time_plot_name": "newkirk_height_time",
    "height_residual_plot_name": "newkirk_height_residual_frequency",
    "height_residual_summary_name": "gaussian_newkirk_height_residual_summary.csv",
    "reverse_frequency_axis": False,
    "color_by": "source_type",
    "drift_time_tolerance_s": 0.75,
    "drift_frequency_tolerance_mhz": "adaptive_half_band_spacing",
    "max_adaptive_frequency_tolerance_mhz": 15.0,
    "min_adaptive_frequency_tolerance_mhz": 5.0,
}
DEFAULT_OUTPUT_CONFIG = {
    "output_dir": None,
    "analysis_subdir": "auto",
    "gaussian_diagnostics_csv": "radio_gaussian_fit_diagnostics.csv",
    "valid_centers_csv": "radio_gaussian_valid_centers.csv",
    "newkirk_csv": "radio_gaussian_newkirk_extrapolated.csv",
    "drift_speed_csv": "radio_drift_newkirk_speed.csv",
    "drift_selection_subdir": "drift_selection",
    "enable_static_summary": None,
    "enable_html_dashboard": None,
}
DEFAULT_DRIFT_SELECTION_PRODUCT_CONFIG = {
    "enable": True,
    "save_raw_preview": True,
    "save_annotated_preview": True,
    "save_selection_csv": True,
    "save_metadata_json": True,
    "save_per_drift_cutouts": True,
    "cutout_time_padding_s": 2.0,
    "cutout_frequency_padding_mhz": 20.0,
    "annotate_drift_rate": True,
    "annotate_endpoints": True,
    "preserve_existing": True,
    "dpi": 200,
    "output_subdir": "drift_selection",
}
DEFAULT_RADIO_DIAGNOSTIC_PRESENTATION_CONFIG = {
    "enable": True,
    "enable_static_summary": True,
    "enable_html_dashboard": True,
    "comparison_frequency_mhz": None,
    "drift_source_type_map": {},
    "drift_time_tolerance_s": 0.75,
    "drift_frequency_tolerance_mhz": "adaptive_half_band_spacing",
    "max_adaptive_frequency_tolerance_mhz": 15.0,
    "min_adaptive_frequency_tolerance_mhz": 5.0,
    "selected_newkirk_multiplier": 2.0,
    "selected_newkirk_harmonic": 2,
    "reference_newkirk_assumption": "2xH2",
    "connect_same_drift_only": False,
    "reverse_frequency_axis": False,
    "enable_debug_center_facets": False,
    "enable_debug_height_time_facets": False,
    "enable_debug_drift_band_matching": False,
    "enable_debug_trajectory_by_frequency": False,
    "enable_event_height_comparison": True,
    "enable_event_speed_frequency": True,
    "best_model_metric": "median_abs_residual_rsun",
    "top_residual_models": 3,
    "summary_panel_name": "newkirk_frequency_priority_summary",
    "center_facets_name": "source_center_frequency_facets",
    "trajectory_by_frequency_name_template": "source_trajectory",
    "drift_band_matching_name": "drift_frequency_band_matching",
    "height_time_facets_name": "height_time_frequency_facets",
    "event_height_comparison_name": "newkirk_height_comparison",
    "event_speed_frequency_name": "newkirk_speed_frequency_scatter",
    "selected_band_newkirk_table_name": "event_selected_band_newkirk_table.csv",
    "physical_consistency_report_name": "newkirk_physical_consistency_report.md",
    "dashboard_name": "radio_newkirk_frequency_priority_dashboard.html",
    "summary_csv_name": "radio_newkirk_frequency_priority_summary.csv",
}


def _normalize_config_module_name(config_name: str | None) -> str:
    if config_name is None or not config_name.strip():
        raise ValueError("an explicit radio event configuration is required")
    name = config_name.strip()
    if name.endswith(".py"):
        name = name[:-3]
    if "." not in name:
        raise ValueError(
            "radio event modules must use an explicit fully qualified name"
        )
    return name


def load_radio_config_module(config_name: str) -> ModuleType:
    """Load an explicitly qualified Python event-config module."""

    return importlib.import_module(_normalize_config_module_name(config_name))


def load_radio_event_config(
    source: ConfigSource,
) -> RadioEventConfig:
    """Load and validate one explicit radio event configuration.

    ``source`` can be a validated object, a mapping of named sections, an
    imported module, or an explicitly qualified module name.  Environment variables
    are intentionally not consulted for scientific settings.
    """

    if isinstance(source, RadioEventConfig):
        return source
    if isinstance(source, Mapping):
        return RadioEventConfig.from_mapping(source)
    module = (
        source if isinstance(source, ModuleType) else load_radio_config_module(source)
    )
    event_config = getattr(module, "EVENT_CONFIG", None)
    if isinstance(event_config, Mapping):
        return RadioEventConfig.from_mapping(event_config)
    sections = {
        name: copy.deepcopy(getattr(module, legacy_name, {}) or {})
        for name, legacy_name in _SECTION_LEGACY_NAMES.items()
        if getattr(module, legacy_name, None) is not None
    }
    return RadioEventConfig.from_mapping(sections)


def load_radio_user_config(config_name: ConfigSource):
    """
    Load a radio event config.

    Returns ``(USER_CONFIG, NEWKIRK_CONFIG)``. The config name may be either a
    fully qualified module path supplied by a local application.
    """
    event = load_radio_event_config(config_name)
    user_config = event.section("user")
    output_config = _load_output_config(event)
    _apply_output_config_to_user_config(user_config, output_config)
    newkirk_config = dict(DEFAULT_NEWKIRK_CONFIG)
    newkirk_config.update(event.section("newkirk"))
    if output_config.get("newkirk_csv"):
        newkirk_config["output_csv"] = output_config["newkirk_csv"]
    if output_config.get("drift_speed_csv"):
        newkirk_config["drift_speed_csv"] = output_config["drift_speed_csv"]
    return user_config, newkirk_config


def _legacy_overlay_config_name(section: str) -> str:
    name = (section or "aia_radio_hmi").strip()
    if name == "aia_radio_hmi":
        return "AIA_RADIO_HMI_CONFIG"
    return f"{name.upper()}_CONFIG"


def load_aia_radio_overlay_user_config(
    config_name: ConfigSource, *, section: str = "aia_radio_hmi"
):
    """Load a named AIA/HMI/radio overlay config section."""
    section_name = (section or "aia_radio_hmi").strip()
    return load_radio_event_config(config_name).section(section_name)


def load_aia_radio_hmi_user_config(config_name: ConfigSource):
    """Load the default AIA/HMI/radio overlay config from a config module."""
    return load_aia_radio_overlay_user_config(config_name, section="aia_radio_hmi")


def load_newkirk_height_comparison_config(
    config_name: ConfigSource,
):
    """Load Gaussian-Newkirk height comparison config."""
    event = load_radio_event_config(config_name)
    config = copy.deepcopy(DEFAULT_NEWKIRK_HEIGHT_COMPARISON_CONFIG)
    config.update(event.section("newkirk_height_comparison"))
    return config


def load_drift_selection_product_config(
    config_name: ConfigSource,
):
    """Load persistent drift-selection product config."""
    event = load_radio_event_config(config_name)
    output_config = _load_output_config(event)
    config = copy.deepcopy(DEFAULT_DRIFT_SELECTION_PRODUCT_CONFIG)
    config.update(event.section("drift_selection_products"))
    if output_config.get("drift_selection_subdir"):
        config["output_subdir"] = output_config["drift_selection_subdir"]
    return config


def load_radio_output_config(config_name: ConfigSource):
    """Load common user-facing output controls from an event config."""
    return _load_output_config(load_radio_event_config(config_name))


def load_radio_diagnostic_presentation_config(
    config_name: ConfigSource,
):
    """Load frequency-priority diagnostic presentation config."""
    event = load_radio_event_config(config_name)
    user_config = event.section("user")
    output_config = _load_output_config(event)
    config = copy.deepcopy(DEFAULT_RADIO_DIAGNOSTIC_PRESENTATION_CONFIG)
    config.update(event.section("diagnostic_presentation"))
    if output_config.get("enable_static_summary") is not None:
        config["enable_static_summary"] = bool(output_config["enable_static_summary"])
    if output_config.get("enable_html_dashboard") is not None:
        config["enable_html_dashboard"] = bool(output_config["enable_html_dashboard"])
    if not config.get("comparison_frequency_mhz"):
        data_cfg = user_config.get("data", {}) if isinstance(user_config, dict) else {}
        freqs = data_cfg.get("multi_band_freqs") or []
        config["comparison_frequency_mhz"] = [float(freq) for freq in freqs]
    return config


def _load_output_config(event: RadioEventConfig) -> dict:
    config = copy.deepcopy(DEFAULT_OUTPUT_CONFIG)
    config.update(event.section("output"))
    return config


def _apply_output_config_to_user_config(user_config: dict, output_config: dict) -> None:
    output = user_config.setdefault("output", {})
    gaussian = user_config.setdefault("gaussian", {})
    for key in ("output_dir", "analysis_subdir"):
        value = output_config.get(key)
        if value not in (None, ""):
            output[key] = value
    gaussian_csv = output_config.get("gaussian_diagnostics_csv")
    if gaussian_csv:
        gaussian["gaussian_diagnostics_csv"] = gaussian_csv
