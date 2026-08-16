"""Shared config builder for 2025-01-24 9-band raw radio source maps."""

from __future__ import annotations

import copy

from . import radio_20250124_config as base_config

__all__ = [
    "FREQUENCIES_9_BAND",
    "FULL_RANGE",
    "OUTPUT_DIR",
    "PREVIEW_RANGE",
    "build_event_config",
]

FREQUENCIES_9_BAND = [149, 164, 190, 223, 238, 285, 300, 324, 309]
OUTPUT_DIR = "outputs/radio/2025-01-24"

PREVIEW_RANGE = (1589, 1590)
FULL_RANGE = (1296, 1884)


def build_event_config(*, polarization: str, phase: str) -> dict:
    """Return a 9-band raw-radio event config for one polarization and phase."""
    if phase not in {"preview", "full"}:
        raise ValueError(f"Unsupported phase: {phase}")
    if polarization not in {"LL", "RR", "RR+LL"}:
        raise ValueError(f"Unsupported polarization: {polarization}")

    event_config = copy.deepcopy(base_config.EVENT_CONFIG)
    output_config = event_config["output"]
    user_config = event_config["user"]
    user_config["study_mode"] = (
        "confirmatory" if phase == "full" else "exploratory"
    )

    label = polarization.replace("+", "_")
    start_idx, end_idx = PREVIEW_RANGE if phase == "preview" else FULL_RANGE

    output_config["output_dir"] = OUTPUT_DIR
    output_config["analysis_subdir"] = f"radio_source_maps_9band_{phase}_{label}"

    data = user_config["data"]
    data.update(
        {
            "multi_band_freqs": FREQUENCIES_9_BAND,
            "polarization": polarization,
            "combine_polarizations": polarization == "RR+LL",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "multi_band_layout": (3, 3),
            "multi_band_time_tolerance_seconds": 0.3,
        }
    )

    user_config["features"].update(
        {
            "gaussian_overlay": False,
            "spectrogram_panel": False,
            "save_gaussian_diagnostics": False,
            "save_background_products": False,
            "save_individual_pols": False,
        }
    )
    user_config["display"].update(
        {
            "color_range_mode": "auto",
            "use_per_band_colormap": True,
            "per_band_range_method": "fixed_percentile",
            "per_band_percentiles": [99.6, 99.99],
        }
    )
    user_config["background"].update(
        {
            "mode": "off",
            "apply_to_display": False,
            "apply_to_fit": False,
            "apply_before_polarization_combine": False,
        }
    )
    user_config["drift_rate"].update(
        {
            "enabled": False,
            "mode": "off",
        }
    )
    user_config["output"].update(
        {
            "output_dir": output_config["output_dir"],
            "analysis_subdir": output_config["analysis_subdir"],
            "show_plot": False,
            "save_plot": True,
            "dpi": 180 if phase == "preview" else 300,
        }
    )

    event_config["diagnostic_presentation"]["comparison_frequency_mhz"] = [
        float(freq) for freq in FREQUENCIES_9_BAND
    ]
    return event_config
