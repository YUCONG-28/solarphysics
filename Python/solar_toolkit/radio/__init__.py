"""Reusable radio-analysis computation APIs.

Event configuration, CLIs, servers, and workflow orchestration live in the
tracked ``solar_apps`` application layer rather than this public package.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "fieldline_association": "solar_toolkit.radio.fieldline_association",
    "source_geometry": "solar_toolkit.radio.source_geometry",
    "centers": "solar_toolkit.radio.centers",
    "config": "solar_toolkit.radio.config",
    "coordinates": "solar_toolkit.radio.coordinates",
    "cso": "solar_toolkit.radio.cso",
    "cso_processing": "solar_toolkit.radio.cso_processing",
    "dart_spectrogram": "solar_toolkit.radio.dart_spectrogram",
    "drift_products": "solar_toolkit.radio.drift_products",
    "drift_rate": "solar_toolkit.radio.drift_rate",
    "existing_fit_overlay": "solar_toolkit.radio.existing_fit_overlay",
    "frequency_priority_diagnostics": "solar_toolkit.radio.frequency_priority_diagnostics",
    "gaussian": "solar_toolkit.radio.gaussian",
    "gaussian_background": "solar_toolkit.radio.gaussian_background",
    "gaussian_diagnostics": "solar_toolkit.radio.gaussian_diagnostics",
    "gaussian_fit": "solar_toolkit.radio.gaussian_fit",
    "gaussian_io": "solar_toolkit.radio.gaussian_io",
    "gaussian_masks": "solar_toolkit.radio.gaussian_masks",
    "gaussian_models": "solar_toolkit.radio.gaussian_models",
    "height_comparison": "solar_toolkit.radio.height_comparison",
    "height_plots": "solar_toolkit.radio.height_plots",
    "io": "solar_toolkit.radio.io",
    "newkirk": "solar_toolkit.radio.newkirk",
    "output_paths": "solar_toolkit.radio.output_paths",
    "physical_diagnostics": "solar_toolkit.radio.physical_diagnostics",
    "provenance": "solar_toolkit.radio.provenance",
    "quicklook": "solar_toolkit.radio.quicklook",
    "quality_autoencoder": "solar_toolkit.radio.quality_autoencoder",
    "quality_ml": "solar_toolkit.radio.quality_ml",
    "quality_science": "solar_toolkit.radio.quality_science",
    "raw_quality": "solar_toolkit.radio.raw_quality",
    "reprojection": "solar_toolkit.radio.reprojection",
    "roi_lightcurve": "solar_toolkit.radio.roi_lightcurve",
    "spectrogram": "solar_toolkit.radio.spectrogram",
    "trajectory": "solar_toolkit.radio.trajectory",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
