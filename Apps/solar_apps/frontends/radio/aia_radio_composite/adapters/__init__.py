"""Scientific API adapters used by the AIA radio composite frontend."""

from __future__ import annotations

from .aia_adapter import AiaSelection, load_aia_selection, scan_aia_catalog
from .radio_adapter import (
    DEFAULT_ROI_FREQUENCIES_MHZ,
    RadioFrameReference,
    RadioGaussianSelection,
    extract_multi_frequency_roi_curve,
    fit_radio_gaussian_frame,
    fit_radio_gaussian_selection,
    load_radio_candidates,
    select_radio_frame_from_candidates,
    select_radio_frame,
)
from .spectrum_adapter import (
    extract_spectrum_flux_curve,
    extract_spectrum_flux_curves,
    load_cso_spectrum_window,
    load_dart_spectrum_window,
    load_spectrum_window,
)

__all__ = [
    "AiaSelection",
    "DEFAULT_ROI_FREQUENCIES_MHZ",
    "RadioFrameReference",
    "RadioGaussianSelection",
    "extract_multi_frequency_roi_curve",
    "fit_radio_gaussian_frame",
    "extract_spectrum_flux_curve",
    "extract_spectrum_flux_curves",
    "fit_radio_gaussian_selection",
    "load_aia_selection",
    "load_radio_candidates",
    "load_cso_spectrum_window",
    "load_dart_spectrum_window",
    "load_spectrum_window",
    "select_radio_frame",
    "select_radio_frame_from_candidates",
    "scan_aia_catalog",
]
