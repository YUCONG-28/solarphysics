"""Static and interactive rendering for the AIA radio composite frontend."""

from __future__ import annotations

from .composite_renderer import (
    TriplePanelArtifact,
    TopPanelArtifact,
    render_composite_result,
    render_top_panel,
    write_composite_artifacts,
)
from .plotly_components import (
    apply_radio_roi_to_request,
    build_dual_flux_figure,
    build_dual_flux_figures,
    build_roi_lightcurve_figure,
    build_spectrum_figure,
    build_spectrum_selection_figure,
    build_top_panel_selection_figure,
    radio_roi_from_selection,
    radio_roi_json,
    spectrum_band_from_selection,
)

__all__ = [
    "TopPanelArtifact",
    "TriplePanelArtifact",
    "apply_radio_roi_to_request",
    "build_dual_flux_figure",
    "build_dual_flux_figures",
    "build_roi_lightcurve_figure",
    "build_spectrum_figure",
    "build_spectrum_selection_figure",
    "build_top_panel_selection_figure",
    "radio_roi_from_selection",
    "radio_roi_json",
    "spectrum_band_from_selection",
    "render_composite_result",
    "render_top_panel",
    "write_composite_artifacts",
]
