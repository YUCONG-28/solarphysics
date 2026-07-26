"""Request and result models for the AIA radio composite frontend."""

from __future__ import annotations

from .schema import (
    AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
    AIA_WAVELENGTHS,
    POLARIZATIONS,
    ROI_CURVE_COLUMNS,
    ROI_TYPES,
    SPECTRUM_SOURCES,
    SPECTRUM_TYPES,
    CompositeRequest,
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumWindow,
    parse_roi_curve_times,
)
from .time_alignment import (
    SpectrumTimeAlignment,
    build_spectrum_time_alignment,
)

__all__ = [
    "AIA_RADIO_COMPOSITE_SCHEMA_VERSION",
    "AIA_WAVELENGTHS",
    "CompositeRequest",
    "CompositeResult",
    "POLARIZATIONS",
    "ROI_CURVE_COLUMNS",
    "ROI_TYPES",
    "SPECTRUM_SOURCES",
    "SPECTRUM_TYPES",
    "SpectrumBand",
    "SpectrumFluxCurve",
    "SpectrumTimeAlignment",
    "SpectrumWindow",
    "build_spectrum_time_alignment",
    "parse_roi_curve_times",
]
