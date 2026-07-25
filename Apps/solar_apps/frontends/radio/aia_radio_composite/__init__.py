"""AIA, radio-source, ROI-lightcurve, and spectrum composite frontend."""

from __future__ import annotations

from .adapters import AiaSelection, RadioGaussianSelection
from .models import (
    CompositeRequest,
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumWindow,
)
from .rendering import TriplePanelArtifact, TopPanelArtifact

FRONTEND_ID = "aia-radio-composite"

__all__ = [
    "FRONTEND_ID",
    "AiaSelection",
    "CompositeRequest",
    "CompositeResult",
    "RadioGaussianSelection",
    "SpectrumBand",
    "SpectrumFluxCurve",
    "SpectrumWindow",
    "TriplePanelArtifact",
    "TopPanelArtifact",
]
