"""Architecture contracts for the AIA radio composite frontend package."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from solar_apps.frontends.radio.aia_radio_composite import FRONTEND_ID

MODULES = (
    "solar_apps.frontends.radio.aia_radio_composite",
    "solar_apps.frontends.radio.aia_radio_composite.cli",
    "solar_apps.frontends.radio.aia_radio_composite.app",
    "solar_apps.frontends.radio.aia_radio_composite.application",
    "solar_apps.frontends.radio.aia_radio_composite.adapters",
    "solar_apps.frontends.radio.aia_radio_composite.adapters.aia_adapter",
    "solar_apps.frontends.radio.aia_radio_composite.adapters.radio_adapter",
    "solar_apps.frontends.radio.aia_radio_composite.adapters.spectrum_adapter",
    "solar_apps.frontends.radio.aia_radio_composite.models",
    "solar_apps.frontends.radio.aia_radio_composite.models.schema",
    "solar_apps.frontends.radio.aia_radio_composite.rendering",
    "solar_apps.frontends.radio.aia_radio_composite.rendering.composite_renderer",
    "solar_apps.frontends.radio.aia_radio_composite.rendering.plotly_components",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_frontend_architecture_module_imports(module_name: str) -> None:
    """Every package boundary imports without loading observation data."""

    assert importlib.import_module(module_name) is not None


def test_frontend_id_matches_planned_public_command() -> None:
    """The package owns one stable public frontend identifier."""

    assert FRONTEND_ID == "aia-radio-composite"


def test_readme_records_scientific_ownership_boundaries() -> None:
    """The package documentation keeps existing science APIs authoritative."""

    package_root = (
        Path(__file__).resolve().parents[3]
        / "solar_apps"
        / "frontends"
        / "radio"
        / "aia_radio_composite"
    )
    readme = (package_root / "README.md").read_text(encoding="utf-8")

    assert "does not implement Gaussian fitting" in readme
    assert "solar_toolkit.radio.dart_spectrogram" in readme
    assert "solar_toolkit.radio.cso" in readme
