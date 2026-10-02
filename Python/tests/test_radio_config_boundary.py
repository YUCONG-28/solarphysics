"""Tests for the installable radio configuration boundary."""

from __future__ import annotations

import pytest

from solar_toolkit.radio.config import (
    RadioEventConfig,
    load_radio_event_config,
    load_radio_output_config,
    load_radio_user_config,
)


def test_mapping_config_is_validated_and_merged_without_script_imports():
    source = {
        "user": {"data": {"multi_band_freqs": [149.0]}, "output": {}},
        "output": {
            "output_dir": "event-output",
            "analysis_subdir": "analysis",
            "gaussian_diagnostics_csv": "diagnostics.csv",
        },
        "newkirk": {"solar_radius_arcsec": 960.0},
    }

    event = load_radio_event_config(source)
    user, newkirk = load_radio_user_config(event)

    assert isinstance(event, RadioEventConfig)
    assert user["output"]["output_dir"] == "event-output"
    assert user["output"]["analysis_subdir"] == "analysis"
    assert user["gaussian"]["gaussian_diagnostics_csv"] == "diagnostics.csv"
    assert newkirk["solar_radius_arcsec"] == 960.0
    assert load_radio_output_config(event)["output_dir"] == "event-output"


def test_event_sections_are_defensive_copies():
    event = RadioEventConfig.from_mapping({"user": {"data": {"value": 1}}})
    first = event.section("user")
    first["data"]["value"] = 2

    assert event.section("user")["data"]["value"] == 1


def test_unknown_event_section_is_rejected():
    with pytest.raises(KeyError, match="Unknown radio event config sections"):
        RadioEventConfig.from_mapping({"mystery": {}})


def test_event_configuration_has_no_implicit_workstation_default():
    with pytest.raises(TypeError):
        load_radio_event_config()
    with pytest.raises(ValueError, match="fully qualified"):
        load_radio_event_config("synthetic_radio_config")


def test_module_configuration_remains_explicit_and_supported():
    from types import ModuleType

    module = ModuleType("synthetic_radio_config")
    module.EVENT_CONFIG = {"user": {"data": {"synthetic": True}}}
    assert load_radio_event_config(module).section("user")["data"]["synthetic"]
    with pytest.raises(ValueError, match="explicit radio"):
        load_radio_event_config(None)
