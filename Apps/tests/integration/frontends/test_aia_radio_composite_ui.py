"""Smoke tests for the AIA radio composite Streamlit page."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _app_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "solar_apps"
        / "frontends"
        / "radio"
        / "aia_radio_composite"
        / "app.py"
    )


def _configure_local_state(
    tmp_path: Path,
    monkeypatch,
) -> Path:
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(data_root))
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    return data_root


def test_page_loads_with_required_three_panel_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_local_state(tmp_path, monkeypatch)

    app = AppTest.from_file(str(_app_path()), default_timeout=40).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "AIA + Radio Composite"
    assert [item.value for item in app.subheader] == [
        "Panel 1 — AIA + Radio Gaussian",
        "Panel 2 — ROI-matched spectrum bands + dual-axis flux",
        "Panel 3 — Composite export",
    ]
    assert app.text_input[0].label == "AIA directory"
    assert app.text_input[1].label == "Radio directory"
    assert app.text_input[2].label == "Spectrum directory or FITS file"
    assert app.selectbox(key="aia-radio-composite_theme_mode").value == "auto"
    assert app.multiselect(key="aia_waves").value == [171]
    assert app.multiselect(key="radio_frequencies").value == [149]
    assert app.selectbox(key="extended_canvas_color").value == "black"
    assert app.radio(key="flux_plot_layout").value == "Combined"
    assert (
        app.multiselect(key="radio_frequencies").label
        == "AIA overlay radio frequencies (MHz)"
    )
    assert len([button for button in app.button if button.label == "Browse"]) == 4
    assert app.checkbox[0].label == "Use custom HPC display range"
    assert {item.label for item in app.number_input} >= {
        "HPLN min (arcsec)",
        "HPLN max (arcsec)",
        "HPLT min (arcsec)",
        "HPLT max (arcsec)",
    }
    assert any(button.label == "Load CSO / DART spectrum" for button in app.button)
    assert any(button.label == "Extract dual-axis flux" for button in app.button)
    assert {item.label for item in app.checkbox} >= {
        "Show Gaussian fitted center",
        "Show Gaussian fitted contour",
    }
    assert app.checkbox(key="gaussian_show_contours").value is False
    assert {item.label for item in app.number_input} >= {
        "Gaussian contour (% of fitted peak)",
        "Radio display low percentile",
        "Radio display high percentile",
        "Video FPS",
        "Matched spectrum bandwidth (MHz)",
        "Spectrum frequency min (MHz)",
        "Spectrum frequency max (MHz)",
        "Spectrum intensity min",
        "Spectrum intensity max",
    }
    assert {item.label for item in app.checkbox} >= {
        "Use custom spectrum frequency range",
        "Use custom spectrum intensity range",
    }
    assert "Video frame count" not in {item.label for item in app.number_input}
    assert {item.label for item in app.text_input} >= {
        "Flux UTC start",
        "Flux UTC end",
        "Spectrum UTC start",
        "Spectrum UTC end",
    }


def test_page_supports_auto_light_and_dark_theme_modes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_local_state(tmp_path, monkeypatch)
    app = AppTest.from_file(str(_app_path()), default_timeout=40).run()
    theme = app.selectbox(key="aia-radio-composite_theme_mode")

    for mode in ("auto", "light", "dark"):
        theme.set_value(mode).run()
        assert len(app.exception) == 0


def test_paths_and_controls_restore_in_a_new_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = _configure_local_state(tmp_path, monkeypatch)
    paths = {
        "aia_directory": data_root / "aia",
        "radio_directory": data_root / "radio",
        "spectrum_path": data_root / "dart",
        "output_directory": data_root / "output",
    }
    for path in paths.values():
        path.mkdir()
    app = AppTest.from_file(str(_app_path()), default_timeout=40).run()
    for key, path in paths.items():
        app.text_input(key=key).set_value(str(path))
    app.selectbox(key="polarization").set_value("LL")
    app.multiselect(key="aia_waves").set_value([193, 94, 171])
    app.multiselect(key="radio_frequencies").set_value([149, 164, 190])
    app.checkbox(key="gaussian_show_center").set_value(False)
    app.checkbox(key="use_custom_fov").set_value(True)
    app.number_input(key="hpln_min").set_value(-800.0)
    app.selectbox(key="extended_canvas_color").set_value("white")
    app.checkbox(key="use_custom_spectrum_frequency_range").set_value(True)
    app.number_input(key="spectrum_frequency_min").set_value(145.0)
    app.number_input(key="spectrum_frequency_max").set_value(210.0)
    app.checkbox(key="use_custom_spectrum_intensity_range").set_value(True)
    app.number_input(key="spectrum_intensity_min").set_value(2.0)
    app.number_input(key="spectrum_intensity_max").set_value(8.0)
    app.radio(key="flux_plot_layout").set_value("One chart per frequency")
    app.run()
    assert len(app.exception) == 0

    restored = AppTest.from_file(str(_app_path()), default_timeout=40).run()

    assert len(restored.exception) == 0
    for key, path in paths.items():
        assert restored.text_input(key=key).value == str(path)
    assert restored.selectbox(key="polarization").value == "LL"
    assert restored.multiselect(key="aia_waves").value == [193, 94, 171]
    assert restored.multiselect(key="radio_frequencies").value == [149, 164, 190]
    assert restored.checkbox(key="gaussian_show_center").value is False
    assert restored.checkbox(key="use_custom_fov").value is True
    assert restored.number_input(key="hpln_min").value == -800.0
    assert restored.selectbox(key="extended_canvas_color").value == "white"
    assert restored.checkbox(key="use_custom_spectrum_frequency_range").value is True
    assert restored.number_input(key="spectrum_frequency_min").value == 145.0
    assert restored.number_input(key="spectrum_frequency_max").value == 210.0
    assert restored.checkbox(key="use_custom_spectrum_intensity_range").value is True
    assert restored.number_input(key="spectrum_intensity_min").value == 2.0
    assert restored.number_input(key="spectrum_intensity_max").value == 8.0
    assert restored.radio(key="flux_plot_layout").value == "One chart per frequency"


def test_cso_mode_offers_file_or_directory_browsing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_local_state(tmp_path, monkeypatch)
    app = AppTest.from_file(str(_app_path()), default_timeout=40).run()

    app.selectbox(key="spectrum_type").set_value("CSO").run()

    assert len(app.exception) == 0
    assert app.radio(key="cso_path_kind").options == ["FITS file", "Directory"]
    assert len([button for button in app.button if button.label == "Browse"]) == 4
