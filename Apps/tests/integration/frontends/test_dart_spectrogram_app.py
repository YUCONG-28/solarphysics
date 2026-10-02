"""Tests for the standalone DART Streamlit application."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, ScalarFormatter
from PIL import Image
from solar_toolkit.radio.dart_spectrogram import (
    extract_dart_narrowband_lightcurves,
    read_dart_spectrogram_window,
)
from streamlit.testing.v1 import AppTest

from solar_apps.frontends.radio.dart_spectrogram.dart_spectrogram_app import (
    DYNAMIC_SPECTRUM_FILENAME,
    DYNAMIC_SPECTRUM_PRODUCT_KEY,
    LIGHTCURVE_FILENAME,
    LIGHTCURVE_PRODUCT_KEY,
    SELECTED_SPECTRUM_FILENAME,
    SELECTED_SPECTRUM_PRODUCT_KEY,
    _display_stokes_i,
    build_dart_artifact_filenames,
    build_dynamic_spectrum_figure,
    build_dynamic_spectrum_png,
    build_narrowband_figure,
    build_narrowband_png,
    build_zip_bytes,
    inspect_dart_dataset,
    parse_center_frequencies,
    parse_marked_times,
    resolve_display_limits,
    resolve_selected_frequency_range,
    save_artifacts,
)


def _write_dataset(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    frequency = np.asarray([147.0, 148.0, 149.0, 150.0, 151.0])
    time_rows = np.asarray(
        [
            [0, 1, 1, 4, 45, 0.0],
            [0, 1, 1, 4, 45, 1.0],
            [0, 1, 1, 4, 45, 2.0],
            [0, 1, 1, 4, 45, 3.0],
        ]
    )
    stokes_i = np.asarray(
        [
            [4.0, 5.0, 6.0, 7.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
        ]
    )
    stokes_vp = np.asarray(
        [
            [-0.10, -0.05, 0.00, 0.05],
            [-0.08, -0.04, 0.01, 0.06],
            [-0.06, -0.02, 0.02, 0.08],
            [-0.04, 0.00, 0.04, 0.10],
            [-0.02, 0.02, 0.06, 0.12],
        ]
    )
    payloads = {
        "2000-01-01_SpecDataIdB.fits": stokes_i,
        "2000-01-01_SpecDataVP.fits": stokes_vp,
        "2000-01-01_SpecFrequency.fits": frequency[None, :],
        "2000-01-01_SpecTime.fits": time_rows,
    }
    for name, payload in payloads.items():
        fits.PrimaryHDU(data=np.asarray(payload, dtype=np.float64)).writeto(
            folder / name
        )
    return folder


def _analysis_products(folder: Path):
    window = read_dart_spectrogram_window(folder)
    narrowband = extract_dart_narrowband_lightcurves(folder, [149.0], 2.0)
    return window, narrowband


def test_parse_center_frequencies_enforces_limit_and_uniqueness() -> None:
    assert parse_center_frequencies("149, 150.5") == (149.0, 150.5)
    with pytest.raises(ValueError, match="duplicates"):
        parse_center_frequencies("149,149")
    with pytest.raises(ValueError, match="At most 20"):
        parse_center_frequencies(",".join(str(value) for value in range(21)))
    with pytest.raises(ValueError, match="separated by commas"):
        parse_center_frequencies("149,")


def test_parse_marked_times_normalizes_deduplicates_and_validates_range() -> None:
    observation_range = (
        datetime(2000, 1, 1, 4, 45, tzinfo=UTC),
        datetime(2000, 1, 1, 4, 45, 3, tzinfo=UTC),
    )

    assert parse_marked_times(
        "04:45:01.5, 2000-01-01T05:45:02+01:00, 04:45:01.500",
        observation_range,
    ) == (
        datetime(2000, 1, 1, 4, 45, 1, 500000, tzinfo=UTC),
        datetime(2000, 1, 1, 4, 45, 2, tzinfo=UTC),
    )
    assert parse_marked_times("", observation_range) == ()
    assert parse_marked_times(
        "2000-01-01T04:45:03",
        observation_range,
    ) == (datetime(2000, 1, 1, 4, 45, 3, tzinfo=UTC),)

    with pytest.raises(ValueError, match="outside the observation range"):
        parse_marked_times("04:46:00", observation_range)
    with pytest.raises(ValueError, match="HH:MM:SS"):
        parse_marked_times("not-a-time", observation_range)
    with pytest.raises(ValueError, match="more than one observation date"):
        parse_marked_times(
            "04:45:01",
            (
                datetime(2000, 1, 1, tzinfo=UTC),
                datetime(2000, 1, 2, 23, 59, 59, tzinfo=UTC),
            ),
        )


def test_stokes_i_display_conversion_and_color_limit_resolution(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    window, _narrowband = _analysis_products(folder)

    source = np.asarray([0.0, 10.0, 20.0])
    assert np.array_equal(_display_stokes_i(source, "db"), source)
    assert np.allclose(_display_stokes_i(source, "linear"), [1.0, 10.0, 100.0])

    percentile_limits = resolve_display_limits(
        window,
        display_mode="linear",
        limit_mode="percentile",
        stokes_i_bounds=(1.0, 99.5),
    )
    assert percentile_limits.stokes_i == pytest.approx(
        tuple(
            np.percentile(
                np.power(10.0, window.stokes_i_db / 10.0),
                [1.0, 99.5],
            )
        )
    )
    assert not hasattr(percentile_limits, "stokes_v_over_i_percent")

    direct_limits = resolve_display_limits(
        window,
        display_mode="db",
        limit_mode="direct",
        stokes_i_bounds=(5.0, 40.0),
    )
    assert direct_limits.stokes_i == (5.0, 40.0)

    with pytest.raises(ValueError, match="lower value below"):
        resolve_display_limits(
            window,
            limit_mode="direct",
            stokes_i_bounds=(10.0, 10.0),
        )
    with pytest.raises(ValueError, match="0 <= lower"):
        resolve_display_limits(
            window,
            limit_mode="percentile",
            stokes_i_bounds=(99.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        resolve_display_limits(
            window,
            limit_mode="direct",
            stokes_i_bounds=(np.nan, 10.0),
        )


def test_custom_time_ticks_and_short_markers_apply_to_all_plot_types(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    window, narrowband = _analysis_products(folder)
    markers = (window.time_utc[1], window.time_utc[2])

    spectrum = build_dynamic_spectrum_figure(
        window,
        x_tick_interval_seconds=1.5,
        marked_times_utc=markers,
    )
    spectrum_axes = [axis for axis in spectrum.axes if axis.images]
    assert isinstance(spectrum_axes[1].xaxis.get_major_locator(), FixedLocator)
    tick_locations = spectrum_axes[1].xaxis.get_majorticklocs()
    assert len(tick_locations) == 3
    assert np.diff(tick_locations) * 86400.0 == pytest.approx([1.5, 1.5])
    for axis in spectrum_axes:
        marker_lines = [line for line in axis.lines if line.get_color() == "#facc15"]
        assert len(marker_lines) == 2
        assert all(
            line.get_ydata() == pytest.approx([0.0, 0.06]) for line in marker_lines
        )
    labels = {text.get_text() for text in spectrum_axes[1].texts}
    assert {"04:45:01", "04:45:02"} <= labels

    lightcurve = build_narrowband_figure(
        narrowband,
        x_tick_interval_seconds=1.5,
        marked_times_utc=(*markers, window.time_utc[-1] + timedelta(seconds=10)),
    )
    marker_lines = [
        line for line in lightcurve.axes[0].lines if line.get_color() == "#facc15"
    ]
    assert len(marker_lines) == 2
    assert {text.get_text() for text in lightcurve.axes[0].texts} == {
        "04:45:01",
        "04:45:02",
    }

    with pytest.raises(ValueError, match="major ticks"):
        build_dynamic_spectrum_figure(window, x_tick_interval_seconds=0.01)


def test_selected_frequency_range_combines_all_bands_and_validates_override(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    narrowband = extract_dart_narrowband_lightcurves(
        folder,
        [148.0, 150.0],
        1.0,
    )
    assert resolve_selected_frequency_range(
        narrowband,
        (147.0, 151.0),
    ) == (147.5, 150.5)
    assert resolve_selected_frequency_range(
        narrowband,
        (147.0, 151.0),
        (147.0, 151.0),
    ) == (147.0, 151.0)
    with pytest.raises(ValueError, match="contain every requested narrowband"):
        resolve_selected_frequency_range(
            narrowband,
            (147.0, 151.0),
            (148.0, 150.0),
        )


def test_figures_share_limits_show_units_annotations_and_scientific_scale(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    full_window = read_dart_spectrogram_window(folder)
    selected_window = read_dart_spectrogram_window(
        folder,
        frequency_range_mhz=(148.0, 150.0),
    )
    narrowband = extract_dart_narrowband_lightcurves(folder, [149.0], 2.0)
    limits = resolve_display_limits(full_window, display_mode="linear")
    selected_time = (selected_window.time_utc[0], selected_window.time_utc[-1])

    full_figure = build_dynamic_spectrum_figure(
        full_window,
        narrowband,
        display_mode="linear",
        display_limits=limits,
        selected_frequency_range_mhz=(148.0, 150.0),
        selected_time_range_utc=selected_time,
    )
    selected_figure = build_dynamic_spectrum_figure(
        selected_window,
        narrowband,
        display_mode="linear",
        display_limits=limits,
        region_label="Selected region 148-150 MHz",
    )
    full_image_axes = [axis for axis in full_figure.axes if axis.images]
    selected_image_axes = [axis for axis in selected_figure.axes if axis.images]
    assert full_image_axes[0].images[0].get_clim() == pytest.approx(limits.stokes_i)
    assert selected_image_axes[0].images[0].get_clim() == pytest.approx(limits.stokes_i)
    assert full_image_axes[1].images[0].get_clim() == (-1.0, 1.0)
    assert selected_image_axes[1].images[0].get_clim() == (-1.0, 1.0)
    assert np.array_equal(
        np.asarray(full_image_axes[1].images[0].get_array()),
        full_window.stokes_v_over_i,
    )
    assert np.array_equal(
        np.asarray(selected_image_axes[1].images[0].get_array()),
        selected_window.stokes_v_over_i,
    )
    assert isinstance(full_image_axes[1].images[0].norm, TwoSlopeNorm)
    assert full_image_axes[1].images[0].norm.vcenter == 0.0
    assert any(isinstance(patch, Rectangle) for patch in full_image_axes[0].patches)
    full_text = " ".join(
        text.get_text() for axis in full_figure.axes for text in axis.texts
    )
    selected_text = " ".join(
        text.get_text() for axis in selected_figure.axes for text in axis.texts
    )
    assert "Selected: 148-150 MHz" in full_text
    assert "148-150 MHz" in selected_text
    axis_labels = " ".join(
        axis.get_ylabel() for axis in full_figure.axes if axis.get_ylabel()
    )
    assert "dimensionless" in axis_labels
    assert "Stokes V/I = (R-L)/(R+L) (dimensionless)" in axis_labels
    assert "%" not in axis_labels

    high_intensity_narrowband = extract_dart_narrowband_lightcurves(
        folder,
        [150.5],
        1.0,
    )
    lightcurve_figure = build_narrowband_figure(
        high_intensity_narrowband,
        display_mode="linear",
    )
    lightcurve_axis = lightcurve_figure.axes[0]
    assert isinstance(lightcurve_axis.yaxis.get_major_formatter(), ScalarFormatter)
    assert "dimensionless" in lightcurve_axis.get_ylabel()
    lightcurve_figure.canvas.draw()
    assert "10^{" in lightcurve_axis.yaxis.get_offset_text().get_text()

    direct_limits = resolve_display_limits(
        full_window,
        limit_mode="direct",
        stokes_i_bounds=(4.0, 43.0),
    )
    direct_figure = build_dynamic_spectrum_figure(
        full_window,
        narrowband,
        display_limits=direct_limits,
    )
    direct_vp_axis = [axis for axis in direct_figure.axes if axis.images][1]
    assert direct_vp_axis.images[0].get_clim() == (-1.0, 1.0)
    assert isinstance(direct_vp_axis.images[0].norm, TwoSlopeNorm)


def test_full_spectrum_figure_needs_no_narrowband_or_selection(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    window = read_dart_spectrogram_window(folder)

    figure = build_dynamic_spectrum_figure(
        window,
        selected_frequency_range_mhz=(148.0, 150.0),
        selected_time_range_utc=(window.time_utc[1], window.time_utc[2]),
    )
    image_axes = [axis for axis in figure.axes if axis.images]

    assert len(image_axes) == 2
    assert len(image_axes[0].patches) == 0
    assert len(image_axes[0].texts) == 0
    assert image_axes[1].images[0].get_clim() == (-1.0, 1.0)
    assert isinstance(image_axes[1].images[0].norm, TwoSlopeNorm)
    assert np.array_equal(
        np.asarray(image_axes[1].images[0].get_array()),
        window.stokes_v_over_i,
    )
    assert image_axes[0].get_ylim()[0] < float(window.frequency_mhz[0])
    assert image_axes[0].get_ylim()[1] > float(window.frequency_mhz[-1])
    assert image_axes[0].get_xlim() == pytest.approx(image_axes[1].get_xlim())

    payload = build_dynamic_spectrum_png(window, dpi=100)
    rendered = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    assert rendered.shape[0] > 400
    assert rendered.shape[1] > 800
    assert float(np.std(rendered)) > 5.0


def test_dataset_inspection_and_figures_are_valid_nonblank_pngs(
    tmp_path: Path,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    summary = inspect_dart_dataset(folder)
    assert summary.matrix_shape == (5, 4)
    assert summary.frequency_range_mhz == (147.0, 151.0)
    assert summary.time_samples == 4

    window, narrowband = _analysis_products(folder)
    figure = build_dynamic_spectrum_figure(window, narrowband)
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 2
    assert all(axis.images[0].origin == "lower" for axis in image_axes)
    assert all(
        axis.images[0].get_extent()[2] < axis.images[0].get_extent()[3]
        for axis in image_axes
    )
    assert image_axes[1].images[0].get_clim() == (-1.0, 1.0)
    assert isinstance(image_axes[1].images[0].norm, TwoSlopeNorm)

    spectrum_png = build_dynamic_spectrum_png(window, narrowband, dpi=100)
    selected_png = build_dynamic_spectrum_png(
        window,
        narrowband,
        dpi=100,
        region_label="Selected region 148-150 MHz",
    )
    lightcurve_png = build_narrowband_png(narrowband, dpi=100)
    for payload in (spectrum_png, selected_png, lightcurve_png):
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        image.verify()
        rendered = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
        assert rendered.shape[0] > 400
        assert rendered.shape[1] > 800
        assert float(np.std(rendered)) > 5.0


def test_zip_and_local_save_reuse_exact_png_bytes(tmp_path: Path) -> None:
    folder = _write_dataset(tmp_path / "data")
    window, narrowband = _analysis_products(folder)
    artifacts = {
        DYNAMIC_SPECTRUM_FILENAME: build_dynamic_spectrum_png(
            window, narrowband, dpi=100
        ),
        SELECTED_SPECTRUM_FILENAME: build_dynamic_spectrum_png(
            window,
            narrowband,
            dpi=100,
            region_label="Selected region 148-150 MHz",
        ),
        LIGHTCURVE_FILENAME: build_narrowband_png(narrowband, dpi=100),
    }

    zip_payload = build_zip_bytes(artifacts)
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        assert set(archive.namelist()) == set(artifacts)
        for filename, payload in artifacts.items():
            assert archive.read(filename) == payload

    first = save_artifacts(artifacts, tmp_path / "outputs")
    second = save_artifacts(artifacts, tmp_path / "outputs")
    assert first != second
    for directory in (first, second):
        for filename, payload in artifacts.items():
            assert (directory / filename).read_bytes() == payload

    missing_selected = dict(artifacts)
    missing_selected.pop(SELECTED_SPECTRUM_FILENAME)
    with pytest.raises(ValueError, match="either the full spectrum only"):
        build_zip_bytes(missing_selected)

    full_only = {DYNAMIC_SPECTRUM_FILENAME: build_dynamic_spectrum_png(window, dpi=100)}
    full_zip = build_zip_bytes(full_only)
    with zipfile.ZipFile(io.BytesIO(full_zip)) as archive:
        assert archive.namelist() == [DYNAMIC_SPECTRUM_FILENAME]
        assert (
            archive.read(DYNAMIC_SPECTRUM_FILENAME)
            == full_only[DYNAMIC_SPECTRUM_FILENAME]
        )
    full_saved = save_artifacts(full_only, tmp_path / "full-outputs")
    assert (full_saved / DYNAMIC_SPECTRUM_FILENAME).read_bytes() == full_only[
        DYNAMIC_SPECTRUM_FILENAME
    ]


def test_dynamic_names_match_download_zip_and_local_save(tmp_path: Path) -> None:
    folder = _write_dataset(tmp_path / "data")
    summary = inspect_dart_dataset(folder)
    window, narrowband = _analysis_products(folder)
    artifacts = {
        DYNAMIC_SPECTRUM_PRODUCT_KEY: build_dynamic_spectrum_png(
            window, narrowband, dpi=100
        ),
        SELECTED_SPECTRUM_PRODUCT_KEY: build_dynamic_spectrum_png(
            window,
            narrowband,
            dpi=100,
            region_label="Selected region 148-150 MHz",
        ),
        LIGHTCURVE_PRODUCT_KEY: build_narrowband_png(narrowband, dpi=100),
    }
    filenames = build_dart_artifact_filenames(summary)
    full_only_filenames = build_dart_artifact_filenames(
        summary,
        product_keys=(DYNAMIC_SPECTRUM_PRODUCT_KEY,),
    )

    assert list(filenames) == [
        DYNAMIC_SPECTRUM_PRODUCT_KEY,
        SELECTED_SPECTRUM_PRODUCT_KEY,
        LIGHTCURVE_PRODUCT_KEY,
    ]
    assert filenames[DYNAMIC_SPECTRUM_PRODUCT_KEY] == (
        "0001_20000101T044500Z-20000101T044503Z_"
        "dart_stokes_i_v_over_i_dynamic_spectrum.png"
    )
    assert filenames[SELECTED_SPECTRUM_PRODUCT_KEY] == (
        "0002_20000101T044500Z-20000101T044503Z_"
        "dart_stokes_i_v_over_i_selected_spectrum.png"
    )
    assert filenames[LIGHTCURVE_PRODUCT_KEY].startswith(
        "0003_20000101T044500Z-20000101T044503Z_dart_stokes_i_"
    )
    assert full_only_filenames == {
        DYNAMIC_SPECTRUM_PRODUCT_KEY: filenames[DYNAMIC_SPECTRUM_PRODUCT_KEY]
    }

    zip_payload = build_zip_bytes(artifacts, filenames=filenames)
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        assert archive.namelist() == list(filenames.values())
        for product_key, filename in filenames.items():
            assert archive.read(filename) == artifacts[product_key]

    saved = save_artifacts(
        artifacts,
        tmp_path / "outputs",
        filenames=filenames,
    )
    assert sorted(path.name for path in saved.iterdir()) == sorted(filenames.values())


def test_streamlit_app_load_validation_generation_and_downloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    folder = _write_dataset(tmp_path / "data")
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    app_path = (
        Path(__file__).resolve().parents[3]
        / "solar_apps"
        / "frontends"
        / "radio"
        / "dart_spectrogram"
        / "dart_spectrogram_app.py"
    )
    app = AppTest.from_file(str(app_path), default_timeout=40).run()
    assert len(app.exception) == 0
    assert app.title[0].value == "DART Spectrogram Analysis"

    app.text_input(key="input_dir").set_value(str(folder))
    app.button(key="load_dataset").click().run()
    assert len(app.exception) == 0
    assert len(app.dataframe) == 2
    assert not any("V/I" in widget.label for widget in app.number_input)
    assert app.radio(key="analysis_mode").value == "Full spectrum only"
    assert not any(widget.key == "center_frequencies" for widget in app.text_input)
    assert app.radio(key="x_tick_mode").value == "Auto"
    assert app.text_input(key="marked_times") is not None
    assert not any(
        widget.key == "x_tick_interval_seconds" for widget in app.number_input
    )

    app.radio(key="x_tick_mode").set_value("Custom").run()
    app.number_input(key="x_tick_interval_seconds").set_value(1.5).run()
    app.text_input(key="marked_times").set_value(
        "04:45:01.5, 2000-01-01T04:45:02Z"
    ).run()

    app.button(key="generate_figures").click().run(timeout=40)
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.image) == 1
    assert len(app.download_button) == 1
    assert app.download_button(key="download_spectrum") is not None
    assert not any(button.key == "download_zip" for button in app.download_button)

    app.number_input(key="x_tick_interval_seconds").set_value(1.0).run()
    assert len(app.image) == 0
    assert any("Parameters changed" in warning.value for warning in app.warning)
    app.button(key="generate_figures").click().run(timeout=40)
    assert len(app.image) == 1

    app.radio(key="display_mode").set_value("Relative linear").run()
    assert len(app.image) == 0
    assert len(app.download_button) == 0
    assert any("Parameters changed" in warning.value for warning in app.warning)

    app.radio(key="limit_mode").set_value("Direct").run()
    assert not any("V/I" in widget.label for widget in app.number_input)
    app.button(key="generate_figures").click().run(timeout=40)
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.image) == 1
    assert len(app.download_button) == 1

    app.radio(key="analysis_mode").set_value(
        "Full spectrum + selected region + narrowband intensity"
    ).run()
    assert len(app.image) == 0
    assert len(app.download_button) == 0
    assert any("Parameters changed" in warning.value for warning in app.warning)
    assert app.text_input(key="center_frequencies") is not None
    app.checkbox(key="limit_time").check().run()
    app.text_input(key="time_start").set_value("2000-01-01T04:45:01+00:00").run()
    app.text_input(key="time_end").set_value("2000-01-01T04:45:03+00:00").run()
    app.text_input(key="marked_times").set_value("04:45:00, 04:45:02").run()

    app.text_input(key="center_frequencies").set_value("149,149").run()
    app.button(key="generate_figures").click().run()
    assert any("duplicates" in error.value for error in app.error)

    app.text_input(key="center_frequencies").set_value("149").run()
    app.button(key="generate_figures").click().run(timeout=40)
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.image) == 3
    assert len(app.download_button) == 4
    assert app.download_button(key="download_spectrum") is not None
    assert app.download_button(key="download_selected_spectrum") is not None
    assert app.download_button(key="download_lightcurves") is not None
    assert app.download_button(key="download_zip") is not None
    assert any("appear only" in warning.value for warning in app.warning)

    app.radio(key="analysis_mode").set_value("Full spectrum only").run()
    assert len(app.image) == 0
    assert len(app.download_button) == 0
    app.button(key="generate_figures").click().run(timeout=40)
    assert len(app.exception) == 0
    assert len(app.error) == 0
    assert len(app.image) == 1
    assert len(app.download_button) == 1
    assert not any(button.key == "download_zip" for button in app.download_button)
