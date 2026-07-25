"""Focused contracts for the Radio Composite Figure application layer."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from PIL import Image
from streamlit.testing.v1 import AppTest

from solar_apps.frontends.radio.composite_figure.composite_figure_app import (
    _apply_pending_band,
    _dart_band_overrides,
    _extract_dart_results_by_frequency,
    _invalidate_composite,
    _invalidate_if_controls_changed,
    _pause_sequence_preview,
    prepare_single_panel_render,
)
from solar_apps.frontends.radio.composite_figure.composite_figure_application import (
    COMPOSITE_SCHEMA_VERSION,
    MAP_TIME_COLOR,
    FrequencyBand,
    annotate_source_map_png,
    build_centered_frequency_bands,
    build_composite_artifacts,
    build_composite_frame_template,
    build_composite_figure,
    build_dart_selection_figure,
    build_request_signature,
    frequency_band_from_selection,
    render_cached_composite_frame,
    save_composite_bundle,
    select_dart_time_overlap,
)
from solar_apps.frontends.radio.source_map.artifacts import (
    validate_source_map_artifact,
)
from solar_apps.workflows.radio import source_map_workflow
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandCurve,
    DartNarrowbandResult,
    DartSpectrogramWindow,
    extract_dart_narrowband_lightcurves,
)
from solar_toolkit.radio.roi_lightcurve import RadioRoi, extract_radio_roi_lightcurve


def _source_map_png() -> bytes:
    image = Image.new("RGB", (200, 100), color=(245, 245, 245))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _source_map_metadata() -> dict:
    return {
        "schema_version": 1,
        "image": {"width": 200, "height": 100, "sha256": "synthetic"},
        "display": {"transform": "linear", "cmap": "hot"},
        "panels": [
            {
                "id": "radio-0",
                "bbox_normalized": [0.1, 0.1, 0.9, 0.9],
                "xlim_arcsec": [-10.0, 10.0],
                "ylim_arcsec": [-5.0, 5.0],
            }
        ],
    }


def _roi() -> RadioRoi:
    return RadioRoi.from_box(-5.0, -2.5, 5.0, 2.5, label="Core")


def _radio_frame(start: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": [
                (start + timedelta(seconds=index)).isoformat() for index in range(5)
            ],
            "freq_mhz": [149.0] * 5,
            "polarization": ["RR+LL"] * 5,
            "raw_sum": [10.0, 12.0, 14.0, 13.0, 11.0],
            "bunit": ["K"] * 5,
            "quality_flag": ["ok"] * 5,
        }
    )


def _dart_result(start: datetime) -> DartNarrowbandResult:
    times = tuple(start + timedelta(seconds=index) for index in (1, 2, 3))
    curve = DartNarrowbandCurve(
        center_frequency_mhz=149.0,
        bandwidth_mhz=2.0,
        requested_frequency_range_mhz=(148.0, 150.0),
        sampled_frequency_range_mhz=(148.5, 149.5),
        channel_count=2,
        stokes_i_db=np.asarray([21.0, 22.5, 22.0], dtype=float),
    )
    return DartNarrowbandResult(time_utc=times, curves=(curve,))


def test_frequency_band_selection_and_original_channel_validation() -> None:
    selected = frequency_band_from_selection(
        {"selection": {"box": [{"y0": 150.0, "y1": 148.0}]}}
    )

    assert selected == FrequencyBand(148.0, 150.0)
    assert selected.validate_observed_range([147.0, 148.5, 149.5, 151.0]) is selected
    with pytest.raises(ValueError, match="contains no original frequency channel"):
        FrequencyBand(149.6, 149.9).validate_observed_range(
            [147.0, 148.5, 149.5, 151.0]
        )
    with pytest.raises(ValueError, match="lower bound"):
        FrequencyBand(150.0, 150.0)


def test_centered_dart_bands_follow_all_selected_radio_frequencies() -> None:
    frequencies = [149.0, 164.0, 190.0, 205.0, 223.0, 238.0]
    bands = build_centered_frequency_bands(
        frequencies,
        2.0,
        {190.0: 4.0},
    )

    assert list(bands) == frequencies
    assert all(bands[frequency].center_mhz == frequency for frequency in frequencies)
    assert bands[149.0] == FrequencyBand(148.0, 150.0)
    assert bands[190.0] == FrequencyBand(188.0, 192.0)


def test_dart_spectrum_overlays_every_band_and_highlights_active_frequency() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    frequencies = [149.0, 164.0, 190.0, 205.0, 223.0, 238.0]
    bands = build_centered_frequency_bands(frequencies, 2.0)
    window = DartSpectrogramWindow(
        stokes_i_db=np.ones((7, 3), dtype=float),
        stokes_v_over_i=np.zeros((7, 3), dtype=float),
        frequency_mhz=np.asarray([140.0, *frequencies]),
        time_utc=tuple(start + timedelta(seconds=index) for index in range(3)),
    )

    figure = build_dart_selection_figure(
        window,
        bands=bands,
        active_frequency_mhz=190.0,
    )

    assert len(figure.layout.shapes) == len(frequencies)
    active_shape = next(
        shape
        for shape in figure.layout.shapes
        if float(shape.y0) == pytest.approx(189.0)
    )
    assert active_shape.line.width == 3


def test_dart_override_state_prunes_unselected_frequencies() -> None:
    overrides = _dart_band_overrides(
        '{"149":2.5,"164":3.0,"999":8.0,"bad":"value"}',
        [149.0, 164.0],
    )

    assert overrides == {149.0: 2.5, 164.0: 3.0}


def test_pending_plotly_band_synchronizes_numeric_inputs() -> None:
    class StreamlitStub:
        session_state = {"_pending_dart_band": {"low_mhz": 148.25, "high_mhz": 149.75}}

    st = StreamlitStub()
    _apply_pending_band(st)

    assert st.session_state["dart_band_low"] == 148.25
    assert st.session_state["dart_band_high"] == 149.75
    assert "_pending_dart_band" not in st.session_state


def test_pending_plotly_width_updates_only_the_active_frequency() -> None:
    class StreamlitStub:
        session_state = {
            "dart_band_overrides_json": '{"149":2.0}',
            "_pending_dart_bandwidth": {
                "frequency_mhz": 164.0,
                "bandwidth_mhz": 3.5,
            },
        }

    st = StreamlitStub()
    _apply_pending_band(st)

    assert json.loads(st.session_state["dart_band_overrides_json"]) == {
        "149": 2.0,
        "164": 3.5,
    }
    assert st.session_state["dart_override_enabled_164"] is True
    assert st.session_state["dart_override_width_164"] == 3.5


def test_dart_extraction_batches_equal_widths_and_splits_results(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    calls: list[tuple[tuple[float, ...], float]] = []

    def fake_extractor(directory, centers, bandwidth, *, time_range_utc):
        assert Path(directory) == tmp_path
        calls.append((tuple(float(value) for value in centers), float(bandwidth)))
        times = tuple(start + timedelta(seconds=index) for index in range(2))
        curves = tuple(
            DartNarrowbandCurve(
                center_frequency_mhz=float(center),
                bandwidth_mhz=float(bandwidth),
                requested_frequency_range_mhz=(
                    float(center) - float(bandwidth) / 2.0,
                    float(center) + float(bandwidth) / 2.0,
                ),
                sampled_frequency_range_mhz=(float(center), float(center)),
                channel_count=1,
                stokes_i_db=np.asarray([float(center), float(center) + 1.0]),
            )
            for center in centers
        )
        return DartNarrowbandResult(time_utc=times, curves=curves)

    bands = build_centered_frequency_bands(
        [149.0, 164.0, 190.0],
        2.0,
        {190.0: 4.0},
    )
    split, combined = _extract_dart_results_by_frequency(
        tmp_path,
        bands,
        time_range_utc=(start, start + timedelta(seconds=1)),
        extractor=fake_extractor,
    )

    assert calls == [((149.0, 164.0), 2.0), ((190.0,), 4.0)]
    assert list(split) == [149.0, 164.0, 190.0]
    assert [result.curves[0].center_frequency_mhz for result in split.values()] == [
        149.0,
        164.0,
        190.0,
    ]
    assert [curve.center_frequency_mhz for curve in combined.curves] == [
        149.0,
        164.0,
        190.0,
    ]


def test_dart_overlap_preserves_partial_gaps_and_rejects_no_samples() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    samples = tuple(start + timedelta(seconds=value) for value in (1, 2, 3))

    overlap_start, overlap_end, partial = select_dart_time_overlap(
        samples, start, start + timedelta(seconds=4)
    )

    assert overlap_start == samples[0]
    assert overlap_end == samples[-1]
    assert partial is True
    with pytest.raises(ValueError, match="contains no DART sample"):
        select_dart_time_overlap(
            samples,
            start + timedelta(seconds=10),
            start + timedelta(seconds=20),
        )


def test_roi_is_mapped_onto_full_source_map_png() -> None:
    annotated = annotate_source_map_png(
        _source_map_png(), _source_map_metadata(), _roi()
    )

    with Image.open(io.BytesIO(annotated)) as image:
        pixels = image.convert("RGB")
        assert pixels.getpixel((140, 30)) == (0, 212, 255)
        assert pixels.size == (200, 100)


def test_three_rows_share_utc_axis_and_map_time_line() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    marker = start + timedelta(seconds=2)
    figure = build_composite_figure(
        annotate_source_map_png(_source_map_png(), _source_map_metadata(), _roi()),
        _radio_frame(start),
        _dart_result(start),
        roi=_roi(),
        map_time=marker,
        map_frequency_mhz=149.0,
        polarization="RR+LL",
        time_start=start,
        time_end=start + timedelta(seconds=4),
    )

    assert len(figure.axes) == 3
    radio_axis, dart_axis = figure.axes[1:]
    assert dart_axis.get_shared_x_axes().joined(radio_axis, dart_axis)
    marker_positions = []
    for axis in (radio_axis, dart_axis):
        marker_line = next(
            line for line in axis.lines if line.get_color() == MAP_TIME_COLOR
        )
        assert marker_line.get_linewidth() == pytest.approx(0.9)
        assert marker_line.get_linestyle() == "--"
        marker_positions.append(mdates.date2num(marker_line.get_xdata()[0]))
    assert marker_positions[0] == pytest.approx(marker_positions[1], abs=1e-12)
    assert marker_positions[0] == pytest.approx(mdates.date2num(marker), abs=1e-12)
    assert all(not label.get_visible() for label in radio_axis.get_xticklabels())


def test_artifact_bundle_uses_v1_schema_and_conflict_safe_save(tmp_path: Path) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    bundle = build_composite_artifacts(
        _source_map_png(),
        _source_map_metadata(),
        _radio_frame(start),
        _dart_result(start),
        roi=_roi(),
        map_time=start + timedelta(seconds=2),
        map_frequency_mhz=149.0,
        polarization="RR+LL",
        time_start=start,
        time_end=start + timedelta(seconds=4),
        request_signature="a" * 64,
        source_context={"radio_directory": "synthetic-radio"},
        generated_at=start,
        dpi=100,
    )

    assert bundle.files["composite_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert bundle.files["radio_curve_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert bundle.files["dart_curve_png"].startswith(b"\x89PNG\r\n\x1a\n")
    metadata = json.loads(bundle.files["metadata_json"].decode("utf-8"))
    assert metadata["schema_version"] == COMPOSITE_SCHEMA_VERSION
    assert metadata["radio_curve"]["metric"] == "raw_sum"
    assert metadata["dart_curve"]["representation"] == "source Stokes I dB intensity"
    stem = Path(bundle.filenames["composite_png"]).stem
    assert "radio_composite" in stem
    assert all(
        Path(filename).stem.startswith(stem)
        for key, filename in bundle.filenames.items()
        if key != "composite_png"
    )
    assert bundle.zip_filename == f"{stem}.zip"
    with zipfile.ZipFile(io.BytesIO(bundle.zip_bytes)) as archive:
        assert set(archive.namelist()) == set(bundle.filenames.values())

    first = save_composite_bundle(bundle, tmp_path)
    second = save_composite_bundle(bundle, tmp_path)
    assert first != second
    assert second.name.endswith("_002")
    assert (first / bundle.zip_filename).is_file()


def test_curve_template_is_marker_free_and_cached_frame_markers_align() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    annotated = annotate_source_map_png(
        _source_map_png(), _source_map_metadata(), _roi()
    )
    template = build_composite_frame_template(
        annotated,
        _radio_frame(start),
        _dart_result(start),
        roi=_roi(),
        map_frequency_mhz=149.0,
        polarization="RR+LL",
        time_start=start,
        time_end=start + timedelta(seconds=4),
        dpi=100,
    )
    marker_rgb = np.asarray(
        tuple(int(MAP_TIME_COLOR[index : index + 2], 16) for index in (1, 3, 5)),
        dtype=np.uint8,
    )
    for payload in (template.radio_curve_png, template.dart_curve_png):
        with Image.open(io.BytesIO(payload)) as opened:
            pixels = np.asarray(opened.convert("RGB"))
        assert not np.any(np.all(pixels == marker_rgb, axis=2))

    rendered = render_cached_composite_frame(
        template,
        annotated,
        map_time=start + timedelta(seconds=2),
        include_png=True,
    )

    assert rendered.png_bytes is not None
    assert rendered.marker_x_pixels is not None
    assert (
        rendered.marker_x_pixels["radio_curve"]
        == rendered.marker_x_pixels["dart_curve"]
    )
    with Image.open(io.BytesIO(rendered.png_bytes)) as opened:
        assert np.array_equal(rendered.rgb, np.asarray(opened.convert("RGB")))


def test_request_signature_invalidates_for_controls_and_file_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "radio.fits"
    source.write_bytes(b"first")
    original = build_request_signature({"frequency": 149.0}, source_paths=[source])
    changed_control = build_request_signature(
        {"frequency": 150.0}, source_paths=[source]
    )
    source.write_bytes(b"second-payload")
    changed_file = build_request_signature({"frequency": 149.0}, source_paths=[source])

    assert len({original, changed_control, changed_file}) == 3


def test_control_change_clears_downstream_but_same_controls_preserve_it() -> None:
    class StreamlitStub:
        session_state = {"downstream": "valid"}

    st = StreamlitStub()
    invalidations: list[str] = []

    def invalidate(target) -> None:
        invalidations.append(target.session_state.pop("downstream", "missing"))

    first = _invalidate_if_controls_changed(
        st, "controls", {"frequency": 149.0}, invalidate
    )
    same = _invalidate_if_controls_changed(
        st, "controls", {"frequency": 149.0}, invalidate
    )
    changed = _invalidate_if_controls_changed(
        st, "controls", {"frequency": 164.0}, invalidate
    )

    assert first == same
    assert changed != same
    assert invalidations == ["valid"]


def test_composite_invalidation_clears_curve_template_cache() -> None:
    class StreamlitStub:
        session_state = {
            "curve_plot_cache_by_frequency": {149.0: object()},
            "sequence_bundle": object(),
            "composite_bundle": object(),
        }

    st = StreamlitStub()
    _invalidate_composite(st)

    assert "curve_plot_cache_by_frequency" not in st.session_state
    assert "sequence_bundle" not in st.session_state


def test_prepare_single_panel_render_handles_linear_and_log_candidates(
    tmp_path: Path,
) -> None:
    candidate = {
        "id": "frame-1",
        "mode": "multi_band",
        "frequencies_mhz": [149.0, 164.0],
        "slot": [["rr-149.fits", "ll-149.fits"], ["rr-164.fits", "ll-164.fits"]],
        "observation_time": "2025-01-24T04:48:32+00:00",
    }
    config = {"spatial_display": {"cmap": "hot"}}

    linear_config, linear_candidate = prepare_single_panel_render(
        config,
        candidate,
        164.0,
        transform="linear",
        output_directory=tmp_path,
    )
    log_config, log_candidate = prepare_single_panel_render(
        config,
        candidate,
        149.0,
        transform="log10",
        output_directory=tmp_path,
    )

    assert linear_config["enable_spectrogram_panel"] is False
    assert linear_config["spatial_display"]["transform"] == "linear"
    assert linear_candidate["paths"] == ["rr-164.fits", "ll-164.fits"]
    assert log_config["mode"] == "multi_band"
    assert log_config["multi_band_freqs"] == [149.0]
    assert log_candidate["slot"] == [["rr-149.fits", "ll-149.fits"]]


def test_synthetic_fits_end_to_end_uses_public_radio_and_dart_algorithms(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    radio_dir = tmp_path / "radio" / "149MHz" / "RR"
    radio_dir.mkdir(parents=True)
    radio_paths: list[Path] = []
    for index, value in enumerate((2.0, 3.0, 4.0)):
        path = radio_dir / f"20250124T04483{index}_RR.fits"
        _write_radio_fits(path, value=value, observed=start + timedelta(seconds=index))
        radio_paths.append(path)

    dart_dir = tmp_path / "dart"
    _write_dart_fits(dart_dir)
    roi = RadioRoi.from_box(-35.0, -35.0, 35.0, 35.0, label="Synthetic")
    radio = extract_radio_roi_lightcurve(
        tmp_path / "radio",
        roi,
        files=radio_paths,
        freqs=[149.0],
        polarization="RR",
    )
    dart = extract_dart_narrowband_lightcurves(
        dart_dir,
        [149.0],
        2.0,
        time_range_utc=(start, start + timedelta(seconds=2)),
    )

    assert radio["quality_flag"].tolist() == ["ok", "ok", "ok"]
    np.testing.assert_allclose(
        radio["raw_sum"] / radio["valid_pixel_count"], [2.0, 3.0, 4.0]
    )
    np.testing.assert_allclose(dart.curves[0].stokes_i_db, [20.0, 21.0, 22.0])

    config = dict(source_map_workflow.DEFAULT_CONFIG)
    config.update(
        {
            "enable_gaussian_overlay": False,
            "enable_spectrogram_panel": False,
            "enable_raw_quality_filter": False,
            "radio_background_strategy": "off",
            "radio_background_force_off": True,
            "enable_radio_background_subtraction": False,
            "save_background_diagnostics": False,
            "show_plot": False,
            "save_plot": True,
            "write_source_map_sidecar": True,
            "fig_size": (5, 4),
            "dpi": 80,
            "analysis_subdir": "source_maps",
            "polarization": "RR",
            "combine_polarizations": False,
        }
    )
    map_output = source_map_workflow.plot_single_band(
        str(radio_paths[1]), str(tmp_path / "map-output"), config, write_sidecar=True
    )
    map_metadata = validate_source_map_artifact(map_output)
    bundle = build_composite_artifacts(
        Path(map_output).read_bytes(),
        map_metadata,
        radio,
        dart,
        roi=roi,
        map_time=start + timedelta(seconds=1),
        map_frequency_mhz=149.0,
        polarization="RR",
        time_start=start,
        time_end=start + timedelta(seconds=2),
        request_signature="b" * 64,
        generated_at=start,
        dpi=100,
    )

    assert bundle.files["composite_png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(pd.read_csv(io.BytesIO(bundle.files["radio_csv"]))) == 3
    assert len(pd.read_csv(io.BytesIO(bundle.files["dart_csv"]))) == 3


def test_streamlit_page_loads_in_auto_light_and_dark_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOLAR_APPS_ALLOWED_ROOTS", str(tmp_path))
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(tmp_path / "Local"))
    app_path = (
        Path(__file__).resolve().parents[3]
        / "solar_apps"
        / "frontends"
        / "radio"
        / "composite_figure"
        / "composite_figure_app.py"
    )
    app = AppTest.from_file(str(app_path), default_timeout=40).run()

    assert len(app.exception) == 0
    assert app.title[0].value == "Radio Composite Figure"
    theme = app.selectbox(key="radio-composite_theme_mode")
    for mode in ("auto", "light", "dark"):
        theme.set_value(mode).run()
        assert len(app.exception) == 0


def test_sequence_preview_pause_callback_updates_state_before_widget_render() -> None:
    class FakeStreamlit:
        session_state = {"sequence_preview_playing": True}

    fake = FakeStreamlit()
    _pause_sequence_preview(fake)

    assert fake.session_state["sequence_preview_playing"] is False


def _write_radio_fits(path: Path, *, value: float, observed: datetime) -> None:
    data = np.full((24, 24), float(value), dtype=np.float32)
    header = fits.Header()
    header["DATE-OBS"] = observed.replace(tzinfo=None).isoformat()
    header["FREQ"] = 149.0
    header["POLAR"] = "RR"
    header["BUNIT"] = "K"
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CRPIX1"] = 12.5
    header["CRPIX2"] = 12.5
    header["CDELT1"] = 10.0
    header["CDELT2"] = 10.0
    header["RSUN_OBS"] = 960.0
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def _write_dart_fits(folder: Path) -> None:
    folder.mkdir(parents=True)
    frequency = np.asarray([147.0, 148.0, 149.0, 150.0, 151.0])
    sample_index = np.arange(3, dtype=float)
    stokes_i = frequency[:, None] - 129.0 + sample_index[None, :]
    time_rows = np.asarray(
        [[25.0, 1.0, 24.0, 4.0, 48.0, 30.0 + index] for index in sample_index]
    )
    payloads = {
        "synthetic_SpecDataIdB.fits": stokes_i,
        "synthetic_SpecDataVP.fits": stokes_i / 100.0,
        "synthetic_SpecFrequency.fits": frequency[None, :],
        "synthetic_SpecTime.fits": time_rows,
    }
    for name, payload in payloads.items():
        fits.PrimaryHDU(data=np.asarray(payload, dtype=np.float64)).writeto(
            folder / name
        )
