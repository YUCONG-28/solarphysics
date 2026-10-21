"""Three-panel renderer and artifact export contracts."""

from __future__ import annotations

import io
import json
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from solar_toolkit.aia.background import AiaBackground

from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite.adapters import AiaSelection
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumTimeAlignment,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    render_composite_result,
    render_top_panel,
    write_composite_artifacts,
)


def _top_png() -> bytes:
    image = Image.new("RGB", (320, 220), color=(20, 30, 40))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_expanded_gaussian_evaluates_native_model_through_inverse_wcs():
    from astropy.wcs import WCS
    from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d

    from solar_apps.frontends.radio.aia_radio_composite.rendering.composite_renderer import (
        _expanded_gaussian_model,
    )

    header = {
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRPIX1": 8,
        "CRPIX2": 8,
        "CRVAL1": 0.01,
        "CRVAL2": -0.01,
        "CD1_1": 0.001,
        "CD1_2": -0.002,
        "CD2_1": 0.002,
        "CD2_2": 0.001,
    }
    result = SimpleNamespace(
        amplitude=12,
        center_pixel=(8.2, 6.8),
        sigma_pixel=(2.1, 1.4),
        theta_rad=0.3,
        coordinate_header=header,
    )
    x, y, model = _expanded_gaussian_model(
        result,
        (-100, 100, -100, 100),
        (16, 16),
        "lower",
        (-80, 150, -150, 80),
        samples_per_axis=80,
    )
    xp, yp = WCS(header).world_to_pixel_values(x / 3600, y / 3600)
    expected = elliptical_gaussian_2d(
        (xp, yp),
        12,
        8.2,
        6.8,
        2.1,
        1.4,
        0.3,
    )
    np.testing.assert_allclose(model, expected, rtol=1e-12, atol=1e-12)


def _result() -> tuple[CompositeResult, datetime]:
    start = datetime(2000, 1, 1, 4, 48, 30, tzinfo=UTC)
    curve = pd.DataFrame(
        {
            "time": [
                start.isoformat(),
                (start + timedelta(seconds=2)).isoformat(),
                (start + timedelta(seconds=4)).isoformat(),
            ],
            "frequency": [149.0, 149.0, 164.0],
            "polarization": ["RR", "RR", "RR"],
            "raw_sum": [10.0, 12.0, 20.0],
            "raw_mean": [1.0, 1.2, 2.0],
            "raw_peak": [3.0, 3.2, 4.0],
            "quality_flag": ["ok", "low_coverage", "ok"],
            "quality_detail": ["", "synthetic", ""],
        }
    )
    spectrum = SpectrumWindow(
        data=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=tuple(start + timedelta(seconds=value) for value in (1, 2, 3)),
        polarization="Stokes I",
        unit="dB",
        source="DART",
        metadata={"reader": "mock"},
    )
    spectrum_flux = SpectrumFluxCurve(
        time_utc=tuple(start + timedelta(seconds=value) for value in (1, 3)),
        values=np.asarray([100.0, 120.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(140.0, 160.0),
        sampled_frequency_range_mhz=(141.0, 159.0),
        channel_count=5,
    )
    return (
        CompositeResult(
            top_image=_top_png(),
            roi_curve=curve,
            spectrum=spectrum,
            metadata={"map_time_utc": (start + timedelta(seconds=2)).isoformat()},
            spectrum_flux_curve=spectrum_flux,
        ),
        start,
    )


def test_renderer_generates_three_panel_png_json_and_csv() -> None:
    result, start = _result()
    marker = start + timedelta(seconds=2)

    artifact = render_composite_result(
        result,
        map_time=marker,
        metric="raw_sum",
        dpi=80,
    )

    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(artifact.image_png)) as image:
        assert image.width > 500
        assert image.height > image.width
    metadata = json.loads(artifact.metadata_json)
    assert metadata["product"] == "aia-radio-composite"
    assert metadata["render"]["metric"] == "raw_sum"
    assert metadata["render"]["map_time_utc"] == marker.isoformat()
    assert metadata["render"]["shared_time_range_utc"] == [
        start.isoformat(),
        (start + timedelta(seconds=4)).isoformat(),
    ]
    assert metadata["render"]["quality_flagged_rows"] == 1
    assert metadata["render"]["dual_axis_flux"] is True
    assert metadata["render"]["main_axis_x_alignment"] is True
    flux_bbox = metadata["render"]["main_axis_bbox_normalized"]["flux"]
    spectrum_bbox = metadata["render"]["main_axis_bbox_normalized"]["spectrum"]
    assert flux_bbox[0] == pytest.approx(spectrum_bbox[0], abs=1e-12)
    assert flux_bbox[2] == pytest.approx(spectrum_bbox[2], abs=1e-12)
    assert metadata["result"]["spectrum_flux_curve"]["channel_count"] == 5
    exported_curve = pd.read_csv(
        io.BytesIO(artifact.roi_curve_csv),
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(exported_curve, result.roi_curve)
    exported_spectrum = pd.read_csv(io.BytesIO(artifact.spectrum_flux_csv))
    assert exported_spectrum["value"].tolist() == [100.0, 120.0]
    assert exported_spectrum["unit"].unique().tolist() == ["dB"]


def test_renderer_supports_all_raw_metrics_without_derived_values() -> None:
    result, start = _result()

    for metric in ("raw_sum", "raw_mean", "raw_peak"):
        artifact = render_composite_result(
            result,
            map_time=start,
            metric=metric,
            dpi=60,
        )
        assert artifact.metadata["render"]["metric"] == metric


def test_renderer_applies_spectrum_display_ranges_without_changing_data() -> None:
    result, start = _result()
    original_data = result.spectrum.data.copy()
    result.metadata["spectrum_display_frequency_range_mhz"] = [120.0, 180.0]
    result.metadata["spectrum_display_intensity_range"] = [2.0, 5.0]

    artifact = render_composite_result(result, map_time=start, dpi=60)

    render = artifact.metadata["render"]
    assert render["spectrum_display_frequency_range_mhz"] == [120.0, 180.0]
    assert render["spectrum_display_intensity_range"] == [2.0, 5.0]
    np.testing.assert_array_equal(result.spectrum.data, original_data)


def test_renderer_aligns_dart_display_but_exports_original_flux_utc() -> None:
    result, start = _result()
    original_first = result.spectrum_flux_curve.time_utc[0]
    reference = start + timedelta(seconds=1, microseconds=-125)
    alignment = SpectrumTimeAlignment(
        reference_radio_time_utc=reference,
        nearest_spectrum_time_utc=original_first,
        display_offset_seconds=-0.000125,
        native_cadence_seconds=1.0,
        nearest_delta_seconds=0.000125,
    )
    result.metadata["spectrum_time_alignment"] = alignment.to_dict()

    artifact = render_composite_result(
        result,
        map_time=reference,
        dpi=60,
    )

    render = artifact.metadata["render"]
    assert render["time_alignment"] == "shared_utc_dart_display_offset_no_interpolation"
    assert render["spectrum_time_alignment"] == alignment.to_dict()
    exported = pd.read_csv(io.BytesIO(artifact.spectrum_flux_csv))
    assert exported.loc[0, "time_utc"] == original_first.isoformat()
    assert result.spectrum_flux_curve.time_utc[0] == original_first
    assert result.spectrum.frequency_mhz.tolist() == [100.0, 200.0]


def test_renderer_retains_but_does_not_plot_untimed_quality_row() -> None:
    result, start = _result()
    result.roi_curve.loc[1, "time"] = ""

    artifact = render_composite_result(result, map_time=start, dpi=60)

    assert artifact.metadata["render"]["untimed_quality_rows"] == 1
    assert b"low_coverage" in artifact.roi_curve_csv


def test_artifact_writer_is_path_based_and_conflict_safe(tmp_path: Path) -> None:
    result, start = _result()
    artifact = render_composite_result(result, map_time=start, dpi=60)

    first = write_composite_artifacts(artifact, tmp_path)
    second = write_composite_artifacts(artifact, tmp_path)

    assert {path.suffix for path in first.values()} == {".png", ".json", ".csv"}
    assert all(path.is_file() for path in first.values())
    assert first["png"].stem == "aia_radio_composite"
    assert second["png"].stem == "aia_radio_composite_002"
    assert json.loads(first["json"].read_text(encoding="utf-8"))["schema_version"] == 3
    assert first["csv"].read_bytes() == artifact.roi_curve_csv
    assert first["spectrum_csv"].read_bytes() == artifact.spectrum_flux_csv


def test_renderer_exports_multiple_matched_spectrum_bands() -> None:
    result, start = _result()
    second = SpectrumFluxCurve(
        time_utc=tuple(start + timedelta(seconds=value) for value in (1, 3)),
        values=np.asarray([200.0, 220.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(163.0, 165.0),
        sampled_frequency_range_mhz=(163.1, 164.9),
        channel_count=4,
    )
    alignments = {
        150.0: SpectrumTimeAlignment(
            reference_radio_time_utc=start + timedelta(seconds=1),
            nearest_spectrum_time_utc=start + timedelta(seconds=1),
            display_offset_seconds=0.0,
            native_cadence_seconds=1.0,
            nearest_delta_seconds=0.0,
        ),
        164.0: SpectrumTimeAlignment(
            reference_radio_time_utc=start + timedelta(seconds=1, microseconds=100000),
            nearest_spectrum_time_utc=start + timedelta(seconds=1),
            display_offset_seconds=0.1,
            native_cadence_seconds=1.0,
            nearest_delta_seconds=0.1,
        ),
    }
    multi_result = CompositeResult(
        top_image=result.top_image,
        roi_curve=result.roi_curve,
        spectrum=result.spectrum,
        metadata={
            **result.metadata,
            "spectrum_flux_time_alignments": {
                f"{frequency:g}": alignment.to_dict()
                for frequency, alignment in alignments.items()
            },
        },
        spectrum_flux_curve=result.spectrum_flux_curve,
        spectrum_flux_curves=(result.spectrum_flux_curve, second),
    )

    artifact = render_composite_result(multi_result, map_time=start, dpi=60)
    exported = pd.read_csv(io.BytesIO(artifact.spectrum_flux_csv))

    assert artifact.metadata["render"]["spectrum_flux_curve_count"] == 2
    assert artifact.metadata["render"]["spectrum_flux_time_alignments"] == {
        "150": alignments[150.0].to_dict(),
        "164": alignments[164.0].to_dict(),
    }
    assert exported["band_index"].unique().tolist() == [0, 1]
    assert exported["center_mhz"].unique().tolist() == [150.0, 164.0]
    assert (
        exported.loc[exported["center_mhz"].eq(164.0), "time_utc"].iloc[0]
        == (start + timedelta(seconds=1)).isoformat()
    )


def test_renderer_separates_flux_axes_for_video_frame_layout() -> None:
    result, start = _result()
    first = SpectrumFluxCurve(
        time_utc=tuple(start + timedelta(seconds=value) for value in (1, 3)),
        values=np.asarray([100.0, 120.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(148.0, 150.0),
        sampled_frequency_range_mhz=(148.5, 149.5),
        channel_count=4,
    )
    second = SpectrumFluxCurve(
        time_utc=tuple(start + timedelta(seconds=value) for value in (1, 3)),
        values=np.asarray([200.0, 220.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(163.0, 165.0),
        sampled_frequency_range_mhz=(163.5, 164.5),
        channel_count=4,
    )
    separate_result = CompositeResult(
        top_image=result.top_image,
        roi_curve=result.roi_curve,
        spectrum=result.spectrum,
        metadata={**result.metadata, "flux_plot_layout": "separate"},
        spectrum_flux_curve=first,
        spectrum_flux_curves=(first, second),
    )

    artifact = render_composite_result(separate_result, map_time=start, dpi=60)

    render = artifact.metadata["render"]
    assert render["flux_plot_layout"] == "separate"
    assert render["flux_axis_count"] == 2
    assert len(render["main_axis_bbox_normalized"]["fluxes"]) == 2
    assert render["main_axis_x_alignment"] is True
    assert render["figure_size_inches"] == [11.0, 16.2]


def test_video_export_animates_shared_marker_without_data_interpolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, start = _result()
    markers: list[datetime] = []

    def fake_render(
        value: CompositeResult,
        *,
        map_time: datetime,
        **_kwargs: object,
    ):
        assert value is result
        markers.append(map_time)
        return type("Frame", (), {"image_png": _top_png()})()

    def fake_encode(
        paths: list[str],
        output_path: str,
        **_kwargs: object,
    ) -> bool:
        assert len(paths) == 4
        Path(output_path).write_bytes(b"mock-mp4")
        return True

    monkeypatch.setattr(application, "render_composite_result", fake_render)
    monkeypatch.setattr(application, "write_video_from_paths", fake_encode)

    video = application.build_composite_video(
        result,
        time_start=start,
        time_end=start + timedelta(seconds=3),
        fps=2,
        frame_count=4,
        dpi=60,
    )

    assert video.video_mp4 == b"mock-mp4"
    assert len(markers) == 4
    assert markers[0] == start
    assert markers[-1] == start + timedelta(seconds=3)
    assert video.metadata["timeline_mode"] == (
        "shared_utc_marker_no_data_interpolation"
    )
    assert video.metadata["top_panel_mode"] == "fixed_reference_frame"


def test_renderer_uses_explicit_display_window_without_sample_expansion() -> None:
    result, start = _result()
    display_start = start + timedelta(seconds=1)
    display_end = start + timedelta(seconds=3)

    artifact = render_composite_result(
        result,
        map_time=start + timedelta(seconds=2),
        display_time_range_utc=(display_start, display_end),
        dpi=60,
    )

    assert artifact.metadata["render"]["shared_time_range_utc"] == [
        display_start.isoformat(),
        display_end.isoformat(),
    ]


def test_renderer_preserves_a_display_window_three_nanoseconds_wide() -> None:
    result, _ = _result()
    start = "2025-01-24T04:48:30.000000001Z"
    marker = "2025-01-24T04:48:30.000000002Z"
    end = "2025-01-24T04:48:30.000000004Z"

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        artifact = render_composite_result(
            result,
            map_time=marker,
            display_time_range_utc=(start, end),
            dpi=40,
        )

    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert not any(
        "identical low and high xlims" in str(item.message) for item in recorded
    )
    bounds = [
        pd.Timestamp(value)
        for value in artifact.metadata["render"]["shared_time_range_utc"]
    ]
    assert bounds[1].value - bounds[0].value == 3
    assert (
        pd.Timestamp(artifact.metadata["render"]["map_time_utc"]).value
        - bounds[0].value
        == 1
    )
    assert artifact.metadata["render"]["axis_time_padding_applied"] is True
    axis = [
        pd.Timestamp(value)
        for value in artifact.metadata["render"]["axis_display_time_range_utc"]
    ]
    assert axis[0] < bounds[0] < bounds[1] < axis[1]


def test_top_panel_draws_radio_match_quality_and_actual_utc(monkeypatch) -> None:
    from matplotlib.text import Text

    observed = pd.Timestamp("2025-01-24T04:48:30.000000007Z")
    background = AiaBackground(
        "aia.fits",
        np.ones((3, 3)),
        np.array([-1.0, 0.0, 1.0]),
        np.array([-1.0, 0.0, 1.0]),
        "AIA 171",
        observed,
        "171",
    )
    aia = AiaSelection(background, observed, observed, 0.0, 1)
    actual = "2025-01-24T04:48:28.750000007"
    matching = {
        "master_time_utc": observed.isoformat(),
        "sources": {
            "radio:149": {
                "actual_time_utc": actual,
                "delta_ns": -1_250_000_000,
                "within_tolerance": False,
                "reused": True,
                "status": "outside_tolerance",
            },
            "radio:164": {
                "actual_time_utc": None,
                "delta_ns": None,
                "within_tolerance": None,
                "reused": False,
                "status": "missing",
            },
        },
    }
    drawn = []
    original_draw = Text.draw

    def capture_draw(self, renderer):
        if self.get_visible():
            drawn.append(self.get_text())
        return original_draw(self, renderer)

    monkeypatch.setattr(Text, "draw", capture_draw)
    artifact = render_top_panel(
        aia,
        (),
        allow_missing_radio=True,
        time_match=matching,
        anchor_time_utc=observed,
        dpi=40,
    )
    labels = "\n".join(drawn)
    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.metadata["time_match"] == matching
    assert actual in labels
    assert "149" in labels and "-1.250" in labels
    assert "164" in labels and ("NO RADIO DATA" in labels or "MISSING" in labels)
    assert "OVER" in labels or "OUTSIDE" in labels
    assert "reus" in labels.lower()


def test_dynamic_video_preserves_real_times_with_empty_secondary_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, start = _result()
    result.metadata["flux_plot_layout"] = "separate"
    request = application.CompositeRequest(
        aia_directory=tmp_path / "aia",
        aia_wave=171,
        aia_time=start,
        radio_directory=tmp_path / "radio",
        radio_frequency=149.0,
        polarization="RR",
        roi_type="box",
        roi_vertices_arcsec=(
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        ),
        spectrum_type="dart",
        spectrum_path=tmp_path / "dart",
    )
    primary_times = [
        start.replace(tzinfo=None) + timedelta(seconds=value)
        for value in (0.0, 0.7, 1.9, 3.4)
    ]
    candidates = tuple(SimpleNamespace(obs_time=value) for value in primary_times)
    rendered_times: list[datetime] = []

    monkeypatch.setattr(
        application,
        "load_radio_candidates",
        lambda _request, **_kwargs: candidates,
    )
    monkeypatch.setattr(
        application,
        "scan_aia_catalog",
        lambda _request: pd.DataFrame(),
    )

    def fake_top(_request, *, primary_frame, **_kwargs):
        assert _kwargs["plan_row"].selections["aia:193"].frame is None
        return SimpleNamespace(
            image_png=_top_png(),
            metadata={"anchor": primary_frame.obs_time.isoformat()},
        )

    def fake_render(_result, *, map_time, **kwargs):
        assert _result.metadata["flux_plot_layout"] == "separate"
        assert kwargs["display_time_range_utc"] == (
            start,
            start + timedelta(seconds=4),
        )
        rendered_times.append(map_time)
        return SimpleNamespace(image_png=_top_png())

    def fake_encode(paths: list[str], output_path: str, **_kwargs) -> bool:
        assert len(paths) == 4
        Path(output_path).write_bytes(b"dynamic-mp4")
        return True

    monkeypatch.setattr(application, "_build_matched_top_panel", fake_top)
    monkeypatch.setattr(application, "render_composite_result", fake_render)
    monkeypatch.setattr(application, "write_video_from_paths", fake_encode)

    video = application.build_dynamic_composite_video(
        result,
        request,
        aia_waves=(171, 193),
        radio_frequencies_mhz=(149.0,),
        time_start=start,
        time_end=start + timedelta(seconds=4),
        fps=5,
    )

    assert rendered_times == [
        primary_times[index].replace(tzinfo=UTC) for index in (0, 1, 2, 3)
    ]
    assert video.metadata["frame_count"] == 4
    assert video.metadata["skipped_frame_count"] == 0
    assert (
        video.metadata["match_plan_summary"]["sources"]["aia:193"]["missing_count"] == 4
    )
    assert video.metadata["aia_waves"] == [171, 193]
    assert video.metadata["timeline_mode"] == (
        "monotone_full_coverage_no_interpolation"
    )
