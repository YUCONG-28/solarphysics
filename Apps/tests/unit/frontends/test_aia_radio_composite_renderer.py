"""Three-panel renderer and artifact export contracts."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    render_composite_result,
    write_composite_artifacts,
)


def _top_png() -> bytes:
    image = Image.new("RGB", (320, 220), color=(20, 30, 40))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _result() -> tuple[CompositeResult, datetime]:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
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
    multi_result = CompositeResult(
        top_image=result.top_image,
        roi_curve=result.roi_curve,
        spectrum=result.spectrum,
        metadata=result.metadata,
        spectrum_flux_curve=result.spectrum_flux_curve,
        spectrum_flux_curves=(result.spectrum_flux_curve, second),
    )

    artifact = render_composite_result(multi_result, map_time=start, dpi=60)
    exported = pd.read_csv(io.BytesIO(artifact.spectrum_flux_csv))

    assert artifact.metadata["render"]["spectrum_flux_curve_count"] == 2
    assert exported["band_index"].unique().tolist() == [0, 1]
    assert exported["center_mhz"].unique().tolist() == [150.0, 164.0]


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


def test_dynamic_video_uses_real_primary_times_and_skips_incomplete_matches(
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
        if primary_frame.obs_time == primary_times[1]:
            raise application._FrameMatchError("AIA 193 Å", "missing")
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
        assert len(paths) == 3
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
        primary_times[index].replace(tzinfo=UTC) for index in (0, 2, 3)
    ]
    assert video.metadata["frame_count"] == 3
    assert video.metadata["skipped_frame_count"] == 1
    assert video.metadata["skipped_frames"][0]["missing_sources"] == ["AIA 193 Å"]
    assert video.metadata["aia_waves"] == [171, 193]
    assert video.metadata["timeline_mode"] == (
        "primary_radio_observations_no_interpolation"
    )
