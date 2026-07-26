"""Multi-frequency ROI lightcurve adapter and Plotly contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    DEFAULT_ROI_FREQUENCIES_MHZ,
    extract_multi_frequency_roi_curve,
)
from solar_apps.frontends.radio.aia_radio_composite.adapters import radio_adapter
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeRequest,
    ROI_CURVE_COLUMNS,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumTimeAlignment,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    build_dual_flux_figure,
    build_dual_flux_figures,
    build_roi_lightcurve_figure,
    spectrum_band_from_selection,
)
from solar_toolkit.radio.roi_lightcurve import RadioRoi


def _request(tmp_path: Path, *, polarization: str = "RR+LL") -> CompositeRequest:
    return CompositeRequest(
        aia_directory=tmp_path / "aia",
        aia_wave=171,
        aia_time=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        radio_directory=tmp_path / "radio",
        radio_frequency=149.0,
        polarization=polarization,
        roi_type="box",
        roi_vertices_arcsec=(
            (-20.0, -20.0),
            (20.0, -20.0),
            (20.0, 20.0),
            (-20.0, 20.0),
        ),
        spectrum_type="dart",
        spectrum_path=tmp_path / "dart",
    )


def _roi() -> RadioRoi:
    return RadioRoi.from_box(-20.0, -10.0, 30.0, 40.0, label="Burst")


def _extractor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": [
                "2025-01-24T04:48:30Z",
                "2025-01-24T04:48:31Z",
                "2025-01-24T04:48:32Z",
            ],
            "freq_mhz": [149.0, 164.0, 164.0],
            "polarization": ["RR+LL", "RR+LL", "RR+LL"],
            "raw_sum": [10.25, 20.5, 30.75],
            "raw_mean": [1.25, 2.5, 3.75],
            "raw_peak": [4.25, 5.5, 6.75],
            "quality_flag": ["ok", "ok", "low_coverage"],
            "quality_detail": ["", "", "coverage below threshold"],
            "filepath": ["/data/a.fits", "/data/b.fits", "/data/c.fits"],
        }
    )


def test_adapter_delegates_to_canonical_multi_frequency_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontend maps columns but never recomputes raw ROI statistics."""

    source = _extractor_frame()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_extract(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((args, kwargs))
        return source

    monkeypatch.setattr(
        radio_adapter,
        "extract_radio_roi_lightcurve",
        fake_extract,
    )
    request = _request(tmp_path)
    roi = _roi()

    curve = extract_multi_frequency_roi_curve(request, roi)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (request.radio_directory, roi)
    assert kwargs["freqs"] == list(DEFAULT_ROI_FREQUENCIES_MHZ)
    assert kwargs["polarization"] == "RR+LL"
    assert list(curve.columns[: len(ROI_CURVE_COLUMNS)]) == list(ROI_CURVE_COLUMNS)
    assert curve["time"].tolist() == source["obs_time"].tolist()
    assert curve["frequency"].tolist() == source["freq_mhz"].tolist()
    pd.testing.assert_series_equal(
        curve["raw_sum"],
        source["raw_sum"],
        check_names=False,
    )
    assert "filepath" in curve
    assert "obs_time" in curve
    assert curve.attrs["scientific_extractor"].endswith("extract_radio_roi_lightcurve")


def test_adapter_accepts_custom_unique_frequencies_and_time_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_extract(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return _extractor_frame()

    monkeypatch.setattr(
        radio_adapter,
        "extract_radio_roi_lightcurve",
        fake_extract,
    )
    request = _request(tmp_path, polarization="LL")

    application.build_roi_lightcurve(
        request,
        _roi(),
        frequencies_mhz=[205, 149, 205],
        time_start="2025-01-24T04:48:00Z",
        time_end="2025-01-24T04:49:00Z",
    )

    assert captured["freqs"] == [205.0, 149.0]
    assert captured["polarization"] == "LL"
    assert captured["time_start"] == "2025-01-24T04:48:00Z"
    assert captured["time_end"] == "2025-01-24T04:49:00Z"


@pytest.mark.parametrize("frequencies", [[], [0.0], [np.nan], "149"])
def test_adapter_rejects_invalid_frequency_selection(
    tmp_path: Path,
    frequencies: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_multi_frequency_roi_curve(
            _request(tmp_path),
            _roi(),
            frequencies_mhz=frequencies,
        )


def test_adapter_rejects_incomplete_extractor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = _extractor_frame().drop(columns=["quality_flag"])
    monkeypatch.setattr(
        radio_adapter,
        "extract_radio_roi_lightcurve",
        lambda *args, **kwargs: incomplete,
    )

    with pytest.raises(ValueError, match="quality_flag"):
        extract_multi_frequency_roi_curve(_request(tmp_path), _roi())


def test_plotly_curve_groups_frequencies_and_marks_quality_flags() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )

    figure = build_roi_lightcurve_figure(
        curve,
        metric="raw_peak",
        theme_mode="dark",
    )

    assert figure.layout.xaxis.title.text == "Time (UTC)"
    assert figure.layout.yaxis.title.text == "raw_peak"
    assert figure.layout.paper_bgcolor == "#0b1120"
    assert [trace.name for trace in figure.data] == [
        "149 MHz | RR+LL",
        "164 MHz | RR+LL",
        "164 MHz | RR+LL (quality flagged)",
    ]
    assert list(figure.data[0].y) == [4.25]
    assert list(figure.data[1].y) == [5.5]
    assert list(figure.data[2].y) == [6.75]
    assert figure.data[2].marker.symbol == "x"
    assert figure.layout.meta["row_count"] == 3


def test_plotly_curve_reports_flagged_rows_without_inventing_values() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    curve.loc[2, "raw_sum"] = np.nan

    figure = build_roi_lightcurve_figure(curve)

    assert len(figure.data) == 2
    assert figure.layout.meta["quality_flagged_without_numeric_value"] == 1


def test_plotly_curve_omits_untimed_quality_rows_without_failing() -> None:
    """Reader diagnostics with no DATE-OBS stay in data but not on the x-axis."""

    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    curve.loc[2, "time"] = ""
    curve.loc[2, "raw_sum"] = np.nan

    figure = build_roi_lightcurve_figure(curve)

    assert figure.layout.meta["row_count"] == 3
    assert figure.layout.meta["plotted_row_count"] == 2
    assert figure.layout.meta["untimed_quality_rows"] == 1


def test_plotly_curve_rejects_untimed_ok_quality_rows() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    curve.loc[0, "time"] = ""

    with pytest.raises(ValueError, match="ok-quality"):
        build_roi_lightcurve_figure(curve)


def test_plotly_curve_rejects_derived_or_unknown_metrics() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )

    with pytest.raises(ValueError, match="raw_sum"):
        build_roi_lightcurve_figure(curve, metric="normalized_flux")


def test_dual_flux_plot_keeps_original_times_and_assigns_secondary_axis() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    spectrum_flux = SpectrumFluxCurve(
        time_utc=(
            datetime(2025, 1, 24, 4, 48, 29, tzinfo=UTC),
            datetime(2025, 1, 24, 4, 48, 35, tzinfo=UTC),
        ),
        values=np.asarray([100.0, np.nan]),
        source="CSO",
        polarization="RR",
        unit="sfu",
        requested_band=SpectrumBand(145.0, 155.0),
        sampled_frequency_range_mhz=(146.0, 154.0),
        channel_count=4,
    )

    figure = build_dual_flux_figure(curve, spectrum_flux, metric="raw_mean")

    spectrum_trace = figure.data[-1]
    assert spectrum_trace.yaxis == "y2"
    assert list(spectrum_trace.x) == list(spectrum_flux.time_utc)
    assert len(spectrum_trace.y) == 2
    assert spectrum_trace.connectgaps is False
    assert figure.layout.yaxis.title.text == "raw_mean"
    assert figure.layout.yaxis2.title.text == "CSO RR mean (sfu)"
    assert figure.layout.xaxis.range[0] == spectrum_flux.time_utc[0]
    assert figure.layout.xaxis.range[1] == spectrum_flux.time_utc[-1]
    assert figure.layout.meta["time_alignment"] == "shared_utc_no_interpolation"


def test_dual_flux_plot_uses_requested_shared_range_and_reference_line() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    spectrum_flux = SpectrumFluxCurve(
        time_utc=(
            datetime(2025, 1, 24, 4, 48, 31, tzinfo=UTC),
            datetime(2025, 1, 24, 4, 48, 34, tzinfo=UTC),
        ),
        values=np.asarray([10.0, 20.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(145.0, 155.0),
        sampled_frequency_range_mhz=(146.0, 154.0),
        channel_count=4,
    )
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    end = datetime(2025, 1, 24, 4, 49, 0, tzinfo=UTC)
    marker = datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC)

    figure = build_dual_flux_figure(
        curve,
        spectrum_flux,
        map_time=marker,
        display_time_range_utc=(start, end),
    )

    assert figure.layout.xaxis.range == (start, end)
    reference_lines = [
        shape
        for shape in figure.layout.shapes
        if shape.type == "line" and shape.x0 == shape.x1
    ]
    assert len(reference_lines) == 1
    assert reference_lines[0].line.dash == "dash"


def test_dual_flux_plot_applies_dart_display_offset_without_changing_values() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    original_times = (
        datetime(2025, 1, 24, 4, 48, 30, 312134, tzinfo=UTC),
        datetime(2025, 1, 24, 4, 48, 30, 713508, tzinfo=UTC),
    )
    spectrum_flux = SpectrumFluxCurve(
        time_utc=original_times,
        values=np.asarray([10.0, np.nan]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(148.0, 150.0),
        sampled_frequency_range_mhz=(148.1, 149.9),
        channel_count=8,
    )
    reference = datetime(2025, 1, 24, 4, 48, 30, 312000, tzinfo=UTC)
    alignment = SpectrumTimeAlignment(
        reference_radio_time_utc=reference,
        nearest_spectrum_time_utc=original_times[0],
        display_offset_seconds=-0.000134,
        native_cadence_seconds=0.401374,
        nearest_delta_seconds=0.000134,
    )

    figure = build_dual_flux_figure(
        curve,
        spectrum_flux,
        map_time=reference,
        time_alignment=alignment,
    )

    spectrum_trace = figure.data[-1]
    assert spectrum_trace.x[0] == reference
    np.testing.assert_allclose(
        np.asarray(spectrum_trace.y, dtype=float),
        np.asarray([10.0, np.nan]),
        equal_nan=True,
    )
    assert spectrum_flux.time_utc == original_times
    assert (
        figure.layout.meta["time_alignment"]
        == "shared_utc_dart_display_offset_no_interpolation"
    )
    assert figure.layout.meta["spectrum_time_alignment"] == alignment.to_dict()


def test_dual_flux_plot_adds_one_secondary_trace_per_matched_roi_frequency() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    start = datetime(2025, 1, 24, 4, 48, 29, tzinfo=UTC)
    fluxes = tuple(
        SpectrumFluxCurve(
            time_utc=(start, start + timedelta(seconds=6)),
            values=np.asarray([center, center + 1.0]),
            source="DART",
            polarization="Stokes I",
            unit="dB",
            requested_band=SpectrumBand(center - 1.0, center + 1.0),
            sampled_frequency_range_mhz=(center - 0.5, center + 0.5),
            channel_count=3,
        )
        for center in (149.0, 164.0)
    )

    figure = build_dual_flux_figure(curve, fluxes)
    secondary = [trace for trace in figure.data if trace.yaxis == "y2"]

    assert len(secondary) == 2
    assert all(trace.connectgaps is False for trace in secondary)
    assert [
        item["requested_band"]["center_mhz"]
        for item in (figure.layout.meta["spectrum_flux_curves"])
    ] == [149.0, 164.0]


def test_dual_flux_plot_applies_independent_frequency_time_offsets() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    dart_start = datetime(2025, 1, 24, 4, 48, 30, 312134, tzinfo=UTC)
    fluxes = tuple(
        SpectrumFluxCurve(
            time_utc=(dart_start, dart_start + timedelta(seconds=0.401374)),
            values=np.asarray([center, center + 1.0]),
            source="DART",
            polarization="Stokes I",
            unit="dB",
            requested_band=SpectrumBand(center - 1.0, center + 1.0),
            sampled_frequency_range_mhz=(center - 0.5, center + 0.5),
            channel_count=3,
        )
        for center in (164.0, 238.0)
    )
    radio_times = {
        164.0: dart_start - timedelta(microseconds=134),
        238.0: dart_start + timedelta(microseconds=99866),
    }
    alignments = {
        frequency: SpectrumTimeAlignment(
            reference_radio_time_utc=radio_time,
            nearest_spectrum_time_utc=dart_start,
            display_offset_seconds=(radio_time - dart_start).total_seconds(),
            native_cadence_seconds=0.401374,
            nearest_delta_seconds=abs((radio_time - dart_start).total_seconds()),
        )
        for frequency, radio_time in radio_times.items()
    }

    figure = build_dual_flux_figure(
        curve,
        fluxes,
        time_alignments=alignments,
    )

    secondary = [trace for trace in figure.data if trace.yaxis == "y2"]
    assert secondary[0].x[0] == radio_times[164.0]
    assert secondary[1].x[0] == radio_times[238.0]
    assert figure.layout.meta["spectrum_flux_time_alignments"] == {
        "164": alignments[164.0].to_dict(),
        "238": alignments[238.0].to_dict(),
    }
    assert fluxes[0].time_utc[0] == dart_start
    assert fluxes[1].time_utc[0] == dart_start


def test_dual_flux_plots_can_separate_each_matched_frequency() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    start = datetime(2025, 1, 24, 4, 48, 29, tzinfo=UTC)
    fluxes = tuple(
        SpectrumFluxCurve(
            time_utc=(start, start + timedelta(seconds=6)),
            values=np.asarray([center, center + 1.0]),
            source="DART",
            polarization="Stokes I",
            unit="dB",
            requested_band=SpectrumBand(center - 1.0, center + 1.0),
            sampled_frequency_range_mhz=(center - 0.5, center + 0.5),
            channel_count=3,
        )
        for center in (149.0, 164.0)
    )

    figures = build_dual_flux_figures(
        curve,
        fluxes,
        separate_by_frequency=True,
    )

    assert len(figures) == 2
    for figure, center in zip(figures, (149.0, 164.0), strict=True):
        assert figure.layout.meta["flux_plot_layout"] == "separate"
        assert figure.layout.meta["roi_frequency_mhz"] == center
        assert f"{center:g} MHz" in figure.layout.title.text
        primary = [trace for trace in figure.data if trace.yaxis != "y2"]
        assert primary
        assert all(f"{center:g} MHz" in trace.name for trace in primary)
        secondary = [trace for trace in figure.data if trace.yaxis == "y2"]
        assert len(secondary) == 1
        assert secondary[0].connectgaps is False


def test_dual_flux_plots_keep_existing_combined_layout_by_default() -> None:
    curve = _extractor_frame().assign(
        time=lambda frame: frame["obs_time"],
        frequency=lambda frame: frame["freq_mhz"],
    )
    flux = SpectrumFluxCurve(
        time_utc=(datetime(2025, 1, 24, 4, 48, 29, tzinfo=UTC),),
        values=np.asarray([149.0]),
        source="DART",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(148.0, 150.0),
        sampled_frequency_range_mhz=(148.5, 149.5),
        channel_count=3,
    )

    figures = build_dual_flux_figures(curve, flux)

    assert len(figures) == 1
    assert figures[0].layout.meta["flux_plot_layout"] == "combined"


def test_spectrum_band_selection_reuses_box_event_contract() -> None:
    band = spectrum_band_from_selection({"selection": {"box": [{"y": [160.0, 140.0]}]}})

    assert band == SpectrumBand(140.0, 160.0)
