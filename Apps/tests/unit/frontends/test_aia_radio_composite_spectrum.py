"""DART and CSO spectrum adapter contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from solar_apps.frontends.radio.aia_radio_composite import app as composite_app
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    extract_spectrum_flux_curve,
    extract_spectrum_flux_curves,
    load_cso_spectrum_window,
    load_dart_spectrum_window,
    load_spectrum_window,
)
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    spectrum_adapter,
)
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeRequest,
    SpectrumBand,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    build_spectrum_selection_figure,
)
from solar_toolkit.radio.cso import CSOSpectrogram
from solar_toolkit.radio.dart_spectrogram import (
    DartNarrowbandCurve,
    DartNarrowbandResult,
    DartSpectrogramFiles,
    DartSpectrogramWindow,
)


def _request(
    tmp_path: Path,
    *,
    spectrum_type: str,
    polarization: str = "RR",
) -> CompositeRequest:
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
        spectrum_type=spectrum_type,
        spectrum_path=tmp_path / spectrum_type,
    )


def test_dart_adapter_uses_discovery_and_window_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "dart"
    files = DartSpectrogramFiles(
        stokes_i_db=folder / "SpecDataIdB.fits",
        stokes_v_over_i=folder / "SpecDataVP.fits",
        frequency=folder / "SpecFrequency.fits",
        time=folder / "SpecTime.fits",
    )
    source = DartSpectrogramWindow(
        stokes_i_db=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        stokes_v_over_i=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=(
            datetime(2025, 1, 24, 4, 48, tzinfo=UTC),
            datetime(2025, 1, 24, 4, 49, tzinfo=UTC),
        ),
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        spectrum_adapter,
        "discover_dart_spectrogram_files",
        lambda path: files,
    )

    def fake_read(value: object, **kwargs: object) -> DartSpectrogramWindow:
        calls["files"] = value
        calls.update(kwargs)
        return source

    monkeypatch.setattr(
        spectrum_adapter,
        "read_dart_spectrogram_window",
        fake_read,
    )

    window = load_dart_spectrum_window(
        folder,
        frequency_range_mhz=(90.0, 210.0),
        max_frequency_samples=800,
        max_time_samples=900,
    )

    assert calls["files"] is files
    assert calls["frequency_range_mhz"] == (90.0, 210.0)
    assert calls["max_frequency_samples"] == 800
    assert calls["max_time_samples"] == 900
    np.testing.assert_array_equal(window.data, source.stokes_i_db)
    assert window.polarization == "Stokes I"
    assert window.unit == "dB"
    assert window.source == "DART"
    assert window.metadata["display_plane"] == "stokes_i_db"
    assert set(window.metadata["files"]) == {
        "SpecDataIdB",
        "SpecDataVP",
        "SpecFrequency",
        "SpecTime",
    }


def test_cso_adapter_uses_reader_date_obs_axes_and_polarization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "cso"
    folder.mkdir()
    fits_path = folder / "observation.fits"
    fits_path.touch()
    ll = CSOSpectrogram(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        time=np.asarray([0.0, 2.0]),
        freq=np.asarray([200.0, 100.0]),
        polar="LL",
        dateobs="2025-01-24",
        unit="K",
        dt_base=datetime(2025, 1, 24, 0, 0),
    )
    rr = CSOSpectrogram(
        data=np.asarray([[5.0, 6.0], [7.0, 8.0]]),
        time=np.asarray([0.0, 2.0]),
        freq=np.asarray([200.0, 100.0]),
        polar="RCP",
        dateobs="2025-01-24",
        unit="sfu",
        dt_base=datetime(2025, 1, 24, 0, 0),
    )
    calls: list[Path] = []

    def fake_read(path: Path) -> list[CSOSpectrogram]:
        calls.append(path)
        return [ll, rr]

    monkeypatch.setattr(spectrum_adapter, "read_cso_spectrogram", fake_read)

    window = load_cso_spectrum_window(folder, polarization="RR")

    assert calls == [fits_path]
    assert window.source == "CSO"
    assert window.polarization == "RR"
    assert window.unit == "sfu"
    assert window.frequency_mhz.tolist() == [100.0, 200.0]
    np.testing.assert_array_equal(
        window.data,
        np.asarray([[7.0, 8.0], [5.0, 6.0]]),
    )
    assert window.time_utc == (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 2, tzinfo=UTC),
    )
    assert window.metadata["date_obs"] == "2025-01-24"


def test_request_dispatches_cso_and_applies_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "cso"
    folder.mkdir()
    (folder / "cso.fit").touch()
    spectrum = CSOSpectrogram(
        data=np.arange(12, dtype=float).reshape(3, 4),
        time=np.asarray([0.0, 1.0, 2.0, 3.0]),
        freq=np.asarray([100.0, 150.0, 200.0]),
        polar="LL",
        dateobs="2025-01-24",
        unit=None,
        dt_base=datetime(2025, 1, 24),
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_cso_spectrogram",
        lambda path: [spectrum],
    )

    window = load_spectrum_window(
        _request(tmp_path, spectrum_type="cso", polarization="LL"),
        frequency_range_mhz=(125.0, 225.0),
        time_range_utc=(
            "2025-01-24T00:00:01Z",
            "2025-01-24T00:00:02Z",
        ),
    )

    assert window.frequency_mhz.tolist() == [150.0, 200.0]
    assert window.data.tolist() == [[5.0, 6.0], [9.0, 10.0]]
    assert window.unit == "unknown"


def test_cso_preview_downsamples_without_changing_flux_extraction_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = tmp_path / "cso"
    folder.mkdir()
    (folder / "cso.fit").touch()
    spectrum = CSOSpectrogram(
        data=np.arange(30, dtype=float).reshape(5, 6),
        time=np.arange(6, dtype=float),
        freq=np.linspace(100.0, 200.0, 5),
        polar="RR",
        dateobs="2025-01-24",
        unit="sfu",
        dt_base=datetime(2025, 1, 24),
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_cso_spectrogram",
        lambda path: [spectrum],
    )

    window = load_cso_spectrum_window(
        folder,
        polarization="RR",
        max_frequency_samples=3,
        max_time_samples=2,
    )

    assert window.data.shape == (3, 2)
    assert window.metadata["original_selected_shape"] == [5, 6]
    assert window.metadata["preview_downsampled"] is True


def test_cso_adapter_rejects_unavailable_polarization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cso.fits"
    path.touch()
    spectrum = CSOSpectrogram(
        data=np.ones((2, 2)),
        time=np.asarray([0.0, 1.0]),
        freq=np.asarray([100.0, 200.0]),
        polar="LL",
        dateobs="2025-01-24",
        dt_base=datetime(2025, 1, 24),
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_cso_spectrogram",
        lambda value: [spectrum],
    )

    with pytest.raises(ValueError, match="available: LL"):
        load_cso_spectrum_window(path, polarization="RR")


def test_dart_flux_uses_canonical_original_channel_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )

    def fake_extract(*args: object, **kwargs: object) -> DartNarrowbandResult:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DartNarrowbandResult(
            time_utc=times,
            curves=(
                DartNarrowbandCurve(
                    center_frequency_mhz=150.0,
                    bandwidth_mhz=20.0,
                    requested_frequency_range_mhz=(140.0, 160.0),
                    sampled_frequency_range_mhz=(141.0, 159.0),
                    channel_count=5,
                    stokes_i_db=np.asarray([1.5, 2.5]),
                ),
            ),
        )

    monkeypatch.setattr(
        spectrum_adapter,
        "extract_dart_narrowband_lightcurves",
        fake_extract,
    )
    request = _request(tmp_path, spectrum_type="dart")

    curve = extract_spectrum_flux_curve(
        request,
        SpectrumBand(140.0, 160.0),
    )

    assert captured["args"] == (request.spectrum_path, [150.0], 20.0)
    assert curve.source == "DART"
    assert curve.unit == "dB"
    assert curve.channel_count == 5
    np.testing.assert_array_equal(curve.values, [1.5, 2.5])


def test_dart_flux_extracts_all_matched_centers_in_one_reader_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )
    calls: list[tuple[object, ...]] = []

    def fake_extract(*args: object, **_kwargs: object) -> DartNarrowbandResult:
        calls.append(args)
        return DartNarrowbandResult(
            time_utc=times,
            curves=tuple(
                DartNarrowbandCurve(
                    center_frequency_mhz=center,
                    bandwidth_mhz=2.0,
                    requested_frequency_range_mhz=(center - 1.0, center + 1.0),
                    sampled_frequency_range_mhz=(center - 0.5, center + 0.5),
                    channel_count=3,
                    stokes_i_db=np.asarray([center, center + 1.0]),
                )
                for center in (149.0, 164.0)
            ),
        )

    monkeypatch.setattr(
        spectrum_adapter,
        "extract_dart_narrowband_lightcurves",
        fake_extract,
    )
    request = _request(tmp_path, spectrum_type="dart")
    curves = extract_spectrum_flux_curves(
        request,
        (SpectrumBand(148.0, 150.0), SpectrumBand(163.0, 165.0)),
    )

    assert calls == [(request.spectrum_path, [149.0, 164.0], 2.0)]
    assert [item.requested_band.center_mhz for item in curves] == [149.0, 164.0]
    assert [item.channel_count for item in curves] == [3, 3]


def test_cso_flux_uses_finite_original_channel_mean_and_native_unit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "cso"
    path.mkdir()
    (path / "cso.fits").touch()
    spectrum = CSOSpectrogram(
        data=np.asarray(
            [
                [1.0, np.nan],
                [3.0, np.nan],
                [5.0, 9.0],
            ]
        ),
        time=np.asarray([0.0, 1.0]),
        freq=np.asarray([100.0, 150.0, 200.0]),
        polar="RCP",
        dateobs="2025-01-24",
        unit="sfu",
        dt_base=datetime(2025, 1, 24),
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_cso_spectrogram",
        lambda value: [spectrum],
    )

    curve = extract_spectrum_flux_curve(
        _request(tmp_path, spectrum_type="cso"),
        SpectrumBand(125.0, 200.0),
    )

    assert curve.source == "CSO"
    assert curve.unit == "sfu"
    assert curve.sampled_frequency_range_mhz == (150.0, 200.0)
    assert curve.channel_count == 2
    np.testing.assert_allclose(curve.values, [4.0, 9.0])


def test_spectrum_selection_figure_supports_cso_and_highlights_band() -> None:
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )
    window = SpectrumWindow(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        frequency_mhz=np.asarray([100.0, 150.0, 200.0]),
        time_utc=times,
        polarization="RR",
        unit="sfu",
        source="CSO",
    )

    figure = build_spectrum_selection_figure(
        window,
        band=SpectrumBand(140.0, 160.0),
    )

    assert figure.layout.dragmode == "select"
    assert figure.data[0].type == "heatmap"
    assert figure.data[1].name == "Frequency selection grid"
    assert len(figure.layout.shapes) == 1
    assert figure.layout.shapes[0].y0 == 140.0
    assert figure.layout.shapes[0].y1 == 160.0


def test_spectrum_figure_applies_display_frequency_and_intensity_ranges() -> None:
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )
    window = SpectrumWindow(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        frequency_mhz=np.asarray([100.0, 150.0, 200.0]),
        time_utc=times,
        polarization="RR",
        unit="sfu",
        source="CSO",
    )

    figure = build_spectrum_selection_figure(
        window,
        display_frequency_range_mhz=(120.0, 180.0),
        display_intensity_range=(2.0, 5.0),
    )

    assert figure.layout.yaxis.range == (120.0, 180.0)
    assert figure.data[0].zmin == 2.0
    assert figure.data[0].zmax == 5.0
    assert figure.layout.meta["display_frequency_range_mhz"] == [120.0, 180.0]
    assert figure.layout.meta["display_intensity_range"] == [2.0, 5.0]


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        (
            "display_frequency_range_mhz",
            (180.0, 120.0),
            "display_frequency_range_mhz minimum must be below its maximum",
        ),
        (
            "display_intensity_range",
            (5.0, 2.0),
            "display_intensity_range minimum must be below its maximum",
        ),
    ],
)
def test_spectrum_figure_rejects_invalid_display_ranges(
    keyword: str,
    value: tuple[float, float],
    match: str,
) -> None:
    window = SpectrumWindow(
        data=np.ones((2, 2)),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=(
            datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
            datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
        ),
        polarization="RR",
        unit="sfu",
        source="CSO",
    )

    with pytest.raises(ValueError, match=match):
        build_spectrum_selection_figure(window, **{keyword: value})


def test_spectrum_selection_figure_highlights_all_roi_matched_bands() -> None:
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )
    window = SpectrumWindow(
        data=np.ones((4, 2)),
        frequency_mhz=np.asarray([140.0, 149.0, 164.0, 170.0]),
        time_utc=times,
        polarization="RR",
        unit="sfu",
        source="CSO",
    )

    figure = build_spectrum_selection_figure(
        window,
        bands=(SpectrumBand(148.0, 150.0), SpectrumBand(163.0, 165.0)),
    )

    assert [(shape.y0, shape.y1) for shape in figure.layout.shapes] == [
        (148.0, 150.0),
        (163.0, 165.0),
    ]


def test_roi_frequencies_automatically_define_same_center_spectrum_bands() -> None:
    times = (
        datetime(2025, 1, 24, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 24, 0, 0, 1, tzinfo=UTC),
    )
    window = SpectrumWindow(
        data=np.ones((6, 2)),
        frequency_mhz=np.asarray([147.0, 149.0, 151.0, 162.0, 164.0, 166.0]),
        time_utc=times,
        polarization="Stokes I",
        unit="dB",
        source="DART",
    )

    bands = composite_app._matched_spectrum_bands([149, 164], 4.0, window)

    assert [band.center_mhz for band in bands] == [149.0, 164.0]
    assert [band.bandwidth_mhz for band in bands] == [4.0, 4.0]
    assert [(band.low_mhz, band.high_mhz) for band in bands] == [
        (147.0, 151.0),
        (162.0, 166.0),
    ]


def test_spectrum_selection_figure_aligns_range_and_reference_line() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    end = datetime(2025, 1, 24, 4, 49, 0, tzinfo=UTC)
    marker = datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC)
    window = SpectrumWindow(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=(start, end),
        polarization="Stokes I",
        unit="dB",
        source="DART",
    )

    figure = build_spectrum_selection_figure(
        window,
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
