"""End-to-end contracts requested for the AIA radio composite frontend."""

from __future__ import annotations

import importlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite.adapters import (
    spectrum_adapter,
)
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeRequest,
    CompositeResult,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import (
    TopPanelArtifact,
    render_composite_result,
)
from solar_toolkit.radio.cso import CSOSpectrogram
from solar_toolkit.radio.dart_spectrogram import (
    DartSpectrogramFiles,
    DartSpectrogramWindow,
)


def _request(tmp_path: Path, spectrum_type: str = "dart") -> CompositeRequest:
    return CompositeRequest(
        aia_directory=tmp_path / "aia",
        aia_wave=171,
        aia_time=datetime(2025, 1, 24, 4, 48, 32, tzinfo=UTC),
        radio_directory=tmp_path / "radio",
        radio_frequency=149.0,
        polarization="RR",
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


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 120), color=(20, 30, 40)).save(output, "PNG")
    return output.getvalue()


def _spectrum(source: str = "DART") -> SpectrumWindow:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    return SpectrumWindow(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=(start, start + timedelta(seconds=1)),
        polarization="Stokes I" if source == "DART" else "RR",
        unit="dB" if source == "DART" else "K",
        source=source,
    )


def test_import_success() -> None:
    assert importlib.import_module("solar_apps.frontends.radio.aia_radio_composite")
    assert importlib.import_module("solar_apps.frontends.radio.aia_radio_composite.app")


def test_mock_aia_build_top_panel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    aia = object()
    frame = SimpleNamespace(
        obs_time=datetime(2025, 1, 24, 4, 48, 32),
        path=tmp_path / "radio.fits",
        hdu_index=0,
        freq_mhz=149.0,
        pol="RCP",
    )
    monkeypatch.setattr(application, "load_aia_selection", lambda *a, **k: aia)
    monkeypatch.setattr(application, "scan_aia_catalog", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        application,
        "load_radio_candidates",
        lambda *a, **k: (frame,),
    )
    monkeypatch.setattr(
        application,
        "fit_radio_gaussian_frame",
        lambda *a, **k: object(),
    )

    def fake_render(left, right, **kwargs):
        assert left == (aia,)
        return TopPanelArtifact(_png(), {"radio": {"matched_time_utc": "mock"}})

    monkeypatch.setattr(application, "render_top_panel", fake_render)

    artifact = application.build_top_panel(_request(tmp_path))

    assert artifact.image_png.startswith(b"\x89PNG")


def test_mock_radio_build_top_panel(
    tmp_path: Path,
    monkeypatch,
) -> None:
    radio = object()
    frame = SimpleNamespace(
        obs_time=datetime(2025, 1, 24, 4, 48, 32),
        path=tmp_path / "radio.fits",
        hdu_index=0,
        freq_mhz=149.0,
        pol="RCP",
    )
    monkeypatch.setattr(
        application,
        "load_aia_selection",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(application, "scan_aia_catalog", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        application,
        "load_radio_candidates",
        lambda *a, **k: (frame,),
    )
    monkeypatch.setattr(
        application,
        "fit_radio_gaussian_frame",
        lambda *a, **k: radio,
    )

    def fake_render(left, right, **kwargs):
        assert right == (radio,)
        return TopPanelArtifact(_png(), {"radio": {"matched_time_utc": "mock"}})

    monkeypatch.setattr(application, "render_top_panel", fake_render)

    assert application.build_top_panel(_request(tmp_path)).image_png


def test_mock_dart_reader(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "dart"
    files = DartSpectrogramFiles(
        folder / "SpecDataIdB.fits",
        folder / "SpecDataVP.fits",
        folder / "SpecFrequency.fits",
        folder / "SpecTime.fits",
    )
    raw = DartSpectrogramWindow(
        stokes_i_db=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        stokes_v_over_i=np.zeros((2, 2)),
        frequency_mhz=np.asarray([100.0, 200.0]),
        time_utc=_spectrum().time_utc,
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "discover_dart_spectrogram_files",
        lambda value: files,
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_dart_spectrogram_window",
        lambda *args, **kwargs: raw,
    )

    window = spectrum_adapter.load_spectrum_window(_request(tmp_path))

    assert window.source == "DART"
    np.testing.assert_array_equal(window.data, raw.stokes_i_db)


def test_mock_cso_reader(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "cso"
    path.mkdir()
    (path / "cso.fits").touch()
    raw = CSOSpectrogram(
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        time=np.asarray([0.0, 1.0]),
        freq=np.asarray([100.0, 200.0]),
        polar="RR",
        dateobs="2025-01-24",
        unit="K",
        dt_base=datetime(2025, 1, 24),
    )
    monkeypatch.setattr(
        spectrum_adapter,
        "read_cso_spectrogram",
        lambda value: [raw],
    )

    window = spectrum_adapter.load_spectrum_window(
        _request(tmp_path, spectrum_type="cso")
    )

    assert window.source == "CSO"
    assert window.polarization == "RR"
    assert window.time_utc[0].tzinfo == UTC


def test_generate_three_panel_figure() -> None:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    curve = pd.DataFrame(
        {
            "time": [start.isoformat(), (start + timedelta(seconds=1)).isoformat()],
            "frequency": [149.0, 164.0],
            "raw_sum": [10.0, 20.0],
            "raw_mean": [1.0, 2.0],
            "raw_peak": [3.0, 4.0],
            "quality_flag": ["ok", "ok"],
        }
    )
    result = CompositeResult(
        top_image=_png(),
        roi_curve=curve,
        spectrum=_spectrum(),
        metadata={"mock": True},
    )

    artifact = render_composite_result(
        result,
        map_time=start,
        dpi=60,
    )

    assert artifact.image_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"aia-radio-composite" in artifact.metadata_json
    assert artifact.roi_curve_csv.startswith(b"time,frequency")
