"""Fresh FITS catalogs must agree with the exact plan frozen by the preview."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.aia_radio_composite import application
from solar_apps.frontends.radio.aia_radio_composite.models import (
    CompositeRequest,
    CompositeResult,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.rendering import TopPanelArtifact

START = "2025-01-24T04:48:30.000000001Z"
END = "2025-01-24T04:48:31.000000001Z"


def _write_aia(path, time=START):
    fits.PrimaryHDU(
        np.ones((3, 3)),
        fits.Header(
            {
                "T_OBS": time,
                "TIMESYS": "UTC",
                "WAVELNTH": 171,
                "QUALITY": 0,
                "EXPTIME": 1.0,
            }
        ),
    ).writeto(path)


def _request(tmp_path):
    for name in ("aia", "radio", "spectrum"):
        (tmp_path / name).mkdir()
    return CompositeRequest(
        aia_directory=tmp_path / "aia",
        aia_wave=171,
        aia_time=pd.Timestamp(START),
        radio_directory=tmp_path / "radio",
        radio_frequency=149.0,
        polarization="RR",
        roi_type="box",
        roi_vertices_arcsec=((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)),
        spectrum_type="dart",
        spectrum_path=tmp_path / "spectrum",
    )


def _settings():
    return {
        "radio_frequencies_mhz": [149.0],
        "aia_waves": [171],
        "time_start": START,
        "time_end": END,
    }


def _top_settings():
    return {**_settings(), "aia_max_dt_seconds": 12.0, "radio_max_dt_seconds": 0.1}


def _result():
    return CompositeResult(
        top_image=b"png",
        roi_curve=pd.DataFrame(
            {
                "time": [START],
                "frequency": [149.0],
                "raw_sum": [1.0],
                "raw_mean": [1.0],
                "raw_peak": [1.0],
                "quality_flag": ["ok"],
            }
        ),
        spectrum=SpectrumWindow(
            data=np.ones((1, 2)),
            frequency_mhz=np.array([149.0]),
            time_utc=(pd.Timestamp(START), pd.Timestamp(END)),
            polarization="RR",
            unit="K",
            source="DART",
        ),
    )


def test_fresh_catalog_resolves_the_original_preview_row_object(tmp_path, monkeypatch):
    request = _request(tmp_path)
    _write_aia(request.aia_directory / "frame.fits")
    plan = application.build_composite_match_plan(request, **_settings())
    rendered = []

    def top(_request, **kwargs):
        rendered.append(kwargs["plan_row"])
        return TopPanelArtifact(b"png", {})

    monkeypatch.setattr(application, "_build_matched_top_panel", top)
    artifact = application.build_top_panel(request, match_plan=plan, **_top_settings())

    assert rendered == [plan.rows[0]]
    assert rendered[0] is plan.rows[0]
    assert artifact.metadata["match_plan_fingerprint"] == plan.fingerprint
    assert plan.master_source_id == "aia:171"
    assert plan.rows[0].selections["radio:149"].frame is None


@pytest.mark.parametrize("change", ["mtime", "addition"])
@pytest.mark.parametrize("output", ["preview", "video"])
def test_modified_or_added_fits_rejects_frozen_plan_before_render(
    tmp_path, monkeypatch, change, output
):
    request = _request(tmp_path)
    original = request.aia_directory / "original.fits"
    _write_aia(original)
    plan = application.build_composite_match_plan(request, **_settings())
    if change == "mtime":
        stat = original.stat()
        os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    else:
        _write_aia(
            request.aia_directory / "additional.fits", "2025-01-24T04:48:30.000000002Z"
        )

    def must_not_render(*_args, **_kwargs):
        pytest.fail("A stale plan reached the image renderer")

    monkeypatch.setattr(application, "_build_matched_top_panel", must_not_render)
    monkeypatch.setattr(application, "write_video_from_paths", must_not_render)
    with pytest.raises(RuntimeError, match="rebuild the preview"):
        if output == "preview":
            application.build_top_panel(request, match_plan=plan, **_top_settings())
        else:
            application.build_dynamic_composite_video(
                _result(), request, match_plan=plan, **_settings()
            )


@pytest.mark.parametrize("change", ["mtime", "addition"])
def test_input_change_during_encoding_rejects_the_video(tmp_path, monkeypatch, change):
    request = _request(tmp_path)
    original = request.aia_directory / "original.fits"
    _write_aia(original)
    plan = application.build_composite_match_plan(request, **_settings())
    rendered = []

    def top(_request, **kwargs):
        rendered.append(kwargs["plan_row"])
        return TopPanelArtifact(b"png", {})

    def encode(paths, output_path, **kwargs):
        assert len(paths) == len(plan.rows)
        Path(output_path).write_bytes(b"encoded")
        if change == "mtime":
            stat = original.stat()
            os.utime(original, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        else:
            _write_aia(
                request.aia_directory / "additional.fits",
                "2025-01-24T04:48:30.000000002Z",
            )
        return True

    monkeypatch.setattr(application, "_build_matched_top_panel", top)
    monkeypatch.setattr(
        application,
        "render_composite_result",
        lambda *_args, **_kwargs: SimpleNamespace(image_png=b"png"),
    )
    monkeypatch.setattr(application, "write_video_from_paths", encode)
    with pytest.raises(RuntimeError, match="rebuild the preview"):
        application.build_dynamic_composite_video(
            _result(), request, match_plan=plan, **_settings()
        )
    assert rendered[0] is plan.rows[0]
