"""Data-model contracts for the AIA radio composite frontend."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_apps.frontends.radio.aia_radio_composite import (
    CompositeRequest,
    CompositeResult,
    SpectrumBand,
    SpectrumFluxCurve,
    SpectrumWindow,
)
from solar_apps.frontends.radio.aia_radio_composite.models import (
    AIA_RADIO_COMPOSITE_SCHEMA_VERSION,
    ROI_CURVE_COLUMNS,
)


def _request(tmp_path: Path, **overrides: object) -> CompositeRequest:
    values: dict[str, object] = {
        "aia_directory": tmp_path / "aia",
        "aia_wave": 171,
        "aia_time": "2025-01-24T04:48:32Z",
        "radio_directory": tmp_path / "radio",
        "radio_frequency": 149,
        "polarization": "rr+ll",
        "roi_type": "box",
        "roi_vertices_arcsec": (
            (-20.0, -10.0),
            (20.0, -10.0),
            (20.0, 10.0),
            (-20.0, 10.0),
        ),
        "spectrum_type": "DART",
        "spectrum_path": tmp_path / "spectrum",
    }
    values.update(overrides)
    return CompositeRequest(**values)


def _spectrum(**overrides: object) -> SpectrumWindow:
    start = datetime(2025, 1, 24, 4, 48, 30, tzinfo=UTC)
    values: dict[str, object] = {
        "data": np.arange(6, dtype=np.float32).reshape(2, 3),
        "frequency_mhz": np.asarray([149.0, 150.0]),
        "time_utc": tuple(start + timedelta(seconds=index) for index in range(3)),
        "polarization": "Stokes I",
        "unit": "dB",
        "source": "dart",
        "metadata": {"downsampled": True},
    }
    values.update(overrides)
    return SpectrumWindow(**values)


def _curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [
                "2025-01-24T04:48:30Z",
                "2025-01-24T04:48:31Z",
            ],
            "frequency": [149.0, 164.0],
            "raw_sum": [10.0, 20.0],
            "raw_mean": [1.0, 2.0],
            "raw_peak": [3.0, 4.0],
            "quality_flag": ["ok", "ok"],
        }
    )


def test_composite_request_normalizes_paths_selectors_and_utc(
    tmp_path: Path,
) -> None:
    """A request stores canonical paths, selectors, and UTC time."""

    request = _request(tmp_path)

    assert request.aia_directory == (tmp_path / "aia").resolve()
    assert request.radio_directory == (tmp_path / "radio").resolve()
    assert request.spectrum_path == (tmp_path / "spectrum").resolve()
    assert request.aia_time.tzinfo is UTC
    assert request.polarization == "RR+LL"
    assert request.spectrum_type == "dart"
    payload = request.to_dict()
    assert payload["schema_version"] == AIA_RADIO_COMPOSITE_SCHEMA_VERSION
    assert payload["roi"]["coordinate_system"] == "HPLN/HPLT arcsec"
    assert "pixel" not in json.dumps(payload).lower()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("aia_wave", 171.5, "integer wavelength"),
        ("aia_wave", 170, "must be one of"),
        ("radio_frequency", 0, "greater than zero"),
        ("radio_frequency", float("nan"), "must be finite"),
        ("polarization", "Stokes V", "RR, LL, or RR\\+LL"),
        ("roi_type", "circle", "box or lasso"),
        ("spectrum_type", "other", "dart or cso"),
    ),
)
def test_composite_request_rejects_invalid_controls(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Invalid scientific selectors fail before any reader is called."""

    with pytest.raises((TypeError, ValueError), match=message):
        _request(tmp_path, **{field: value})


def test_composite_request_requires_nondegenerate_arcsec_vertices(
    tmp_path: Path,
) -> None:
    """ROI geometry is stored only as nondegenerate HPLN/HPLT vertices."""

    with pytest.raises(ValueError, match="at least 3"):
        _request(
            tmp_path,
            roi_type="lasso",
            roi_vertices_arcsec=((0.0, 0.0), (1.0, 1.0)),
        )
    with pytest.raises(ValueError, match="degenerate"):
        _request(
            tmp_path,
            roi_vertices_arcsec=(
                (0.0, 0.0),
                (0.0, 1.0),
                (0.0, 2.0),
                (0.0, 3.0),
            ),
        )


def test_spectrum_window_preserves_samples_and_normalizes_metadata() -> None:
    """Spectrum validation aligns axes without changing the sample matrix."""

    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    window = _spectrum(
        data=source,
        time_utc=(
            "2025-01-24T04:48:30",
            "2025-01-24T04:48:31Z",
            "2025-01-24T12:48:32+08:00",
        ),
        metadata={
            "path": Path("/data/dart"),
            "generated": datetime(2025, 1, 24, tzinfo=UTC),
            "bins": np.int64(2),
        },
    )

    np.testing.assert_array_equal(window.data, source)
    assert window.data.dtype == np.float32
    assert all(value.tzinfo is UTC for value in window.time_utc)
    assert window.source == "DART"
    assert window.metadata["path"] == "/data/dart"
    assert window.metadata["bins"] == 2
    assert window.to_metadata_dict()["shape"] == [2, 3]
    json.dumps(window.to_metadata_dict())


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"data": np.ones((2, 2))}, "does not match axes"),
        ({"frequency_mhz": np.asarray([150.0, 149.0])}, "strictly increasing"),
        (
            {
                "time_utc": (
                    "2025-01-24T04:48:30Z",
                    "2025-01-24T04:48:30Z",
                    "2025-01-24T04:48:31Z",
                )
            },
            "strictly increasing",
        ),
        ({"source": "CALLISTO"}, "DART or CSO"),
        ({"unit": ""}, "unit is required"),
    ),
)
def test_spectrum_window_rejects_misaligned_or_ambiguous_data(
    overrides: dict[str, object],
    message: str,
) -> None:
    """A normalized spectrum cannot carry misaligned axes or unknown identity."""

    with pytest.raises(ValueError, match=message):
        _spectrum(**overrides)


def test_composite_result_validates_curve_and_copies_inputs() -> None:
    """A result owns a stable ROI curve and JSON-safe metadata inventory."""

    curve = _curve()
    result = CompositeResult(
        top_image=b"\x89PNG\r\n\x1a\nsynthetic",
        roi_curve=curve,
        spectrum=_spectrum(),
        metadata={"request_id": "abc", "output": Path("/tmp/product.png")},
    )
    curve.loc[0, "raw_sum"] = 999.0

    assert result.roi_curve.loc[0, "raw_sum"] == 10.0
    assert result.metadata["output"] == "/tmp/product.png"
    inventory = result.to_metadata_dict()
    assert inventory["roi_curve_rows"] == 2
    assert inventory["spectrum"]["source"] == "DART"
    json.dumps(inventory)


def test_composite_result_requires_all_curve_columns() -> None:
    """The exported ROI curve contract includes all requested measurements."""

    curve = _curve().drop(columns=["raw_peak"])

    with pytest.raises(ValueError, match="raw_peak"):
        CompositeResult(
            top_image=b"image",
            roi_curve=curve,
            spectrum=_spectrum(),
        )
    assert set(ROI_CURVE_COLUMNS) <= set(_curve().columns)


def test_composite_result_retains_untimed_failed_quality_row() -> None:
    """A failed FITS row without DATE-OBS remains exportable as diagnostics."""

    curve = _curve()
    curve.loc[0, "time"] = ""
    curve.loc[0, "quality_flag"] = "invalid_image"

    result = CompositeResult(
        top_image=b"image",
        roi_curve=curve,
        spectrum=_spectrum(),
    )

    assert result.roi_curve.loc[0, "time"] == ""


def test_composite_result_rejects_untimed_ok_quality_row() -> None:
    curve = _curve()
    curve.loc[0, "time"] = ""

    with pytest.raises(ValueError, match="ok-quality"):
        CompositeResult(
            top_image=b"image",
            roi_curve=curve,
            spectrum=_spectrum(),
        )


def test_spectrum_band_validates_bounds_and_original_channels() -> None:
    band = SpectrumBand(140.0, 160.0)

    np.testing.assert_array_equal(
        band.observed_indices([100.0, 150.0, 200.0]),
        [1],
    )
    assert band.center_mhz == 150.0
    assert band.bandwidth_mhz == 20.0
    with pytest.raises(ValueError, match="below"):
        SpectrumBand(160.0, 140.0)
    with pytest.raises(ValueError, match="outside"):
        SpectrumBand(90.0, 110.0).observed_indices([100.0, 200.0])
    with pytest.raises(ValueError, match="no original"):
        SpectrumBand(120.0, 130.0).observed_indices([100.0, 200.0])


def test_spectrum_flux_curve_preserves_nan_gaps_and_exports_metadata() -> None:
    start = datetime(2025, 1, 24, tzinfo=UTC)
    flux = SpectrumFluxCurve(
        time_utc=(start, start + timedelta(seconds=1)),
        values=np.asarray([2.5, np.nan]),
        source="dart",
        polarization="Stokes I",
        unit="dB",
        requested_band=SpectrumBand(140.0, 160.0),
        sampled_frequency_range_mhz=(141.0, 159.0),
        channel_count=5,
    )

    assert np.isnan(flux.values[1])
    assert flux.to_metadata_dict()["finite_sample_count"] == 1
    assert flux.to_frame()["aggregation"].unique().tolist() == ["finite_channel_mean"]


def test_composite_result_supports_multiple_spectrum_flux_curves_and_legacy_first() -> (
    None
):
    start = datetime(2025, 1, 24, tzinfo=UTC)
    fluxes = tuple(
        SpectrumFluxCurve(
            time_utc=(start, start + timedelta(seconds=1)),
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

    result = CompositeResult(
        top_image=b"image",
        roi_curve=_curve(),
        spectrum=_spectrum(),
        spectrum_flux_curves=fluxes,
    )

    assert result.spectrum_flux_curve is fluxes[0]
    assert result.spectrum_flux_curves == fluxes
    assert len(result.to_metadata_dict()["spectrum_flux_curves"]) == 2


def test_composite_result_rejects_non_json_metadata() -> None:
    """Metadata remains serializable for the required JSON artifact."""

    with pytest.raises(TypeError, match="non-JSON"):
        CompositeResult(
            top_image=b"image",
            roi_curve=_curve(),
            spectrum=_spectrum(),
            metadata={"unsupported": object()},
        )
