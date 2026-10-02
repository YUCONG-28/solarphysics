"""Guard the application-layer scientific image naming contract."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from types import SimpleNamespace

import astropy.units as u

from solar_apps.workflows.common.image_naming import (
    INTERNAL_FIXED_IMAGE_NAMES,
    build_scientific_image_filename,
    configured_scientific_image_path,
)
from solar_apps.frontends.radio.dart_spectrogram.dart_spectrogram_app import (
    DartDatasetSummary,
    build_dart_artifact_filenames,
)
from solar_apps.workflows.visualization.stereo_euvi_overview import (
    out_name as euvi_output_name,
)
from solar_apps.workflows.visualization.suvi_quadrant import SuviSelection, output_name
from solar_apps.frontends.workbench.radio_workspace.contracts import (
    RadioArtifact,
    RadioFigureExport,
)

_CONTRACT_NAME = re.compile(
    r"^\d{4}_\d{8}T\d{6}Z(?:-\d{8}T\d{6}Z)?_"
    r"(?:generated_)?[a-z0-9]+(?:_[a-z0-9]+)*\.png$"
)


def test_application_helper_uses_one_supplied_generated_time() -> None:
    generated_at = dt.datetime(2026, 7, 17, 10, 11, 12, tzinfo=dt.UTC)
    first = build_scientific_image_filename(
        sequence=1,
        start_time=None,
        instrument="SUVI",
        channel="195 Angstrom",
        product="Intensity",
        generated_at=generated_at,
    )
    second = build_scientific_image_filename(
        sequence=2,
        start_time="not-a-time",
        instrument="SUVI",
        channel="195 Angstrom",
        product="Overview",
        generated_at=generated_at,
    )

    assert first == "0001_20260717T101112Z_generated_suvi_195a_intensity.png"
    assert second == "0002_20260717T101112Z_generated_suvi_195a_overview.png"
    assert _CONTRACT_NAME.fullmatch(first)
    assert _CONTRACT_NAME.fullmatch(second)


def test_explicit_complete_output_path_remains_unchanged(tmp_path: Path) -> None:
    explicit = tmp_path / "caller-selected.png"
    resolved = configured_scientific_image_path(
        explicit,
        sequence=1,
        start_time="2000-01-01T04:48:30Z",
        instrument="AIA",
        channel=171,
        product="Intensity",
        generated_at=dt.datetime(2026, 7, 17, tzinfo=dt.UTC),
    )
    assert resolved == explicit


def test_dart_declared_product_order_drives_sequence(tmp_path: Path) -> None:
    summary = DartDatasetSummary(
        directory=tmp_path,
        files=None,  # type: ignore[arg-type]
        matrix_shape=(2, 2),
        frequency_range_mhz=(100.0, 200.0),
        frequency_samples=2,
        time_range_utc=(
            dt.datetime(2000, 1, 1, 4, 48, tzinfo=dt.UTC),
            dt.datetime(2000, 1, 1, 4, 50, tzinfo=dt.UTC),
        ),
        time_samples=2,
    )
    names = build_dart_artifact_filenames(summary)

    assert names["dynamic_spectrum"].startswith(
        "0001_20000101T044800Z-20000101T045000Z_dart_"
    )
    assert names["selected_spectrum"].startswith(
        "0002_20000101T044800Z-20000101T045000Z_dart_"
    )
    assert names["narrowband_lightcurve"].startswith(
        "0003_20000101T044800Z-20000101T045000Z_dart_"
    )
    assert all(_CONTRACT_NAME.fullmatch(name) for name in names.values())


def test_suvi_and_euvi_providers_use_observation_metadata(tmp_path: Path) -> None:
    observed = dt.datetime(2000, 1, 1, 4, 48, 30, 900000, tzinfo=dt.UTC)
    suvi_map = SimpleNamespace(date=observed, wavelength=195 * u.Angstrom)
    euvi_map = SimpleNamespace(date=observed, wavelength=171 * u.Angstrom)
    selection = SuviSelection("goes16", "g16", "195", tmp_path / "source.fits")

    assert output_name(selection, suvi_map) == (
        "0001_20000101T044830Z_suvi_g16_195a_intensity_lower_right_quadrant.png"
    )
    assert euvi_output_name(euvi_map) == (
        "0001_20000101T044830Z_stereo_a_euvi_171a_intensity.png"
    )


def test_web_contracts_expose_actual_download_basename() -> None:
    filename = "0001_20000101T044830Z_radio_149mhz_rcp_source_map.png"
    artifact = RadioArtifact(
        id="artifact1",
        relative_path=f"nested/{filename}",
        kind="image",
        mime_type="image/png",
    )
    exported = RadioFigureExport(
        figure_schema_version=1,
        id="export1",
        workspace_id="workspace1",
        mime_type="image/png",
        output_path=filename,
        thumbnail_path="thumbnail.png",
        preflight_revision="revision1",
        sha256="0" * 64,
        size=100,
        width=10,
        height=10,
        frame_count=1,
        created_at="2000-01-01T04:48:30Z",
    )

    assert artifact.to_dict()["suggested_filename"] == filename
    assert exported.to_dict()["suggested_filename"] == filename
    assert exported.to_dict()["thumbnail_path"] == "thumbnail.png"


def test_internal_fixed_protocol_names_are_narrowly_whitelisted() -> None:
    assert INTERNAL_FIXED_IMAGE_NAMES == {"preview.png", "thumbnail.png"}
    assert all(
        not _CONTRACT_NAME.fullmatch(name) for name in INTERNAL_FIXED_IMAGE_NAMES
    )
