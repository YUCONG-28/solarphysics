"""Backend-neutral Athena bridge and physical-time calibration tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from spike_typeIII_visual import athena
from spike_typeIII_visual.athena_io import (
    AthenaBinarySnapshot,
    MHDFieldDataset,
    build_field_series,
    read_bridge_hdf5,
    write_bridge_hdf5,
)
from spike_typeIII_visual.config import TimeCalibrationConfig
from spike_typeIII_visual.events import EventBundle, build_event_bundle
from spike_typeIII_visual.physics.normalization import PhysicalNormalization
from spike_typeIII_visual.physics.radio import alfven_time_seconds
from spike_typeIII_visual.physics.synthetic_aia import (
    AIAResponse,
    synthesize_aia_intensity,
)
from spike_typeIII_visual.solar_jet_sweep import generate_sobol_cases


def test_athena_source_uses_canonical_fluxrope_demo_tree() -> None:
    simulation_root = Path(__file__).resolve().parents[2]
    expected = simulation_root / "fluxrope_demo" / "athena4.2"
    assert athena.ATHENA_SOURCE == expected
    assert (expected / "src" / "prob" / "fluxrope.c").is_file()
    assert all(path.is_file() for path in athena.ATHENA_INPUTS.values())
    report = athena.doctor()
    assert report["athena_source"] is True
    assert all(report["input_files"].values())


def test_copy_source_excludes_in_source_build_products(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    (source / "bin").mkdir(parents=True)
    (source / "src" / "prob").mkdir(parents=True)
    (source / "src" / "microphysics").mkdir(parents=True)
    keep = {
        source / "configure": "#!/bin/sh\n",
        source / "Makefile.in": "template\n",
        source / "src" / "prob" / "keep.c": "int keep;\n",
    }
    generated = {
        source / "Makefile": "generated\n",
        source / "Makeoptions": "generated\n",
        source / "config.log": "private path\n",
        source / "bin" / "athena": "binary\n",
        source / "src" / "Makedepend": "generated\n",
        source / "src" / "config.h": "generated\n",
        source / "src" / "defs.h": "generated\n",
        source / "src" / "problem.c": "generated\n",
        source / "src" / "microphysics" / "viscosity.o": "object\n",
    }
    for path, content in keep.items() | generated.items():
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(athena, "ATHENA_SOURCE", source)
    destination = tmp_path / "copy"
    athena._copy_source(destination)
    for path in keep:
        assert (destination / path.relative_to(source)).is_file()
    for path in generated:
        assert not (destination / path.relative_to(source)).exists()


def _snapshot(
    time: float,
    amplitude: float = 1.0,
    guide_amplitude: float = 0.25,
) -> AthenaBinarySnapshot:
    nx, ny = 32, 24
    x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(-np.pi, np.pi, ny, endpoint=False)
    _, y_mesh = np.meshgrid(x, y)
    zeros = np.zeros((ny, nx), dtype=float)
    magnetic_x = amplitude * np.sin(y_mesh)
    magnetic_z = guide_amplitude * np.sin(y_mesh)
    return AthenaBinarySnapshot(
        time=time,
        dt=0.1,
        gamma=5.0 / 3.0,
        x=x,
        y=y,
        rho=np.ones_like(zeros),
        pressure=np.ones_like(zeros),
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        magnetic_x=magnetic_x,
        magnetic_y=zeros,
        magnetic_z=magnetic_z,
        precision_bytes=8,
    )


def test_athena_series_current_sign_and_hdf5_roundtrip(tmp_path) -> None:
    series = build_field_series(
        [_snapshot(0.0), _snapshot(0.1, 0.98)],
        resistivity=0.002,
        viscosity=0.002,
        sheet_half_width=0.2,
        sheet_center_fraction=0.25,
        perturbation_kx=1.0,
    )
    expected = -np.cos(series.grid.y_mesh)
    assert np.max(np.abs(series.current_z[0] - expected)) < 0.02
    assert series.divergence_rms < 1.0e-12
    path = tmp_path / "bridge.h5"
    write_bridge_hdf5(
        series,
        path,
        diagnostic_metadata={
            "density_floor": 1.0,
            "pressure_floor": 1.0,
            "resistivity_model": {
                "background": 1.0e-5,
                "anomalous": 2.0e-4,
                "threshold": 0.5,
            },
        },
    )
    loaded = read_bridge_hdf5(path)
    assert loaded.source == "athena-c"
    assert np.array_equal(loaded.times, series.times)
    assert np.array_equal(loaded.current_z, series.current_z)
    assert np.array_equal(loaded.magnetic_z, series.magnetic_z)
    assert np.max(np.abs(loaded.current_x[0] - 0.25 * np.cos(loaded.grid.y_mesh))) < 0.02
    assert loaded.geometry.x_points == series.geometry.x_points
    with MHDFieldDataset(path) as dataset:
        assert dataset.schema_version == 5
        assert dataset.shape == series.rho.shape
        assert np.array_equal(dataset.snapshot(0)["magnetic_z"], series.magnetic_z[0])
    with h5py.File(path, "r") as handle:
        diagnostics = handle["development_diagnostics"]
        assert np.all(diagnostics["maximum_mach"][...] == 0.0)
        assert np.all(
            diagnostics["density_floor_count"][...] == np.prod(series.rho.shape[1:])
        )
        assert np.all(diagnostics["boundary_mass_flux_outward"][...] == 0.0)
        assert np.all(diagnostics["boundary_energy_flux_outward"][...] == 0.0)
        assert np.all(
            diagnostics["local_resistivity_maximum"][...]
            >= diagnostics["local_resistivity_minimum"][...]
        )


def test_schema_v3_is_promoted_with_zero_out_of_plane_fields(tmp_path) -> None:
    series = build_field_series(
        [_snapshot(0.0), _snapshot(0.1)],
        resistivity=0.002,
        viscosity=0.002,
        sheet_half_width=0.2,
        sheet_center_fraction=0.25,
        perturbation_kx=1.0,
    )
    path = tmp_path / "legacy_bridge.h5"
    write_bridge_hdf5(series, path)
    new_names = (
        "velocity_z",
        "magnetic_z",
        "current_x",
        "current_y",
        "omega_x",
        "omega_y",
    )
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = 3
        for name in new_names:
            del handle[name]
    loaded = read_bridge_hdf5(path)
    assert not np.any(loaded.velocity_z)
    assert not np.any(loaded.magnetic_z)
    with MHDFieldDataset(path) as dataset:
        assert not np.any(dataset.snapshot(0)["current_x"])


def test_alfven_calibration_requires_explicit_scales() -> None:
    with pytest.raises(ValueError, match="requires positive"):
        TimeCalibrationConfig(mode="alfven")
    config = TimeCalibrationConfig(
        mode="alfven",
        length_scale_mm=10.0,
        magnetic_field_gauss=20.0,
        electron_density_cm3=1.0e9,
    )
    assert alfven_time_seconds(config) > 0.0
    with pytest.raises(ValueError, match="only with"):
        replace(config, mode="proxy")
    assert TimeCalibrationConfig(mode="event").mode == "event"


def test_physical_normalization_and_sanitized_event_bundle(tmp_path) -> None:
    normalization = PhysicalNormalization.from_solar_units(
        length_mm=10.0,
        magnetic_field_gauss=10.0,
        electron_density_cm3=1.0e9,
    )
    assert normalization.alfven_speed_m_s > 1.0e5
    assert normalization.time_s > 0.0

    config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "event_20250124_sanitized.json"
    )
    output = tmp_path / "event.json"
    bundle = build_event_bundle(config, output)
    reloaded = EventBundle.from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert reloaded.event_id == bundle.event_id
    assert reloaded.duration_s == 30.0
    assert reloaded.to_dict()["bundle_sha256"] == bundle.to_dict()["bundle_sha256"]
    assert all(
        drift.status == "candidate"
        for drift in bundle.drifts
        if drift.drift_id in {"drift_003", "drift_004", "drift_007", "drift_010"}
    )


def test_event_bundle_rejects_private_path() -> None:
    private_path = str(Path("/Users") / "PRIVATE_TEST_USER" / "raw.fits")
    payload = {
        "event_id": "unsafe",
        "core_start_utc": "2025-01-24T00:00:00Z",
        "core_end_utc": "2025-01-24T00:00:01Z",
        "frequency_range_mhz": [100.0, 200.0],
        "cadence_s": 1.0,
        "roi": {"source": private_path},
        "data_ids": ["RAW"],
    }
    with pytest.raises(ValueError, match="personal absolute path"):
        EventBundle.from_dict(payload)


def test_sobol_table_and_aia_forward_model_are_deterministic() -> None:
    assert generate_sobol_cases() == generate_sobol_cases()
    assert len(generate_sobol_cases()) == 16
    response = AIAResponse(
        channel="171",
        log10_temperature_k=np.array([5.5, 6.0, 6.5]),
        response_dn_cm5_s_pixel=np.array([0.0, 1.0e-24, 0.0]),
        calibration_id="unit-test-response",
    )
    density = np.full((4, 5), 1.0e9)
    temperature = np.full((4, 5), 1.0e6)
    intensity = synthesize_aia_intensity(
        density,
        temperature,
        los_depth_cm=1.0e9,
        response=response,
    )
    assert np.allclose(intensity, 1.0e3)
