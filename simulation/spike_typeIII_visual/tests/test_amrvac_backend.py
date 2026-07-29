from __future__ import annotations

import json
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from spike_typeIII_visual.amrvac_io import (
    AMRVACHeader,
    AMRVACUniformSnapshot,
    ingest_amrvac,
    project_dat_to_uniform,
    read_dat_v5_header,
    recover_primitive,
)
from spike_typeIII_visual.athena_io import read_bridge_hdf5

FIELD_NAMES = ("rho", "m1", "m2", "m3", "e", "b1", "b2", "b3")


def _synthetic_header() -> AMRVACHeader:
    return AMRVACHeader(
        version=5,
        offset_tree=0,
        offset_blocks=0,
        nw=8,
        ndir=3,
        ndim=2,
        levmax=1,
        nleafs=1,
        nparents=0,
        iteration=0,
        time=0.0,
        xmin=np.array([0.0, 0.0]),
        xmax=np.array([1.0, 1.0]),
        domain_nx=np.array([2, 2], dtype=np.int32),
        block_nx=np.array([2, 2], dtype=np.int32),
        periodic=np.array([False, False]),
        geometry="Cartesian_2.5D",
        staggered=False,
        field_names=("rho", "m1", "m2", "m3", "e", "b1", "b2", "b3"),
        physics="mhd",
        parameters={"gamma": 5.0 / 3.0},
    )


def _fixed_name(value: str) -> bytes:
    return value.encode("ascii").ljust(16, b"\0")


def _write_dat_v5(path: Path, *, time: float = 0.0) -> Path:
    """Write one compact, deterministic Cartesian 2.5D dat-v5 fixture."""

    gamma = 5.0 / 3.0
    shape = (3, 3)
    conserved = np.zeros((*shape, len(FIELD_NAMES)), dtype=np.float64, order="F")
    conserved[..., 0] = 2.0
    conserved[..., 1] = 1.0
    conserved[..., 2] = 2.0
    conserved[..., 3] = 3.0
    conserved[..., 5] = 0.5
    conserved[..., 6] = 0.75
    conserved[..., 7] = 1.25
    kinetic = 0.5 * (1.0**2 + 2.0**2 + 3.0**2) / 2.0
    magnetic = 0.5 * (0.5**2 + 0.75**2 + 1.25**2)
    conserved[..., 4] = 4.0 / (gamma - 1.0) + kinetic + magnetic

    def header(offset_tree: int, offset_blocks: int) -> bytes:
        return b"".join(
            (
                struct.pack("=i", 5),
                struct.pack(
                    "=9id",
                    offset_tree,
                    offset_blocks,
                    len(FIELD_NAMES),
                    3,
                    2,
                    1,
                    1,
                    0,
                    0,
                    time,
                ),
                struct.pack("=2d", 0.0, 0.0),
                struct.pack("=2d", 1.0, 1.0),
                struct.pack("=2i", *shape),
                struct.pack("=2i", *shape),
                struct.pack("=2i", 0, 0),
                _fixed_name("Cartesian_2.5D"),
                struct.pack("=i", 0),
                b"".join(_fixed_name(name) for name in FIELD_NAMES),
                _fixed_name("mhd"),
                struct.pack("=i", 1),
                struct.pack("=d", gamma),
                _fixed_name("gamma"),
            )
        )

    provisional = header(0, 0)
    offset_tree = len(provisional)
    tree_size = struct.calcsize("=ii2iq")
    offset_blocks = offset_tree + tree_size
    metadata = header(offset_tree, offset_blocks)
    tree = struct.pack("=ii2iq", 1, 1, 1, 1, offset_blocks)
    ghost_bounds = struct.pack("=4i", 0, shape[0] - 1, 0, shape[1] - 1)
    path.write_bytes(
        metadata + tree + ghost_bounds + conserved.tobytes(order="F")
    )
    return path


def test_pressure_recovery_uses_m3_and_b3() -> None:
    gamma = 5.0 / 3.0
    rho = 2.0
    momentum = np.array([1.0, 2.0, 3.0])
    magnetic = np.array([0.5, 0.75, 1.25])
    pressure = 4.0
    energy = (
        pressure / (gamma - 1.0)
        + 0.5 * np.sum(momentum**2) / rho
        + 0.5 * np.sum(magnetic**2)
    )
    conserved = np.zeros((2, 2, 8))
    conserved[..., 0] = rho
    conserved[..., 1:4] = momentum
    conserved[..., 4] = energy
    conserved[..., 5:8] = magnetic
    snapshot = AMRVACUniformSnapshot(
        header=_synthetic_header(),
        x=np.array([0.25, 0.75]),
        y=np.array([0.25, 0.75]),
        conserved=conserved,
        coverage=np.ones((2, 2), dtype=np.int16),
        target_level=1,
    )
    primitive = recover_primitive(
        snapshot,
        {
            "gamma": gamma,
            "background_magnetic_field": {"kind": "zero"},
        },
    )
    np.testing.assert_allclose(primitive.pressure, pressure, atol=1e-12)
    np.testing.assert_allclose(primitive.velocity_z, momentum[2] / rho, atol=1e-12)
    np.testing.assert_allclose(primitive.magnetic_z, magnetic[2], atol=1e-12)


def test_dat_v5_header_and_conservative_projection(tmp_path: Path) -> None:
    dat_path = _write_dat_v5(tmp_path / "fixture.dat")
    header = read_dat_v5_header(dat_path)
    assert header.version == 5
    assert (header.ndim, header.ndir) == (2, 3)
    projected = project_dat_to_uniform(dat_path)
    assert np.all(projected.coverage == 1)
    rho_index = header.field_names.index("rho")
    dx = (header.xmax - header.xmin) / np.asarray(projected.coverage.shape)
    mass = float(np.sum(projected.conserved[..., rho_index]) * np.prod(dx))
    assert np.isfinite(mass)
    assert mass > 0.0


def test_dat_reader_rejects_non_v5(tmp_path: Path) -> None:
    path = tmp_path / "bad.dat"
    path.write_bytes(struct.pack("=i", 4))
    with pytest.raises(ValueError, match="version 5"):
        read_dat_v5_header(path)


def test_dat_matches_vtu_total_field_and_pressure(tmp_path: Path) -> None:
    pyvista = pytest.importorskip("pyvista")
    dat_path = _write_dat_v5(tmp_path / "fixture.dat")
    vtu_path = tmp_path / "fixture.vtu"
    projected = project_dat_to_uniform(dat_path)
    primitive = recover_primitive(
        projected,
        {
            "gamma": 5.0 / 3.0,
            "background_magnetic_field": {
                "kind": "force_free_sheet",
                "amplitude": 50.0,
                "inverse_width": 20.0 / 3.0,
            },
        },
    )
    image = pyvista.ImageData(
        dimensions=(4, 4, 1),
        spacing=(1.0 / 3.0, 1.0 / 3.0, 1.0),
    )
    fixture = image.cast_to_unstructured_grid()
    for name, expected in (
        ("p", primitive.pressure),
        ("b1", primitive.magnetic_x),
        ("b2", primitive.magnetic_y),
        ("b3", primitive.magnetic_z),
    ):
        fixture.cell_data[name] = np.asarray(expected).ravel()
    fixture.save(vtu_path)
    mesh = pyvista.read(vtu_path)
    assert mesh.n_cells == primitive.rho.size
    # Cell ordering differs between the DAT bridge and VTK; compare invariant
    # extrema instead of assuming converter traversal order.
    for name, expected in (
        ("p", primitive.pressure),
        ("b1", primitive.magnetic_x),
        ("b2", primitive.magnetic_y),
        ("b3", primitive.magnetic_z),
    ):
        actual = np.asarray(mesh.cell_data[name], dtype=float)
        np.testing.assert_allclose(
            [actual.min(), actual.max()],
            [expected.min(), expected.max()],
            rtol=2e-5,
            atol=2e-5,
        )


def test_amrvac_ingest_requires_sidecar_and_writes_v5(tmp_path: Path) -> None:
    source = _write_dat_v5(tmp_path / "sfr_2.5d0000.dat")
    second = _write_dat_v5(tmp_path / "sfr_2.5d0001.dat", time=0.1)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gamma": 5.0 / 3.0,
                "resistivity": 0.0,
                "viscosity": 0.0,
                "source_content_hash": "fixture",
                "background_magnetic_field": {
                    "kind": "force_free_sheet",
                    "amplitude": 50.0,
                    "inverse_width": 20.0 / 3.0,
                },
                "diagnostic_geometry": {
                    "kind": "open_solar_jet",
                    "boundary_mode": "open",
                    "centers_normal": [6.0],
                    "x_points": [0.0],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "bridge.h5"
    ingest_amrvac([source, second], sidecar_path=sidecar, output_path=output)
    with h5py.File(output, "r") as handle:
        assert int(handle.attrs["schema_version"]) == 5
        provenance = json.loads(str(handle.attrs["provenance_json"]))
        assert provenance["solver"] == "mpi-amrvac"
        assert provenance["native_format_version"] == 5
        diagnostics = handle["development_diagnostics"]
        assert diagnostics["maximum_mach"].shape == (2,)
        assert diagnostics["boundary_mass_flux_outward"].shape == (2,)
        assert diagnostics["boundary_energy_flux_outward"].shape == (2,)
        assert diagnostics["global_jet_speed_p95"].shape == (2,)
        assert np.isfinite(diagnostics["local_resistivity_minimum"][...]).all()
    loaded = read_bridge_hdf5(output)
    assert loaded.source == "mpi-amrvac"
    assert loaded.velocity_z.shape == loaded.rho.shape
