"""Minimal, audited MPI-AMRVAC dat-v5 reader and schema-v5 bridge adapter.

The project intentionally does not import the vendor snapshot's Python reader:
that historical reader derives 2.5D pressure with ``ndim`` components and can
therefore omit ``m3`` and ``b3``.  This module always treats ``ndir=3`` as a
three-component state, even though the mesh has only two spatial dimensions.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

import h5py
import numpy as np
from numpy.typing import NDArray

from .athena_io import AthenaBinarySnapshot, build_field_series, write_bridge_hdf5
from .physics.fields import MHDFieldSeries

FloatArray = NDArray[np.float64]
NAME_LENGTH = 16


@dataclass(frozen=True)
class AMRVACHeader:
    """Privacy-neutral subset of one MPI-AMRVAC dat-v5 header."""

    version: int
    offset_tree: int
    offset_blocks: int
    nw: int
    ndir: int
    ndim: int
    levmax: int
    nleafs: int
    nparents: int
    iteration: int
    time: float
    xmin: FloatArray
    xmax: FloatArray
    domain_nx: NDArray[np.int32]
    block_nx: NDArray[np.int32]
    periodic: NDArray[np.bool_]
    geometry: str
    staggered: bool
    field_names: tuple[str, ...]
    physics: str
    parameters: dict[str, float]


@dataclass(frozen=True)
class AMRVACUniformSnapshot:
    """One conservative AMR projection on a cell-centered analysis grid."""

    header: AMRVACHeader
    x: FloatArray
    y: FloatArray
    conserved: FloatArray
    coverage: NDArray[np.int16]
    target_level: int


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated MPI-AMRVAC dat-v5 file.")
    return value


def _unpack(stream: BinaryIO, fmt: str) -> tuple[object, ...]:
    size = struct.calcsize("=" + fmt)
    return struct.unpack("=" + fmt, _read_exact(stream, size))


def _name(stream: BinaryIO) -> str:
    return _read_exact(stream, NAME_LENGTH).rstrip(b"\0 ").decode("ascii")


def read_dat_v5_header(path: Path) -> AMRVACHeader:
    """Read and validate a Cartesian 2.5D dat-v5 header."""

    with Path(path).open("rb") as stream:
        (version,) = _unpack(stream, "i")
        if version != 5:
            raise ValueError(f"Expected MPI-AMRVAC dat version 5, found {version}.")
        values = _unpack(stream, "9id")
        (
            offset_tree,
            offset_blocks,
            nw,
            ndir,
            ndim,
            levmax,
            nleafs,
            nparents,
            iteration,
            time,
        ) = values
        if ndim != 2 or ndir != 3:
            raise ValueError(
                f"Expected Cartesian 2.5D (ndim=2, ndir=3), found "
                f"ndim={ndim}, ndir={ndir}."
            )
        xmin = np.asarray(_unpack(stream, "2d"), dtype=float)
        xmax = np.asarray(_unpack(stream, "2d"), dtype=float)
        domain_nx = np.asarray(_unpack(stream, "2i"), dtype=np.int32)
        block_nx = np.asarray(_unpack(stream, "2i"), dtype=np.int32)
        periodic = np.asarray(_unpack(stream, "2i"), dtype=bool)
        geometry = _name(stream)
        (staggered_raw,) = _unpack(stream, "i")
        field_names = tuple(_name(stream) for _ in range(int(nw)))
        physics = _name(stream)
        (parameter_count,) = _unpack(stream, "i")
        parameter_values = _unpack(stream, f"{parameter_count}d")
        parameter_names = tuple(_name(stream) for _ in range(parameter_count))
    if geometry.lower() not in {"cartesian_2.5d", "cartesian"}:
        raise ValueError(f"Unsupported AMRVAC geometry {geometry!r}.")
    if min(*domain_nx, *block_nx, levmax, nleafs) < 1:
        raise ValueError("AMRVAC header contains a non-positive grid size.")
    required = {"rho", "m1", "m2", "m3", "e", "b1", "b2", "b3"}
    if not required.issubset(field_names):
        raise ValueError(
            f"AMRVAC dat file is missing conservative MHD fields: "
            f"{sorted(required - set(field_names))}"
        )
    return AMRVACHeader(
        version=int(version),
        offset_tree=int(offset_tree),
        offset_blocks=int(offset_blocks),
        nw=int(nw),
        ndir=int(ndir),
        ndim=int(ndim),
        levmax=int(levmax),
        nleafs=int(nleafs),
        nparents=int(nparents),
        iteration=int(iteration),
        time=float(time),
        xmin=xmin,
        xmax=xmax,
        domain_nx=domain_nx,
        block_nx=block_nx,
        periodic=periodic,
        geometry=geometry,
        staggered=bool(staggered_raw),
        field_names=field_names,
        physics=physics,
        parameters=dict(zip(parameter_names, parameter_values, strict=True)),
    )


def _tree(stream: BinaryIO, header: AMRVACHeader) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int64],
]:
    stream.seek(header.offset_tree + 4 * (header.nleafs + header.nparents))
    levels = np.asarray(
        _unpack(stream, f"{header.nleafs}i"),
        dtype=np.int32,
    )
    indices = np.asarray(
        _unpack(stream, f"{header.nleafs * header.ndim}i"),
        dtype=np.int32,
    ).reshape(header.nleafs, header.ndim)
    offsets = np.asarray(
        _unpack(stream, f"{header.nleafs}q"),
        dtype=np.int64,
    )
    # Each block starts with lower/upper ghost-cell integer bounds.
    offsets += 2 * header.ndim * 4
    if np.any(levels < 1) or np.any(levels > header.levmax):
        raise ValueError("AMRVAC tree contains an invalid refinement level.")
    return levels, indices, offsets


def project_dat_to_uniform(
    path: Path,
    *,
    target_level: int | None = None,
) -> AMRVACUniformSnapshot:
    """Conservatively project leaf-cell averages to one uniform grid.

    A coarse leaf is replicated into its exact fine-cell footprint.  Because
    all values are cell averages, this piecewise-constant prolongation
    preserves every volume integral to round-off on a Cartesian mesh.
    """

    header = read_dat_v5_header(path)
    level = header.levmax if target_level is None else int(target_level)
    if level < 1 or level > header.levmax:
        raise ValueError("target_level must lie inside the dat tree.")
    shape_xy = header.domain_nx * 2 ** (level - 1)
    data = np.full((*shape_xy, header.nw), np.nan, dtype=float, order="F")
    coverage = np.zeros(tuple(shape_xy), dtype=np.int16)
    with Path(path).open("rb") as stream:
        levels, indices, offsets = _tree(stream, header)
        for leaf_level, index, offset in zip(
            levels,
            indices,
            offsets,
            strict=True,
        ):
            if leaf_level > level:
                raise ValueError(
                    "Requested analysis level is coarser than a leaf block."
                )
            stream.seek(int(offset))
            count = int(np.prod(header.block_nx) * header.nw)
            block = np.frombuffer(
                _read_exact(stream, count * 8),
                dtype=np.dtype("=f8"),
            ).reshape((*header.block_nx, header.nw), order="F")
            factor = 2 ** (level - int(leaf_level))
            expanded = np.repeat(
                np.repeat(block, factor, axis=0),
                factor,
                axis=1,
            )
            start = (index - 1) * header.block_nx * factor
            stop = start + np.asarray(expanded.shape[:2], dtype=int)
            selection = np.s_[start[0] : stop[0], start[1] : stop[1]]
            if np.any(coverage[selection]):
                raise ValueError("AMRVAC leaf blocks overlap on the analysis grid.")
            data[selection] = expanded
            coverage[selection] += 1
    if not np.all(coverage == 1) or not np.isfinite(data).all():
        raise ValueError("AMRVAC leaf blocks do not cover the analysis grid exactly.")
    dx = (header.xmax - header.xmin) / shape_xy
    x = header.xmin[0] + (np.arange(shape_xy[0]) + 0.5) * dx[0]
    y = header.xmin[1] + (np.arange(shape_xy[1]) + 0.5) * dx[1]
    return AMRVACUniformSnapshot(
        header=header,
        x=x,
        y=y,
        conserved=data,
        coverage=coverage,
        target_level=level,
    )


def _background_field(
    x: FloatArray,
    y: FloatArray,
    sidecar: dict[str, object],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    config = sidecar.get("background_magnetic_field")
    if not isinstance(config, dict):
        raise TypeError("AMRVAC sidecar lacks background_magnetic_field.")
    kind = str(config.get("kind", ""))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    if kind == "zero":
        zero = np.zeros_like(xx)
        return zero, zero.copy(), zero.copy()
    if kind == "force_free_sheet":
        amplitude = float(config["amplitude"])
        inverse_width = float(config["inverse_width"])
        return (
            np.zeros_like(xx),
            -amplitude * np.tanh(inverse_width * xx),
            amplitude / np.cosh(inverse_width * xx),
        )
    if kind == "open_dipole":
        amplitude = float(config["amplitude"])
        depth = float(config["depth"])
        null_height = float(config["null_height"])
        guide_ratio = float(config["guide_ratio"])
        yp = yy + depth
        radius2 = xx**2 + yp**2
        moment = amplitude * (null_height + depth) ** 2
        bx = -2.0 * moment * xx * yp / radius2**2
        by = amplitude - moment * (yp**2 - xx**2) / radius2**2
        bz = np.full_like(xx, guide_ratio * amplitude)
        return bx, by, bz
    raise ValueError(f"Unsupported AMRVAC background field {kind!r}.")


def recover_primitive(
    snapshot: AMRVACUniformSnapshot,
    sidecar: dict[str, object],
) -> AthenaBinarySnapshot:
    """Recover 2.5D primitive variables including m3 and b3 energy."""

    header = snapshot.header
    index = {name: position for position, name in enumerate(header.field_names)}
    conserved = snapshot.conserved
    rho = conserved[..., index["rho"]]
    if np.any(rho <= 0.0):
        raise ValueError("AMRVAC density is not strictly positive.")
    momentum = [conserved[..., index[f"m{i}"]] for i in (1, 2, 3)]
    perturbation = [conserved[..., index[f"b{i}"]] for i in (1, 2, 3)]
    energy = conserved[..., index["e"]]
    gamma = float(sidecar.get("gamma", header.parameters.get("gamma", 5.0 / 3.0)))
    kinetic = 0.5 * sum(component**2 for component in momentum) / rho
    # With B0field, AMRVAC evolves perturbation energy; the static background
    # field is added to the Lorentz force and output but not to this state e.
    magnetic_perturbation = 0.5 * sum(component**2 for component in perturbation)
    pressure = (gamma - 1.0) * (energy - kinetic - magnetic_perturbation)
    if np.any(pressure <= 0.0):
        raise ValueError("Recovered AMRVAC pressure is not strictly positive.")
    background = _background_field(snapshot.x, snapshot.y, sidecar)
    total = [
        perturbation[component] + background[component]
        for component in range(3)
    ]
    # AMRVAC arrays are [x,y]; the common bridge is [y,x].
    transpose = lambda value: np.asarray(value.T, dtype=float)
    return AthenaBinarySnapshot(
        time=header.time,
        dt=0.0,
        gamma=gamma,
        x=snapshot.x,
        y=snapshot.y,
        rho=transpose(rho),
        pressure=transpose(pressure),
        velocity_x=transpose(momentum[0] / rho),
        velocity_y=transpose(momentum[1] / rho),
        velocity_z=transpose(momentum[2] / rho),
        magnetic_x=transpose(total[0]),
        magnetic_y=transpose(total[1]),
        magnetic_z=transpose(total[2]),
        precision_bytes=8,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sidecar(path: Path) -> dict[str, object]:
    """Load a mandatory privacy-safe background-field/provenance sidecar."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        raise ValueError("AMRVAC sidecar must use schema_version 1.")
    serialized = json.dumps(data, ensure_ascii=False)
    if any(token in serialized for token in ("/Users/", "/home/", "@")):
        raise ValueError("AMRVAC sidecar contains a private locator.")
    return data


def ingest_amrvac(
    dat_paths: list[Path],
    *,
    sidecar_path: Path,
    output_path: Path,
) -> MHDFieldSeries:
    """Project dat-v5 snapshots and create a schema-v5 common bridge."""

    if len(dat_paths) < 2:
        raise ValueError("AMRVAC ingest requires at least two dat snapshots.")
    sidecar = load_sidecar(sidecar_path)
    projected = [project_dat_to_uniform(path) for path in sorted(dat_paths)]
    snapshots = [recover_primitive(snapshot, sidecar) for snapshot in projected]
    geometry = sidecar.get("diagnostic_geometry", {})
    if not isinstance(geometry, dict):
        raise TypeError("diagnostic_geometry must be a mapping.")
    series = build_field_series(
        snapshots,
        resistivity=float(sidecar.get("resistivity", 0.0)),
        viscosity=float(sidecar.get("viscosity", 0.0)),
        sheet_half_width=float(geometry.get("sheet_half_width", 0.2)),
        sheet_center_fraction=0.25,
        perturbation_kx=1.0,
        geometry_kind=str(geometry.get("kind", "open_solar_jet")),
        boundary_mode=str(geometry.get("boundary_mode", "open")),
        diagnostic_centers_y=tuple(
            float(value) for value in geometry.get("centers_normal", [2.0])
        ),
        diagnostic_x_points=tuple(
            float(value) for value in geometry.get("x_points", [0.0])
        ),
        source="mpi-amrvac",
        sheet_normal=str(geometry.get("sheet_normal", "y")),
        outflow_direction=str(geometry.get("local_outflow_direction", "x")),
        vertical_direction=str(geometry.get("global_jet_direction", "y")),
    )
    # AMRVAC evolves only the perturbation field when B0field is enabled.
    # Powell cleaning applies to that field; the analytic sidecar background
    # is divergence-free by construction and must not be re-differentiated
    # with a coarse cell-centered stencil for this acceptance diagnostic.
    divergence_values: list[float] = []
    for native, primitive in zip(projected, snapshots, strict=True):
        field_index = {
            name: index for index, name in enumerate(native.header.field_names)
        }
        b1 = native.conserved[..., field_index["b1"]]
        b2 = native.conserved[..., field_index["b2"]]
        dx = float(np.median(np.diff(native.x)))
        dy = float(np.median(np.diff(native.y)))
        divergence = np.gradient(b1, dx, axis=0, edge_order=2) + np.gradient(
            b2,
            dy,
            axis=1,
            edge_order=2,
        )
        field_scale = np.sqrt(
            np.mean(
                primitive.magnetic_x**2
                + primitive.magnetic_y**2
                + primitive.magnetic_z**2
            )
        )
        divergence_values.append(
            float(
                np.max(np.abs(divergence))
                * min(dx, dy)
                / max(field_scale, 1.0e-30)
            )
        )
    series = replace(
        series,
        divergence_normalized_rms=np.asarray(divergence_values, dtype=float),
    )
    provenance = {
        "solver": "mpi-amrvac",
        "source_content_hash": str(sidecar.get("source_content_hash", "unknown")),
        "native_format": "MPI-AMRVAC dat",
        "native_format_version": 5,
        "amr_levels": sorted(
            {
                read_dat_v5_header(path).levmax
                for path in dat_paths
            }
        ),
        "analysis_grid": [len(series.grid.x), len(series.grid.y)],
        "projection_method": "volume-conservative piecewise-constant",
        "magnetic_storage": "perturbation plus sidecar B0field",
        "energy_convention": "e excludes static B0field; includes m1..m3 and b1..b3",
    }
    write_bridge_hdf5(
        series,
        output_path,
        source_manifest=[
            {"name": path.name, "sha256": _sha256(path)}
            for path in sorted(dat_paths)
        ],
        unit_metadata=dict(sidecar.get("normalization", {"system": "dimensionless"})),
        provenance_metadata=provenance,
        diagnostic_metadata={
            "resistivity_model": {
                "background": float(sidecar.get("resistivity", 0.0)),
                "anomalous": float(sidecar.get("resistivity", 0.0)),
                "threshold": float("inf"),
            },
        },
    )
    with h5py.File(output_path, "r+") as handle:
        handle.attrs["native_sidecar_sha256"] = _sha256(sidecar_path)
    return series
