"""Athena C binary ingestion and privacy-safe HDF5 bridge files."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Self

import h5py
import numpy as np
from numpy.typing import NDArray

from .physics.fields import FieldGrid, MHDFieldSeries, MHDGeometry

FloatArray = NDArray[np.float64]

_NUMBERED_DUMP = re.compile(r"\.(\d+)\.bin$")


class MHDFieldDataset:
    """Lazy, snapshot-oriented access to a schema-v3/v4/v5 bridge.

    This avoids loading a long 2.5D field history into memory.  The caller owns
    the context manager lifetime; returned snapshots are ordinary NumPy arrays.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle: h5py.File | None = None

    def __enter__(self) -> Self:
        self._handle = h5py.File(self.path, "r")
        schema = int(self._handle.attrs.get("schema_version", 0))
        if schema not in {3, 4, 5}:
            self._handle.close()
            self._handle = None
            raise ValueError(f"Unsupported Athena bridge schema {schema}.")
        return self

    def __exit__(self, *args: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            raise RuntimeError("MHDFieldDataset must be used as a context manager.")
        return self._handle

    @property
    def schema_version(self) -> int:
        return int(self.handle.attrs["schema_version"])

    @property
    def times(self) -> FloatArray:
        return np.asarray(self.handle["time"], dtype=float)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.handle["rho"].shape)

    def snapshot(self, index: int) -> dict[str, FloatArray]:
        """Read one snapshot, promoting absent schema-v3 components to zero."""

        names = (
            "rho",
            "pressure",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
            "current_x",
            "current_y",
            "current_z",
            "psi",
            "omega_x",
            "omega_y",
            "omega",
        )
        zero = np.zeros(self.shape[1:], dtype=np.float32)
        return {
            name: (
                np.asarray(self.handle[name][index], dtype=np.float32)
                if name in self.handle
                else zero.copy()
            )
            for name in names
        }


@dataclass(frozen=True)
class AthenaBinarySnapshot:
    """One primitive-variable Athena C dump."""

    time: float
    dt: float
    gamma: float
    x: FloatArray
    y: FloatArray
    rho: FloatArray
    pressure: FloatArray
    velocity_x: FloatArray
    velocity_y: FloatArray
    velocity_z: FloatArray
    magnetic_x: FloatArray
    magnetic_y: FloatArray
    magnetic_z: FloatArray
    precision_bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump_index(path: Path) -> int:
    match = _NUMBERED_DUMP.search(path.name)
    if match is None:
        raise ValueError(f"Not a numbered Athena binary dump: {path.name}")
    return int(match.group(1))


def discover_binary_dumps(run_dir: Path) -> list[Path]:
    """Return numerically ordered Athena C binary dumps."""

    paths = [
        path
        for path in run_dir.rglob("*.bin")
        if _NUMBERED_DUMP.search(path.name) is not None
    ]
    return sorted(paths, key=lambda path: (_dump_index(path), path.as_posix()))


def _merge_domain_snapshots(
    snapshots: list[AthenaBinarySnapshot],
) -> AthenaBinarySnapshot:
    """Tile one MPI dump time from coordinate-bearing domain files."""

    if not snapshots:
        raise ValueError("Cannot merge an empty Athena snapshot group.")
    if len(snapshots) == 1:
        return snapshots[0]
    first = snapshots[0]
    for snapshot in snapshots[1:]:
        if not np.isclose(snapshot.time, first.time, atol=1e-13):
            raise ValueError("MPI domain dumps do not share one output time.")
        if not np.isclose(snapshot.gamma, first.gamma):
            raise ValueError("MPI domain dumps disagree on gamma.")
    x = np.unique(np.concatenate([snapshot.x for snapshot in snapshots]))
    y = np.unique(np.concatenate([snapshot.y for snapshot in snapshots]))
    names = (
        "rho",
        "pressure",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "magnetic_x",
        "magnetic_y",
        "magnetic_z",
    )
    merged = {
        name: np.full((len(y), len(x)), np.nan, dtype=float) for name in names
    }
    coverage = np.zeros((len(y), len(x)), dtype=np.int16)
    for snapshot in snapshots:
        x_indices = np.searchsorted(x, snapshot.x)
        y_indices = np.searchsorted(y, snapshot.y)
        selection = np.ix_(y_indices, x_indices)
        coverage[selection] += 1
        for name in names:
            merged[name][selection] = getattr(snapshot, name)
    if not np.all(coverage == 1):
        raise ValueError("MPI domain dumps have a gap or overlapping grid cells.")
    return AthenaBinarySnapshot(
        time=first.time,
        dt=min(snapshot.dt for snapshot in snapshots),
        gamma=first.gamma,
        x=x,
        y=y,
        precision_bytes=min(snapshot.precision_bytes for snapshot in snapshots),
        **merged,
    )


def read_dump_series(paths: Iterable[Path]) -> list[AthenaBinarySnapshot]:
    """Read serial files or merge MPI domain files by output index."""

    groups: dict[int, list[AthenaBinarySnapshot]] = {}
    for path in paths:
        groups.setdefault(_dump_index(path), []).append(read_athena_binary(path))
    return [
        _merge_domain_snapshots(groups[index]) for index in sorted(groups)
    ]


def read_athena_binary(path: Path) -> AthenaBinarySnapshot:
    """Read an Athena C v4 primitive binary dump.

    The v4 format contains a native-endian integer header followed by either
    single- or double-precision reals.  Official project outputs are little
    endian on both supported target platforms.
    """

    raw = path.read_bytes()
    if len(raw) < 32:
        raise ValueError(f"Athena dump is too short: {path.name}")
    coordsys, *ndata = struct.unpack_from("<8i", raw, 0)
    nx, ny, nz, nvar, nscalars, self_gravity, particles = ndata
    if coordsys != -1:
        raise ValueError("Only Cartesian Athena C dumps are supported.")
    if min(nx, ny, nz) < 1:
        raise ValueError(f"Invalid Athena grid shape {(nx, ny, nz)}.")
    if nvar != 8 or nscalars != 0 or self_gravity != 0 or particles != 0:
        raise ValueError(
            "The bridge requires adiabatic MHD primitive dumps with "
            "NVAR=8, no scalars, gravity, or particles."
        )

    cell_count = nx * ny * nz
    real_count = 4 + nx + ny + nz + nvar * cell_count
    payload_bytes = len(raw) - 32
    if payload_bytes == real_count * 8:
        dtype = np.dtype("<f8")
    elif payload_bytes == real_count * 4:
        dtype = np.dtype("<f4")
    else:
        raise ValueError(
            f"Unexpected Athena payload size in {path.name}: "
            f"{payload_bytes} bytes for {real_count} values."
        )
    values = np.frombuffer(raw, dtype=dtype, offset=32, count=real_count)
    cursor = 0
    gamma_minus_one = float(values[cursor])
    cursor += 2  # skip isothermal sound speed
    time = float(values[cursor])
    dt = float(values[cursor + 1])
    cursor += 2
    x = np.asarray(values[cursor : cursor + nx], dtype=float)
    cursor += nx
    y = np.asarray(values[cursor : cursor + ny], dtype=float)
    cursor += ny
    cursor += nz  # z coordinates; the supported workflow is two-dimensional
    variables = np.asarray(
        values[cursor : cursor + nvar * cell_count],
        dtype=float,
    ).reshape((nvar, nz, ny, nx))
    if nz != 1:
        raise ValueError("The Spike-Topping bridge currently requires Nx3=1.")
    rho, vx, vy, vz, pressure, bx, by, bz = variables[:, 0]
    if gamma_minus_one <= 0.0:
        raise ValueError("Athena dump does not contain an adiabatic gamma.")
    return AthenaBinarySnapshot(
        time=time,
        dt=dt,
        gamma=gamma_minus_one + 1.0,
        x=x,
        y=y,
        rho=rho,
        pressure=pressure,
        velocity_x=vx,
        velocity_y=vy,
        velocity_z=vz,
        magnetic_x=bx,
        magnetic_y=by,
        magnetic_z=bz,
        precision_bytes=dtype.itemsize,
    )


def _periodic_derivative(
    field: FloatArray,
    spacing: float,
    axis: int,
) -> FloatArray:
    return (
        np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)
    ) / (2.0 * spacing)


def _derivative(
    field: FloatArray,
    spacing: float,
    axis: int,
    *,
    periodic: bool,
) -> FloatArray:
    if periodic:
        return _periodic_derivative(field, spacing, axis)
    return np.gradient(field, spacing, axis=axis, edge_order=2)


def _reconstruct_flux(current_z: FloatArray, dx: float, dy: float) -> FloatArray:
    ny, nx = current_z.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    k_squared = kx_mesh**2 + ky_mesh**2
    current_hat = np.fft.fft2(current_z)
    psi_hat = np.zeros_like(current_hat)
    nonzero = k_squared > 0.0
    psi_hat[nonzero] = -current_hat[nonzero] / k_squared[nonzero]
    return np.fft.ifft2(psi_hat).real


def _reconstruct_open_flux(
    magnetic_x: FloatArray,
    magnetic_y: FloatArray,
    dx: float,
    dy: float,
) -> FloatArray:
    """Reconstruct Az on a non-periodic grid by path integration."""

    ny, nx = magnetic_x.shape
    psi = np.zeros((ny, nx), dtype=float)
    for i in range(1, nx):
        psi[0, i] = psi[0, i - 1] - 0.5 * dx * (
            magnetic_y[0, i - 1] + magnetic_y[0, i]
        )
    for j in range(1, ny):
        psi[j] = psi[j - 1] + 0.5 * dy * (
            magnetic_x[j - 1] + magnetic_x[j]
        )
    return psi


def _expected_xpoints(x: FloatArray, perturbation_kx: float) -> tuple[float, ...]:
    x_min = float(x[0] - 0.5 * np.median(np.diff(x)))
    x_max = float(x[-1] + 0.5 * np.median(np.diff(x)))
    n_min = int(np.floor((perturbation_kx * x_min / np.pi - 1.0) / 2.0)) - 1
    n_max = int(np.ceil((perturbation_kx * x_max / np.pi - 1.0) / 2.0)) + 1
    candidates = [
        (2 * n + 1) * np.pi / perturbation_kx
        for n in range(n_min, n_max + 1)
    ]
    return tuple(value for value in candidates if x_min <= value < x_max)


def _flux_difference(
    psi: FloatArray,
    y: FloatArray,
    sheet_centers: tuple[float, ...],
) -> float:
    differences = []
    for center in sheet_centers:
        row = psi[int(np.argmin(np.abs(y - center)))]
        differences.append(float(np.max(row) - np.min(row)))
    return float(np.mean(differences))


def _xpoint_electric_field(
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    magnetic_x: FloatArray,
    magnetic_y: FloatArray,
    current_z: FloatArray,
    x: FloatArray,
    y: FloatArray,
    x_points: tuple[float, ...],
    sheet_centers: tuple[float, ...],
    resistivity: float,
) -> float:
    electric = (
        -(velocity_x * magnetic_y - velocity_y * magnetic_x)
        + resistivity * current_z
    )
    samples = []
    for center in sheet_centers:
        j = int(np.argmin(np.abs(y - center)))
        for x_point in x_points:
            i = int(np.argmin(np.abs(x - x_point)))
            window = electric[
                max(0, j - 1) : min(len(y), j + 2),
                np.mod(np.arange(i - 1, i + 2), len(x)),
            ]
            samples.append(float(np.median(np.abs(window))))
    return float(np.median(samples))


def build_field_series(
    snapshots: Iterable[AthenaBinarySnapshot],
    *,
    resistivity: float,
    viscosity: float,
    sheet_half_width: float,
    sheet_center_fraction: float,
    perturbation_kx: float,
    geometry_kind: str = "double_harris_periodic",
    boundary_mode: str = "periodic",
    diagnostic_centers_y: tuple[float, ...] | None = None,
    diagnostic_x_points: tuple[float, ...] | None = None,
    source: str = "athena-c",
    sheet_normal: str = "y",
    outflow_direction: str = "x",
    vertical_direction: str = "y",
) -> MHDFieldSeries:
    """Convert ordered Athena snapshots into the common field representation."""

    ordered = sorted(snapshots, key=lambda snapshot: snapshot.time)
    unique: list[AthenaBinarySnapshot] = []
    for snapshot in ordered:
        if unique and np.isclose(snapshot.time, unique[-1].time, atol=1e-14):
            unique[-1] = snapshot
        else:
            unique.append(snapshot)
    if len(unique) < 2:
        raise ValueError("At least two distinct Athena snapshots are required.")

    first = unique[0]
    for snapshot in unique[1:]:
        if not np.array_equal(snapshot.x, first.x) or not np.array_equal(
            snapshot.y,
            first.y,
        ):
            raise ValueError("Athena grid coordinates changed between dumps.")
        if not np.isclose(snapshot.gamma, first.gamma):
            raise ValueError("Athena gamma changed between dumps.")

    x = first.x
    y = first.y
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    lx = dx * len(x)
    ly = dy * len(y)
    periodic = boundary_mode == "periodic"
    if boundary_mode not in {"periodic", "open"}:
        raise ValueError("boundary_mode must be 'periodic' or 'open'.")
    sheet_centers = diagnostic_centers_y or (
        -sheet_center_fraction * ly,
        sheet_center_fraction * ly,
    )
    x_points = diagnostic_x_points or _expected_xpoints(x, perturbation_kx)
    if not x_points:
        raise ValueError("No seeded X points lie inside the Athena domain.")
    geometry = MHDGeometry(
        lx=lx,
        ly=ly,
        sheet_centers_y=sheet_centers,
        sheet_half_width=sheet_half_width,
        x_points=x_points,
        outflow_half_window=lx / 8.0,
        kind=geometry_kind,
        sheet_normal=sheet_normal,
        outflow_direction=outflow_direction,
        vertical_direction=vertical_direction,
    )

    times = np.asarray([snapshot.time for snapshot in unique], dtype=float)
    rho = np.asarray([snapshot.rho for snapshot in unique], dtype=np.float32)
    pressure = np.asarray(
        [snapshot.pressure for snapshot in unique],
        dtype=np.float32,
    )
    velocity_x = np.asarray(
        [snapshot.velocity_x for snapshot in unique],
        dtype=np.float32,
    )
    velocity_y = np.asarray(
        [snapshot.velocity_y for snapshot in unique],
        dtype=np.float32,
    )
    velocity_z = np.asarray(
        [snapshot.velocity_z for snapshot in unique],
        dtype=np.float32,
    )
    magnetic_x = np.asarray(
        [snapshot.magnetic_x for snapshot in unique],
        dtype=np.float32,
    )
    magnetic_y = np.asarray(
        [snapshot.magnetic_y for snapshot in unique],
        dtype=np.float32,
    )
    magnetic_z = np.asarray(
        [snapshot.magnetic_z for snapshot in unique],
        dtype=np.float32,
    )

    current_x_values = []
    current_y_values = []
    current_values = []
    divergence_values = []
    psi_values = []
    omega_x_values = []
    omega_y_values = []
    omega_values = []
    flux_values = []
    electric_values = []
    for index in range(len(times)):
        bx = np.asarray(magnetic_x[index], dtype=float)
        by = np.asarray(magnetic_y[index], dtype=float)
        bz = np.asarray(magnetic_z[index], dtype=float)
        vx = np.asarray(velocity_x[index], dtype=float)
        vy = np.asarray(velocity_y[index], dtype=float)
        vz = np.asarray(velocity_z[index], dtype=float)
        current_x = _derivative(bz, dy, 0, periodic=periodic)
        current_y = -_derivative(bz, dx, 1, periodic=periodic)
        current = _derivative(by, dx, 1, periodic=periodic) - _derivative(
            bx,
            dy,
            0,
            periodic=periodic,
        )
        divergence = _derivative(bx, dx, 1, periodic=periodic) + _derivative(
            by,
            dy,
            0,
            periodic=periodic,
        )
        psi = (
            _reconstruct_flux(current, dx, dy)
            if periodic
            else _reconstruct_open_flux(bx, by, dx, dy)
        )
        omega_x = _derivative(vz, dy, 0, periodic=periodic)
        omega_y = -_derivative(vz, dx, 1, periodic=periodic)
        omega = _derivative(vy, dx, 1, periodic=periodic) - _derivative(
            vx,
            dy,
            0,
            periodic=periodic,
        )
        current_x_values.append(current_x)
        current_y_values.append(current_y)
        current_values.append(current)
        divergence_values.append(divergence)
        psi_values.append(psi)
        omega_x_values.append(omega_x)
        omega_y_values.append(omega_y)
        omega_values.append(omega)
        flux_values.append(_flux_difference(psi, y, sheet_centers))
        electric_values.append(
            _xpoint_electric_field(
                vx,
                vy,
                bx,
                by,
                current,
                x,
                y,
                x_points,
                sheet_centers,
                resistivity,
            )
        )

    current_x = np.asarray(current_x_values, dtype=np.float32)
    current_y = np.asarray(current_y_values, dtype=np.float32)
    current_z = np.asarray(current_values, dtype=np.float32)
    psi = np.asarray(psi_values, dtype=np.float32)
    omega_x = np.asarray(omega_x_values, dtype=np.float32)
    omega_y = np.asarray(omega_y_values, dtype=np.float32)
    omega = np.asarray(omega_values, dtype=np.float32)
    flux_difference = np.asarray(flux_values, dtype=float)
    edge_order = 2 if len(times) >= 3 else 1
    reconnection_rate = np.abs(
        np.gradient(flux_difference, times, edge_order=edge_order)
    )
    xpoint_electric = np.asarray(electric_values, dtype=float)
    magnetic_energy = np.mean(
        0.5
        * (
            magnetic_x.astype(float) ** 2
            + magnetic_y.astype(float) ** 2
            + magnetic_z.astype(float) ** 2
        ),
        axis=(1, 2),
    )
    kinetic_energy = np.mean(
        0.5
        * rho.astype(float)
        * (
            velocity_x.astype(float) ** 2
            + velocity_y.astype(float) ** 2
            + velocity_z.astype(float) ** 2
        ),
        axis=(1, 2),
    )
    internal_energy = np.mean(
        pressure.astype(float) / (first.gamma - 1.0),
        axis=(1, 2),
    )
    total_energy = magnetic_energy + kinetic_energy + internal_energy
    max_current = np.max(np.abs(current_z.astype(float)), axis=(1, 2))
    max_speed = np.max(
        np.sqrt(
            velocity_x.astype(float) ** 2
            + velocity_y.astype(float) ** 2
            + velocity_z.astype(float) ** 2
        ),
        axis=(1, 2),
    )
    divergence_rms = []
    for index, divergence in enumerate(divergence_values):
        field_rms = float(
            np.sqrt(
                np.mean(
                    magnetic_x[index].astype(float) ** 2
                    + magnetic_y[index].astype(float) ** 2
                )
            )
        )
        divergence_rms.append(
            float(np.sqrt(np.mean(divergence**2)) / max(field_rms, 1e-15))
        )

    return MHDFieldSeries(
        source=source,
        grid=FieldGrid.from_coordinates(x, y),
        geometry=geometry,
        times=times,
        rho=rho,
        pressure=pressure,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        magnetic_x=magnetic_x,
        magnetic_y=magnetic_y,
        magnetic_z=magnetic_z,
        current_x=current_x,
        current_y=current_y,
        current_z=current_z,
        psi=psi,
        omega_x=omega_x,
        omega_y=omega_y,
        omega=omega,
        magnetic_energy=np.asarray(magnetic_energy, dtype=float),
        kinetic_energy=np.asarray(kinetic_energy, dtype=float),
        internal_energy=np.asarray(internal_energy, dtype=float),
        total_energy=np.asarray(total_energy, dtype=float),
        max_current=np.asarray(max_current, dtype=float),
        max_speed=np.asarray(max_speed, dtype=float),
        reconnection_proxy=np.asarray(reconnection_rate, dtype=float),
        flux_difference=flux_difference,
        xpoint_electric_field=xpoint_electric,
        divergence_normalized_rms=np.asarray(divergence_rms, dtype=float),
        gamma=float(first.gamma),
        resistivity=float(resistivity),
        viscosity=float(viscosity),
    )


def development_diagnostics(
    series: MHDFieldSeries,
    *,
    density_floor: float = 0.0,
    pressure_floor: float = 0.0,
    resistivity_model: dict[str, float] | None = None,
) -> dict[str, FloatArray]:
    """Return backend-neutral positivity, flux, resistivity, and jet diagnostics.

    Boundary fluxes are outward-positive line integrals on the rectangular
    analysis domain.  They are diagnostics, not a closed energy budget: source
    terms and the discrete solver fluxes still have to be included before an
    open-domain conservation claim is made.
    """

    rho = series.rho.astype(float)
    pressure = series.pressure.astype(float)
    vx = series.velocity_x.astype(float)
    vy = series.velocity_y.astype(float)
    vz = series.velocity_z.astype(float)
    bx = series.magnetic_x.astype(float)
    by = series.magnetic_y.astype(float)
    bz = series.magnetic_z.astype(float)
    speed_squared = vx**2 + vy**2 + vz**2
    sound_speed_squared = series.gamma * pressure / rho
    maximum_mach = np.max(
        np.sqrt(speed_squared / np.maximum(sound_speed_squared, 1.0e-30)),
        axis=(1, 2),
    )

    dx = float(np.median(np.diff(series.grid.x)))
    dy = float(np.median(np.diff(series.grid.y)))
    magnetic_squared = bx**2 + by**2 + bz**2
    total_energy_density = (
        pressure / (series.gamma - 1.0)
        + 0.5 * rho * speed_squared
        + 0.5 * magnetic_squared
    )
    enthalpy_factor = total_energy_density + pressure + 0.5 * magnetic_squared
    velocity_dot_b = vx * bx + vy * by + vz * bz
    energy_flux_x = enthalpy_factor * vx - velocity_dot_b * bx
    energy_flux_y = enthalpy_factor * vy - velocity_dot_b * by
    mass_flux_x = rho * vx
    mass_flux_y = rho * vy
    boundary_mass_flux = (
        np.sum(mass_flux_x[:, :, -1] - mass_flux_x[:, :, 0], axis=1) * dy
        + np.sum(mass_flux_y[:, -1, :] - mass_flux_y[:, 0, :], axis=1) * dx
    )
    boundary_energy_flux = (
        np.sum(energy_flux_x[:, :, -1] - energy_flux_x[:, :, 0], axis=1) * dy
        + np.sum(energy_flux_y[:, -1, :] - energy_flux_y[:, 0, :], axis=1) * dx
    )

    geometry = series.geometry
    vertical_velocity = vy if geometry.vertical_direction == "y" else vx
    vertical_mesh = (
        series.grid.y_mesh
        if geometry.vertical_direction == "y"
        else series.grid.x_mesh
    )
    transverse_mesh = (
        series.grid.x_mesh
        if geometry.vertical_direction == "y"
        else series.grid.y_mesh
    )
    launch = min(geometry.sheet_centers_y)
    plume_mask = (
        (vertical_mesh >= launch)
        & (
            np.abs(transverse_mesh - geometry.x_points[0])
            <= 2.0 * geometry.outflow_half_window
        )
    )
    global_jet_speed = []
    for snapshot in vertical_velocity:
        upward = snapshot[plume_mask]
        upward = upward[upward > 0.0]
        global_jet_speed.append(
            0.0 if upward.size == 0 else float(np.quantile(upward, 0.95))
        )
    global_jet_speed_array = np.asarray(global_jet_speed, dtype=float)
    span = float(np.ptp(global_jet_speed_array))
    global_jet_activity = (
        np.zeros_like(global_jet_speed_array)
        if span <= 1.0e-15
        else np.clip(
            (global_jet_speed_array - global_jet_speed_array[0]) / span,
            0.0,
            1.0,
        )
    )

    eta_model = resistivity_model or {}
    eta_background = float(eta_model.get("background", series.resistivity))
    eta_anomalous = float(eta_model.get("anomalous", eta_background))
    current_threshold = float(eta_model.get("threshold", np.inf))
    excess = np.maximum(np.abs(series.current_z.astype(float)) - current_threshold, 0.0)
    local_eta = eta_background + (eta_anomalous - eta_background) * (
        1.0 - np.exp(-(excess**2))
    )

    density_limit = density_floor * (1.0 + 1.0e-12)
    pressure_limit = pressure_floor * (1.0 + 1.0e-12)
    return {
        "minimum_density": np.min(rho, axis=(1, 2)),
        "minimum_pressure": np.min(pressure, axis=(1, 2)),
        "density_floor_count": np.count_nonzero(
            rho <= density_limit,
            axis=(1, 2),
        ).astype(float),
        "pressure_floor_count": np.count_nonzero(
            pressure <= pressure_limit,
            axis=(1, 2),
        ).astype(float),
        "maximum_mach": np.asarray(maximum_mach, dtype=float),
        "boundary_mass_flux_outward": np.asarray(boundary_mass_flux, dtype=float),
        "boundary_energy_flux_outward": np.asarray(
            boundary_energy_flux,
            dtype=float,
        ),
        "local_resistivity_minimum": np.min(local_eta, axis=(1, 2)),
        "local_resistivity_maximum": np.max(local_eta, axis=(1, 2)),
        "global_jet_speed_p95": global_jet_speed_array,
        "global_jet_activity": global_jet_activity,
    }


def ingest_run_directory(
    run_dir: Path,
    output_path: Path,
    *,
    resistivity: float = 0.002,
    viscosity: float = 0.002,
    sheet_half_width: float = 0.20,
    sheet_center_fraction: float = 0.25,
    perturbation_kx: float = 1.0,
    geometry_kind: str = "double_harris_periodic",
    boundary_mode: str = "periodic",
    diagnostic_centers_y: tuple[float, ...] | None = None,
    diagnostic_x_points: tuple[float, ...] | None = None,
    unit_metadata: dict[str, float | str] | None = None,
    provenance_metadata: dict[str, object] | None = None,
    diagnostic_metadata: dict[str, object] | None = None,
) -> MHDFieldSeries:
    """Read raw dumps, create one bridge file, and return its field series."""

    paths = discover_binary_dumps(run_dir)
    if not paths:
        raise FileNotFoundError("No numbered Athena .bin dumps were found.")
    snapshots = read_dump_series(paths)
    series = build_field_series(
        snapshots,
        resistivity=resistivity,
        viscosity=viscosity,
        sheet_half_width=sheet_half_width,
        sheet_center_fraction=sheet_center_fraction,
        perturbation_kx=perturbation_kx,
        geometry_kind=geometry_kind,
        boundary_mode=boundary_mode,
        diagnostic_centers_y=diagnostic_centers_y,
        diagnostic_x_points=diagnostic_x_points,
    )
    history_paths = sorted(run_dir.rglob("*.hst"))
    if history_paths:
        history = np.loadtxt(history_paths[0], comments="#", ndmin=2)
        if history.shape[1] >= 15:
            history_time = history[:, 0]
            div_b_squared = np.maximum(history[:, 14], 0.0)
            field_rms = np.sqrt(np.maximum(2.0 * series.magnetic_energy, 1e-30))
            ct_divergence = np.interp(
                series.times,
                history_time,
                np.sqrt(div_b_squared),
            ) / field_rms
            series = replace(
                series,
                divergence_normalized_rms=np.asarray(ct_divergence, dtype=float),
            )
    source_manifest = [
        {"name": path.name, "sha256": _sha256(path)} for path in paths
    ]
    write_bridge_hdf5(
        series,
        output_path,
        source_manifest=source_manifest,
        unit_metadata=unit_metadata,
        provenance_metadata=provenance_metadata,
        diagnostic_metadata=diagnostic_metadata,
    )
    return series


def write_bridge_hdf5(
    series: MHDFieldSeries,
    path: Path,
    *,
    source_manifest: list[dict[str, str]] | None = None,
    unit_metadata: dict[str, float | str] | None = None,
    provenance_metadata: dict[str, object] | None = None,
    diagnostic_metadata: dict[str, object] | None = None,
) -> None:
    """Write schema-v5 2.5D MHD fields with snapshot-aligned chunks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 5
        handle.attrs["dimensionality"] = "2.5D"
        handle.attrs["mhd_source"] = series.source
        handle.attrs["gamma"] = series.gamma
        handle.attrs["resistivity"] = series.resistivity
        handle.attrs["viscosity"] = series.viscosity
        handle.attrs["geometry_json"] = json.dumps(
            asdict(series.geometry),
            sort_keys=True,
        )
        handle.attrs["source_manifest_json"] = json.dumps(
            source_manifest or [],
            sort_keys=True,
        )
        handle.attrs["unit_metadata_json"] = json.dumps(
            unit_metadata or {"system": "dimensionless"},
            sort_keys=True,
        )
        handle.attrs["provenance_json"] = json.dumps(
            provenance_metadata
            or series.provenance
            or {
                "solver": series.source,
                "native_format": "Athena C primitive BIN",
                "magnetic_storage": "total field",
                "energy_convention": "three-component total field",
            },
            sort_keys=True,
        )
        handle.create_dataset("x", data=series.grid.x)
        handle.create_dataset("y", data=series.grid.y)
        handle.create_dataset("time", data=series.times)
        metadata = diagnostic_metadata or {}
        diagnostics = development_diagnostics(
            series,
            density_floor=float(metadata.get("density_floor", 0.0)),
            pressure_floor=float(metadata.get("pressure_floor", 0.0)),
            resistivity_model=(
                dict(metadata["resistivity_model"])
                if isinstance(metadata.get("resistivity_model"), dict)
                else None
            ),
        )
        diagnostic_group = handle.create_group("development_diagnostics")
        diagnostic_group.attrs["boundary_flux_convention"] = (
            "outward-positive analysis-grid line integral"
        )
        diagnostic_group.attrs["energy_budget_status"] = (
            "diagnostic-only; source terms and native face fluxes not closed"
        )
        for name, values in diagnostics.items():
            diagnostic_group.create_dataset(name, data=values)
        chunks = (1, len(series.grid.y), len(series.grid.x))
        for name in (
            "rho",
            "pressure",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
            "current_x",
            "current_y",
            "current_z",
            "psi",
            "omega_x",
            "omega_y",
            "omega",
        ):
            handle.create_dataset(
                name,
                data=getattr(series, name),
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
        for name in (
            "magnetic_energy",
            "kinetic_energy",
            "internal_energy",
            "total_energy",
            "max_current",
            "max_speed",
            "reconnection_proxy",
            "flux_difference",
            "xpoint_electric_field",
            "divergence_normalized_rms",
        ):
            handle.create_dataset(name, data=getattr(series, name))


def read_bridge_hdf5(path: Path) -> MHDFieldSeries:
    """Load a schema-v3, schema-v4, or schema-v5 bridge file.

    Schema v3 files are promoted to 2.5D by adding zero-valued out-of-plane,
    in-plane-current, and in-plane-vorticity arrays.
    """

    with h5py.File(path, "r") as handle:
        schema_version = int(handle.attrs.get("schema_version", 0))
        if schema_version not in {3, 4, 5}:
            raise ValueError(
                f"Unsupported Athena bridge schema {schema_version}; "
                "expected 3, 4, or 5."
            )
        geometry_data = json.loads(str(handle.attrs["geometry_json"]))
        geometry_data["sheet_centers_y"] = tuple(
            geometry_data["sheet_centers_y"]
        )
        geometry_data["x_points"] = tuple(geometry_data["x_points"])
        geometry = MHDGeometry(**geometry_data)
        x = np.asarray(handle["x"], dtype=float)
        y = np.asarray(handle["y"], dtype=float)
        legacy_field_names = (
            "rho",
            "pressure",
            "velocity_x",
            "velocity_y",
            "magnetic_x",
            "magnetic_y",
            "current_z",
            "psi",
            "omega",
        )
        fields = {
            name: np.asarray(handle[name], dtype=np.float32)
            for name in legacy_field_names
        }
        zero_shape = fields["rho"].shape
        for name in (
            "velocity_z",
            "magnetic_z",
            "current_x",
            "current_y",
            "omega_x",
            "omega_y",
        ):
            fields[name] = (
                np.asarray(handle[name], dtype=np.float32)
                if name in handle
                else np.zeros(zero_shape, dtype=np.float32)
            )
        diagnostics = {
            name: np.asarray(handle[name], dtype=float)
            for name in (
                "magnetic_energy",
                "kinetic_energy",
                "internal_energy",
                "total_energy",
                "max_current",
                "max_speed",
                "reconnection_proxy",
                "flux_difference",
                "xpoint_electric_field",
                "divergence_normalized_rms",
            )
        }
        return MHDFieldSeries(
            source=str(handle.attrs["mhd_source"]),
            grid=FieldGrid.from_coordinates(x, y),
            geometry=geometry,
            times=np.asarray(handle["time"], dtype=float),
            gamma=float(handle.attrs["gamma"]),
            resistivity=float(handle.attrs["resistivity"]),
            viscosity=float(handle.attrs["viscosity"]),
            provenance=json.loads(str(handle.attrs.get("provenance_json", "{}"))),
            **fields,
            **diagnostics,
        )


def initial_balance_metrics(series: MHDFieldSeries) -> dict[str, float]:
    """Return initial positivity, pressure-balance, and divergence metrics."""

    index = 0
    total_pressure = (
        series.pressure[index].astype(float)
        + 0.5
        * (
            series.magnetic_x[index].astype(float) ** 2
            + series.magnetic_y[index].astype(float) ** 2
            + series.magnetic_z[index].astype(float) ** 2
        )
    )
    residual = total_pressure - float(np.mean(total_pressure))
    return {
        "minimum_density": float(np.min(series.rho[index])),
        "minimum_pressure": float(np.min(series.pressure[index])),
        "total_pressure_max_abs_residual": float(np.max(np.abs(residual))),
        "ct_divergence_normalized_rms": float(
            series.divergence_normalized_rms[index]
        ),
    }


def initial_binary_metrics(snapshot: AthenaBinarySnapshot) -> dict[str, float]:
    """Measure double-precision initial-state positivity and pressure balance."""

    total_pressure = snapshot.pressure + 0.5 * (
        snapshot.magnetic_x**2
        + snapshot.magnetic_y**2
        + snapshot.magnetic_z**2
    )
    residual = total_pressure - float(np.mean(total_pressure))
    return {
        "binary_precision_bytes": float(snapshot.precision_bytes),
        "minimum_density": float(np.min(snapshot.rho)),
        "minimum_pressure": float(np.min(snapshot.pressure)),
        "total_pressure_max_abs_residual": float(np.max(np.abs(residual))),
    }
