"""Solver-independent 2D and 2.5D MHD field containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]


@dataclass(frozen=True)
class FieldGrid:
    """Cartesian cell-center coordinates used by all MHD backends."""

    x: FloatArray
    y: FloatArray
    x_mesh: FloatArray
    y_mesh: FloatArray

    @classmethod
    def from_coordinates(cls, x: FloatArray, y: FloatArray) -> FieldGrid:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        x_mesh, y_mesh = np.meshgrid(x_array, y_array)
        return cls(x_array, y_array, x_mesh, y_mesh)


@dataclass(frozen=True)
class MHDGeometry:
    """Geometry needed by sheet-localized reconnection diagnostics."""

    lx: float
    ly: float
    sheet_centers_y: tuple[float, ...]
    sheet_half_width: float
    x_points: tuple[float, ...]
    outflow_half_window: float
    sheet_normal: str = "y"
    outflow_direction: str = "x"
    kind: str = "double_harris_periodic"
    vertical_direction: str = "y"

    def __post_init__(self) -> None:
        for name in ("sheet_normal", "outflow_direction", "vertical_direction"):
            if getattr(self, name) not in {"x", "y"}:
                raise ValueError(f"{name} must be 'x' or 'y'.")
        if self.sheet_normal == self.outflow_direction:
            raise ValueError("Sheet normal and local outflow direction must differ.")


@dataclass(frozen=True)
class MHDFieldSeries:
    """Full-MHD fields and diagnostics in a backend-neutral representation.

    The spatial grid is two-dimensional, while all three vector components are
    retained.  This is the standard 2.5D convention ``d/dz = 0``.
    """

    source: str
    grid: FieldGrid
    geometry: MHDGeometry
    times: FloatArray
    rho: Float32Array
    pressure: Float32Array
    velocity_x: Float32Array
    velocity_y: Float32Array
    velocity_z: Float32Array
    magnetic_x: Float32Array
    magnetic_y: Float32Array
    magnetic_z: Float32Array
    current_x: Float32Array
    current_y: Float32Array
    current_z: Float32Array
    psi: Float32Array
    omega_x: Float32Array
    omega_y: Float32Array
    omega: Float32Array
    magnetic_energy: FloatArray
    kinetic_energy: FloatArray
    internal_energy: FloatArray
    total_energy: FloatArray
    max_current: FloatArray
    max_speed: FloatArray
    reconnection_proxy: FloatArray
    flux_difference: FloatArray
    xpoint_electric_field: FloatArray
    divergence_normalized_rms: FloatArray
    gamma: float
    resistivity: float
    viscosity: float
    provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        expected = (len(self.times), len(self.grid.y), len(self.grid.x))
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
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{name} has shape {value.shape}; expected {expected}.")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        if np.any(self.rho <= 0.0) or np.any(self.pressure <= 0.0):
            raise ValueError("Density and pressure must remain strictly positive.")
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
            value = getattr(self, name)
            if value.shape != self.times.shape:
                raise ValueError(
                    f"{name} has shape {value.shape}; expected {self.times.shape}."
                )
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values.")
        if not np.all(np.diff(self.times) > 0.0):
            raise ValueError("MHD times must be strictly increasing.")

    @property
    def divergence_rms(self) -> float:
        """Maximum normalized cell-centered divergence diagnostic."""

        return float(np.max(self.divergence_normalized_rms))

    def snapshot_fields(
        self,
        index: int,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """Return fields using the legacy RMHD plotting tuple."""

        return (
            np.asarray(self.magnetic_x[index], dtype=float),
            np.asarray(self.magnetic_y[index], dtype=float),
            np.asarray(self.velocity_x[index], dtype=float),
            np.asarray(self.velocity_y[index], dtype=float),
            np.asarray(self.current_z[index], dtype=float),
            np.asarray(self.psi[index], dtype=float),
        )

    def snapshot_vectors(
        self,
        index: int,
    ) -> dict[str, tuple[FloatArray, FloatArray, FloatArray]]:
        """Return all vector components for one 2.5D snapshot."""

        return {
            "velocity": (
                np.asarray(self.velocity_x[index], dtype=float),
                np.asarray(self.velocity_y[index], dtype=float),
                np.asarray(self.velocity_z[index], dtype=float),
            ),
            "magnetic": (
                np.asarray(self.magnetic_x[index], dtype=float),
                np.asarray(self.magnetic_y[index], dtype=float),
                np.asarray(self.magnetic_z[index], dtype=float),
            ),
            "current": (
                np.asarray(self.current_x[index], dtype=float),
                np.asarray(self.current_y[index], dtype=float),
                np.asarray(self.current_z[index], dtype=float),
            ),
            "vorticity": (
                np.asarray(self.omega_x[index], dtype=float),
                np.asarray(self.omega_y[index], dtype=float),
                np.asarray(self.omega[index], dtype=float),
            ),
        }
