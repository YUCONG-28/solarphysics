r"""Pseudo-spectral two-dimensional incompressible reduced MHD.

The normalized sign convention used by the implementation is

    B = (-partial_y psi, partial_x psi)
    v = (-partial_y phi, partial_x phi)
    j = -laplacian(psi)
    omega = laplacian(phi)

and the evolved equations are

    partial_t psi + [phi, psi] = eta laplacian(psi)
    partial_t omega + [phi, omega] = [j, psi] + nu laplacian(omega).

With ``j = -laplacian(psi)``, the physical Lorentz convention is ``[j, psi]``.
The former ``[psi, j]`` sign remains available only as a named legacy
diagnostic.  A periodic double-Harris equilibrium is used so FFT derivatives
remain consistent with the periodic pseudo-spectral grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import MHDConfig

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpectralGrid:
    """Periodic Cartesian grid and Fourier wave numbers."""

    x: FloatArray
    y: FloatArray
    x_mesh: FloatArray
    y_mesh: FloatArray
    kx_mesh: FloatArray
    ky_mesh: FloatArray
    k_squared: FloatArray
    dealias_mask: NDArray[np.bool_]

    @classmethod
    def from_config(cls, config: MHDConfig) -> SpectralGrid:
        x = np.linspace(-config.lx / 2.0, config.lx / 2.0, config.nx, endpoint=False)
        y = np.linspace(-config.ly / 2.0, config.ly / 2.0, config.ny, endpoint=False)
        dx = config.lx / config.nx
        dy = config.ly / config.ny
        kx = 2.0 * np.pi * np.fft.fftfreq(config.nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(config.ny, d=dy)
        x_mesh, y_mesh = np.meshgrid(x, y)
        kx_mesh, ky_mesh = np.meshgrid(kx, ky)
        k_squared = kx_mesh**2 + ky_mesh**2
        mode_x = np.fft.fftfreq(config.nx) * config.nx
        mode_y = np.fft.fftfreq(config.ny) * config.ny
        mode_x_mesh, mode_y_mesh = np.meshgrid(mode_x, mode_y)
        # Strict quadratic 2/3 truncation: do not retain the exact N/3 modes.
        dealias_mask = (np.abs(mode_x_mesh) < config.nx / 3.0) & (
            np.abs(mode_y_mesh) < config.ny / 3.0
        )
        return cls(
            x=x,
            y=y,
            x_mesh=x_mesh,
            y_mesh=y_mesh,
            kx_mesh=kx_mesh,
            ky_mesh=ky_mesh,
            k_squared=k_squared,
            dealias_mask=dealias_mask,
        )

    def filter(self, field: FloatArray) -> FloatArray:
        transformed = np.fft.fft2(field) * self.dealias_mask
        return np.fft.ifft2(transformed).real

    def derivative_x(self, field: FloatArray) -> FloatArray:
        return np.fft.ifft2(1j * self.kx_mesh * np.fft.fft2(field)).real

    def derivative_y(self, field: FloatArray) -> FloatArray:
        return np.fft.ifft2(1j * self.ky_mesh * np.fft.fft2(field)).real

    def laplacian(self, field: FloatArray) -> FloatArray:
        return np.fft.ifft2(-self.k_squared * np.fft.fft2(field)).real

    def poisson_solve(self, source: FloatArray) -> FloatArray:
        source_hat = np.fft.fft2(source)
        solution_hat = np.zeros_like(source_hat)
        nonzero = self.k_squared > 0.0
        solution_hat[nonzero] = -source_hat[nonzero] / self.k_squared[nonzero]
        return np.fft.ifft2(solution_hat).real

    def bracket(self, first: FloatArray, second: FloatArray) -> FloatArray:
        return self.derivative_x(first) * self.derivative_y(second) - self.derivative_y(
            first
        ) * self.derivative_x(second)

    def fields(
        self, psi: FloatArray, omega: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        phi = self.poisson_solve(omega)
        magnetic_x = -self.derivative_y(psi)
        magnetic_y = self.derivative_x(psi)
        velocity_x = -self.derivative_y(phi)
        velocity_y = self.derivative_x(phi)
        current = -self.laplacian(psi)
        return magnetic_x, magnetic_y, velocity_x, velocity_y, current, phi


@dataclass(frozen=True)
class MHDResult:
    """Saved MHD states and scalar diagnostics."""

    grid: SpectralGrid
    times: FloatArray
    psi: FloatArray
    omega: FloatArray
    magnetic_energy: FloatArray
    kinetic_energy: FloatArray
    max_current: FloatArray
    max_speed: FloatArray
    reconnection_proxy: FloatArray
    divergence_rms: float
    flux_difference: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    xpoint_electric_field: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    island_width_proxy: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    ohmic_dissipation: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    viscous_dissipation: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    energy_budget_residual: FloatArray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    execution_backend: str = "numpy"
    execution_device: str = "cpu"
    execution_precision: str = "float64"
    peak_device_memory_bytes: int = 0

    def snapshot_fields(
        self, index: int
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        if self.execution_backend == "torch" and self.execution_device == "cuda":
            from .rmhd_torch import snapshot_fields_cuda

            return snapshot_fields_cuda(
                self.psi[index],
                self.omega[index],
                self.grid,
                precision=self.execution_precision,
            )
        return self.grid.fields(self.psi[index], self.omega[index])


def double_harris_flux(config: MHDConfig, grid: SpectralGrid) -> FloatArray:
    """Return a periodic double-Harris magnetic flux equilibrium."""

    width = config.sheet_half_width
    center = config.sheet_center_fraction * config.ly
    y_lower = -center
    y_upper = center
    y = grid.y_mesh
    flux = (
        -width * np.log(np.cosh((y - y_lower) / width))
        + width * np.log(np.cosh((y - y_upper) / width))
        + y
    )
    envelope = np.exp(-(((y - y_lower) / config.perturbation_width) ** 2)) - np.exp(
        -(((y - y_upper) / config.perturbation_width) ** 2)
    )
    perturbation = config.perturbation_amplitude * np.cos(grid.x_mesh) * envelope
    return grid.filter(flux + perturbation)


def _rhs(
    psi: FloatArray,
    omega: FloatArray,
    config: MHDConfig,
    grid: SpectralGrid,
) -> tuple[FloatArray, FloatArray]:
    phi = grid.poisson_solve(omega)
    current = -grid.laplacian(psi)
    d_psi = -grid.bracket(phi, psi) + config.resistivity * grid.laplacian(psi)
    if config.lorentz_convention == "physical":
        lorentz = grid.bracket(current, psi)
    else:
        lorentz = grid.bracket(psi, current)
    d_omega = (
        -grid.bracket(phi, omega) + lorentz + config.viscosity * grid.laplacian(omega)
    )
    return grid.filter(d_psi), grid.filter(d_omega)


def ideal_energy_exchange_residual(
    psi: FloatArray,
    omega: FloatArray,
    config: MHDConfig,
    grid: SpectralGrid,
) -> float:
    """Return the instantaneous ideal total-energy exchange residual."""

    ideal_config = replace(config, resistivity=0.0, viscosity=0.0)
    d_psi, d_omega = _rhs(psi, omega, ideal_config, grid)
    d_phi = grid.poisson_solve(d_omega)
    magnetic_rate = float(np.mean((-grid.laplacian(psi)) * d_psi))
    kinetic_rate = float(-np.mean(omega * d_phi))
    return magnetic_rate + kinetic_rate


def _rk4_step(
    psi: FloatArray,
    omega: FloatArray,
    config: MHDConfig,
    grid: SpectralGrid,
) -> tuple[FloatArray, FloatArray]:
    dt = config.dt
    p1, w1 = _rhs(psi, omega, config, grid)
    p2, w2 = _rhs(psi + 0.5 * dt * p1, omega + 0.5 * dt * w1, config, grid)
    p3, w3 = _rhs(psi + 0.5 * dt * p2, omega + 0.5 * dt * w2, config, grid)
    p4, w4 = _rhs(psi + dt * p3, omega + dt * w3, config, grid)
    next_psi = psi + (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4)
    next_omega = omega + (dt / 6.0) * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
    return grid.filter(next_psi), grid.filter(next_omega)


def _diagnostics(
    psi: FloatArray,
    omega: FloatArray,
    config: MHDConfig,
    grid: SpectralGrid,
) -> tuple[float, float, float, float, float]:
    magnetic_x, magnetic_y, velocity_x, velocity_y, current, _ = grid.fields(psi, omega)
    magnetic_energy = float(np.mean(magnetic_x**2 + magnetic_y**2) / 2.0)
    kinetic_energy = float(np.mean(velocity_x**2 + velocity_y**2) / 2.0)
    max_current = float(np.max(np.abs(current)))
    max_speed = float(np.max(np.hypot(velocity_x, velocity_y)))
    reconnection_proxy = config.resistivity * max_current
    return (
        magnetic_energy,
        kinetic_energy,
        max_current,
        max_speed,
        reconnection_proxy,
    )


def _extended_diagnostics(
    psi: FloatArray,
    omega: FloatArray,
    config: MHDConfig,
    grid: SpectralGrid,
) -> tuple[float, float, float, float, float]:
    """Return topology and dissipation diagnostics for one saved state.

    The island-width value is the standard slab estimate
    ``4 sqrt(|psi_O-psi_X| / max|j_sheet|)``.  It is a reduced-MHD diagnostic,
    not a geometrical contour fit.
    """

    _, _, _, _, current, _ = grid.fields(psi, omega)
    sheet_center = config.sheet_center_fraction * config.ly
    row = int(np.argmin(np.abs(grid.y - sheet_center)))
    sheet_flux = psi[row]
    o_index = int(np.argmax(sheet_flux))
    x_index = int(np.argmin(sheet_flux))
    flux_difference = float(abs(sheet_flux[o_index] - sheet_flux[x_index]))
    xpoint_electric_field = float(config.resistivity * current[row, x_index])
    current_scale = max(float(np.max(np.abs(current[row]))), 1.0e-15)
    island_width = float(4.0 * np.sqrt(flux_difference / current_scale))
    ohmic = float(config.resistivity * np.mean(current**2))
    viscous = float(config.viscosity * np.mean(omega**2))
    return flux_difference, xpoint_electric_field, island_width, ohmic, viscous


def _energy_budget_residual(
    times: FloatArray,
    magnetic_energy: FloatArray,
    kinetic_energy: FloatArray,
    ohmic: FloatArray,
    viscous: FloatArray,
) -> FloatArray:
    """Return ``E(t)-E(0)+integral(eta*j^2+nu*omega^2) dt``."""

    total = magnetic_energy + kinetic_energy
    dissipation = ohmic + viscous
    integral = np.zeros_like(total)
    if len(total) > 1:
        increments = 0.5 * (dissipation[1:] + dissipation[:-1]) * np.diff(times)
        integral[1:] = np.cumsum(increments)
    return total - total[0] + integral


def solve_rmhd(
    config: MHDConfig,
    progress: Callable[[int, int], None] | None = None,
    *,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    resume: bool = False,
    stop_after_step: int | None = None,
) -> MHDResult:
    """Run the deterministic reduced-MHD simulation."""

    grid = SpectralGrid.from_config(config)
    from ..checkpoint import load_checkpoint, save_checkpoint

    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative.")
    psi = double_harris_flux(config, grid)
    omega = np.zeros_like(psi)
    snapshot_steps = list(range(0, config.steps + 1, config.snapshot_stride))
    if snapshot_steps[-1] != config.steps:
        snapshot_steps.append(config.steps)
    saved_step_set = set(snapshot_steps)

    saved_times: list[float] = []
    saved_psi: list[FloatArray] = []
    saved_omega: list[FloatArray] = []
    magnetic_energy: list[float] = []
    kinetic_energy: list[float] = []
    max_current: list[float] = []
    max_speed: list[float] = []
    reconnection_proxy: list[float] = []
    flux_difference: list[float] = []
    xpoint_electric_field: list[float] = []
    island_width_proxy: list[float] = []
    ohmic_dissipation: list[float] = []
    viscous_dissipation: list[float] = []

    def save_state(step: int) -> None:
        diagnostics = _diagnostics(psi, omega, config, grid)
        saved_times.append(step * config.dt)
        saved_psi.append(psi.copy())
        saved_omega.append(omega.copy())
        magnetic_energy.append(diagnostics[0])
        kinetic_energy.append(diagnostics[1])
        max_current.append(diagnostics[2])
        max_speed.append(diagnostics[3])
        reconnection_proxy.append(diagnostics[4])
        extended = _extended_diagnostics(psi, omega, config, grid)
        flux_difference.append(extended[0])
        xpoint_electric_field.append(extended[1])
        island_width_proxy.append(extended[2])
        ohmic_dissipation.append(extended[3])
        viscous_dissipation.append(extended[4])

    history_lists = {
        "times": saved_times,
        "saved_psi": saved_psi,
        "saved_omega": saved_omega,
        "magnetic_energy": magnetic_energy,
        "kinetic_energy": kinetic_energy,
        "max_current": max_current,
        "max_speed": max_speed,
        "reconnection_proxy": reconnection_proxy,
        "flux_difference": flux_difference,
        "xpoint_electric_field": xpoint_electric_field,
        "island_width_proxy": island_width_proxy,
        "ohmic_dissipation": ohmic_dissipation,
        "viscous_dissipation": viscous_dissipation,
    }
    start_step = 0
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path.")
        restored = load_checkpoint(
            checkpoint_path, config=config, engine="numpy", precision="float64"
        )
        start_step = int(restored.pop("step"))
        psi = np.asarray(restored.pop("psi"), dtype=float)
        omega = np.asarray(restored.pop("omega"), dtype=float)
        for name, destination in history_lists.items():
            destination.extend(np.asarray(restored[name]).tolist())
    else:
        save_state(0)

    last_step = config.steps
    if stop_after_step is not None:
        last_step = min(last_step, stop_after_step)
    for step in range(start_step + 1, last_step + 1):
        psi, omega = _rk4_step(psi, omega, config, grid)
        if not np.isfinite(psi).all() or not np.isfinite(omega).all():
            raise FloatingPointError(f"Non-finite MHD state at step {step}.")
        if step in saved_step_set:
            save_state(step)
        if progress is not None:
            progress(step, config.steps)
        if (
            checkpoint_path is not None
            and checkpoint_every
            and (step % checkpoint_every == 0 or step == last_step)
        ):
            save_checkpoint(
                checkpoint_path,
                config=config,
                engine="numpy",
                precision="float64",
                step=step,
                psi=psi,
                omega=omega,
                history=history_lists,
            )

    magnetic_x, magnetic_y, *_ = grid.fields(psi, omega)
    divergence = grid.derivative_x(magnetic_x) + grid.derivative_y(magnetic_y)
    field_rms = float(np.sqrt(np.mean(magnetic_x**2 + magnetic_y**2)))
    divergence_rms = float(np.sqrt(np.mean(divergence**2)) / max(field_rms, 1e-15))

    saved_times_array = np.asarray(saved_times, dtype=float)
    magnetic_energy_array = np.asarray(magnetic_energy, dtype=float)
    kinetic_energy_array = np.asarray(kinetic_energy, dtype=float)
    ohmic_array = np.asarray(ohmic_dissipation, dtype=float)
    viscous_array = np.asarray(viscous_dissipation, dtype=float)
    return MHDResult(
        grid=grid,
        times=saved_times_array,
        psi=np.asarray(saved_psi, dtype=float),
        omega=np.asarray(saved_omega, dtype=float),
        magnetic_energy=magnetic_energy_array,
        kinetic_energy=kinetic_energy_array,
        max_current=np.asarray(max_current, dtype=float),
        max_speed=np.asarray(max_speed, dtype=float),
        reconnection_proxy=np.asarray(reconnection_proxy, dtype=float),
        divergence_rms=divergence_rms,
        flux_difference=np.asarray(flux_difference, dtype=float),
        xpoint_electric_field=np.asarray(xpoint_electric_field, dtype=float),
        island_width_proxy=np.asarray(island_width_proxy, dtype=float),
        ohmic_dissipation=ohmic_array,
        viscous_dissipation=viscous_array,
        energy_budget_residual=_energy_budget_residual(
            saved_times_array,
            magnetic_energy_array,
            kinetic_energy_array,
            ohmic_array,
            viscous_array,
        ),
    )
