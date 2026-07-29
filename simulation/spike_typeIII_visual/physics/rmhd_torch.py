"""PyTorch/CUDA pseudo-spectral reduced-MHD backend.

The evolved state remains in the half-complex Fourier representation.  Each
right-hand-side evaluation performs batched inverse transforms for all spatial
derivatives and one batched forward transform for the nonlinear brackets.
Only requested snapshots cross the device boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import MHDConfig
from .rmhd import (
    MHDResult,
    SpectralGrid,
    _diagnostics,
    _energy_budget_residual,
    _extended_diagnostics,
    double_harris_flux,
)

_FIELD_OPERATOR_CACHE: dict[tuple[object, ...], tuple[object, ...]] = {}


def _torch_module():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "The torch RMHD engine requires PyTorch. Activate the WSL "
            "'torch-cuda' environment."
        ) from exc
    return torch


def cuda_available() -> bool:
    """Return whether the optional PyTorch runtime can access CUDA."""

    try:
        torch = _torch_module()
    except RuntimeError:
        return False
    return bool(torch.cuda.is_available())


def snapshot_fields_cuda(
    psi: np.ndarray,
    omega: np.ndarray,
    grid: SpectralGrid,
    *,
    precision: str = "float64",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one saved snapshot with two batched CUDA FFT calls."""

    torch = _torch_module()
    if not torch.cuda.is_available():
        return grid.fields(psi, omega)
    real_dtype = torch.float64 if precision == "float64" else torch.float32
    key = (
        grid.x.size,
        grid.y.size,
        float(grid.x[-1] - grid.x[0] + (grid.x[1] - grid.x[0])),
        float(grid.y[-1] - grid.y[0] + (grid.y[1] - grid.y[0])),
        precision,
    )
    operators = _FIELD_OPERATOR_CACHE.get(key)
    if operators is None:
        dx = float(grid.x[1] - grid.x[0])
        dy = float(grid.y[1] - grid.y[0])
        kx = 2.0 * np.pi * np.fft.rfftfreq(grid.x.size, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(grid.y.size, d=dy)
        kx_mesh, ky_mesh = np.meshgrid(kx, ky)
        k2 = kx_mesh**2 + ky_mesh**2
        operators = (
            torch.as_tensor(kx_mesh, dtype=real_dtype, device="cuda"),
            torch.as_tensor(ky_mesh, dtype=real_dtype, device="cuda"),
            torch.as_tensor(k2, dtype=real_dtype, device="cuda"),
        )
        _FIELD_OPERATOR_CACHE[key] = operators
    kx_t, ky_t, k2_t = operators
    state = torch.as_tensor(
        np.stack((psi, omega)),
        dtype=real_dtype,
        device="cuda",
    )
    state_hat = torch.fft.rfft2(state)
    psi_hat, omega_hat = state_hat
    phi_hat = torch.zeros_like(omega_hat)
    nonzero = k2_t > 0.0
    phi_hat[nonzero] = -omega_hat[nonzero] / k2_t[nonzero]
    output_shape = (grid.y.size, grid.x.size)
    fields_hat = torch.stack(
        (
            -1j * ky_t * psi_hat,
            1j * kx_t * psi_hat,
            -1j * ky_t * phi_hat,
            1j * kx_t * phi_hat,
            k2_t * psi_hat,
            phi_hat,
        )
    )
    fields = torch.fft.irfft2(fields_hat, s=output_shape)
    arrays = fields.detach().cpu().numpy().astype(np.float64, copy=False)
    return tuple(arrays[index] for index in range(6))  # type: ignore[return-value]


def solve_rmhd_torch(
    config: MHDConfig,
    *,
    device: str = "auto",
    precision: str = "float64",
    progress: Callable[[int, int], None] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    resume: bool = False,
) -> MHDResult:
    """Run reduced MHD with a resident spectral PyTorch state."""

    torch = _torch_module()
    if precision not in {"float64", "float32"}:
        raise ValueError("precision must be 'float64' or 'float32'.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")

    real_dtype = torch.float64 if precision == "float64" else torch.float32
    complex_dtype = torch.complex128 if precision == "float64" else torch.complex64
    torch_device = torch.device(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    grid = SpectralGrid.from_config(config)
    from ..checkpoint import load_checkpoint, save_checkpoint

    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative.")
    initial_psi = double_harris_flux(config, grid)
    psi_physical = torch.as_tensor(initial_psi, dtype=real_dtype, device=torch_device)
    omega_physical = torch.zeros_like(psi_physical)
    psi_hat = torch.fft.rfft2(psi_physical).to(complex_dtype)
    omega_hat = torch.fft.rfft2(omega_physical).to(complex_dtype)

    dx = config.lx / config.nx
    dy = config.ly / config.ny
    kx = 2.0 * np.pi * np.fft.rfftfreq(config.nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(config.ny, d=dy)
    kx_mesh, ky_mesh = np.meshgrid(kx, ky)
    k_squared = kx_mesh**2 + ky_mesh**2
    mode_x = np.fft.rfftfreq(config.nx) * config.nx
    mode_y = np.fft.fftfreq(config.ny) * config.ny
    mode_x_mesh, mode_y_mesh = np.meshgrid(mode_x, mode_y)
    mask_np = (np.abs(mode_x_mesh) < config.nx / 3.0) & (
        np.abs(mode_y_mesh) < config.ny / 3.0
    )
    kx_t = torch.as_tensor(kx_mesh, dtype=real_dtype, device=torch_device)
    ky_t = torch.as_tensor(ky_mesh, dtype=real_dtype, device=torch_device)
    k2_t = torch.as_tensor(k_squared, dtype=real_dtype, device=torch_device)
    mask_t = torch.as_tensor(mask_np, dtype=real_dtype, device=torch_device)
    nonzero_t = k2_t > 0.0
    output_shape = (config.ny, config.nx)

    def filtered(value):
        return value * mask_t

    psi_hat = filtered(psi_hat)
    omega_hat = filtered(omega_hat)

    def rhs(current_psi_hat, current_omega_hat):
        phi_hat = torch.zeros_like(current_omega_hat)
        phi_hat[nonzero_t] = -current_omega_hat[nonzero_t] / k2_t[nonzero_t]
        current_hat = k2_t * current_psi_hat
        fields_hat = torch.stack(
            (phi_hat, current_psi_hat, current_omega_hat, current_hat)
        )
        derivative_x = torch.fft.irfft2(
            1j * kx_t.unsqueeze(0) * fields_hat,
            s=output_shape,
        )
        derivative_y = torch.fft.irfft2(
            1j * ky_t.unsqueeze(0) * fields_hat,
            s=output_shape,
        )
        phi_x, psi_x, omega_x, current_x = derivative_x
        phi_y, psi_y, omega_y, current_y = derivative_y
        bracket_phi_psi = phi_x * psi_y - phi_y * psi_x
        bracket_phi_omega = phi_x * omega_y - phi_y * omega_x
        if config.lorentz_convention == "physical":
            lorentz = current_x * psi_y - current_y * psi_x
        else:
            lorentz = psi_x * current_y - psi_y * current_x
        nonlinear_hat = torch.fft.rfft2(
            torch.stack(
                (
                    -bracket_phi_psi,
                    -bracket_phi_omega + lorentz,
                )
            )
        )
        dpsi_hat = nonlinear_hat[0] - config.resistivity * k2_t * current_psi_hat
        domega_hat = nonlinear_hat[1] - config.viscosity * k2_t * current_omega_hat
        return filtered(dpsi_hat), filtered(domega_hat)

    def rk4(current_psi_hat, current_omega_hat):
        dt = config.dt
        p1, w1 = rhs(current_psi_hat, current_omega_hat)
        p2, w2 = rhs(
            current_psi_hat + 0.5 * dt * p1,
            current_omega_hat + 0.5 * dt * w1,
        )
        p3, w3 = rhs(
            current_psi_hat + 0.5 * dt * p2,
            current_omega_hat + 0.5 * dt * w2,
        )
        p4, w4 = rhs(current_psi_hat + dt * p3, current_omega_hat + dt * w3)
        next_psi = current_psi_hat + (dt / 6.0) * (
            p1 + 2.0 * p2 + 2.0 * p3 + p4
        )
        next_omega = current_omega_hat + (dt / 6.0) * (
            w1 + 2.0 * w2 + 2.0 * w3 + w4
        )
        return filtered(next_psi), filtered(next_omega)

    snapshot_steps = list(range(0, config.steps + 1, config.snapshot_stride))
    if snapshot_steps[-1] != config.steps:
        snapshot_steps.append(config.steps)
    saved_step_set = set(snapshot_steps)
    saved_times: list[float] = []
    saved_psi: list[np.ndarray] = []
    saved_omega: list[np.ndarray] = []
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

    def to_numpy(value_hat) -> np.ndarray:
        value = torch.fft.irfft2(value_hat, s=output_shape)
        return value.detach().cpu().numpy().astype(np.float64, copy=True)

    def save_state(step: int) -> None:
        psi_np = to_numpy(psi_hat)
        omega_np = to_numpy(omega_hat)
        base = _diagnostics(psi_np, omega_np, config, grid)
        extended = _extended_diagnostics(psi_np, omega_np, config, grid)
        saved_times.append(step * config.dt)
        saved_psi.append(psi_np)
        saved_omega.append(omega_np)
        magnetic_energy.append(base[0])
        kinetic_energy.append(base[1])
        max_current.append(base[2])
        max_speed.append(base[3])
        reconnection_proxy.append(base[4])
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
            checkpoint_path, config=config, engine="torch", precision=precision
        )
        start_step = int(restored.pop("step"))
        psi_physical = torch.as_tensor(
            restored.pop("psi"), dtype=real_dtype, device=torch_device
        )
        omega_physical = torch.as_tensor(
            restored.pop("omega"), dtype=real_dtype, device=torch_device
        )
        psi_hat = filtered(torch.fft.rfft2(psi_physical).to(complex_dtype))
        omega_hat = filtered(torch.fft.rfft2(omega_physical).to(complex_dtype))
        for name, destination in history_lists.items():
            destination.extend(np.asarray(restored[name]).tolist())

    with torch.inference_mode():
        if not resume:
            save_state(0)
        for step in range(start_step + 1, config.steps + 1):
            psi_hat, omega_hat = rk4(psi_hat, omega_hat)
            if not bool(torch.isfinite(psi_hat).all()) or not bool(
                torch.isfinite(omega_hat).all()
            ):
                raise FloatingPointError(f"Non-finite CUDA MHD state at step {step}.")
            if step in saved_step_set:
                save_state(step)
            if progress is not None:
                progress(step, config.steps)
            if (
                checkpoint_path is not None
                and checkpoint_every
                and (step % checkpoint_every == 0 or step == config.steps)
            ):
                save_checkpoint(
                    checkpoint_path,
                    config=config,
                    engine="torch",
                    precision=precision,
                    step=step,
                    psi=to_numpy(psi_hat),
                    omega=to_numpy(omega_hat),
                    history=history_lists,
                )

    final_psi = saved_psi[-1]
    final_omega = saved_omega[-1]
    magnetic_x, magnetic_y, *_ = grid.fields(final_psi, final_omega)
    divergence = grid.derivative_x(magnetic_x) + grid.derivative_y(magnetic_y)
    field_rms = float(np.sqrt(np.mean(magnetic_x**2 + magnetic_y**2)))
    divergence_rms = float(
        np.sqrt(np.mean(divergence**2)) / max(field_rms, 1.0e-15)
    )
    times = np.asarray(saved_times, dtype=float)
    magnetic = np.asarray(magnetic_energy, dtype=float)
    kinetic = np.asarray(kinetic_energy, dtype=float)
    ohmic = np.asarray(ohmic_dissipation, dtype=float)
    viscous = np.asarray(viscous_dissipation, dtype=float)
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    )
    return MHDResult(
        grid=grid,
        times=times,
        psi=np.asarray(saved_psi),
        omega=np.asarray(saved_omega),
        magnetic_energy=magnetic,
        kinetic_energy=kinetic,
        max_current=np.asarray(max_current),
        max_speed=np.asarray(max_speed),
        reconnection_proxy=np.asarray(reconnection_proxy),
        divergence_rms=divergence_rms,
        flux_difference=np.asarray(flux_difference),
        xpoint_electric_field=np.asarray(xpoint_electric_field),
        island_width_proxy=np.asarray(island_width_proxy),
        ohmic_dissipation=ohmic,
        viscous_dissipation=viscous,
        energy_budget_residual=_energy_budget_residual(
            times,
            magnetic,
            kinetic,
            ohmic,
            viscous,
        ),
        execution_backend="torch",
        execution_device=device,
        execution_precision=precision,
        peak_device_memory_bytes=peak_memory,
    )
