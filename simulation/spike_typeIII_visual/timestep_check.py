"""Isolated CUDA time-step halving check at fixed spatial resolution."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .physics.jet import reconnection_flux_rate
from .physics.rmhd_torch import solve_rmhd_torch
from .storage import read_rmhd_hdf5


def check(
    reference_hdf5: Path, *, engine: str = "torch", device: str = "cuda"
) -> dict[str, object]:
    reference, config, _ = read_rmhd_hdf5(reference_hdf5, lazy=True)
    coarse_time_config = replace(
        config,
        dt=2.0 * config.dt,
        steps=config.steps // 2,
        snapshot_stride=max(1, config.snapshot_stride // 2),
    )
    if not np.isclose(
        coarse_time_config.dt * coarse_time_config.steps,
        config.dt * config.steps,
    ):
        raise ValueError("Reference step count must be divisible by two.")
    if engine == "torch":
        candidate = solve_rmhd_torch(
            coarse_time_config, device=device, precision="float64"
        )
    else:
        from .physics.rmhd import solve_rmhd

        candidate = solve_rmhd(coarse_time_config)

    def relative(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1.0e-15)

    candidate_rate = reconnection_flux_rate(candidate)
    reference_rate = reconnection_flux_rate(reference)
    metrics = {
        "peak_max_speed": relative(
            float(np.max(candidate.max_speed)),
            float(np.max(reference.max_speed)),
        ),
        "final_flux_difference": relative(
            float(candidate.flux_difference[-1]),
            float(reference.flux_difference[-1]),
        ),
        "peak_reconnection_flux_rate": relative(
            float(np.max(candidate_rate)),
            float(np.max(reference_rate)),
        ),
        "final_psi_l2": float(
            np.linalg.norm(candidate.psi[-1] - reference.psi[-1])
            / max(float(np.linalg.norm(reference.psi[-1])), 1.0e-15)
        ),
        "final_omega_l2": float(
            np.linalg.norm(candidate.omega[-1] - reference.omega[-1])
            / max(float(np.linalg.norm(reference.omega[-1])), 1.0e-15)
        ),
    }
    return {
        "schema": "rmhd-timestep-halving-v1",
        "grid": [config.nx, config.ny],
        "coarse_dt": coarse_time_config.dt,
        "fine_dt": config.dt,
        "end_time": config.dt * config.steps,
        "relative_changes": metrics,
        "core_diagnostics_below_1_percent": all(
            metrics[name] < 0.01
            for name in (
                "peak_max_speed",
                "final_flux_difference",
                "peak_reconnection_flux_rate",
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    report = check(args.reference, engine=args.engine, device=args.device)
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["core_diagnostics_below_1_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
