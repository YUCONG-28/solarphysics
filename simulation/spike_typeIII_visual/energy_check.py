"""Verify ideal energy exchange and dissipative energy-budget behavior."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import profile_config
from .physics.rmhd_torch import solve_rmhd_torch
from .storage import read_rmhd_hdf5


def check(
    dissipative_hdf5: Path, *, engine: str = "torch", device: str = "cuda"
) -> dict[str, object]:
    dissipative, _, _ = read_rmhd_hdf5(dissipative_hdf5, lazy=True)
    ideal_config = replace(
        profile_config("quick", 20260726).mhd,
        resistivity=0.0,
        viscosity=0.0,
        dt=1.0e-4,
        steps=100,
        snapshot_stride=1,
    )
    if engine == "torch":
        ideal = solve_rmhd_torch(
            ideal_config, device=device, precision="float64"
        )
    else:
        from .physics.rmhd import solve_rmhd

        ideal = solve_rmhd(ideal_config)
    ideal_energy = ideal.magnetic_energy + ideal.kinetic_energy
    ideal_exchange_residual = float(
        np.max(np.abs(ideal_energy - ideal_energy[0]))
        / max(float(ideal_energy[0]), 1.0e-15)
    )
    dissipative_energy = (
        dissipative.magnetic_energy + dissipative.kinetic_energy
    )
    maximum_increase = float(
        np.max(np.diff(dissipative_energy))
        / max(float(dissipative_energy[0]), 1.0e-15)
    )
    budget = float(
        np.max(np.abs(dissipative.energy_budget_residual))
        / max(float(dissipative_energy[0]), 1.0e-15)
    )
    return {
        "schema": "rmhd-energy-gates-v1",
        "ideal_exchange_residual_fraction": ideal_exchange_residual,
        "ideal_below_1e-10": ideal_exchange_residual < 1.0e-10,
        "dissipative_maximum_single_step_increase_fraction": maximum_increase,
        "dissipative_nonincreasing": maximum_increase <= 1.0e-12,
        "dissipative_budget_max_abs_fraction": budget,
        "passed": (
            ideal_exchange_residual < 1.0e-10
            and maximum_increase <= 1.0e-12
            and budget < 2.0e-4
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dissipative", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    report = check(args.dissipative, engine=args.engine, device=args.device)
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
