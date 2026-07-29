"""Compare aligned CUDA RMHD convergence runs without interpolating in time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .physics.jet import reconnection_flux_rate
from .storage import read_rmhd_hdf5


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-15)


def compare(coarser: Path, finer: Path) -> dict[str, object]:
    coarse, coarse_config, coarse_meta = read_rmhd_hdf5(coarser)
    fine, fine_config, fine_meta = read_rmhd_hdf5(finer)
    if len(coarse.times) != len(fine.times) or not np.allclose(
        coarse.times, fine.times, rtol=0.0, atol=1.0e-13
    ):
        raise ValueError("Runs must have identical saved times.")
    if (
        fine_config.nx % coarse_config.nx
        or fine_config.ny % coarse_config.ny
    ):
        raise ValueError("Fine grid must be an integer refinement.")
    step_x = fine_config.nx // coarse_config.nx
    step_y = fine_config.ny // coarse_config.ny
    fine_psi_on_coarse = fine.psi[-1, ::step_y, ::step_x]
    fine_omega_on_coarse = fine.omega[-1, ::step_y, ::step_x]
    coarse_rate = reconnection_flux_rate(coarse)
    fine_rate = reconnection_flux_rate(fine)
    coarse_rate_index = int(np.argmax(coarse_rate))
    fine_rate_index = int(np.argmax(fine_rate))
    duration = max(float(fine.times[-1] - fine.times[0]), 1.0e-15)
    metrics = {
        "peak_max_speed": _relative(
            float(np.max(coarse.max_speed)), float(np.max(fine.max_speed))
        ),
        "final_flux_difference": _relative(
            float(coarse.flux_difference[-1]),
            float(fine.flux_difference[-1]),
        ),
        "peak_reconnection_flux_rate": _relative(
            float(coarse_rate[coarse_rate_index]),
            float(fine_rate[fine_rate_index]),
        ),
        "peak_reconnection_time_fraction": abs(
            float(coarse.times[coarse_rate_index])
            - float(fine.times[fine_rate_index])
        )
        / duration,
        "peak_xpoint_electric_field": _relative(
            float(np.max(np.abs(coarse.xpoint_electric_field))),
            float(np.max(np.abs(fine.xpoint_electric_field))),
        ),
        "final_island_width_proxy": _relative(
            float(coarse.island_width_proxy[-1]),
            float(fine.island_width_proxy[-1]),
        ),
        "final_psi_l2": float(
            np.linalg.norm(coarse.psi[-1] - fine_psi_on_coarse)
            / max(float(np.linalg.norm(fine_psi_on_coarse)), 1.0e-15)
        ),
        "final_omega_l2": float(
            np.linalg.norm(coarse.omega[-1] - fine_omega_on_coarse)
            / max(float(np.linalg.norm(fine_omega_on_coarse)), 1.0e-15)
        ),
    }
    return {
        "schema": "rmhd-convergence-v1",
        "coarse": {
            "profile": coarse_meta.get("profile"),
            "grid": [coarse_config.nx, coarse_config.ny],
            "dt": coarse_config.dt,
        },
        "fine": {
            "profile": fine_meta.get("profile"),
            "grid": [fine_config.nx, fine_config.ny],
            "dt": fine_config.dt,
        },
        "relative_changes": metrics,
        "core_diagnostics_below_5_percent": all(
            metrics[name] < 0.05
            for name in (
                "peak_max_speed",
                "final_flux_difference",
                "peak_reconnection_flux_rate",
                "peak_reconnection_time_fraction",
                "final_island_width_proxy",
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = compare(args.coarse, args.fine)
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["core_diagnostics_below_5_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
