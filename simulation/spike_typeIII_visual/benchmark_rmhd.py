"""CUDA RMHD correctness and synchronized timing benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import profile_config
from .physics.rmhd import solve_rmhd
from .physics.rmhd_torch import solve_rmhd_torch


def benchmark(
    profile: str = "quick",
    *,
    steps: int = 4,
    repeats: int = 5,
) -> dict[str, object]:
    config = profile_config(profile, 20260726).mhd
    config = replace(
        config,
        steps=steps,
        snapshot_stride=max(1, steps),
    )
    reference = solve_rmhd(config)
    warmup = solve_rmhd_torch(config, device="cuda", precision="float64")
    del warmup

    import torch

    timings: list[float] = []
    result = None
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        result = solve_rmhd_torch(config, device="cuda", precision="float64")
        torch.cuda.synchronize()
        timings.append(perf_counter() - started)
    assert result is not None
    denominator = max(float(np.linalg.norm(reference.psi[-1])), 1.0e-30)
    return {
        "profile": profile,
        "grid": [config.nx, config.ny],
        "steps": steps,
        "repeats": repeats,
        "timings_s": timings,
        "mean_s": float(np.mean(timings)),
        "std_s": float(np.std(timings)),
        "psi_relative_l2": float(
            np.linalg.norm(reference.psi[-1] - result.psi[-1]) / denominator
        ),
        "omega_absolute_l2": float(
            np.linalg.norm(reference.omega[-1] - result.omega[-1])
        ),
        "divergence_normalized_rms": result.divergence_rms,
        "peak_device_memory_bytes": result.peak_device_memory_bytes,
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "precision": "float64",
        "tf32": False,
        "amp": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="quick")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    text = json.dumps(
        benchmark(args.profile, steps=args.steps, repeats=args.repeats),
        indent=2,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
