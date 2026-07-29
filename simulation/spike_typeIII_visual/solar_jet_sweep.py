"""Generate deterministic, privacy-safe Sobol case tables for v4 solar jets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import qmc

PARAMETER_RANGES = {
    "magnetic_field_gauss": (7.0, 20.0),
    "electron_density_cm3": (5.0e8, 3.0e9),
    "coronal_temperature_mk": (1.0, 2.5),
    "guide_field_ratio": (0.0, 1.0),
    "null_height_mm": (15.0, 35.0),
    "drive_converge_va": (0.01, 0.04),
    "effective_lundquist": (3000.0, 10000.0),
}


def generate_sobol_cases(
    *,
    count: int = 16,
    seed: int = 20260726,
) -> list[dict[str, float | int | str]]:
    """Return a deterministic power-of-two Sobol design."""

    if count < 2 or count & (count - 1):
        raise ValueError("Sobol case count must be a power of two >= 2.")
    names = tuple(PARAMETER_RANGES)
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    unit = sampler.random_base2(m=count.bit_length() - 1)
    lower = [PARAMETER_RANGES[name][0] for name in names]
    upper = [PARAMETER_RANGES[name][1] for name in names]
    scaled = qmc.scale(unit, lower, upper)
    return [
        {
            "case_id": f"sobol_{index + 1:02d}",
            "seed": seed,
            **{
                name: float(value)
                for name, value in zip(names, row, strict=True)
            },
        }
        for index, row in enumerate(scaled)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = {
        "schema_version": 1,
        "status": "planned",
        "cases": generate_sobol_cases(count=args.count, seed=args.seed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(payload['cases'])} planned cases to {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
