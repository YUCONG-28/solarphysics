"""Verify radio-proxy ridge, strict topping, and RNG stream separation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import profile_config
from .physics.jet import diagnose_jet
from .physics.radio import synthesize_radio_proxy
from .storage import read_rmhd_hdf5


def check(hdf5: Path, seed: int = 20260726) -> dict[str, object]:
    result, mhd_config, metadata = read_rmhd_hdf5(hdf5)
    profile = str(metadata.get("profile", "cuda-fine"))
    config = profile_config(profile, seed)
    jet = diagnose_jet(result, mhd_config, config.jet)
    first = synthesize_radio_proxy(
        result,
        config.radio,
        seed,
        jet_result=jet,
        jet_config=config.jet,
        spike_coupling="jet",
    )
    resampled_config = replace(
        config.radio,
        time_samples=721,
        frequency_samples=512,
    )
    second = synthesize_radio_proxy(
        result,
        resampled_config,
        seed,
        jet_result=jet,
        jet_config=config.jet,
        spike_coupling="jet",
    )
    catalog_identical = np.array_equal(
        first.spike_catalog,
        second.spike_catalog,
    )
    ridge_monotonic = bool(np.all(np.diff(first.ridge_frequency_mhz) < 0.0))
    strict_topping = bool(
        first.event_status == "no_event"
        or np.all(first.topping_margin_mhz > 0.0)
    )
    return {
        "schema": "radio-proxy-gates-v1",
        "seed": seed,
        "primary_shape": list(first.intensity.shape),
        "resampled_shape": list(second.intensity.shape),
        "event_count": len(first.spike_catalog),
        "jet_coincidence_fraction": first.jet_coincidence_fraction,
        "catalog_identical_across_sampling": catalog_identical,
        "ridge_strictly_decreasing": ridge_monotonic,
        "minimum_topping_margin_mhz": (
            None
            if first.topping_margin_mhz.size == 0
            else float(np.min(first.topping_margin_mhz))
        ),
        "strict_topping": strict_topping,
        "passed": (
            catalog_identical
            and ridge_monotonic
            and strict_topping
            and first.event_status == "events"
            and len(first.spike_catalog) == config.radio.spike_count
            and first.jet_coincidence_fraction == 1.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = check(args.hdf5, args.seed)
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
