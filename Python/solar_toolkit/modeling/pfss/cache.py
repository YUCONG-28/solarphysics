"""Stage keys keep display and density parameters outside magnetic geometry."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from .bundle import file_hash


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def cache_keys(config):
    """Hash scientific inputs, backend versions and stage implementation bytes."""
    from dataclasses import asdict

    from .solver import PFSSConfig

    physics = asdict(PFSSConfig(**config.get("pfss", {})))
    versions = {
        name: version(name)
        for name in [
            "numpy",
            "scipy",
            "sunpy",
            "astropy",
            "sunkit-magex",
            "streamtracer",
        ]
    }
    root = Path(__file__).parent
    code = {
        name: file_hash(root / f"{name}.py")
        for name in ["boundary", "solver", "tracing", "projection", "cache"]
    }
    boundary = _digest(
        {
            "input": file_hash(config["boundary_fits"]),
            "shape": [physics["nphi"], physics["ns"]],
            "correction": config.get("net_flux_correction", "none"),
            "code": code["boundary"],
            "versions": versions,
        }
    )
    solve = _digest(
        {
            "boundary": boundary,
            "nr": physics["nr"],
            "rss": physics["rss"],
            "code": code["solver"],
        }
    )
    trace = _digest(
        {
            "solve": solve,
            "code": code["tracing"],
            "seed_policy_version": 1,
            "event": config["event_utc"],
            "box": config.get("seed_hpc_box"),
            "local_seed_side": config.get("local_seed_side", 8),
            "step": config.get("trace_step_size", 0.25),
            "max_steps": config.get("trace_max_steps", 4096),
        }
    )
    projection = _digest(
        {
            "trace": trace,
            "aia": file_hash(config["aia_fits"]),
            "code": code["projection"],
        }
    )
    association = _digest(
        {
            "projection": projection,
            "inputs": {
                key: file_hash(config[key])
                for key in [
                    "sources_csv",
                    "jet_axis_csv",
                    "burst_matches_csv",
                    "drifts_csv",
                ]
                if config.get(key)
            },
            "window": [config.get("window_start_utc"), config.get("window_end_utc")],
            "candidate_limit": config.get("candidate_limit", 20),
        }
    )
    return dict(
        boundary=boundary,
        solve=solve,
        trace=trace,
        projection=projection,
        association=association,
    )


__all__ = ["cache_keys"]
