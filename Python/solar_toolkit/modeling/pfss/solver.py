"""Lazy public-backend adapter; no density or application dependencies."""

import math
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class PFSSConfig:
    nphi: int = 360
    ns: int = 180
    nr: int = 35
    rss: float = 2.5
    max_seeds: int = 512

    def __post_init__(self):
        for name in ("nphi", "ns", "nr", "max_seeds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or int(value) != value
                or value < (4 if name != "max_seeds" else 1)
            ):
                raise ValueError(
                    f"{name} must be a positive integer within the supported grid"
                )
        if not math.isfinite(self.rss) or self.rss <= 1:
            raise ValueError(
                "rss must be a finite heliocentric radius greater than one"
            )


def backend_status():
    packages = {}
    for name in ("sunkit-magex", "streamtracer"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"available": all(packages.values()), "packages": packages}


def solve_pfss(boundary, config=None):
    """Return the public backend Output and a numerical-run receipt."""
    config = config or PFSSConfig()
    status = backend_status()
    if not status["available"]:
        raise ImportError(
            "PFSS requires the optional sunkit-magex and streamtracer backend; existing results remain readable"
        )
    from sunkit_magex import pfss

    if boundary.data.shape != (config.ns, config.nphi):
        raise ValueError("Prepared boundary dimensions disagree with PFSSConfig")
    start = time.monotonic()
    result = pfss.pfss(pfss.Input(boundary, config.nr, config.rss))
    return result, {
        **status,
        "elapsed_s": time.monotonic() - start,
        "geometry_method": "conditional_pfss_field_line",
        "independent_geometry_valid": False,
        "input_boundary_mean_G": float(boundary.data.mean()),
        "solver_monopole_policy": "sunkit-magex excludes the monopole term, independently of input preprocessing",
    }


__all__ = ["PFSSConfig", "backend_status", "solve_pfss"]
