"""Physical and proxy models used by the simulation."""

from .radio import RadioResult, synthesize_radio_proxy
from .rmhd import MHDResult, SpectralGrid, solve_rmhd

__all__ = [
    "MHDResult",
    "RadioResult",
    "SpectralGrid",
    "solve_rmhd",
    "synthesize_radio_proxy",
]
