"""Optional global PFSS modelling; importing this namespace needs no backend.

Positions inferred from a magnetic model are conditional, not independently
measured radio-source heights. The density model never enters the solver.
"""

from .boundary import prepare_boundary
from .bundle import load_bundle, save_bundle
from .projection import project_fieldlines
from .solver import PFSSConfig, backend_status, solve_pfss
from .tracing import trace_fieldlines, uniform_seeds

__all__ = [
    "PFSSConfig",
    "backend_status",
    "prepare_boundary",
    "solve_pfss",
    "trace_fieldlines",
    "uniform_seeds",
    "project_fieldlines",
    "load_bundle",
    "save_bundle",
]
