"""Top-level public API for the solar physics toolkit.

The science-domain namespaces are imported lazily so that ``import
solar_toolkit`` remains lightweight and free of workflow side effects.
"""

from __future__ import annotations

from importlib import import_module as _import_module

from ._version import __version__

__author__ = "Solar Physics Research Team"
__email__ = "solar-physics@example.com"

_SUBMODULES = {
    "aia": "solar_toolkit.aia",
    "cme": "solar_toolkit.cme",
    "coordinates": "solar_toolkit.coordinates",
    "cso": "solar_toolkit.cso",
    "data": "solar_toolkit.data",
    "exceptions": "solar_toolkit.exceptions",
    "gaussian": "solar_toolkit.gaussian",
    "hmi": "solar_toolkit.hmi",
    "io": "solar_toolkit.io",
    "map": "solar_toolkit.map",
    "modeling": "solar_toolkit.modeling",
    "net": "solar_toolkit.net",
    "radio": "solar_toolkit.radio",
    "solar_analysis_utils": "solar_toolkit.solar_analysis_utils",
    "time": "solar_toolkit.time",
    "timeseries": "solar_toolkit.timeseries",
    "visualization": "solar_toolkit.visualization",
    "xray_dem": "solar_toolkit.xray_dem",
}

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    *_SUBMODULES,
]


def __getattr__(name: str):
    """Lazily import a public top-level namespace."""

    if name in _SUBMODULES:
        module = _import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the stable public namespace advertised by this package."""

    return sorted(__all__)
