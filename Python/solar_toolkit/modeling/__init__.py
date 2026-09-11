"""Reusable mathematical and physical model namespace.

中文：与具体仪器工作流解耦的高斯模型和密度模型公共边界。
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "pfss": "solar_toolkit.modeling.pfss",
    "gaussian": "solar_toolkit.modeling.gaussian",
    "newkirk": "solar_toolkit.radio.newkirk",
}

__all__ = sorted(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
