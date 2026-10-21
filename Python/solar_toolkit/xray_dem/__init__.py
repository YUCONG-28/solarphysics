"""Reusable X-ray and HXI numerical helpers.

English: Reusable helpers for loading SXR products, smoothing flux arrays, and
calculating finite-difference derivatives.

中文：用于加载软 X 射线产品、平滑通量数组和计算有限差分导数的可复用工具。
"""

from __future__ import annotations

from importlib import import_module

from .processing import calculate_derivative, smooth_flux_data
from .sxr import load_sxr_data

_SUBMODULES = {
    "brightness_temperature": "solar_toolkit.xray_dem.brightness_temperature",
    "hxi": "solar_toolkit.xray_dem.hxi",
    "processing": "solar_toolkit.xray_dem.processing",
    "sxr": "solar_toolkit.xray_dem.sxr",
}

__all__ = [
    "calculate_derivative",
    "load_sxr_data",
    "smooth_flux_data",
    *_SUBMODULES,
]


def __getattr__(name: str):
    target = _SUBMODULES.get(name)
    if target is not None:
        module = import_module(target)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
