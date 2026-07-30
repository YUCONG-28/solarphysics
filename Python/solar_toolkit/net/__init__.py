"""Network and archive helper namespace.

English: Small reusable helpers for collecting links, filtering archive
results, and downloading files into explicit user-selected locations.

中文：用于收集链接、筛选归档结果，并将文件下载到用户明确指定位置的轻量网络工具。
"""

from __future__ import annotations

from importlib import import_module

from .downloads import DownloadResult, download_url
from .links import collect_links, filter_links

_SUBMODULES = {
    "downloads": "solar_toolkit.net.downloads",
    "jsoc": "solar_toolkit.net.jsoc",
    "links": "solar_toolkit.net.links",
    "observations": "solar_toolkit.net.observations",
    "soar": "solar_toolkit.net.soar",
    "stereo": "solar_toolkit.net.stereo",
    "suvi": "solar_toolkit.net.suvi",
}

__all__ = [
    "DownloadResult",
    "collect_links",
    "download_url",
    "filter_links",
    *_SUBMODULES,
]


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
