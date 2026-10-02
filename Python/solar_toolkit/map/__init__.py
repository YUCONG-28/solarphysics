"""Map-oriented helpers built around SunPy-compatible metadata.

English: Thin utilities for display extent, observation time, ROI cropping,
and normalization. They do not replace ``sunpy.map.Map``.

中文：围绕 SunPy Map 或类似 FITS header 的显示范围、观测时间、ROI 裁剪和归一化工具；
这些工具不替代 ``sunpy.map.Map``。
"""

from .image import crop_roi, normalize_image
from .metadata import get_display_extent, get_map_obs_time

__all__ = [
    "crop_roi",
    "get_display_extent",
    "get_map_obs_time",
    "normalize_image",
]


def __getattr__(name):
    """Load optional jet geometry only when the public function is requested."""
    if name == "fit_jet_axis":
        from .jet_axis import fit_jet_axis

        globals()[name] = fit_jet_axis
        return fit_jet_axis
    if name in {"fit_jet_geometry", "freeze_jet_documents"}:
        from . import jet_geometry_fit

        function = getattr(jet_geometry_fit, name)
        globals()[name] = function
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
